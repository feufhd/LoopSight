"""LoopSight with RGB patch inputs and a pooled 100-class prediction head."""

import math
from numbers import Integral

from timm.layers import PatchEmbed
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from loopsight.backbone import AsymptoticStepEmbedding, ConditionedEncoderLayer, LoopStateRMSNorm
from loopsight.gate import ContextWriteGate


class LoopSightImageNet100(nn.Module):
    """Classify a normalized RGB image by pooling the final patch memory."""

    def __init__(self, *, image_size=224, patch_size=16, embed_dim=256, core_depth=6,
                 num_heads=8, mlp_dim=512, loops=6, dropout=0.2, drop_path_rate=0.0,
                 checkpoint_blocks=False):
        super().__init__()
        for name, value in dict(image_size=image_size, patch_size=patch_size, embed_dim=embed_dim,
                                core_depth=core_depth, num_heads=num_heads, mlp_dim=mlp_dim).items():
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if image_size % patch_size:
            raise ValueError("patch_size must divide image_size")
        if embed_dim % 32:
            raise ValueError("Write gate rank D/16 must be divisible by two fusion heads")
        if any(not math.isfinite(p) or not 0 <= p < 1 for p in (dropout, drop_path_rate)):
            raise ValueError("Dropout and drop-path rates must be finite and in [0, 1)")
        self.image_size = image_size
        self.loops = self._validate_depth(loops)
        self.drop_path_rate = drop_path_rate
        self.checkpoint_blocks = checkpoint_blocks

        self.patch_embed = PatchEmbed(image_size, patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.context_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        self.core_layers = nn.ModuleList([
            ConditionedEncoderLayer(embed_dim, num_heads, mlp_dim, dropout, image_size // patch_size, 1, 64)
            for _ in range(core_depth)
        ])
        self.clock = AsymptoticStepEmbedding(embed_dim, core_depth, 8, min(3, core_depth))
        self.loop_state_norm = LoopStateRMSNorm()
        self.fusion_projection = nn.Linear(embed_dim, embed_dim)
        nn.init.eye_(self.fusion_projection.weight)
        nn.init.zeros_(self.fusion_projection.bias)
        self.fusion_gate = ContextWriteGate(embed_dim, rank=embed_dim // 16)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, 100)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.context_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _validate_depth(depth):
        if isinstance(depth, bool) or not isinstance(depth, Integral) or depth < 1:
            raise ValueError("depth must be a positive integer")
        return int(depth)

    def forward_features(self, images, depth=None):
        depth = self.loops if depth is None else self._validate_depth(depth)
        if images.shape[1:] != (3, self.image_size, self.image_size):
            raise ValueError(f"Expected [B,3,{self.image_size},{self.image_size}] RGB images")
        patches = self.patch_embed(images) + self.pos_embed
        context = self.context_token.expand(len(images), -1, -1)
        hidden = self.pos_drop(torch.cat((context, patches), dim=1))
        memory = None
        recompute = self.checkpoint_blocks and self.training and torch.is_grad_enabled()
        for step in range(1, depth + 1):
            conditions = self.clock(step)
            for index, (layer, condition) in enumerate(zip(self.core_layers, conditions)):
                rate = self.drop_path_rate * index / max(1, len(self.core_layers) - 1)
                if recompute:
                    hidden = checkpoint(layer, hidden, use_reentrant=False,
                                        condition=condition, drop_path_rate=rate)
                else:
                    hidden = layer(hidden, condition=condition, drop_path_rate=rate)
            hidden = self.loop_state_norm(hidden)
            current = self.fusion_projection(hidden).float()[:, 1:]
            alpha = self.fusion_gate(current, memory, hidden[:, 0].float(), step)
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                memory = current * alpha if memory is None else memory + alpha * (current - memory)
        return memory

    def forward(self, images, depth=None):
        memory = self.forward_features(images, depth)
        return self.head(self.norm(memory.mean(dim=1)))
