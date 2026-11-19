"""LoopSight with grayscale board input and grayscale pixel reconstruction."""

import math
from numbers import Integral

from timm.layers import PatchEmbed
import torch
from torch import nn

from loopsight.backbone import AsymptoticStepEmbedding, ConditionedEncoderLayer, LoopStateRMSNorm
from loopsight.gate import ContextWriteGate


class _GrayscaleCore(nn.Module):
    def __init__(self, image_size, patch_size, embed_dim, core_depth, heads,
                 mlp_dim, dropout, fusion_heads, fusion_logit_bound):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.task_token_embed = nn.Embedding(1, embed_dim)
        self.patch_embed = PatchEmbed(image_size, patch_size, 1, embed_dim, bias=True)
        self.positional_embed = nn.Parameter(torch.zeros(1, self.grid_size ** 2, embed_dim))
        self.core_layers = nn.ModuleList([
            ConditionedEncoderLayer(embed_dim, heads, mlp_dim, dropout, self.grid_size, 1, 64)
            for _ in range(core_depth)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, patch_size ** 2)
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        self.loop_state_norm = LoopStateRMSNorm()
        self.asymptotic_clock = AsymptoticStepEmbedding(embed_dim, core_depth, 8, min(3, core_depth))
        self.fusion_projection = nn.Linear(embed_dim, embed_dim)
        nn.init.eye_(self.fusion_projection.weight)
        nn.init.zeros_(self.fusion_projection.bias)
        with torch.random.fork_rng(devices=[]):
            self.fusion_gate = ContextWriteGate(embed_dim, embed_dim // 16, fusion_logit_bound, fusion_heads)

    def _decode(self, projected_task, memory):
        patches = self.norm(torch.cat((projected_task, memory), dim=1))[:, 1:]
        pixels = self.head(patches).reshape(
            -1, self.grid_size, self.grid_size, self.patch_size, self.patch_size,
        )
        return pixels.permute(0, 1, 3, 2, 4).reshape(-1, 1, self.image_size, self.image_size)

    def forward(self, images, depth, return_intermediates):
        if images.shape[1:] != (1, self.image_size, self.image_size) or not images.is_floating_point():
            raise ValueError(f"Expected floating-point [B,1,{self.image_size},{self.image_size}] images")
        patches = self.patch_embed(images) + self.positional_embed
        task_ids = torch.zeros(len(images), dtype=torch.long, device=images.device)
        task = self.task_token_embed(task_ids).unsqueeze(1)
        hidden = self.dropout(torch.cat((task, patches), dim=1))
        memory = None
        intermediates = []
        for step in range(1, depth + 1):
            conditions = self.asymptotic_clock(step)
            for layer, condition in zip(self.core_layers, conditions):
                hidden = layer(hidden, condition=condition)
            hidden = self.loop_state_norm(hidden)

            projected = self.fusion_projection(hidden).float()
            current = projected[:, 1:]
            alpha = self.fusion_gate(current, memory, hidden[:, 0].float(), step)
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                memory = current * alpha if memory is None else memory + alpha * (current - memory)
            if return_intermediates or step == depth:
                pixels = self._decode(projected[:, :1], memory)
                if return_intermediates:
                    intermediates.append(pixels)
        return (pixels, intermediates) if return_intermediates else pixels


class LoopSightVisualSudoku(nn.Module):
    """Predict a completed grayscale Sudoku board from a partial board image."""

    def __init__(self, *, image_size=252, patch_size=14, embed_dim=256, core_depth=8,
                 heads=8, mlp_dim=512, dropout=0.3, depth=12,
                 fusion_heads=2, fusion_logit_bound=math.log(16)):
        super().__init__()
        if image_size not in (252, 256) or patch_size < 1 or image_size % patch_size:
            raise ValueError("Use image_size 252 or 256 and a positive patch_size dividing it")
        if embed_dim < 16 or embed_dim % 16:
            raise ValueError("embed_dim must be a positive multiple of 16 for rank D/16")
        if min(core_depth, mlp_dim) < 1 or not 0 <= dropout < 1:
            raise ValueError("Invalid core depth, MLP width or dropout")
        self.depth = self._validate_depth(depth)
        self.core = _GrayscaleCore(image_size, patch_size, embed_dim, core_depth, heads,
                                   mlp_dim, dropout, fusion_heads, fusion_logit_bound)

    @staticmethod
    def _validate_depth(depth):
        if isinstance(depth, bool) or not isinstance(depth, Integral) or depth < 1:
            raise ValueError("depth must be a positive integer")
        return int(depth)

    def forward(self, images, depth=None, *, return_intermediates=False):
        depth = self.depth if depth is None else self._validate_depth(depth)
        return self.core(images, depth, return_intermediates)
