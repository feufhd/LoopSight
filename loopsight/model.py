"""LoopSight with a shared recurrent core and a gated prediction memory."""

from dataclasses import dataclass
import math
from numbers import Integral

import torch
from timm.layers import PatchEmbed
from torch import nn

from .backbone import AsymptoticStepEmbedding, ConditionedEncoderLayer, LoopStateRMSNorm
from .gate import ContextWriteGate


@dataclass
class LoopForwardMetadata:
    gate_probs: list[torch.Tensor]
    max_steps: int
    exit_steps: torch.Tensor


class LoopSight(nn.Module):
    """Decode pixel logits from the prediction memory at the final loop."""

    def __init__(
        self,
        num_tasks: int,
        *,
        image_size: int = 64,
        patch_size: int = 2,
        num_colors: int = 12,
        num_task_tokens: int = 1,
        embed_dim: int = 256,
        loop_core_depth: int = 6,
        max_loop_steps: int = 12,
        num_heads: int = 8,
        mlp_dim: int = 512,
        dropout: float = 0.1,
        asymptotic_channels: int = 8,
        local_clock_frequencies: int = 3,
        modulation_hidden_dim: int = 64,
        fusion_gate_rank: int = 16,
        fusion_gate_heads: int = 2,
        fusion_gate_logit_bound: float = math.log(16),
        loop_state_rmsnorm: bool = True,
        loop_drop_path_rate: float = 0.0,
        loop_drop_path_depth: int = 6,
    ):
        super().__init__()
        if image_size < 1 or patch_size < 1 or image_size % patch_size:
            raise ValueError("image_size must be a positive multiple of patch_size")
        if min(num_tasks, num_colors, num_task_tokens, loop_core_depth) < 1:
            raise ValueError("Task, color, task-token and core-layer counts must be positive")
        if isinstance(max_loop_steps, bool) or not isinstance(max_loop_steps, Integral) or max_loop_steps < 1:
            raise ValueError("max_loop_steps must be a positive integer")
        if not math.isfinite(loop_drop_path_rate) or not 0 <= loop_drop_path_rate < 1:
            raise ValueError("loop_drop_path_rate must be finite and in [0, 1)")
        if (isinstance(loop_drop_path_depth, bool) or not isinstance(loop_drop_path_depth, Integral)
                or loop_drop_path_depth < 2):
            raise ValueError("loop_drop_path_depth must be an integer >= 2")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_colors = num_colors
        self.embed_dim = embed_dim
        self.num_task_tokens = num_task_tokens
        self.max_loop_steps = int(max_loop_steps)
        self.loop_core_depth = loop_core_depth
        self.fusion_mode = "task_lowrank_channel"
        self.fusion_gate_rank = fusion_gate_rank
        self.fusion_gate_heads = fusion_gate_heads
        self.fusion_gate_logit_bound = fusion_gate_logit_bound
        self.loop_drop_path_rate = float(loop_drop_path_rate)
        self.loop_drop_path_depth = int(loop_drop_path_depth)

        self.color_embed = nn.Embedding(num_colors, embed_dim)
        self.task_token_embed = nn.Embedding(num_tasks, num_task_tokens * embed_dim)
        self.patch_embed = PatchEmbed(image_size, patch_size, embed_dim, embed_dim, bias=True)
        self.positional_embed = nn.Parameter(torch.zeros(1, self.grid_size ** 2, embed_dim))
        self.core_layers = nn.ModuleList([
            ConditionedEncoderLayer(
                embed_dim, num_heads, mlp_dim, dropout, self.grid_size,
                num_task_tokens, modulation_hidden_dim,
            )
            for _ in range(loop_core_depth)
        ])
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_colors * patch_size ** 2)
        nn.init.trunc_normal_(self.positional_embed, std=0.02)
        nn.init.trunc_normal_(self.task_token_embed.weight, std=0.02)
        nn.init.trunc_normal_(self.color_embed.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

        self.loop_state_norm = LoopStateRMSNorm() if loop_state_rmsnorm else nn.Identity()
        self.asymptotic_clock = AsymptoticStepEmbedding(
            embed_dim, loop_core_depth, asymptotic_channels, local_clock_frequencies,
        )
        self.fusion_projection = nn.Linear(embed_dim, embed_dim)
        nn.init.eye_(self.fusion_projection.weight)
        nn.init.zeros_(self.fusion_projection.bias)
        with torch.random.fork_rng(devices=[]):
            self.fusion_gate = ContextWriteGate(
                embed_dim, fusion_gate_rank, fusion_gate_logit_bound, fusion_gate_heads,
            )

    def _encode_inputs(self, pixel_values, task_ids):
        if pixel_values.ndim != 3 or pixel_values.shape[1:] != (self.image_size, self.image_size):
            raise ValueError("pixel_values must have shape [B, image_size, image_size]")
        batch = pixel_values.size(0)
        image = self.color_embed(pixel_values.long()).permute(0, 3, 1, 2)
        image = self.patch_embed(image) + self.positional_embed
        task = self.task_token_embed(task_ids.long()).reshape(batch, self.num_task_tokens, -1)
        return self.dropout(torch.cat((task, image), dim=1))

    def _prepare_attention_mask(self, attention_mask, pixel_values, device):
        if attention_mask is None:
            return None
        if attention_mask.shape != pixel_values.shape:
            raise ValueError("attention_mask must match pixel_values")
        batch = pixel_values.size(0)
        patches = attention_mask.reshape(
            batch, self.grid_size, self.patch_size, self.grid_size, self.patch_size,
        )
        patch_validity = patches.max(dim=2).values.max(dim=3).values
        padding = ~patch_validity.flatten(1).bool()
        task_padding = torch.zeros(batch, self.num_task_tokens, device=device, dtype=torch.bool)
        return torch.cat((task_padding, padding), dim=1)

    def _apply_loop_drop_path(self, loop_input, loop_output, step):
        rate = self.loop_drop_path_rate * (step - 1) / (self.loop_drop_path_depth - 1)
        if not self.training or rate == 0:
            return loop_output
        if rate >= 1:
            raise ValueError("Scheduled loop drop path probability must remain below 1 during training")
        keep = 1.0 - rate
        mask = loop_output.new_empty((loop_output.size(0), 1, 1)).bernoulli_(keep)
        return loop_input + (loop_output - loop_input) * (mask / keep)

    def _project_to_logits(self, hidden_states):
        image = self.norm(hidden_states)[:, self.num_task_tokens:]
        logits = self.head(image).reshape(
            -1, self.grid_size, self.grid_size, self.patch_size, self.patch_size, self.num_colors,
        )
        logits = logits.permute(0, 1, 3, 2, 4, 5)
        logits = logits.reshape(hidden_states.size(0), self.image_size, self.image_size, self.num_colors)
        return logits.permute(0, 3, 1, 2)

    def forward(
        self, pixel_values, task_ids, attention_mask=None, *, override_max_steps=None,
        return_intermediates=False, return_raw_intermediates=False, anchor_step=None,
        dynamic_exit=None, gate_threshold=None,
    ):
        """Return final pixel logits and loop metadata, optionally with earlier readouts."""
        if dynamic_exit:
            raise ValueError("LoopSight requires a fixed runtime loop budget")
        steps = self.max_loop_steps if override_max_steps is None else override_max_steps
        if isinstance(steps, bool) or not isinstance(steps, Integral) or steps < 1:
            raise ValueError("loop count must be a positive integer")
        if anchor_step is not None:
            if (isinstance(anchor_step, bool) or not isinstance(anchor_step, Integral)
                    or not 1 <= anchor_step < steps):
                raise ValueError("anchor_step must satisfy 1 <= anchor_step < loop count")
            if return_intermediates or return_raw_intermediates:
                raise ValueError("anchor_step cannot be combined with intermediate-output flags")

        hidden = self._encode_inputs(pixel_values, task_ids)
        key_padding_mask = self._prepare_attention_mask(attention_mask, pixel_values, hidden.device)
        patch_mask = None if key_padding_mask is None else key_padding_mask[:, self.num_task_tokens:]
        memory = anchor_logits = None
        intermediate_logits, raw_logits = [], []

        for step in range(1, int(steps) + 1):
            loop_input = hidden
            conditions = self.asymptotic_clock(step)
            for layer, condition in zip(self.core_layers, conditions):
                hidden = layer(hidden, key_padding_mask=key_padding_mask, condition=condition)
            hidden = self.loop_state_norm(self._apply_loop_drop_path(loop_input, hidden, step))

            projected = self.fusion_projection(hidden).float()
            current = projected[:, self.num_task_tokens:]
            task = hidden[:, :self.num_task_tokens].float().mean(dim=1)
            alpha = self.fusion_gate(current, memory, task, step, patch_mask)
            # Only hidden feeds the next core iteration. Memory is for the readout.
            with torch.autocast(device_type=hidden.device.type, enabled=False):
                memory = current * alpha if memory is None else memory + alpha * (current - memory)

            if return_intermediates or step in (anchor_step, steps):
                readout = torch.cat((projected[:, :self.num_task_tokens], memory), dim=1)
                logits = self._project_to_logits(readout)
                if return_intermediates:
                    intermediate_logits.append(logits)
                if step == anchor_step:
                    anchor_logits = logits
            if return_raw_intermediates:
                raw_logits.append(self._project_to_logits(projected))

        metadata = LoopForwardMetadata(
            gate_probs=[], max_steps=int(steps),
            exit_steps=torch.full((hidden.size(0),), int(steps), dtype=torch.long, device=hidden.device),
        )
        if anchor_step is not None:
            return logits, metadata, anchor_logits
        if return_raw_intermediates:
            return logits, metadata, intermediate_logits, raw_logits
        if return_intermediates:
            return logits, metadata, intermediate_logits
        return logits, metadata
