"""Shared transformer blocks, spatial rotary embeddings, and the loop clock."""

import math

import torch
from torch import nn
from torch.nn import functional as F


def drop_path(x, drop_prob, training):
    """Drop a residual branch per sample, shared across tokens and channels."""
    if not training or drop_prob == 0.0:
        return x
    keep_prob = 1.0 - drop_prob
    mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = (keep_prob + torch.rand(mask_shape, dtype=x.dtype, device=x.device)).floor_()
    return x * mask / keep_prob


class RotaryEmbedding2D(nn.Module):
    """Apply row and column rotary embeddings adapted from EVA-02."""

    def __init__(self, head_dim, grid_size, num_task_tokens):
        super().__init__()
        if head_dim % 4:
            raise ValueError("2D rotary embeddings require head_dim to be divisible by four")
        dim = head_dim // 2
        frequencies = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(grid_size) / grid_size * grid_size
        angles = torch.einsum("..., f -> ... f", positions, frequencies).repeat_interleave(2, dim=-1)
        row_angles = angles[:, None, :].expand(grid_size, grid_size, dim)
        column_angles = angles[None, :, :].expand(grid_size, grid_size, dim)
        angles = torch.cat((row_angles, column_angles), dim=-1)
        self.register_buffer("freqs_cos", angles.cos().view(-1, head_dim))
        self.register_buffer("freqs_sin", angles.sin().view(-1, head_dim))
        self.num_task_tokens = num_task_tokens

    def forward(self, x):
        image = x[:, :, self.num_task_tokens:]
        even, odd = image.unflatten(-1, (-1, 2)).unbind(dim=-1)
        rotated = torch.stack((-odd, even), dim=-1).flatten(-2)
        image = image * self.freqs_cos + rotated * self.freqs_sin
        return torch.cat((x[:, :, :self.num_task_tokens], image), dim=2)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        return self.scale * x * torch.rsqrt(norm_x + self.eps)


class LoopStateRMSNorm(nn.Module):
    """Normalize the recurrent state in FP32 without learned parameters."""

    def forward(self, hidden):
        with torch.autocast(device_type=hidden.device.type, enabled=False):
            state = hidden.float()
            return state * torch.rsqrt(state.square().mean(dim=-1, keepdim=True) + 1e-6)


class ConvolutionalGLU(nn.Module):
    """Apply depthwise convolution to the image-token portion of the gate."""

    def __init__(self, embed_dim, mlp_dim, dropout, num_task_tokens):
        super().__init__()
        hidden_dim = int(2 * mlp_dim / 3)
        self.num_task_tokens = num_task_tokens
        self.fc1 = nn.Linear(embed_dim, hidden_dim * 2)
        self.dwconv = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, grid_size):
        gate, value = self.fc1(x).chunk(2, dim=-1)
        task_gate = gate[:, :self.num_task_tokens]
        image_gate = gate[:, self.num_task_tokens:]
        batch, _, channels = image_gate.shape
        image_gate = image_gate.transpose(1, 2).reshape(batch, channels, grid_size, grid_size)
        image_gate = self.dwconv(image_gate).flatten(2).transpose(1, 2)
        gate = torch.cat((task_gate, image_gate), dim=1)
        return self.drop(self.fc2(self.drop(self.act(gate) * value)))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, grid_size, dropout, num_task_tokens):
        super().__init__()
        if num_heads < 1 or embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by a positive num_heads")
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)
        self.rotary = RotaryEmbedding2D(self.head_dim, grid_size, num_task_tokens)

    def forward(self, x, key_padding_mask=None):
        batch, seq_len, embed_dim = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        q, k = self.rotary(q), self.rotary(k)
        attn_mask = None if key_padding_mask is None else ~key_padding_mask[:, None, None, :]
        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
        )
        context = context.transpose(1, 2).reshape(batch, seq_len, embed_dim)
        return self.proj_dropout(self.proj(context))


class AsymptoticStepEmbedding(nn.Module):
    """Combine a decaying loop clock with a fixed within-loop layer clock."""

    def __init__(self, embed_dim, core_depth, channels=8, local_frequencies=3):
        super().__init__()
        if channels < 1 or core_depth < 1 or not 1 <= local_frequencies <= core_depth:
            raise ValueError("Positive clock dimensions and local frequencies <= core depth are required")
        completion_steps = torch.linspace(6.0, 12.0, channels)
        initial_decay = math.log(100.0) / completion_steps
        self.decay_logits = nn.Parameter(torch.log(torch.expm1(initial_decay)))
        self.limit = nn.Parameter(torch.zeros(embed_dim))
        self.loop_projection = nn.Linear(channels, embed_dim, bias=False)
        self.local_projection = nn.Linear(local_frequencies, embed_dim, bias=False)
        positions = (torch.arange(core_depth, dtype=torch.float32) + 0.5) / core_depth
        frequencies = torch.arange(1, local_frequencies + 1, dtype=torch.float32)
        self.register_buffer("local_clock", torch.cos(math.pi * positions[:, None] * frequencies))
        nn.init.normal_(self.loop_projection.weight, std=0.02)
        nn.init.normal_(self.local_projection.weight, std=0.02)

    def forward(self, step):
        with torch.autocast(device_type=self.decay_logits.device.type, enabled=False):
            decay = F.softplus(self.decay_logits.float())
            geometric = -torch.expm1(-decay) * torch.exp(-decay * (step - 1))
        outer = self.limit + self.loop_projection(geometric)
        return outer.unsqueeze(0) + self.local_projection(self.local_clock)

class ConditionedEncoderLayer(nn.Module):
    """Condition both pre-normalizations on the loop and layer clocks."""

    def __init__(self, embed_dim, num_heads, mlp_dim, dropout, grid_size,
                 num_task_tokens, modulation_hidden_dim):
        super().__init__()
        self.grid_size = grid_size
        self.self_attn = MultiHeadSelfAttention(embed_dim, num_heads, grid_size, dropout, num_task_tokens)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = RMSNorm(embed_dim)
        self.mlp = ConvolutionalGLU(embed_dim, mlp_dim, dropout, num_task_tokens)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.time_modulation = nn.Sequential(
            nn.Linear(embed_dim, modulation_hidden_dim), nn.SiLU(),
            nn.Linear(modulation_hidden_dim, 4 * embed_dim),
        )
        nn.init.zeros_(self.time_modulation[-1].weight)
        nn.init.zeros_(self.time_modulation[-1].bias)

    def forward(self, x, key_padding_mask=None, drop_path_rate=0.0, *, condition, fuse_norm=False):
        gamma_attn, beta_attn, gamma_ffn, beta_ffn = self.time_modulation(condition).chunk(4, dim=-1)
        if fuse_norm:
            from ._training_fusion import conditioned_rms, residual_conditioned_rms

            normalized = conditioned_rms(x, self.norm1.scale, gamma_attn, beta_attn, self.norm1.eps)
        else:
            normalized = self.norm1(x) * (1 + gamma_attn) + beta_attn

        attention = self.self_attn(normalized, key_padding_mask=key_padding_mask)
        branch = drop_path(self.dropout1(attention), drop_path_rate, self.training)
        if fuse_norm:
            residual, normalized = residual_conditioned_rms(
                x, branch, self.norm2.scale, gamma_ffn, beta_ffn, self.norm2.eps,
            )
        else:
            residual = x + branch
            normalized = self.norm2(residual) * (1 + gamma_ffn) + beta_ffn

        output = self.mlp(normalized, grid_size=self.grid_size)
        return residual + drop_path(self.dropout2(output), drop_path_rate, self.training)
