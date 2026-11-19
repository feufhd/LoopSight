"""Cross-attention write gate for LoopSight's prediction memory."""

import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


class ContextWriteGate(nn.Module):
    """Compute a write coefficient for every output patch and channel."""

    def __init__(self, embed_dim, rank=16, logit_bound=math.log(16), num_heads=2):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, Integral) or rank < 1:
            raise ValueError("fusion_gate_rank must be a positive integer")
        if not math.isfinite(logit_bound) or logit_bound < 0:
            raise ValueError("fusion_gate_logit_bound must be finite and nonnegative")
        self.rank = int(rank)
        self.logit_bound = float(logit_bound)
        self.down = nn.Linear(3 * embed_dim, self.rank, bias=False)
        self.activation = nn.SiLU()
        self.up = nn.Linear(self.rank, embed_dim, bias=True)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

        if isinstance(num_heads, bool) or not isinstance(num_heads, Integral) or num_heads < 1:
            raise ValueError("fusion_gate_heads must be a positive integer")
        if self.rank % num_heads:
            raise ValueError("fusion_gate_rank must be divisible by fusion_gate_heads")
        self.num_heads = int(num_heads)
        self.query = nn.Linear(self.rank, self.rank, bias=True)
        self.key = nn.Linear(self.rank, self.rank, bias=True)
        self.value = nn.Linear(self.rank, self.rank, bias=True)
        self.output = nn.Linear(self.rank, self.rank, bias=True)

    @staticmethod
    def _normalize(state):
        state = state.float()
        return state * torch.rsqrt(state.square().mean(dim=-1, keepdim=True) + 1e-6)

    @staticmethod
    def _linear(state, layer):
        return F.linear(state, layer.weight.float(), layer.bias.float())

    def forward(self, current, previous, task, step, padding_mask=None):
        """Read all previous memory patches and the current post-core task state."""
        if isinstance(step, bool) or not isinstance(step, Integral) or step < 1:
            raise ValueError("The write gate requires a positive one-based loop index")
        if step > 1 and previous is None:
            raise ValueError("A later loop requires the previous prediction memory")
        if current.ndim != 3 or task.shape != (current.shape[0], current.shape[-1]):
            raise ValueError("Expected current [B, P, D] and task [B, D]")
        if previous is not None and previous.shape != current.shape:
            raise ValueError("Current and previous memory patches must have identical shapes")
        if padding_mask is not None and padding_mask.shape != current.shape[:2]:
            raise ValueError("Expected patch padding mask [B, P]")

        with torch.autocast(device_type=current.device.type, enabled=False):
            previous = current if previous is None else previous
            p, f, t = map(self._normalize, (current, previous, task))
            batch, patches, width = p.shape
            wp, wf, wt = self.down.weight.float().split(width, dim=1)
            p_low = F.linear(p, wp)
            f_low = F.linear(f, wf)
            t_low = F.linear(t.unsqueeze(1), wt)
            local = p_low + f_low + t_low

            context = self._normalize(torch.cat((f_low, t_low), dim=1))
            q = self._linear(self._normalize(p_low), self.query)
            k = self._linear(context, self.key)
            v = self._linear(context, self.value)
            head_dim = self.rank // self.num_heads
            q = q.reshape(batch, patches, self.num_heads, head_dim).transpose(1, 2)
            k = k.reshape(batch, patches + 1, self.num_heads, head_dim).transpose(1, 2)
            v = v.reshape(batch, patches + 1, self.num_heads, head_dim).transpose(1, 2)
            # Every output position is valid memory context, including input padding.
            attended = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
            attended = attended.transpose(1, 2).reshape(batch, patches, self.rank)
            mixed = local + self._linear(attended, self.output)
            score = F.linear(self.activation(mixed), self.up.weight.float(), self.up.bias.float())

            if step == 1:
                # Keep the gate in the gradient graph when this is the only loop.
                return torch.ones_like(score) + score * 0.0
            return torch.sigmoid(-math.log(step - 1) + self.logit_bound * torch.tanh(score))
