"""A reproducible discrete Gaussian budget shared by an entire DDP update."""

import torch
import torch.distributed as dist


class LoopDepthSampler:
    def __init__(self, minimum=6, maximum=18, mean=12.0, std=3.0, seed=42):
        self.depths = torch.arange(minimum, maximum + 1, dtype=torch.long)
        self.probabilities = torch.softmax(-0.5 * ((self.depths.double() - mean) / std).square(), dim=0)
        self.seed = seed

    def sample_update(self, update_index, device, distributed=False):
        # Index-based private RNG also reproduces resumed runs without consuming
        # the model/dropout RNG. All accumulation micro-batches reuse this draw.
        shared = distributed and dist.is_initialized()
        budget = torch.zeros((), dtype=torch.long, device=device)
        if not shared or dist.get_rank() == 0:
            generator = torch.Generator().manual_seed((self.seed + 104729 * (update_index + 1)) % (2**63 - 1))
            index = torch.multinomial(self.probabilities, 1, generator=generator).item()
            budget.fill_(int(self.depths[index]))
        if shared:
            dist.broadcast(budget, src=0)
        return int(budget.item())

    def config(self):
        return {int(k): float(p) for k, p in zip(self.depths, self.probabilities)}
