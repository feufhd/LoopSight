"""Autograd-preserving fusion for LoopSight's normalization paths."""

import torch


def _conditioned_rms(x, scale, gain, beta, eps):
    variance = x.square().mean(dim=-1, keepdim=True)
    normalized = (scale * x) * torch.rsqrt(variance + eps)
    return normalized * gain + beta


_compiled_conditioned_rms = torch.compile(_conditioned_rms, fullgraph=True, dynamic=False)


@torch.compile(fullgraph=True, dynamic=False)
def _compiled_residual_rms(x, branch, scale, gain, beta, eps):
    residual = x + branch.float()
    return residual, _conditioned_rms(residual, scale, gain, beta, eps)


@torch.compile(fullgraph=True, dynamic=False)
def _compiled_state_rms(x):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + 1e-6)


def conditioned_rms(x, scale, gamma, beta, eps):
    # Round the gain in the projection dtype, then promote it like eager FP32 RMSNorm.
    gain = (1 + gamma).float()
    with torch.autocast(device_type="cuda", enabled=False):
        return _compiled_conditioned_rms(x, scale, gain, beta.float(), eps)


def residual_conditioned_rms(x, branch, scale, gamma, beta, eps):
    gain = (1 + gamma).float()
    with torch.autocast(device_type="cuda", enabled=False):
        return _compiled_residual_rms(x, branch, scale, gain, beta.float(), eps)


def state_rms(x):
    with torch.autocast(device_type="cuda", enabled=False):
        return _compiled_state_rms(x)
