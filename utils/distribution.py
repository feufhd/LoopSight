"""Ordinary execution or one distributed worker per device with torchrun."""
import os

import torch.distributed as dist

from loopsight.runtime import resolve_device


def setup_execution(device="auto"):
    selected = resolve_device(device)
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=dist.get_default_backend_for_device(selected),
            init_method="env://", rank=rank, world_size=world,
        )
    return selected, rank, world


def init_distributed_mode(args):
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if getattr(args, "independent_process", False):
        args.distributed = False
        return False, 0, 1, local_rank, resolve_device()
    device, rank, world = setup_execution(getattr(args, "device", "auto"))
    args.distributed = world > 1
    return args.distributed, rank, world, local_rank, device
