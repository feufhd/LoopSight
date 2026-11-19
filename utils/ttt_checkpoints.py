"""Persist complete adapted weights, including the new task-token vocabulary."""

from datetime import datetime, timezone
from pathlib import Path
import os

import torch


def save_ttt_checkpoint(model, args, attempt_idx, *, task_lookup,
                        weight_type='raw', ema=None):
    task = Path(args.train_split).name
    directory = Path(args.ttt_checkpoint_dir)
    if weight_type != 'raw':
        directory = directory.with_name(directory.name + '_' + weight_type)
    destination = directory / task / f"attempt_{attempt_idx}.pt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    module = model
    while hasattr(module, "module") or hasattr(module, "_orig_mod"):
        module = module.module if hasattr(module, "module") else module._orig_mod
    checkpoint = {
        "format_version": 1,
        "model_state": {key: value.detach().cpu().clone()
                        for key, value in module.state_dict().items()},
        "config": dict(vars(args)),
        "task_lookup": dict(task_lookup),
        "num_tasks": len(task_lookup),
        "task": task,
        "attempt_idx": attempt_idx,
        "epoch": args.epochs,
        "source_checkpoint": args.resume_checkpoint,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "restore_task_tokens": True,
        "weight_type": weight_type,
    }
    if ema is not None:
        checkpoint.update(ema_decay=ema.decay, ema_num_updates=ema.num_updates)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination
