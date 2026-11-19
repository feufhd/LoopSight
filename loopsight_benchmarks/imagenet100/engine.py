"""Training utilities with epoch-boundary resume and explicit final EMA exports."""
import copy
import json
import math
from pathlib import Path
import random

import numpy as np
import torch
from loopsight.runtime import capture_device_rng, restore_device_rng
from utils.distribution import setup_execution
import torch.distributed as dist


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def unwrap(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in unwrap(model).state_dict().items()}


def save_checkpoint(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".partial")
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class EMA:
    def __init__(self, model, decay):
        self.module = copy.deepcopy(unwrap(model)).eval().requires_grad_(False)
        self.decay, self.step = decay, 0

    @torch.no_grad()
    def update(self, model, step):
        for target, source in zip(self.module.parameters(), unwrap(model).parameters()):
            target.lerp_(source.detach(), 1 - self.decay)
        for target, source in zip(self.module.buffers(), unwrap(model).buffers()):
            target.copy_(source)
        self.step = step

    def export_final(self, path, config, step, epoch):
        if self.step != step:
            raise ValueError("EMA does not contain the final optimizer update")
        state = cpu_state(self.module)
        save_checkpoint(path, dict(model_config=config.to_dict(), model=state,
            weights="final_step_ema", ema_decay=self.decay, step=step, epoch=epoch))
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if loaded["step"] != step or any(not torch.equal(state[k], loaded["model"][k]) for k in state):
            raise ValueError("Final EMA reload verification failed")
        self.module.load_state_dict(loaded["model"], strict=True)


def capture_rng(device):
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch=torch.get_rng_state(), device=capture_device_rng(device))


def restore_rng(state, device):
    random.setstate(state["python"])
    rng = state["numpy"]
    np.random.set_state((rng[0], np.asarray(rng[1], dtype=np.uint32), *rng[2:]))
    torch.set_rng_state(state["torch"].cpu())
    restore_device_rng(state["device"], device)


def gather_rng(device, world):
    local = capture_rng(device)
    states = [None] * world
    if dist.is_initialized():
        dist.all_gather_object(states, local)
    else:
        states[0] = local
    return states


def optimizer_groups(model, weight_decay):
    decay, no_decay = [], []
    for name, value in model.named_parameters():
        if value.requires_grad:
            excluded = value.ndim <= 1 or any(s in name for s in ("pos_embed", "context_token", "cls_token", "step_embed"))
            (no_decay if excluded else decay).append(value)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def learning_rate(update, total, warmup, peak, minimum):
    if warmup and update < warmup:
        return peak * (update + 1) / warmup
    progress = min(1.0, max(0.0, (update - warmup) / max(1, total - warmup - 1)))
    return minimum + (peak - minimum) * (1 + math.cos(math.pi * progress)) / 2


def accumulation_window(index, batches, accumulation):
    start = index // accumulation * accumulation
    return min(accumulation, batches - start)


def compatible_resume(saved, current):
    # Training horizon, optimizer, data and effective-batch settings stay fixed.
    saved = dict(saved)
    selection = saved.get("dataset_selection")
    if selection is not None:
        saved["dataset_selection"] = {key: value for key, value in selection.items()
                                      if key in current.get("dataset_selection", {})}
    allowed = {"resume", "output", "workers", "log_every", "device", "dry_run", "cpu_threads"}
    changed = {key: (saved.get(key), current.get(key)) for key in saved.keys() | current.keys()
               if key not in allowed and saved.get(key) != current.get(key)}
    if changed:
        raise ValueError(f"Resume recipe differs: {changed}")
