"""Automatic device selection and portable random-state handling."""

from contextlib import contextmanager
import os

import torch


def resolve_device(device="auto"):
    if device is not None and str(device) != "auto":
        return torch.device(device)
    selected = torch.accelerator.current_accelerator(check_available=True)
    if selected is not None and int(os.environ.get("WORLD_SIZE", "1")) > 1:
        index = int(os.environ.get("LOCAL_RANK", "0"))
        if index >= torch.accelerator.device_count():
            raise ValueError("The process count exceeds the number of visible devices")
        torch.accelerator.set_device_index(index)
        return torch.device(selected.type, index)
    return selected if selected is not None else torch.device("cpu")


def capture_device_rng(device="auto"):
    selected = resolve_device(device)
    if selected.type == "cpu":
        return None
    backend = getattr(torch, selected.type)
    return backend.get_rng_state(selected).cpu()


def restore_device_rng(state, device="auto"):
    selected = resolve_device(device)
    if state is not None and selected.type != "cpu":
        getattr(torch, selected.type).set_rng_state(state.cpu(), selected)


def synchronize(device="auto"):
    selected = resolve_device(device)
    if selected.type != "cpu":
        torch.accelerator.synchronize(selected)


@contextmanager
def preserve_rng(device="auto"):
    selected = resolve_device(device)
    host_state = torch.get_rng_state()
    device_state = capture_device_rng(selected)
    try:
        yield
    finally:
        torch.set_rng_state(host_state)
        restore_device_rng(device_state, selected)
