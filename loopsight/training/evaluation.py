"""ARC pixel and whole-grid validation."""
from __future__ import annotations
import random
from itertools import islice
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader
from loopsight.data.arc import IGNORE_INDEX
from loopsight.runtime import preserve_rng

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def evaluate(model, loader: DataLoader, device: torch.device, args, *, distributed: bool = False) -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_pixels = 0
    total_exact = 0
    total_examples = 0
    avg_steps_accumulator = 0.0

    dataset = getattr(loader, "dataset", None)
    if dataset is not None:
        dataset.disable_translation()
        dataset.disable_resolution_augmentation(fix_scale_factor=args.fix_scale_factor)

    with torch.no_grad():
        for batch in islice(loader, getattr(args, "max_eval_batches", None)):
            task_ids_cpu = batch["task_ids"]
            inputs = batch["inputs"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            task_ids = task_ids_cpu.to(device)

            logits, metadata = model(
                inputs,
                task_ids,
                attention_mask=attention_mask,
                dynamic_exit=args.eval_dynamic_exit,
                gate_threshold=args.exit_gate_threshold,
                override_max_steps=getattr(args, "eval_loop_steps", None),
            )

            num_colors = logits.size(1)
            logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, num_colors)
            loss = F.cross_entropy(
                logits_flat,
                targets.view(-1),
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )
            total_loss += loss.item()
            total_pixels += (targets != IGNORE_INDEX).sum().item()

            predictions = logits.argmax(dim=1)
            batch_size = predictions.size(0)
            avg_steps_accumulator += metadata.exit_steps.float().sum().item()
            for idx in range(batch_size):
                target = targets[idx]
                prediction = predictions[idx]
                valid = target != IGNORE_INDEX
                is_exact = bool(torch.equal(prediction[valid], target[valid])) if valid.any() else False
                total_exact += int(is_exact)
                total_examples += 1
    if distributed and dist.is_initialized():
        totals = torch.tensor(
            [total_loss, total_pixels, total_exact, total_examples, avg_steps_accumulator],
            dtype=torch.float64,
            device=device,
        )
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total_pixels, total_exact, total_examples, avg_steps_accumulator = totals.tolist()

    avg_loss = total_loss / max(total_pixels, 1)
    accuracy = total_exact / max(total_examples, 1)
    mean_steps = avg_steps_accumulator / max(total_examples, 1)

    if dataset is not None:
        if not args.disable_translation:
            dataset.enable_translation()
        if not args.disable_resolution_augmentation:
            dataset.enable_resolution_augmentation()

    return avg_loss, accuracy, mean_steps


def evaluate_depths(model, loader, device, *, depths, fix_scale_factor=2, distributed=False):
    """Return loss and grid accuracy at the depths used for checkpoint selection."""
    if not depths or any(isinstance(depth, bool) or not isinstance(depth, int) or depth < 1
                         for depth in depths):
        raise ValueError("Evaluation depths must be positive integers")
    depths = sorted(set(depths))
    is_distributed = distributed and dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0
    world_size = dist.get_world_size() if is_distributed else 1
    dataset = loader.dataset
    augmentation = {
        name: getattr(dataset, name) for name in
        ("translation_enabled", "resolution_enabled", "fix_scale_factor")
        if hasattr(dataset, name)
    }
    training_flags = [(module, module.training) for module in model.modules()]
    python_rng, numpy_rng = random.getstate(), np.random.get_state()
    totals = torch.zeros(len(depths), 4, dtype=torch.float64, device=device)
    try:
        model.eval()
        dataset.disable_translation()
        dataset.disable_resolution_augmentation(fix_scale_factor=fix_scale_factor)
        evaluation_loader = DataLoader(
            dataset, batch_size=loader.batch_size,
            sampler=range(rank, len(dataset), world_size),
            num_workers=loader.num_workers, collate_fn=loader.collate_fn,
            pin_memory=loader.pin_memory, drop_last=False,
            generator=torch.Generator().manual_seed(0),
        )
        with preserve_rng(device), torch.inference_mode():
            for batch in evaluation_loader:
                targets = batch["targets"].to(device)
                valid = targets != IGNORE_INDEX
                _, _, logits_by_step = model(
                    batch["inputs"].to(device), batch["task_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    override_max_steps=max(depths), return_intermediates=True,
                )
                for index, depth in enumerate(depths):
                    logits = logits_by_step[depth - 1].float()
                    totals[index, 0] += F.cross_entropy(
                        logits, targets, ignore_index=IGNORE_INDEX, reduction="sum"
                    ).double()
                    totals[index, 1] += valid.sum()
                    exact = ((logits.argmax(dim=1) == targets) | ~valid).flatten(1).all(dim=1)
                    exact &= valid.flatten(1).any(dim=1)
                    totals[index, 2] += exact.sum()
                    totals[index, 3] += targets.size(0)
            if is_distributed:
                dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        return [
            dict(step=depth, ce=loss / max(pixels, 1), exact_accuracy=exact / max(count, 1))
            for depth, (loss, pixels, exact, count) in zip(depths, totals.cpu().tolist())
        ]
    finally:
        for module, training in training_flags:
            module.training = training
        for name, value in augmentation.items():
            setattr(dataset, name, value)
        random.setstate(python_rng)
        np.random.set_state(numpy_rng)
