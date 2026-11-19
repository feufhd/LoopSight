"""Task-specific adaptation with paired raw and final-EMA predictions."""
from __future__ import annotations
import argparse
from copy import deepcopy
from pathlib import Path
import random
import sys
import time
import os
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from utils.distribution import init_distributed_mode
from .common import load_optimizer, compute_training_objective


from loopsight.data.arc import build_dataloaders, IGNORE_INDEX
from utils.eval_utils_ttt import generate_predictions, get_eval_rot_transform_resolver
from utils.ttt_checkpoints import save_ttt_checkpoint
from utils.ttt_ema import TTTEMA, capture_rng, restore_rng, record_attempt


def _run_model_forward(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    task_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    eval_mode: bool,
    return_intermediates: bool = False,
    **kwargs,
) -> torch.Tensor | tuple[torch.Tensor, Any, List[torch.Tensor]] | tuple[
    torch.Tensor, Any, List[torch.Tensor], List[torch.Tensor]
]:
    outputs = model(
        inputs,
        task_ids,
        attention_mask=attention_mask,
        return_intermediates=return_intermediates,
        **kwargs,
    )
    if return_intermediates or kwargs.get("return_raw_intermediates", False):
        return outputs
        
    if isinstance(outputs, tuple):
        logits, _ = outputs
    else:
        logits = outputs
    return logits


def _format_eta(seconds: float) -> str:
    total_seconds = int(max(seconds, 0))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}h{minutes:02d}m{secs:02d}s"


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _prediction_save_name(cur_attempt_idx: int, *, ema: bool = False, variant: str = '') -> str:
    if not args.eval_save_name:
        raise ValueError("--eval-save-name must be set when generating TTT predictions.")

    tag = variant or ('ema' if ema else '')
    attempt_name = f"{args.eval_save_name}{'_' + tag if tag else ''}_attempt_{cur_attempt_idx}"
    if not args.output_subdir:
        return attempt_name

    output_subdir = Path(args.output_subdir)
    if output_subdir.is_absolute() or ".." in output_subdir.parts:
        raise ValueError("--output-subdir must be a relative path under outputs/.")
    if output_subdir == Path("."):
        return attempt_name
    return str(output_subdir / attempt_name)


def ttt_once(model, device, distributed, rank, train_loader, train_sampler, eval_loader, cur_attempt_idx,
             runtime_max_steps: Optional[int] = None, # [NEW]
             ):
    if args.epochs < 1:
        raise ValueError("--epochs must be positive for test-time training.")
    if getattr(args, 'ttt_ema', False) and args.num_workers != 0:
        raise ValueError('Paired raw/EMA prediction RNG requires --num-workers 0')
    ema = (TTTEMA(model, args.ttt_ema_decay, getattr(args, 'ttt_ema_decays', None))
           if getattr(args, 'ttt_ema', False) else None)
    autocast_device_type = device.type
    is_main_process = (not distributed) or rank == 0

    global_start = time.time()
    previous_total_steps = 0
    optimizer, scaler, scheduler = load_optimizer(
        model=model, args=args, device=device, distributed=distributed, rank=rank
    )
    resume_epochs = bool(getattr(args, 'ttt_resume_dir', None))
    start_epoch = 1
    if resume_epochs:
        from utils import ttt_resume
        start_epoch = ttt_resume.restore(model, optimizer, scaler, scheduler, args,
                                         cur_attempt_idx, train_loader, device,
                                         **({'ema': ema} if ema is not None else {}))
        previous_total_steps = (start_epoch - 1) * len(train_loader)
    session_start_steps = previous_total_steps
    try:
        for epoch in range(start_epoch, args.epochs + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            sample_count = 0
            total_batches = len(train_loader)
            epoch_start = time.time()
            train_exact = 0
            train_examples = 0

            for step, batch in enumerate(train_loader, 1):
                inputs = batch["inputs"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                task_ids = batch["task_ids"].to(device)

                optimizer.zero_grad(set_to_none=True)
                
                # Use automatic mixed precision
                with autocast(device_type=autocast_device_type, enabled=scaler.is_enabled()):
                    outputs = _run_model_forward(
                        model,
                        inputs,
                        task_ids,
                        attention_mask,
                        eval_mode=False,
                        return_intermediates=False,
                    )
                    if isinstance(outputs, tuple):
                        logits = outputs[0]
                        intermediates = outputs[2] if len(outputs) >= 3 else None
                        raw_intermediates = outputs[3] if len(outputs) == 4 else None
                    else:
                        logits, intermediates = outputs, None
                        raw_intermediates = None
                    loss, _ = compute_training_objective(
                        logits, intermediates, targets, args, ignore_index=IGNORE_INDEX,
                        raw_intermediates=raw_intermediates,
                    )

                batch_size = inputs.size(0)
                predictions = logits.argmax(dim=1)
                for idx in range(batch_size):
                    target = targets[idx]
                    prediction = predictions[idx]
                    valid = target != IGNORE_INDEX
                    if valid.any():
                        is_exact = bool(torch.equal(prediction[valid], target[valid]))
                    else:
                        is_exact = False
                    train_exact += int(is_exact)
                    train_examples += 1

                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Unscale gradients before clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step with scaler
                previous_scale = scaler.get_scale() if ema is not None else None
                scaler.step(optimizer)
                scaler.update()
                if ema is not None and scaler.get_scale() >= previous_scale:
                    ema.update(model)

                running_loss += loss.item() * batch_size
                sample_count += batch_size

                if total_batches > 0 and is_main_process and not args.quiet_progress and step % 10 == 0:
                    elapsed = time.time() - epoch_start
                    avg_step_time = elapsed / step
                    steps_completed = previous_total_steps + step
                    total_steps = len(train_loader) * args.epochs
                    remaining_steps = total_steps - steps_completed
                    elapsed_global = time.time() - global_start
                    avg_time_per_step_global = elapsed_global / max(steps_completed - session_start_steps, 1)
                    eta = remaining_steps * avg_time_per_step_global
                    bar_length = 30
                    progress_ratio = steps_completed / total_steps if total_steps else 0
                    filled = int(bar_length * progress_ratio)
                    bar = "#" * filled + "-" * (bar_length - filled)
                    progress = 100.0 * progress_ratio
                    sys.stdout.write(
                        f"\rEpoch {epoch} [{bar}] {progress:5.1f}% ETA {_format_eta(eta)}"
                    )
                    sys.stdout.flush()

            if total_batches > 0 and is_main_process and not args.quiet_progress:
                sys.stdout.write("\n")
            previous_total_steps += total_batches

            epoch_duration = time.time() - epoch_start if total_batches > 0 else 0.0

            train_totals = torch.tensor(
                [running_loss, sample_count, train_exact, train_examples],
                dtype=torch.float64,
                device=device,
            )
            if distributed and dist.is_initialized():
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
            running_loss_total, sample_count_total, train_exact_total, train_examples_total = train_totals.tolist()
            avg_train_loss = running_loss_total / max(sample_count_total, 1)
            train_acc = train_exact_total / max(train_examples_total, 1)

            total_elapsed = time.time() - global_start
            total_steps = len(train_loader) * args.epochs
            steps_completed = min(previous_total_steps, total_steps)
            remaining_steps = total_steps - steps_completed
            avg_time_per_step_global = total_elapsed / max(steps_completed - session_start_steps, 1)
            total_eta = remaining_steps * avg_time_per_step_global

            log_parts = [
                f"epoch={epoch}",
                f"train_loss={avg_train_loss:.4f}",
                f"train_acc={train_acc:.4f}",
                f"epoch_time={epoch_duration:.1f}s",
                f"eta_total={_format_eta(total_eta)}",
            ]

            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else args.learning_rate
            log_parts.append(f"lr={current_lr:.6f}")           
            if is_main_process and not args.quiet_progress:
                print(" | ".join(log_parts))

            if scheduler is not None:
                scheduler.step()
            if resume_epochs and is_main_process:
                ttt_resume.save(model, optimizer, scaler, scheduler, args, cur_attempt_idx,
                                train_loader, device, epoch,
                                **({'ema': ema} if ema is not None else {}))

    finally:
        if distributed and dist.is_initialized():
            dist.barrier()

    if distributed and dist.is_initialized():
        dist.destroy_process_group()

    if is_main_process and getattr(args, "ttt_checkpoint_dir", None):
        save_ttt_checkpoint(
            model, args, cur_attempt_idx,
            task_lookup=train_loader.dataset.task_lookup,
        )

    prediction_kwargs = dict(
        img_size=args.image_size,
        attempt_nums=args.num_attempts,
        task_transform_resolver=get_eval_rot_transform_resolver(),
        fix_scale_factor=args.fix_scale_factor,
        disable_translation=args.disable_translation,
        if_fix_scale=args.disable_resolution_augmentation,
        eval_split=args.eval_split,
        task_type=args.data_root.split("/")[-1],  # e.g., "ARC-AGI"
        forward_fn=lambda model_ref, inp, task, mask, **kwargs: _run_model_forward(
            model_ref,
            inp,
            task,
            mask,
            eval_mode=True,
            return_intermediates=False,
            **kwargs # Pass runtime overrides like override_max_steps
        ),
        override_max_steps=runtime_max_steps, # [NEW] Pass runtime override
        show_progress=not args.quiet_progress,
    )
    raw_name = _prediction_save_name(cur_attempt_idx)
    before_prediction = capture_rng(eval_loader, device) if ema is not None else None
    generate_predictions(model, eval_loader, device, save_name=raw_name, **prediction_kwargs)
    if ema is not None:
        after_prediction = capture_rng(eval_loader, device)
        prediction_names = {'raw': raw_name}
        try:
            for variant, average in ema.variants.items():
                name = _prediction_save_name(cur_attempt_idx, variant=variant)
                prediction_names[variant] = name
                restore_rng(before_prediction, eval_loader, device)
                with average.apply_to(model):
                    if is_main_process and getattr(args, 'ttt_checkpoint_dir', None):
                        save_ttt_checkpoint(model, args, cur_attempt_idx,
                                            task_lookup=train_loader.dataset.task_lookup,
                                            weight_type=variant, ema=average)
                    generate_predictions(model, eval_loader, device, save_name=name,
                                         **prediction_kwargs)
        finally:
            # Additional EMA predictions must not alter the next attempt's RNG.
            restore_rng(after_prediction, eval_loader, device)
        if is_main_process and getattr(args, 'ttt_checkpoint_dir', None):
            record_attempt(args, cur_attempt_idx, ema, prediction_names)
    if resume_epochs and is_main_process:
        # Keep the final epoch checkpoint until prediction export succeeds, so
        # interruption during prediction can retry without training again.
        ttt_resume.progress_path(args, cur_attempt_idx).unlink(missing_ok=True)

def train(args: argparse.Namespace, *, model_loader=None, model_preparer=None) -> None:
    distributed, rank, world_size, local_rank, device = init_distributed_mode(args)
    set_seed(args.seed + (rank if distributed else 0))

    train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler = build_dataloaders(
        args,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    total_train_examples = len(train_dataset)

    if ((not distributed) or rank == 0) and not args.quiet_progress:
        print(f"Total training examples: {total_train_examples}")


    model_original = model_loader(
        args=args, train_dataset=train_dataset, device=device, distributed=distributed, rank=rank, local_rank=local_rank
    )
    
    if not 0 <= args.ttt_start_attempt <= args.ttt_num_each:
        raise ValueError("--ttt-start-attempt must be between 0 and --ttt-num-each")

    for attempt_idx in range(args.ttt_start_attempt, args.ttt_num_each):
        model = deepcopy(model_original)
        if model_preparer is not None:
            model = model_preparer(model, args)
        if not args.quiet_progress:
            print(f"Starting test-time training attempt {attempt_idx + 1}/{args.ttt_num_each}...")
        ttt_once(model=model, device=device, distributed=distributed, rank=rank,
                train_loader=train_loader, train_sampler=train_sampler,
                eval_loader=eval_loader, cur_attempt_idx=attempt_idx,
                runtime_max_steps=args.runtime_max_steps)
