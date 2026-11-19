"""Train RGB ImageNet classifiers and evaluate the labeled val set after epochs."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time

import torch
from loopsight.runtime import resolve_device
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

from utils.loop_depth_sampling import LoopDepthSampler
from .dataset import DEFAULT_DATA, DistributedEvalSampler, evaluate, image_folder, make_loader
from .engine import (EMA, accumulation_window, compatible_resume, cpu_state,
    gather_rng, learning_rate, optimizer_groups, restore_rng, save_checkpoint,
    seed_everything, setup_execution, write_json)
from .config import LOOP_MODELS, MODEL_NAMES, ModelConfig, build_model, forward_logits
from .subsets import DATASETS, dataset_spec


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=MODEL_NAMES, default="loopsight_small")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--dataset", choices=DATASETS, default="imagenet100")
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=32, help="Images per microbatch")
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--eval-batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4, help="Peak learning rate at global batch 512")
    p.add_argument("--lr-scaling", choices=("linear", "none"), default="linear",
                   help="Use none for an explicitly specified absolute peak LR")
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=20)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--clip-grad", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--drop-path", type=float, default=0.1)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--mixup", type=float, default=0.8)
    p.add_argument("--cutmix", type=float, default=1.0)
    p.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ema", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--loops", type=int, default=12)
    p.add_argument("--step-table-size", type=int, default=18)
    p.add_argument("--depth-sampling", choices=("fixed", "gaussian"))
    p.add_argument("--min-loops", type=int, default=6)
    p.add_argument("--max-loops", type=int, default=18)
    p.add_argument("--depth-std", type=float, default=3.0)
    p.add_argument("--eval-depths", type=int, nargs="+", default=[12])
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--checkpoint-blocks", action="store_true")
    p.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    p.set_defaults(device="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--dry-run", action="store_true", help="Print plan; no model forward, data scan or output writes")
    p.set_defaults(output=Path("outputs/imagenet100/loopsight"), epochs=100,
        batch_size=256, accumulation_steps=4, eval_batch_size=384, loops=6,
        depth_sampling="fixed", eval_depths=[6], warmup_epochs=5, lr_scaling="none",
        dropout=0.2, drop_path=0.0, ema=True, ema_decay=0.9999, eval_every=10)
    a = p.parse_args(argv)
    if min(a.epochs, a.batch_size, a.accumulation_steps, a.eval_batch_size, a.loops,
           a.step_table_size, a.eval_every, a.cpu_threads, a.log_every, *a.eval_depths) < 1:
        p.error("Counts, loop depths and batch sizes must be positive")
    if a.workers < 0 or a.seed < 0 or not 0 <= a.warmup_epochs < a.epochs:
        p.error("Invalid workers, seed or warmup horizon")
    numeric = [a.lr, a.min_lr, a.weight_decay, a.clip_grad, a.dropout, a.drop_path,
               a.label_smoothing, a.mixup, a.cutmix, a.ema_decay, a.depth_std]
    if not all(math.isfinite(v) for v in numeric):
        p.error("Numeric hyperparameters must be finite")
    if min(a.lr, a.depth_std) <= 0 or min(a.min_lr, a.weight_decay, a.clip_grad, a.mixup, a.cutmix) < 0:
        p.error("Invalid learning rate, depth spread, clipping or regularization")
    if any(not 0 <= v < 1 for v in (a.dropout, a.drop_path, a.label_smoothing, a.ema_decay)):
        p.error("Dropout, smoothing and EMA decay must be in [0,1)")
    if a.min_loops < 1 or a.max_loops < a.min_loops or len(set(a.eval_depths)) != len(a.eval_depths):
        p.error("Invalid loop range or duplicate evaluation depths")
    if a.depth_sampling is None:
        a.depth_sampling = "gaussian" if a.model in LOOP_MODELS else "fixed"
    if (a.mixup or a.cutmix) and a.batch_size % 2:
        p.error("Mixup/CutMix requires an even microbatch size")
    if resolve_device(a.device).type == "cpu" and a.amp == "fp16":
        p.error("FP16 AMP requires an accelerator")
    if a.output is None:
        a.output = Path("outputs") / ("imagenet-100" if a.dataset == "imagenet100" else "imagenet-1k") / (a.model + time.strftime("_%Y%m%d_%H%M%S"))
    return a


def recipe(a, world):
    subset = dataset_spec(a.dataset)
    config = ModelConfig(name=a.model, loops=a.loops, step_table_size=a.step_table_size,
        dropout=a.dropout, drop_path_rate=a.drop_path, checkpoint_blocks=a.checkpoint_blocks,
        num_classes=subset["classes"])
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(a).items()}
    settings.update(world_size=world, global_batch=a.batch_size * a.accumulation_steps * world,
                    image_size=224, patch_size=16, classes=subset["classes"], dataset_selection=subset,
                    train_split="train", evaluation_split="val", validation_crop="resize256_center224",
                    train_input_shape=[3, 224, 224], validation_input_shape=[3, 224, 224],
                    preprocessing_reference="https://github.com/facebookresearch/deit/blob/main/datasets.py",
                    normalization_mean=[0.485, 0.456, 0.406], normalization_std=[0.229, 0.224, 0.225],
                    randaugment="rand-m9-mstd0.5-inc1" if a.augment else None,
                    random_erasing_probability=0.25 if a.augment else 0,
                    random_erasing_mode="pixel", color_jitter=0.3 if a.augment else 0,
                    pretrained=False, best_selection_depth=a.eval_depths[0])
    settings["scaled_peak_lr"] = a.lr * settings["global_batch"] / 512 if a.lr_scaling == "linear" else a.lr
    if a.min_lr > settings["scaled_peak_lr"]:
        raise ValueError("min-lr exceeds the scaled peak learning rate")
    return config, settings


def train_epoch(model, loader, optimizer, scaler, ema, criterion, mixup, sampler, a,
                epoch, step, total_updates, warmup_updates, peak_lr, device, rank):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros(2, device=device, dtype=torch.float64)
    started = time.monotonic()
    depth = a.loops
    for index, (images, targets) in enumerate(loader):
        if index % a.accumulation_steps == 0:
            depth = sampler.sample_update(step, device, dist.is_initialized()) if sampler else a.loops
            lr = learning_rate(step, total_updates, warmup_updates, peak_lr, a.min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        if mixup:
            images, targets = mixup(images, targets)
        window = accumulation_window(index, len(loader), a.accumulation_steps)
        boundary = (index + 1) % a.accumulation_steps == 0 or index + 1 == len(loader)
        sync = model.no_sync() if isinstance(model, DDP) and not boundary else nullcontext()
        with sync:
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16 if a.amp == "bf16" else torch.float16,
                                enabled=a.amp != "none"):
                logits = forward_logits(model, images, depth)
                loss = criterion(logits, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite loss at epoch {epoch}, batch {index}")
            scaler.scale(loss / window).backward()
        total_loss[0] += loss.detach().double() * len(images)
        total_loss[1] += len(images)
        if boundary:
            scaler.unscale_(optimizer)
            if a.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip_grad)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= old_scale:
                step += 1
                if ema:
                    ema.update(model, step)
            optimizer.zero_grad(set_to_none=True)
            if rank == 0 and (step % a.log_every == 0 or index + 1 == len(loader)):
                print(json.dumps(dict(event="train", epoch=epoch + 1, step=step,
                    depth=depth if a.model in LOOP_MODELS else None, lr=lr, loss=float(loss.detach()),
                    epoch_elapsed_seconds=round(time.monotonic() - started, 3),
                    images_per_second=round(float(total_loss[1]) / (time.monotonic() - started), 2),
)), flush=True)
    if dist.is_initialized():
        dist.all_reduce(total_loss)
    return step, float(total_loss[0] / total_loss[1])


def run(a):
    config, settings = recipe(a, int(os.environ.get("WORLD_SIZE", "1")))
    if a.dry_run:
        print(json.dumps(dict(mode="plan_only", model_config=config.to_dict(), training=settings), indent=2))
        return
    torch.set_num_threads(a.cpu_threads)
    device, rank, world = setup_execution(a.device)
    # Construct identical raw/EMA initial weights before DDP broadcasts.
    seed_everything(a.seed)
    if rank == 0:
        if a.output.exists() and any(a.output.iterdir()) and not a.resume:
            raise FileExistsError("Nonempty output directory requires an explicit --resume")
        a.output.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()
    started = time.monotonic()
    if rank == 0:
        print(json.dumps(dict(event="dataset_scan_start", data_root=str(a.data_root))), flush=True)
    subset = settings["dataset_selection"]
    train = image_folder(a.data_root, "train", classes=subset["classes"], augment=a.augment,
                         class_ids=subset["class_ids"], expected_rows=subset["expected_rows"]["train"])
    validation = image_folder(a.data_root, "val", classes=subset["classes"],
                              class_ids=subset["class_ids"], expected_rows=subset["expected_rows"]["val"])
    if train.class_to_idx != validation.class_to_idx:
        raise ValueError("Training and validation class indices differ")
    if rank == 0:
        write_json(a.output / "dataset.json", dict(**subset, train_images=len(train), val_images=len(validation),
            class_to_idx=train.class_to_idx, original_class_to_idx=getattr(train, "original_class_to_idx", None)))
    if rank == 0:
        print(json.dumps(dict(event="dataset_ready", train_images=len(train), val_images=len(validation),
            seconds=round(time.monotonic() - started, 3))), flush=True)
    train_sampler = DistributedSampler(train, num_replicas=world, rank=rank, seed=a.seed, drop_last=True) if world > 1 else None
    generator = torch.Generator()
    train_loader = make_loader(train, a.batch_size, a.workers, device, train_sampler, True, generator)
    val_loader = make_loader(validation, a.eval_batch_size, a.workers, device,
                            DistributedEvalSampler(validation, rank, world))
    if not len(train_loader):
        raise ValueError("Training loader has no full batch")
    updates_per_epoch = math.ceil(len(train_loader) / a.accumulation_steps)
    total_updates, warmup_updates = a.epochs * updates_per_epoch, a.warmup_epochs * updates_per_epoch
    model = build_model(config).to(device)
    if rank == 0:
        print(json.dumps(dict(event="model_ready", model_config=config.to_dict(),
            parameters=sum(p.numel() for p in model.parameters()),
            training=settings, total_updates=total_updates, warmup_updates=warmup_updates)), flush=True)
    ema = EMA(model, a.ema_decay) if a.ema else None
    optimizer = torch.optim.AdamW(optimizer_groups(model, a.weight_decay), lr=settings["scaled_peak_lr"])
    scaler = torch.amp.GradScaler(device.type, enabled=a.amp == "fp16")
    first_epoch = step = 0
    best = {"raw": -1.0, "ema": -1.0}
    resume_rng = None
    if a.resume:
        saved = torch.load(a.resume, map_location="cpu", weights_only=True)
        if saved["model_config"] != config.to_dict() or saved["world_size"] != world:
            raise ValueError("Resume model configuration or world size differs")
        compatible_resume(saved["training_config"], settings)
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        first_epoch, step, best = saved["epoch"] + 1, saved["step"], saved["best"]
        if ema:
            ema.module.load_state_dict(saved["ema"], strict=True)
            ema.step = saved["ema_step"]
            if ema.step != step:
                raise ValueError("Resume EMA step differs from optimizer step")
        resume_rng = saved["rng_by_rank"][rank]
    if first_epoch >= a.epochs:
        raise ValueError("Checkpoint already reached the requested training horizon")
    if world > 1:
        model = DDP(
            model, device_ids=[device.index] if device.type != "cpu" else None,
            broadcast_buffers=False,
            find_unused_parameters=(a.min_loops == 1 if a.depth_sampling == "gaussian" else a.loops == 1),
        )
    seed_everything(a.seed + rank)
    if resume_rng:
        restore_rng(resume_rng, device)
    mixup = Mixup(mixup_alpha=a.mixup, cutmix_alpha=a.cutmix, label_smoothing=a.label_smoothing,
                  num_classes=config.num_classes) if a.mixup or a.cutmix else None
    criterion = SoftTargetCrossEntropy() if mixup else LabelSmoothingCrossEntropy(a.label_smoothing)
    depth_sampler = LoopDepthSampler(a.min_loops, a.max_loops, a.loops, a.depth_std, a.seed) if a.depth_sampling == "gaussian" else None
    if rank == 0:
        write_json(a.output / "config.json", dict(model_config=config.to_dict(), training=settings,
            updates_per_epoch=updates_per_epoch, depth_probabilities=depth_sampler.config() if depth_sampler else None))
    for epoch in range(first_epoch, a.epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        generator.manual_seed(a.seed + epoch * world + rank)
        step, train_loss = train_epoch(model, train_loader, optimizer, scaler, ema, criterion,
            mixup, depth_sampler, a, epoch, step, total_updates, warmup_updates,
            settings["scaled_peak_lr"], device, rank)
        final = epoch + 1 == a.epochs
        if final and rank == 0:
            save_checkpoint(a.output / "final.pt", dict(model_config=config.to_dict(), model=cpu_state(model),
                weights="final_raw", step=step, epoch=epoch))
            if ema:
                ema.export_final(a.output / "ema_final.pt", config, step, epoch)
        if final and dist.is_initialized():
            dist.barrier()
        if final and ema and rank != 0:
            exported = torch.load(a.output / "ema_final.pt", map_location="cpu", weights_only=True)
            if exported["step"] != step or exported["weights"] != "final_step_ema":
                raise ValueError("Invalid final EMA export")
            ema.module.load_state_dict(exported["model"], strict=True)
        results = {}
        if final or (epoch + 1) % a.eval_every == 0:
            for name, evaluated in [("raw", model)] + ([("ema", ema.module)] if ema else []):
                results[name] = {}
                for depth in a.eval_depths if a.model in LOOP_MODELS else [0]:
                    result = evaluate(evaluated, val_loader, device, depth, a.amp)
                    if result["examples"] != len(validation):
                        raise ValueError("Validation coverage differs from the dataset")
                    if final:
                        result["checkpoint"] = str(a.output / ("ema_final.pt" if name == "ema" else "final.pt"))
                    results[name][str(depth)] = result
                    if rank == 0:
                        print(json.dumps(dict(event="val", epoch=epoch + 1, weights=name, **result)), flush=True)
                primary = str(a.eval_depths[0] if a.model in LOOP_MODELS else 0)
                score = results[name][primary]["top1"]
                if score > best[name]:
                    best[name] = score
                    if rank == 0:
                        save_checkpoint(a.output / f"best_{name}.pt", dict(model_config=config.to_dict(),
                            model=cpu_state(evaluated), weights=f"validation_best_{name}", step=step,
                            epoch=epoch, selection=results[name][primary]))
        rng_states = gather_rng(device, world)
        if rank == 0:
            save_checkpoint(a.output / "last.pt", dict(model_config=config.to_dict(), training_config=settings,
                model=cpu_state(model), ema=cpu_state(ema.module) if ema else None,
                ema_step=ema.step if ema else None, optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                epoch=epoch, step=step, best=best, world_size=world, rng_by_rank=rng_states))
            event = dict(epoch=epoch + 1, step=step, train_loss=train_loss, validation=results)
            with (a.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(event) + "\n")
            if final:
                write_json(a.output / "final_metrics.json", event)
        if dist.is_initialized():
            dist.barrier()


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
