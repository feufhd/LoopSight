"""LoopSight ARC training with Gaussian loop budgets and final-step EMA."""
import argparse
import json
import math
import time
from contextlib import nullcontext
from itertools import islice
from pathlib import Path
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from .evaluation import evaluate, evaluate_depths, set_seed
from .common import compute_training_objective, validate_model_args, enable_checkpointing
from loopsight.presets import model_from_args
from loopsight.checkpoint import ARCHITECTURE, CONTEXT, FUSION_MODE, validate_task_context_checkpoint, load_task_lowrank_state
from loopsight.data.arc import IGNORE_INDEX, build_dataloaders
from utils.loop_depth_sampling import LoopDepthSampler
from utils.model_ema import ModelEMA, MultiModelEMA
from utils.args import add_resume_checkpoints
from utils.distribution import init_distributed_mode
from utils.lr_scheduler import get_cosine_schedule_with_warmup, resolve_warmup_epochs
from utils.progress import TotalProgressReporter

TRAINING_SCHEME = "task_context_cross_attention_silu_gaussian_final_ce_v1"

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.set_defaults(distributed=False)
    add_resume_checkpoints(parser)
    parser.add_argument("--train-loop-sampling", choices=("gaussian",), default="gaussian")
    parser.add_argument("--train-loop-min", type=int, default=6)
    parser.add_argument("--train-loop-max", type=int, default=18)
    parser.add_argument("--train-loop-mean", type=float, default=12.0)
    parser.add_argument("--train-loop-std", type=float, default=3.0)

    parser.add_argument("--data-root", default="raw_data/ARC-AGI")
    parser.add_argument("--train-split", default="training")
    parser.add_argument("--eval-split", default="training")
    parser.add_argument("--eval-subset", choices=("train", "test"), default="test")
    parser.add_argument("--include-rearc", action="store_true")
    parser.add_argument("--rearc-path", default="raw_data/re_arc")
    parser.add_argument("--rearc-limit", type=int, default=-1)
    parser.add_argument("--disable-translation", action="store_true")
    parser.add_argument("--disable-resolution-augmentation", action="store_true")
    parser.add_argument("--fix-scale-factor", type=int, default=2)

    architecture = ARCHITECTURE
    parser.add_argument("--architecture", choices=(architecture,), default=architecture)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--patch-size", type=int, default=2)
    parser.add_argument("--num-colors", type=int, default=12)
    parser.add_argument("--num-task-tokens", type=int, default=1)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--loop-core-depth", type=int, default=6)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--step-embedding-type", choices=("asymptotic",),
                        default="asymptotic")
    parser.add_argument("--learned-loop-steps", type=int, default=6,
                        help="Learned additive loop-table length; later loops reuse its last entry.")
    parser.add_argument("--step-embedding-scale", type=float, default=1.0)
    parser.add_argument("--asymptotic-channels", type=int, default=8)
    parser.add_argument("--local-clock-frequencies", type=int, default=3)
    parser.add_argument("--modulation-hidden-dim", type=int, default=64)
    parser.add_argument("--fusion-mode", choices=(FUSION_MODE,),
                        default=FUSION_MODE)
    parser.add_argument("--fusion-gate-rank", type=int, default=16,
                        help="Width of source projections and the cross-attention write gate.")
    parser.add_argument("--fusion-gate-heads", type=int, default=2)
    parser.set_defaults(fusion_gate_context=CONTEXT, fusion_gate_transform="bounded_tanh")
    parser.add_argument("--fusion-gate-logit-bound", type=float, default=math.log(16))
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--loop-drop-path-rate", type=float, default=0.0)
    parser.add_argument("--loop-drop-path-depth", type=int, default=6)
    parser.add_argument("--use-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--checkpoint-blocks", type=int, default=0)
    parser.add_argument("--checkpoint-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--loop-state-rmsnorm", action=argparse.BooleanOptionalAction, default=True,
                        help="Use FP32 per-token normalization at every loop boundary (default: enabled).")

    parser.add_argument("--batch-size", type=int, default=8, help="Micro-batch size per rank.")
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stop-after-epoch", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", choices=("none", "cosine"), default="cosine")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--no-compile", action="store_true", default=True,
                        help="This variable-depth entry point uses eager execution.")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--max-eval-batches", type=int, default=None)
    parser.add_argument("--eval-loop-steps", type=int, default=12,
                        help="Default inference and best-checkpoint selection depth.")
    parser.add_argument("--save-path", default=None, help="Final training checkpoint.")
    parser.add_argument("--best-save-path", default=None, help="Best default-depth validation checkpoint.")
    parser.add_argument("--last-save-path", default=None, help="Optional resume checkpoint saved each epoch.")
    parser.add_argument("--ema", action=argparse.BooleanOptionalAction, default=False,
                        help="Average successful optimizer updates and validate EMA every epoch.")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-decays", type=float, nargs="+", default=None,
                        help="Track independent EMA decays in one run; requires --ema. "
                             "The first decay is primary and retains legacy EMA filenames/metrics.")
    parser.add_argument("--ema-save-path", default=None, help="EMA inference export (default: final_ema.pt beside final.pt).")
    parser.add_argument("--ema-best-save-path", default=None, help="Best EMA inference export (default: best_ema.pt beside best.pt).")
    parser.add_argument("--best-checkpoint-depths", type=int, nargs="+", default=None,
                        help="Select each raw/EMA best across these validation depths every epoch; default: eval-loop-steps only.")
    return parser


def validate_training_args(args):
    # Validate before constructing datasets, starting execution.
    if args.ema_decays is not None:
        if not args.ema:
            raise ValueError("--ema-decays requires --ema")
        if (not args.ema_decays or len(set(args.ema_decays)) != len(args.ema_decays)
                or any(not math.isfinite(d) or not 0 <= d < 1 for d in args.ema_decays)):
            raise ValueError("--ema-decays must contain distinct finite values in [0, 1)")
        args.ema_decay = args.ema_decays[0]
    if not math.isfinite(args.ema_decay) or not 0 <= args.ema_decay < 1:
        raise ValueError("--ema-decay must be finite and in [0, 1)")
    if args.best_checkpoint_depths is not None:
        if not args.best_checkpoint_depths or any(d < 1 for d in args.best_checkpoint_depths):
            raise ValueError("--best-checkpoint-depths must contain positive depths")
        if not args.eval_split:
            raise ValueError("--best-checkpoint-depths requires a nonempty --eval-split")
    if not args.ema and (args.ema_save_path or args.ema_best_save_path):
        raise ValueError("EMA checkpoint paths require --ema")
    if args.ema:
        if not args.eval_split:
            raise ValueError("--ema requires a nonempty --eval-split for epoch validation")
        if args.save_path and not args.ema_save_path:
            args.ema_save_path = str(Path(args.save_path).with_name("final_ema.pt"))
        if args.best_save_path and not args.ema_best_save_path:
            args.ema_best_save_path = str(Path(args.best_save_path).with_name("best_ema.pt"))
    if args.eval_loop_steps < 1:
        raise ValueError("--eval-loop-steps must be positive")
    if args.resume_reset_epoch != args.resume_reset_optimizer:
        raise ValueError("Use --resume-reset-epoch and --resume-reset-optimizer together for a new training run")
    if args.resume_skip_task_token and not args.resume_reset_epoch:
        raise ValueError("Skipping task tokens requires both resume reset flags")
    for name in ("batch_size", "accumulation_steps", "epochs"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be nonnegative")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("--learning-rate must be finite and positive")
    for name in ("weight_decay", "max_grad_norm"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and nonnegative")
    if args.best_save_path and not args.eval_split:
        raise ValueError("--best-save-path requires a nonempty --eval-split")
    paths = [Path(p).resolve() for p in (args.save_path, args.best_save_path, args.last_save_path,
                                        args.ema_save_path, args.ema_best_save_path) if p]
    for decay in (args.ema_decays or [])[1:]:
        paths.extend(Path(_ema_checkpoint_path(p, decay, args.ema_decay)).resolve()
                     for p in (args.ema_save_path, args.ema_best_save_path) if p)
    if len(set(paths)) != len(paths):
        raise ValueError("Final, best, last and EMA checkpoint paths must be distinct")
    if not args.resume_checkpoint and any(path.exists() for path in paths):
        raise FileExistsError("A checkpoint destination already exists; choose a new run directory or explicitly resume")

    args.training_scheme = TRAINING_SCHEME
    args.fusion_gate_activation = "silu"
    args.fusion_gate_context = CONTEXT
    args.fusion_gate_transform = "bounded_tanh"
    if args.fusion_gate_heads < 1 or args.fusion_gate_rank % args.fusion_gate_heads:
        raise ValueError("Gate rank must be divisible by positive gate heads")
    resolve_warmup_epochs(args.epochs, args.warmup_epochs)
    if not (1 if args.use_checkpoint else 0) <= args.checkpoint_blocks <= args.loop_core_depth:
        raise ValueError("Checkpoint block count must lie within the shared core")
    rate = args.loop_drop_path_rate
    if not math.isfinite(rate) or not 0 <= rate < 1 or args.loop_drop_path_depth < 2:
        raise ValueError("Invalid loop drop-path configuration")
    if rate * max(args.train_loop_max - 1, 0) / (args.loop_drop_path_depth - 1) >= 1:
        raise ValueError("Scheduled loop drop probability must remain below one")
    args.max_loop_steps = args.eval_loop_steps
    args.min_loop_steps = 1
    args.disable_exit_gate = True
    args.train_dynamic_exit = args.eval_dynamic_exit = False
    args.exit_gate_threshold = 0.6
    args.gate_entropy_weight = args.loop_penalty_weight = 0.0
    args.no_step_embedding = False
    args.drop_path_rate = 0.0
    args.direct_loss_weight = 0.0
    args.progressive_loss = "none"
    args.progressive_loss_weight = 0.0
    args.progressive_kappa = 0.25
    if args.num_task_tokens < 1:
        raise ValueError("--num-task-tokens must be positive for task-conditioned fusion")
    if args.fusion_gate_rank < 1:
        raise ValueError("--fusion-gate-rank must be positive")
    if args.architecture != ARCHITECTURE or args.fusion_mode != FUSION_MODE:
        raise ValueError("This entry point requires the task-conditioned low-rank variant")
    if args.embed_dim < 1 or args.num_heads < 1 or args.embed_dim % args.num_heads:
        raise ValueError("--embed-dim must be positive and divisible by --num-heads")
    if args.patch_size < 1 or args.image_size < 1 or args.image_size % args.patch_size:
        raise ValueError("--image-size must be positive and divisible by --patch-size")
    if args.loop_core_depth < 1 or args.mlp_dim < 1 or args.num_colors < 1:
        raise ValueError("Model dimensions must be positive")
    if not math.isfinite(args.dropout) or not 0 <= args.dropout < 1:
        raise ValueError("--dropout must be finite and in [0, 1)")
    validate_model_args(args)


def validate_resume_config(args, checkpoint):
    if args.resume_reset_epoch and args.resume_reset_optimizer:
        return
    saved = checkpoint.get("config", {})
    if saved.get("ema", False) != args.ema:
        raise ValueError("Resume requires the same EMA setting; use both resume reset flags to start a new run")
    if args.ema and saved.get("ema_decay") != args.ema_decay:
        raise ValueError("Resume requires the same EMA decay")
    if args.ema and ((saved.get("ema_decays") or [saved.get("ema_decay")])
                     != (args.ema_decays or [args.ema_decay])):
        raise ValueError("Resume requires the same EMA decays")
    if saved.get("best_checkpoint_depths") != args.best_checkpoint_depths:
        raise ValueError("Resume requires the same best-checkpoint-depths")
    fields = ("training_scheme", "train_loop_sampling", "train_loop_min", "train_loop_max",
              "train_loop_mean", "train_loop_std", "eval_loop_steps",
              "seed", "batch_size", "accumulation_steps", "epochs", "max_train_batches",
              "optimizer_updates_per_epoch", "training_world_size", "lr_scheduler",
              "no_amp", "learning_rate", "weight_decay", "max_grad_norm",
              "warmup_epochs", "use_checkpoint", "checkpoint_blocks", "checkpoint_gate",
              "loop_drop_path_rate", "loop_drop_path_depth")
    mismatches = [name for name in fields if saved.get(name) != getattr(args, name, None)]
    if mismatches:
        raise ValueError("Resume training configuration differs: " + ", ".join(mismatches)
                         + ". To initialize a new run from these weights, set both "
                         "--resume-reset-epoch and --resume-reset-optimizer.")


def _resume_if_needed(model, optimizer, scaler, scheduler, args, device, ema=None, best_selection=None):
    if not args.resume_checkpoint:
        return 1, float("-inf")
    checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    validate_task_context_checkpoint(args, checkpoint)
    validate_resume_config(args, checkpoint)
    raw_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    load_task_lowrank_state(raw_model, checkpoint, skip_task_token=args.resume_skip_task_token)
    if ema is not None:
        if args.resume_reset_optimizer:
            ema.reset(raw_model)
        elif checkpoint.get("ema_state") is None:
            raise ValueError("Resume checkpoint is missing ema_state; use a raw training checkpoint")
        else:
            ema.load_state_dict(checkpoint["ema_state"])
    if args.resume_reset_optimizer:
        return 1, float("-inf")
    if best_selection is not None:
        best_selection['depth'] = checkpoint.get('best_validation_depth', args.eval_loop_steps)
    for key in ("optimizer_state", "epoch"):
        if checkpoint.get(key) is None:
            raise ValueError(f"Resume checkpoint is missing {key}")
    optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler.is_enabled():
        if not checkpoint.get("scaler_state"):
            raise ValueError("AMP resume checkpoint is missing scaler_state")
        scaler.load_state_dict(checkpoint["scaler_state"])
    if scheduler is not None:
        if checkpoint.get("scheduler_state") is None:
            raise ValueError("Resume checkpoint is missing scheduler_state")
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    return checkpoint["epoch"] + 1, checkpoint.get("best_eval_accuracy", float("-inf"))


def _build_model(args, train_dataset, device, distributed, rank, local_rank):
    model = enable_checkpointing(model_from_args(args, train_dataset.num_tasks), args).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[device.index] if device.type != "cpu" else None,
            output_device=device.index if device.type != "cpu" else None,
            find_unused_parameters=args.train_loop_min == 1,
        )
    return model, model.module if distributed else model


def _unpad_eval_loader(loader, rank, world_size):
    """Count each validation example once, including uneven rank partitions."""
    if loader is None:
        return None
    return torch.utils.data.DataLoader(
        loader.dataset, batch_size=loader.batch_size,
        sampler=range(rank, len(loader.dataset), world_size),
        num_workers=loader.num_workers, collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory, drop_last=False,
        generator=torch.Generator().manual_seed(0),
    )


def _save_last(args, model, optimizer, scaler, scheduler, epoch, best_accuracy, ema=None, best_depth=None):
    _save_checkpoint(args.last_save_path, args, model, optimizer, scaler, scheduler, epoch, best_accuracy, ema, best_depth)


def _save_checkpoint(destination, args, model, optimizer, scaler, scheduler, epoch, best_accuracy, ema=None, best_depth=None):
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model_state=model.state_dict(), config=vars(args), epoch=epoch,
                   best_eval_accuracy=best_accuracy, optimizer_state=optimizer.state_dict(),
                   scaler_state=scaler.state_dict() if scaler.is_enabled() else None)
    if scheduler is not None:
        payload["scheduler_state"] = scheduler.state_dict()
    if ema is not None:
        payload["ema_state"] = ema.state_dict()
    payload['best_validation_depth'] = best_depth
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_ema_checkpoint(destination, args, ema, epoch):
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(model_state=ema.shadow, config=vars(args), epoch=epoch,
                   best_eval_accuracy=ema.best_accuracy, weight_type="ema",
                   ema_decay=ema.decay, ema_num_updates=ema.num_updates,
                   best_validation_depth=ema.best_depth)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _ema_models(ema):
    return getattr(ema, 'averages', (ema,)) if ema is not None else ()


def _ema_checkpoint_path(destination, decay, primary_decay):
    if not destination or decay == primary_decay:
        return destination
    path = Path(destination)
    return str(path.with_name(f"{path.stem}_{decay}{path.suffix}"))


def _save_all_ema_checkpoints(destination, args, ema, epoch):
    for average in _ema_models(ema):
        _save_ema_checkpoint(_ema_checkpoint_path(destination, average.decay, ema.decay),
                             args, average, epoch)


def train(args: argparse.Namespace) -> None:
    validate_training_args(args)
    for name in ("max_train_batches", "max_eval_batches", "eval_loop_steps"):
        value = getattr(args, name, None)
        if value is not None and value < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.accumulation_steps < 1:
        raise ValueError("--accumulation-steps must be at least 1.")
    if args.stop_after_epoch is not None and not 1 <= args.stop_after_epoch <= args.epochs:
        raise ValueError("--stop-after-epoch must be between 1 and --epochs.")
    end_epoch = args.stop_after_epoch if args.stop_after_epoch is not None else args.epochs

    parser_args = args
    distributed, rank, world_size, local_rank, device = init_distributed_mode(parser_args)
    set_seed(args.seed + (rank if distributed else 0))

    train_dataset, train_loader, eval_dataset, eval_loader, train_sampler, eval_sampler = build_dataloaders(
        args,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    if distributed:
        eval_loader = _unpad_eval_loader(eval_loader, rank, world_size)

    if args.disable_translation:
        train_dataset.disable_translation()
        if eval_dataset is not None:
            eval_dataset.disable_translation()
    else:
        train_dataset.enable_translation()
        if eval_dataset is not None:
            eval_dataset.enable_translation()

    if args.disable_resolution_augmentation:
        train_dataset.disable_resolution_augmentation(fix_scale_factor=args.fix_scale_factor)
        if eval_dataset is not None:
            eval_dataset.disable_resolution_augmentation(fix_scale_factor=args.fix_scale_factor)
    else:
        train_dataset.enable_resolution_augmentation()
        if eval_dataset is not None:
            eval_dataset.enable_resolution_augmentation()

    model, model_for_eval = _build_model(args, train_dataset, device, distributed, rank, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = GradScaler(device.type, enabled=(device.type != "cpu" and not args.no_amp))
    scheduler = None
    if args.lr_scheduler == "cosine":
        optimizer_steps_per_epoch = max(
            (min(len(train_loader), args.max_train_batches or len(train_loader))
             + args.accumulation_steps - 1) // args.accumulation_steps,
            1,
        )
        total_training_steps = optimizer_steps_per_epoch * max(args.epochs, 1)
        warmup_epochs = resolve_warmup_epochs(args.epochs, args.warmup_epochs)
        warmup_steps = warmup_epochs * optimizer_steps_per_epoch
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

    args.optimizer_updates_per_epoch = max(
        (min(len(train_loader), args.max_train_batches or len(train_loader))
         + args.accumulation_steps - 1) // args.accumulation_steps, 1,
    )
    args.training_world_size = world_size
    ema = ((MultiModelEMA(model_for_eval, args.ema_decays) if args.ema_decays
            else ModelEMA(model_for_eval, args.ema_decay)) if args.ema else None)
    best_selection = dict(depth=None)
    start_epoch, best_eval_acc = _resume_if_needed(model, optimizer, scaler, scheduler, args, device, ema, best_selection)
    if start_epoch > end_epoch:
        if (not distributed) or rank == 0:
            if end_epoch == args.epochs:
                # Recover final.pt if the previous job stopped after writing
                # the completed last.pt but before publishing final.pt.
                _save_checkpoint(args.save_path, args, model_for_eval, optimizer, scaler,
                                 scheduler, start_epoch - 1, best_eval_acc, ema, best_selection['depth'])
                if ema is not None:
                    _save_all_ema_checkpoints(args.ema_save_path, args, ema, start_epoch - 1)
            print(f"Checkpoint already completed epoch {start_epoch - 1}; target epoch {end_epoch} reached.", flush=True)
        if distributed and dist.is_initialized():
            dist.destroy_process_group()
        return
    is_main_process = (not distributed) or rank == 0

    batches_per_epoch = min(len(train_loader), args.max_train_batches or len(train_loader))
    updates_per_epoch = (batches_per_epoch + args.accumulation_steps - 1) // args.accumulation_steps
    loop_sampler = LoopDepthSampler(
        args.train_loop_min, args.train_loop_max, args.train_loop_mean, args.train_loop_std, args.seed,
    )
    if is_main_process and loop_sampler is not None:
        print("[loop-depth-distribution] " + json.dumps(loop_sampler.config()), flush=True)
    if is_main_process:
        parameter_count = sum(p.numel() for name, p in model_for_eval.named_parameters()
                              if p.requires_grad and "task_token" not in name)
        print(f"[gaussian-objective] K=gaussian[{args.train_loop_min},{args.train_loop_max}] "
              f"mean={args.train_loop_mean} std={args.train_loop_std} "
              f"loss=CE_K fusion_queries=all gate_activation=silu "
              f"gate_transform={getattr(args, 'fusion_gate_transform', 'bounded_tanh')} "
              f"eval_depth={args.eval_loop_steps}", flush=True)
        if args.save_path:
            config_path = Path(args.save_path).parent / "training_config.json"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(vars(args), indent=2) + "\n")
        print(f"[training-config] world_size={world_size} batch_size={args.batch_size} "
              f"accumulation_steps={args.accumulation_steps} global_batch="
              f"{world_size * args.batch_size * args.accumulation_steps} "
              f"parameters_excluding_task_tokens={parameter_count}", flush=True)
        if ema is not None:
            print("[ema-config] raw=true " + json.dumps({
                str(average.decay): {
                    "best": _ema_checkpoint_path(args.ema_best_save_path, average.decay, ema.decay),
                    "final": _ema_checkpoint_path(args.ema_save_path, average.decay, ema.decay),
                } for average in _ema_models(ema)
            }), flush=True)
    all_steps = max(batches_per_epoch * args.epochs, 1)
    resumed_steps = min(max(start_epoch - 1, 0), args.epochs) * batches_per_epoch
    progress_reporter = None
    if is_main_process:
        progress_reporter = TotalProgressReporter(
            "TRAIN",
            all_steps,
            initial_completed=resumed_steps,
            resolution_percent=0.1,
        )
        progress_reporter.start()

    try:
        for epoch in range(start_epoch, end_epoch + 1):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            model.train()
            running_loss = 0.0
            objective_totals = {}
            sample_count = 0
            train_exact = 0
            train_examples = 0
            avg_steps_accumulator = 0.0
            epoch_start = time.time()
            total_batches = batches_per_epoch
            optimizer.zero_grad(set_to_none=True)
            epoch_depth_counts = {}
            sampled_depth = None

            for step, batch in enumerate(islice(train_loader, total_batches), 1):
                inputs = batch["inputs"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                targets = batch["targets"].to(device)
                task_ids = batch["task_ids"].to(device)

                group_start = ((step - 1) // args.accumulation_steps) * args.accumulation_steps + 1
                micro_batches_in_group = min(
                    args.accumulation_steps,
                    total_batches - group_start + 1,
                )
                should_update = step % args.accumulation_steps == 0 or step == total_batches
                if loop_sampler is not None and step == group_start:
                    update_index = (epoch - 1) * updates_per_epoch + (step - 1) // args.accumulation_steps
                    sampled_depth = loop_sampler.sample_update(update_index, device, distributed)
                    epoch_depth_counts[sampled_depth] = epoch_depth_counts.get(sampled_depth, 0) + 1
                    if is_main_process and update_index < 4:
                        print(f"[loop-depth] update={update_index} depth={sampled_depth} "
                              f"micro_batches={micro_batches_in_group}", flush=True)
                sync_context = (
                    model.no_sync()
                    if distributed and not should_update
                    else nullcontext()
                )

                with sync_context:
                    autocast_device_type = device.type
                    with autocast(device_type=autocast_device_type, enabled=scaler.is_enabled()):
                        logits, metadata = model(
                            inputs, task_ids, attention_mask=attention_mask,
                            override_max_steps=sampled_depth,
                        )
                        loss, objective_metrics = compute_training_objective(
                            logits, None, targets, args, ignore_index=IGNORE_INDEX,
                        )

                    loss_for_backward = loss / micro_batches_in_group
                    scaler.scale(loss_for_backward).backward()

                if should_update:
                    scale_before = scaler.get_scale()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    if ema is not None and scaler.get_scale() >= scale_before:
                        ema.update(model_for_eval)
                    optimizer.zero_grad(set_to_none=True)

                    if scheduler is not None:
                        scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                for key, value in objective_metrics.items():
                    objective_totals[key] = objective_totals.get(key, 0.0) + value.item() * inputs.size(0)
                sample_count += inputs.size(0)
                avg_steps_accumulator += metadata.exit_steps.float().sum().item()

                predictions = logits.argmax(dim=1)
                for idx in range(inputs.size(0)):
                    target = targets[idx]
                    prediction = predictions[idx]
                    valid = target != IGNORE_INDEX
                    is_exact = bool(torch.equal(prediction[valid], target[valid])) if valid.any() else False
                    train_exact += int(is_exact)
                    train_examples += 1

                if progress_reporter is not None:
                    completed_steps = (epoch - 1) * batches_per_epoch + step
                    progress_reporter.update(completed_steps)

            epoch_duration = time.time() - epoch_start if len(train_loader) > 0 else 0.0
            if distributed and dist.is_initialized():
                train_totals = torch.tensor(
                    [running_loss, sample_count, train_exact, train_examples, avg_steps_accumulator],
                    dtype=torch.float64,
                    device=device,
                )
                dist.all_reduce(train_totals, op=dist.ReduceOp.SUM)
                running_loss, sample_count, train_exact, train_examples, avg_steps_accumulator = train_totals.tolist()
                metric_keys = sorted(objective_totals)
                metric_values = torch.tensor([objective_totals[key] for key in metric_keys],
                                             dtype=torch.float64, device=device)
                dist.all_reduce(metric_values)
                objective_totals = dict(zip(metric_keys, metric_values.tolist()))

            avg_train_loss = running_loss / max(sample_count, 1)
            train_acc = train_exact / max(train_examples, 1)
            avg_train_steps = avg_steps_accumulator / max(train_examples, 1)
            objective_means = {key: value / max(sample_count, 1)
                               for key, value in objective_totals.items()}

            log_parts = [
                f"epoch={epoch}",
                f"train_loss={avg_train_loss:.4f}",
                f"train_acc={train_acc:.4f}",
                f"avg_train_steps={avg_train_steps:.2f}",
                f"epoch_time={epoch_duration:.1f}s",
            ]
            log_parts.extend(f"{key}={value:.6f}" for key, value in sorted(objective_means.items()))

            current_lr = optimizer.param_groups[0]["lr"] if optimizer.param_groups else args.learning_rate
            log_parts.append(f"lr={current_lr:.6f}")

            eval_loss = None
            eval_acc = None
            eval_steps = None
            save_best_checkpoint = False
            depth_rows = []
            ema_depth_rows = []
            ema_eval_loss = ema_eval_acc = None
            ema_results = {}
            if eval_loader is not None:
                eval_loss, eval_acc, eval_steps = evaluate(
                    # Uneven validation partitions must bypass DDP's per-forward
                    # collectives; evaluate() reduces the final totals once.
                    model_for_eval,
                    eval_loader,
                    device,
                    args,
                    distributed=distributed,
                )
                if is_main_process:
                    log_parts.append(f"eval_loss={eval_loss:.4f}")
                    log_parts.append(f"eval_acc={eval_acc:.4f}")
                    log_parts.append(f"eval_steps={eval_steps:.2f}")
                if args.best_checkpoint_depths:
                    depth_rows = evaluate_depths(
                        model_for_eval, eval_loader, device,
                        depths=args.best_checkpoint_depths, fix_scale_factor=args.fix_scale_factor,
                        distributed=distributed,
                    )
                    if is_main_process:
                        for row in depth_rows:
                            log_parts.append(f"depth{row['step']}_ce={row['ce']:.6f}")
                            log_parts.append(f"depth{row['step']}_acc={row['exact_accuracy']:.6f}")

                raw_score, raw_depth = eval_acc, args.eval_loop_steps
                if args.best_checkpoint_depths:
                    best_row = max((r for r in depth_rows if r['step'] in args.best_checkpoint_depths),
                                   key=lambda row: row['exact_accuracy'])
                    raw_score, raw_depth = best_row['exact_accuracy'], best_row['step']
                if raw_score is not None and raw_score > best_eval_acc and args.best_save_path and is_main_process:
                    best_eval_acc = raw_score
                    best_selection['depth'] = raw_depth
                    save_best_checkpoint = True

                if ema is not None:
                    ema_depths = sorted({args.eval_loop_steps} | set(args.best_checkpoint_depths or []))
                    selection_depths = args.best_checkpoint_depths or [args.eval_loop_steps]
                    for average in _ema_models(ema):
                        label = "ema" if average is ema else f"ema_{average.decay}"
                        with average.apply_to(model_for_eval):
                            rows = evaluate_depths(
                                model_for_eval, eval_loader, device, depths=ema_depths,
                                fix_scale_factor=args.fix_scale_factor, distributed=distributed,
                            )
                        selection_row = max((row for row in rows if row['step'] in selection_depths),
                                            key=lambda row: row['exact_accuracy'])
                        score = selection_row["exact_accuracy"]
                        if score > average.best_accuracy:
                            average.best_accuracy = score
                            average.best_depth = selection_row['step']
                            if is_main_process:
                                _save_ema_checkpoint(
                                    _ema_checkpoint_path(args.ema_best_save_path, average.decay, ema.decay),
                                    args, average, epoch)
                        ema_results[str(average.decay)] = dict(
                            eval_loss=selection_row["ce"], eval_accuracy=score,
                            best_eval_accuracy=average.best_accuracy, updates=average.num_updates,
                            best_validation_depth=average.best_depth,
                            depth_validation=rows,
                        )
                        if average is ema:
                            ema_depth_rows = rows
                            ema_eval_loss, ema_eval_acc = selection_row["ce"], score
                        if is_main_process:
                            for row in rows:
                                log_parts.append(f"{label}_depth{row['step']}_ce={row['ce']:.6f}")
                                log_parts.append(f"{label}_depth{row['step']}_acc={row['exact_accuracy']:.6f}")

            if save_best_checkpoint and is_main_process:
                _save_checkpoint(args.best_save_path, args, model_for_eval, optimizer, scaler,
                                 scheduler, epoch, best_eval_acc, ema, best_selection['depth'])

            if distributed and dist.is_initialized():
                save_flag = torch.tensor(int(save_best_checkpoint), device=device)
                dist.broadcast(save_flag, src=0)
                save_best_checkpoint = bool(save_flag.item())

                best_acc_tensor = torch.tensor(best_eval_acc if is_main_process else 0.0, device=device)
                dist.broadcast(best_acc_tensor, src=0)
                if not is_main_process:
                    best_eval_acc = best_acc_tensor.item()

                if save_best_checkpoint:
                    dist.barrier()


            if is_main_process:
                if loop_sampler is not None:
                    epoch_metrics = {
                        "epoch": epoch, "train_loss": avg_train_loss, "train_accuracy": train_acc,
                        "average_train_loops": avg_train_steps, "objective": objective_means,
                        "loop_update_counts": epoch_depth_counts, "loop_probabilities": loop_sampler.config(),
                        "eval_loss": eval_loss, "eval_accuracy": eval_acc,
                        "best_eval_accuracy": best_eval_acc, "best_validation_depth": best_selection['depth'],
                        "depth_validation": depth_rows,
                    }
                    if ema is not None:
                        epoch_metrics.update(
                            ema_eval_loss=ema_eval_loss, ema_eval_accuracy=ema_eval_acc,
                            ema_best_eval_accuracy=ema.best_accuracy, ema_updates=ema.num_updates,
                            ema_best_validation_depth=ema.best_depth,
                            ema_depth_validation=ema_depth_rows,
                            ema_results=ema_results,
                        )
                    metrics_dir = Path(args.save_path or args.best_save_path or "outputs/run.pt").parent / "epoch_metrics"
                    metrics_dir.mkdir(parents=True, exist_ok=True)
                    (metrics_dir / f"epoch_{epoch:04d}.json").write_text(json.dumps(epoch_metrics, indent=2) + "\n")
                _save_last(args, model_for_eval, optimizer, scaler, scheduler, epoch, best_eval_acc, ema, best_selection['depth'])
                print(" | ".join(log_parts), flush=True)

        if progress_reporter is not None and end_epoch == args.epochs:
            progress_reporter.finish()

    finally:
        if distributed and dist.is_initialized():
            dist.barrier()

    if args.save_path and ((not distributed) or rank == 0):
        _save_checkpoint(args.save_path, args, model_for_eval, optimizer, scaler,
                         scheduler, end_epoch, best_eval_acc, ema, best_selection['depth'])
        if ema is not None:
            _save_all_ema_checkpoints(args.ema_save_path, args, ema, end_epoch)

    if distributed and dist.is_initialized():
        dist.destroy_process_group()
