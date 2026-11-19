"""Shared validation, final-output supervision and activation recomputation."""
from functools import wraps
import math
import torch
from torch.amp import GradScaler
from utils.lr_scheduler import get_cosine_schedule_with_warmup, resolve_warmup_epochs
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def validate_model_args(args):
    if args.step_embedding_type != "asymptotic":
        raise ValueError("LoopSight uses asymptotic loop conditioning")
    if args.asymptotic_channels < 1 or args.modulation_hidden_dim < 1:
        raise ValueError("Clock and modulation dimensions must be positive")
    if not 1 <= args.local_clock_frequencies <= args.loop_core_depth:
        raise ValueError("Local clock frequencies must be between one and the core depth")
    if not math.isfinite(args.fusion_gate_logit_bound) or args.fusion_gate_logit_bound < 0:
        raise ValueError("Gate logit bound must be finite and nonnegative")
    if getattr(args, "train_loop_sampling", "fixed") == "gaussian":
        if args.train_loop_min < 1 or args.train_loop_max < args.train_loop_min:
            raise ValueError("Invalid training loop range")
        if not math.isfinite(args.train_loop_mean) or not math.isfinite(args.train_loop_std) or args.train_loop_std <= 0:
            raise ValueError("Loop sampling requires a finite mean and positive finite spread")

def compute_training_objective(logits, intermediates, targets, args, *, ignore_index,
                               raw_intermediates=None, cross_entropy_fn=None):
    """Supervise only the final prediction memory's pixel logits."""
    criterion = cross_entropy_fn or F.cross_entropy
    loss = (criterion(logits.float(), targets, ignore_index=ignore_index)
            if (targets != ignore_index).any() else logits.float().sum() * 0.0)
    return loss, {"task_ce": loss.detach()}

def checkpoint_forward(module):
    """Recompute a module without changing its parameters or checkpoint keys."""
    original = module.forward

    @wraps(original)
    def forward(*args, **kwargs):
        if module.training and torch.is_grad_enabled():
            return checkpoint(original, *args, use_reentrant=False,
                              preserve_rng_state=True, **kwargs)
        return original(*args, **kwargs)

    module.forward = forward

def enable_checkpointing(model, args):
    if args.use_checkpoint:
        for layer in list(model.core_layers)[:args.checkpoint_blocks]:
            checkpoint_forward(layer)
    if args.checkpoint_gate:
        checkpoint_forward(model.fusion_gate)
    return model

def load_optimizer(args, model, device, distributed, rank):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    scaler = GradScaler(device.type, enabled=(device.type != "cpu" and not args.no_amp))
    if (not distributed) or rank == 0:
        if scaler.is_enabled():
            print("Using automatic mixed precision (AMP) training")
        else:
            print("AMP disabled (CPU or --no-amp flag)")
    if args.lr_scheduler == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, resolve_warmup_epochs(args.epochs, getattr(args, "warmup_epochs", None)), args.epochs)
    else:
        scheduler = None

    return optimizer, scaler, scheduler
