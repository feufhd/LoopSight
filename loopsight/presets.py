"""Paper configurations and checkpoint-compatible model construction."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .model import LoopSight


@dataclass(frozen=True)
class ScaleConfig:
    embed_dim: int
    core_depth: int
    gate_rank: int
    loops: int
    train_min: int
    train_max: int
    train_std: float
    learning_rate: float


SCALES = {
    "small": ScaleConfig(256, 6, 16, 12, 6, 18, 3.0, 3e-4),
    "medium": ScaleConfig(384, 8, 24, 6, 3, 9, 1.5, 3e-4),
    "large": ScaleConfig(512, 10, 32, 6, 3, 9, 1.5, 2.5e-4),
}


def build_loopsight(scale="small", *, num_tasks=400, **overrides):
    config = SCALES[scale]
    options = dict(embed_dim=config.embed_dim, loop_core_depth=config.core_depth,
                   fusion_gate_rank=config.gate_rank, max_loop_steps=config.loops)
    options.update(overrides)
    return LoopSight(num_tasks=num_tasks, **options)


def loopsight_small(**kwargs):
    return build_loopsight("small", **kwargs)


def loopsight_medium(**kwargs):
    return build_loopsight("medium", **kwargs)


def loopsight_large(**kwargs):
    return build_loopsight("large", **kwargs)


def model_from_args(args, num_tasks):
    if args.step_embedding_type != "asymptotic":
        raise ValueError("LoopSight uses asymptotic loop conditioning")
    if args.fusion_mode != "task_lowrank_channel":
        raise ValueError("LoopSight uses the context cross-attention channel gate")
    return LoopSight(
        num_tasks=num_tasks, image_size=args.image_size, patch_size=args.patch_size,
        num_colors=args.num_colors, num_task_tokens=args.num_task_tokens,
        embed_dim=args.embed_dim, loop_core_depth=args.loop_core_depth,
        max_loop_steps=args.max_loop_steps, num_heads=args.num_heads,
        mlp_dim=args.mlp_dim, dropout=args.dropout,
        asymptotic_channels=args.asymptotic_channels,
        local_clock_frequencies=args.local_clock_frequencies,
        modulation_hidden_dim=args.modulation_hidden_dim,
        fusion_gate_rank=args.fusion_gate_rank, fusion_gate_heads=args.fusion_gate_heads,
        fusion_gate_logit_bound=args.fusion_gate_logit_bound,
        loop_state_rmsnorm=args.loop_state_rmsnorm,
        loop_drop_path_rate=getattr(args, "loop_drop_path_rate", 0.0),
        loop_drop_path_depth=getattr(args, "loop_drop_path_depth", 6),
    )


def load_checkpoint(path, *, device="cpu"):
    from .checkpoint import validate_task_context_checkpoint
    from .checkpoint import normalize_model_state

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = SimpleNamespace(**checkpoint["config"])
    validate_task_context_checkpoint(args, checkpoint)
    state = normalize_model_state(checkpoint["model_state"])
    model = model_from_args(args, state["task_token_embed.weight"].shape[0])
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), checkpoint["config"]
