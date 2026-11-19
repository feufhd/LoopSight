"""Command-line arguments for LoopSight training and adaptation."""
import argparse
import math
from loopsight.checkpoint import ARCHITECTURE

def add_resume_checkpoints(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--resume-skip-task-token",
        action="store_true",
        help="Ignore task token embedding weights in the checkpoint when resuming.",
    )
    parser.add_argument(
        "--resume-reset-optimizer",
        action="store_true",
        help="Do not load optimizer/scheduler/scaler states when resuming a checkpoint.",
    )
    parser.add_argument(
        "--resume-reset-epoch",
        action="store_true",
        help="Ignore the stored epoch in the checkpoint and restart training from epoch 1.",
    )


def add_speed_optimizer_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--no-compile", action="store_true", help="Disable torch.compile optimization (useful for debugging).")
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="reduce-overhead",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="torch.compile mode: 'default' (balanced), 'reduce-overhead' (faster), 'max-autotune' (slowest compile, fastest runtime).",
    )
    parser.add_argument("--no-amp", action="store_true", help="Disable automatic mixed precision training.")

def add_model_options(parser):
    parser.add_argument("--fusion-mode", choices=("task_lowrank_channel",), default="task_lowrank_channel")
    parser.add_argument("--fusion-gate-rank", type=int, default=16)
    parser.add_argument("--fusion-gate-heads", type=int, default=2)
    parser.add_argument("--fusion-gate-logit-bound", type=float, default=math.log(16))
    parser.add_argument("--step-embedding-type", choices=("asymptotic",), default="asymptotic")
    parser.add_argument("--asymptotic-channels", type=int, default=8)
    parser.add_argument("--local-clock-frequencies", type=int, default=3)
    parser.add_argument("--modulation-hidden-dim", type=int, default=64)
    parser.add_argument("--loop-state-rmsnorm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--runtime-max-steps", type=int, default=12)
    parser.set_defaults(fusion_gate_activation="silu", fusion_gate_context="all_memory_and_task",
        fusion_gate_transform="bounded_tanh", train_loop_sampling="fixed", progressive_loss="none",
        progressive_loss_weight=0.0, direct_loss_weight=0.0, no_step_embedding=False,
        train_dynamic_exit=False, eval_dynamic_exit=False, disable_exit_gate=True,
        exit_gate_threshold=0.6, exit_on_entropy_stable=False, entropy_patience=2,
        entropy_threshold_pct=0.05, image_size=64, num_colors=12, embed_dim=256,
        mlp_dim=512, num_heads=8, loop_core_depth=6, min_loop_steps=12, max_loop_steps=12,
        batch_size=8, epochs=100, no_compile=True)

def build_parser():
    parser = argparse.ArgumentParser(description="Train ARC-specific Vision Transformer.")
    parser.set_defaults(distributed=False)
    add_resume_checkpoints(parser)
    add_speed_optimizer_args(parser)
    add_model_options(parser)

    parser.add_argument("--data-root", type=str, default="raw_data/ARC-AGI")
    parser.add_argument(
        "--train-split",
        dest="train_split",
        default="training",
        help="Dataset split to use for training.",
    )
    parser.add_argument(
        "--eval-split",
        dest="eval_split",
        default="training",
        help="Dataset split to evaluate. Use '' to disable evaluation.",
    )
    parser.add_argument(
        "--eval-subset",
        dest="eval_subset",
        choices=("train", "test"),
        default="test",
        help="Which example subset to use when evaluating a split.",
    )
    parser.set_defaults(architecture=ARCHITECTURE)
    parser.add_argument("--image-size", type=int, default=30)
    parser.add_argument("--num-colors", type=int, default=10)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--num-task-tokens", type=int, default=1, help="Number of task tokens to use in the model.")
    parser.add_argument("--loop-core-depth", type=int, default=2, help="Number of unique transformer blocks reused per loop iteration (LoopARCViT only).")
    parser.add_argument("--max-loop-steps", type=int, default=6, help="Maximum number of loop iterations for LoopARCViT inference/training.")
    parser.add_argument("--min-loop-steps", type=int, default=2, help="Minimum loop iterations before the exit gate may trigger.")
    parser.add_argument("--lr-scheduler", type=str, default="cosine", choices=("none", "cosine"))
    parser.add_argument("--warmup-epochs", type=int, default=None,
                        help="Override cosine warmup; default is min(epochs // 5, 10).")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-path", type=str, default=None, help="Optional path to store the final trained checkpoint")
    parser.add_argument("--best-save-path", type=str, default=None, help="Optional path to store the checkpoint achieving the best evaluation accuracy.")

    parser.add_argument("--ttt-num-each", type=int, default=2)
    parser.add_argument('--ttt-ema', action='store_true',
                        help='Also export and predict with EMA of successful TTT optimizer updates.')
    parser.add_argument('--ttt-ema-decay', type=float, default=0.99,
                        help='TTT EMA decay per successful optimizer update (default: 0.99).')
    parser.add_argument('--ttt-ema-decays', type=float, nargs='+', default=None,
                        help='Track several EMA decays in one TTT run; the last is the primary EMA.')
    parser.add_argument(
        "--ttt-checkpoint-dir", type=str, default=None,
        help="Save each adapted model and task-token mapping under DIR/TASK/attempt_N.pt.",
    )
    parser.add_argument('--ttt-resume-dir', default=None,
                        help='Atomically save each TTT attempt after each epoch and resume it automatically.')
    parser.add_argument(
        "--ttt-start-attempt",
        type=int,
        default=0,
        help="First TTT attempt index to run; used to resume partially completed tasks.",
    )
    parser.add_argument(
        "--quiet-progress",
        action="store_true",
        help="Suppress per-task/epoch progress so an outer runner can print total progress only.",
    )
    parser.add_argument("--patch-size", type=int, default=2)

    parser.add_argument(
        "--include-rearc",
        action="store_true",
        help="Add tasks from the Re-ARC dataset to the training set.",
    )
    parser.add_argument(
        "--rearc-path",
        type=str,
        default="raw_data/re_arc",
        help="Path to the Re-ARC dataset root.",
    )
    parser.add_argument(
        "--rearc-limit",
        type=int,
        default=-1,
        help="Maximum number of Re-ARC examples to include per task (use -1 for all examples).",
    )
    
  
    parser.add_argument('--disable-translation', action='store_true')
    parser.add_argument('--disable-resolution-augmentation', action='store_true')
    parser.add_argument("--fix-scale-factor", type=int, default=1)
    parser.add_argument('--num-attempts', type=int, default=10, help="Number of attempts per evaluation example.")
    parser.add_argument('--eval-save-name', type=str, default=None, help="Name for saving evaluation predictions.")
    parser.add_argument(
        '--output-subdir',
        type=str,
        default=None,
        help=(
            "Optional relative subdirectory under outputs/ used to group TTT results "
            "(for example: --output-subdir test_ttt10e)."
        ),
    )
    
    parser.set_defaults(image_size=64, num_colors=12, loop_core_depth=6, min_loop_steps=12, max_loop_steps=12, epochs=100, batch_size=8)
    return parser
