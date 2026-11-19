import math
from pathlib import Path
from loopsight.checkpoint import ARCHITECTURE, FUSION_MODE
from .common import validate_model_args

def validate_ttt_args(args):
    if args.architecture != ARCHITECTURE or args.fusion_mode != FUSION_MODE:
        raise ValueError("This TTT entry point requires the LoopSight architecture")
    if args.architecture == ARCHITECTURE:
        if args.fusion_gate_heads < 1 or args.fusion_gate_rank % args.fusion_gate_heads:
            raise ValueError("--fusion-gate-rank must be divisible by positive --fusion-gate-heads")
    if args.distributed:
        raise ValueError("Run independent task processes; distributed TTT is not supported here")
    if not args.resume_checkpoint or not Path(args.resume_checkpoint).is_file():
        raise FileNotFoundError(f"LoopSight checkpoint not found: {args.resume_checkpoint}")
    if not (args.resume_skip_task_token and args.resume_reset_epoch and args.resume_reset_optimizer):
        raise ValueError("TTT requires --resume-skip-task-token and both resume reset flags")
    for name in ("epochs", "batch_size", "fusion_gate_rank", "num_task_tokens", "runtime_max_steps",
                 "min_loop_steps", "max_loop_steps", "ttt_num_each"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.min_loop_steps != args.max_loop_steps or args.train_loop_sampling != "fixed":
        raise ValueError("LoopSight TTT requires a fixed training loop budget")
    if args.progressive_loss != "none" or args.progressive_loss_weight != 0 or args.direct_loss_weight != 0:
        raise ValueError("LoopSight TTT supervises only the final fused CE")
    if args.no_step_embedding:
        raise ValueError("LoopSight requires asymptotic loop conditioning")
    if not 0 <= args.ttt_start_attempt <= args.ttt_num_each:
        raise ValueError("--ttt-start-attempt must be between 0 and --ttt-num-each")
    validate_model_args(args)
