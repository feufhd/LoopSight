"""Task-specific LoopSight adaptation with paired raw and final EMA predictions."""

import argparse
import copy
import json
import os
from pathlib import Path

import torch

from loopsight import SCALES
from loopsight.presets import model_from_args
from loopsight.training import ttt
from loopsight.training.ttt_args import validate_ttt_args
from utils.args import build_parser
from loopsight.checkpoint import validate_task_context_checkpoint, load_task_lowrank_state


def parse_args(argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--scale", choices=SCALES, default="small")
    probe.add_argument("--dataset", choices=("arc1", "arc2"), default="arc1")
    selected = probe.parse_known_args(argv)[0]
    config = SCALES[selected.scale]
    parser = build_parser()
    parser.description = __doc__
    parser.add_argument("--scale", choices=SCALES, default=selected.scale)
    parser.add_argument("--dataset", choices=("arc1", "arc2"), default=selected.dataset)
    parser.add_argument("--checkpoint", dest="resume_checkpoint")
    parser.add_argument("--tasks", nargs="+", help="Task IDs; defaults to the entire evaluation set")
    parser.add_argument("--ttt-split", default="eval_color_permute_ttt_9")
    parser.add_argument("--dry-run", action="store_true")
    output = f"outputs/loopsight_{selected.scale}/ttt/{selected.dataset}"
    parser.set_defaults(
        data_root="raw_data/ARC-AGI" if selected.dataset == "arc1" else "raw_data/ARC-AGI-2",
        embed_dim=config.embed_dim, loop_core_depth=config.core_depth,
        fusion_gate_rank=config.gate_rank, min_loop_steps=config.loops,
        max_loop_steps=config.loops, runtime_max_steps=config.loops,
        epochs=100, learning_rate=3e-4, warmup_epochs=10, batch_size=8,
        resume_checkpoint=f"outputs/loopsight_{selected.scale}/offline/ema_final.pt",
        resume_skip_task_token=True, resume_reset_epoch=True, resume_reset_optimizer=True,
        ttt_ema=True, ttt_ema_decay=0.998, ttt_num_each=2, num_attempts=10,
        ttt_checkpoint_dir=f"{output}/checkpoints", ttt_resume_dir=f"{output}/resume",
        output_subdir=f"loopsight_{selected.scale}/ttt/{selected.dataset}",
        eval_save_name="predictions", eval_split="evaluation", train_loop_sampling="fixed",
        no_compile=True, num_workers=0,
        independent_process=True,
    )
    return parser.parse_args(argv)


def load_model(args, train_dataset, device, distributed, rank, local_rank):
    if distributed:
        raise ValueError("Run task-specific TTT in a single-device process")
    checkpoint = torch.load(args.resume_checkpoint, map_location="cpu", weights_only=False)
    validate_task_context_checkpoint(args, checkpoint)
    model = model_from_args(args, train_dataset.num_tasks)
    load_task_lowrank_state(model, checkpoint, skip_task_token=True)
    return model.to(device)


def export_final_ema(args, task):
    """Give every completed adaptation attempt an explicit final-EMA export."""
    root = Path(args.ttt_checkpoint_dir)
    root = root.with_name(root.name + "_ema") / task
    for attempt in range(args.ttt_num_each):
        source = root / f"attempt_{attempt}.pt"
        checkpoint = torch.load(source, map_location="cpu", weights_only=False)
        if (checkpoint.get("weight_type") != "ema" or checkpoint.get("epoch") != args.epochs
                or checkpoint.get("attempt_idx") != attempt or checkpoint.get("ema_num_updates", 0) < 1):
            raise ValueError("Expected final-epoch EMA weights for the adaptation attempt")
        destination = root / f"attempt_{attempt}" / "ema_final.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        # A hard link shares the immutable export and adds no duplicate tensor storage.
        temporary = destination.with_name(f".ema_final.{os.getpid()}.tmp")
        try:
            temporary.hardlink_to(source)
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    evaluation = Path(args.data_root) / "data" / args.eval_split
    available = {path.stem for path in evaluation.glob("*.json")}
    tasks = sorted(available) if args.tasks is None else args.tasks
    if not tasks or len(tasks) != len(set(tasks)) or set(tasks) - available:
        raise ValueError("Select unique task IDs present in the evaluation split")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tasks = tasks[rank::world]
    for task in tasks:
        current = copy.copy(args)
        current.train_split = f"{args.ttt_split}/{task}"
        if not (Path(current.data_root) / "data" / current.train_split).is_dir():
            raise FileNotFoundError("Prepare the task's TTT augmentations before adaptation")
        validate_ttt_args(current)
        ttt.args = current
        ttt.train(current, model_loader=load_model)
        if current.ttt_ema:
            export_final_ema(current, task)


if __name__ == "__main__":
    main()
