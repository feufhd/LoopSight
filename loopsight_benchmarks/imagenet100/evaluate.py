"""Evaluate raw, best, or final-EMA LoopSight ImageNet-100 checkpoints."""
import argparse
import json
from pathlib import Path
import time

import torch
from loopsight.runtime import resolve_device
import torch.distributed as dist

from .dataset import DEFAULT_DATA, DistributedEvalSampler, evaluate, image_folder, make_loader
from .engine import setup_execution, write_json
from .config import LOOP_MODELS, ModelConfig, build_model
from .subsets import DATASETS, dataset_spec


def select_weights(payload, requested):
    kind = payload.get("weights", "raw_resume")
    is_ema = "ema" in kind
    if requested == "ema":
        if payload.get("ema") is not None:
            return payload["ema"], "ema"
        if not is_ema:
            raise ValueError("Checkpoint contains no EMA weights")
    elif requested == "raw" and is_ema:
        raise ValueError("This inference export contains EMA weights only")
    return payload["model"], kind


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--weights", choices=("auto", "raw", "ema"), default="auto")
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--dataset", choices=DATASETS, help="Default: infer the fixed selection from checkpoint class count")
    p.add_argument("--depths", type=int, nargs="+")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    p.set_defaults(device="auto")
    p.add_argument("--amp", choices=("none", "bf16", "fp16"), default="bf16")
    p.add_argument("--cpu-threads", type=int, default=4)
    p.add_argument("--output", type=Path)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    if min(a.batch_size, a.cpu_threads) < 1 or a.workers < 0 or (a.depths and min(a.depths) < 1):
        p.error("Invalid batch size, workers, threads or loop depth")
    if resolve_device(a.device).type == "cpu" and a.amp == "fp16":
        p.error("FP16 AMP requires an accelerator")
    if a.dry_run:
        print(json.dumps(dict(mode="validation_plan_only", checkpoint=str(a.checkpoint),
            weights=a.weights, data=str(a.data_root / "val"), image_size=224,
            depths=a.depths, batch_size=a.batch_size, metrics=["loss", "top1", "top5"],
            partition="strided_without_padding_or_duplicates"), indent=2))
        return
    if a.checkpoint is None:
        p.error("--checkpoint is required for evaluation")
    torch.set_num_threads(a.cpu_threads)
    try:
        device, rank, world = setup_execution(a.device)
        payload = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
        config = ModelConfig(**payload["model_config"])
        state, weights = select_weights(payload, a.weights)
        model = build_model(config).to(device)
        model.load_state_dict(state, strict=True)
        subset = dataset_spec(a.dataset or ("imagenet100" if config.num_classes == 100 else "imagenet1k"))
        if subset["classes"] != config.num_classes:
            raise ValueError("Selected dataset and classifier output size differ")
        dataset = image_folder(a.data_root, "val", config.num_classes, class_ids=subset["class_ids"],
                               expected_rows=subset["expected_rows"]["val"])
        loader = make_loader(dataset, a.batch_size, a.workers, device,
                             DistributedEvalSampler(dataset, rank, world))
        depths = (a.depths or [config.loops]) if config.name in LOOP_MODELS else [0]
        started = time.monotonic()
        results = {str(depth): evaluate(model, loader, device, depth, a.amp) for depth in depths}
        if any(result["examples"] != len(dataset) for result in results.values()):
            raise ValueError("Validation coverage differs from the dataset")
        report = dict(checkpoint=str(a.checkpoint), weights=weights,
            checkpoint_step=payload.get("step"), model_config=config.to_dict(), split="val",
            data_root=str(a.data_root), dataset_selection=subset, world_size=world, results=results,
            elapsed_seconds=round(time.monotonic() - started, 3))
        if rank == 0:
            print(json.dumps(report, indent=2))
            if a.output:
                a.output.parent.mkdir(parents=True, exist_ok=True)
                write_json(a.output, report)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
