"""Train LoopSight on ARC with sampled depths and final-step EMA exports."""

import argparse
import json

from loopsight.training import arc as shared
from loopsight import SCALES


def parse_args(argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--scale", choices=SCALES, default="small")
    scale = probe.parse_known_args(argv)[0].scale
    config = SCALES[scale]
    parser = shared.build_parser()
    parser.description = __doc__
    parser.add_argument("--scale", choices=SCALES, default=scale)
    parser.add_argument("--dry-run", action="store_true")
    output = f"outputs/loopsight_{scale}/offline"
    parser.set_defaults(
        embed_dim=config.embed_dim, loop_core_depth=config.core_depth,
        fusion_gate_rank=config.gate_rank, train_loop_min=config.train_min,
        train_loop_max=config.train_max, train_loop_mean=config.loops,
        train_loop_std=config.train_std, eval_loop_steps=config.loops,
        max_loop_steps=config.loops, min_loop_steps=config.loops,
        learning_rate=config.learning_rate, ema=True, ema_decay=0.9999,
        save_path=f"{output}/final.pt", best_save_path=f"{output}/best.pt",
        last_save_path=f"{output}/last.pt", ema_save_path=f"{output}/ema_final.pt",
        ema_best_save_path=f"{output}/best_ema.pt",
        distributed=False, accumulation_steps=32,
        include_rearc=True, rearc_path="raw_data/re_arc", rearc_limit=-1,
        use_checkpoint=False, checkpoint_blocks=0, checkpoint_gate=False,
        no_compile=True, num_workers=4,
    )
    args = parser.parse_args(argv)
    args.model_name = f"LoopSight-{scale.capitalize()}"
    shared.validate_training_args(args)
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        print(json.dumps(vars(args), indent=2))
        return
    shared.train(args)


if __name__ == "__main__":
    main()
