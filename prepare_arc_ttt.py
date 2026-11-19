"""Prepare reproducible per-task ARC evaluation augmentations for TTT."""

import argparse

from utils.data_augmentation import augment_raw_data_split_per_task


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="raw_data/ARC-AGI-2")
    parser.add_argument("--output-subdir", default="eval_color_permute_ttt_9")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    augment_raw_data_split_per_task(
        args.data_root, "evaluation", output_subdir=args.output_subdir,
        num_permuate=9, seed=args.seed,
    )


if __name__ == "__main__":
    main()
