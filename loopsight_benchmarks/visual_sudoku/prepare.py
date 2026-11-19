"""Prepare image-task metadata while reusing every shared Sudoku board file."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from .boards import DEFAULT_DATA, SudokuDataset
from .prepare_boards import validate_boards
from .dataset import PREPARED
from .fetch_assets import MNIST_PATH, write_json


def prepare(board_root=DEFAULT_DATA, mnist_path=MNIST_PATH, output=PREPARED,
            hard_count=1000, hard_seed=0, verify=False, image_size=252):
    if image_size not in (252, 256):
        raise ValueError("Visual-Sudoku image_size must be 252 or 256")
    board_root, mnist_path, output = map(Path, (board_root, mnist_path, output))
    datasets = {k: SudokuDataset("sudoku-ood", v, board_root) for k, v in
                {"train": "train", "id_test": "validation", "ood_test": "test"}.items()}
    for name, data in datasets.items():
        validate_boards(data.inputs, data.labels)
        expected = {"train": 9000, "id_test": 1000, "ood_test": 18000}[name]
        if len(data) != expected:
            raise ValueError(f"Unexpected {name} count: {len(data)}")
    if not 0 < hard_count <= len(datasets["ood_test"]) or hard_seed < 0:
        raise ValueError("Invalid hard-test count/seed")
    train_keys = {b.tobytes() for b in datasets["train"].inputs}
    for key in ("id_test", "ood_test"):
        if any(b.tobytes() in train_keys for b in datasets[key].inputs):
            raise ValueError(f"Exact train/{key} board overlap")
    with np.load(mnist_path, allow_pickle=False) as mnist:
        for pool, count in (("train", 60000), ("test", 10000)):
            if mnist[f"x_{pool}"].shape != (count, 28, 28) or mnist[f"x_{pool}"].dtype != np.uint8:
                raise ValueError(f"Invalid MNIST {pool} images")
            if mnist[f"y_{pool}"].shape != (count,) or set(mnist[f"y_{pool}"].tolist()) != set(range(10)):
                raise ValueError(f"Invalid MNIST {pool} labels")
        prototypes = np.zeros((10, 28, 28), dtype=np.float32)
        for digit in range(1, 10):
            prototypes[digit] = mnist["x_test"][mnist["y_test"] == digit].mean(axis=0) / 255.0
    rows = np.sort(np.random.default_rng(hard_seed).choice(18000, hard_count, replace=False))
    padding = image_size - 252
    expected = dict(schema_version=1, task="Visual-Sudoku", image_size=image_size, digit_size=28,
                    padding={"top": 0, "left": 0, "bottom": padding, "right": padding},
                    blank_input="uniform(-0.1,0.1), following the official inference notebook",
                    target="independently sampled MNIST glyphs; zero padding",
                    mnist_pools={"train": "test", "validation_and_test": "train", "prototypes": "test"},
                    counts={"train": 9000, "validation": 1000, "easy_test": 1000,
                            "hard_test": hard_count, "hard_test_full": 18000},
                    hard_seed=hard_seed, hard_count=hard_count,
                    hard_selection="local reproducible random subset; paper's exact 1000 indices are unspecified",
                    validation_note="validation and easy_test are the same SATNet held-out 1000 boards, not independent splits")
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if any(existing.get(k) != v for k, v in expected.items()):
            raise ValueError("Prepared protocol differs; select a new --output rather than changing test rows")
        if not np.array_equal(np.load(output / "hard_test_rows.npy", allow_pickle=False), rows):
            raise ValueError("Hard-test indices differ from the recorded seed")
        with np.load(output / "prototypes.npz", allow_pickle=False) as stored:
            if not np.array_equal(stored["images"], prototypes):
                raise ValueError("Prototype values differ")
    elif verify:
        raise FileNotFoundError(manifest_path)
    else:
        output.mkdir(parents=True, exist_ok=True)
        np.save(output / "hard_test_rows.npy", rows, allow_pickle=False)
        np.savez_compressed(output / "prototypes.npz", images=prototypes)
        write_json(manifest_path, expected)
    print(json.dumps({"counts": expected["counts"], "shared_boards": str(board_root),
                      "mnist": str(mnist_path), "verified": True}, ensure_ascii=False), flush=True)
    return expected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--board-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--mnist-path", type=Path, default=MNIST_PATH)
    parser.add_argument("--output", type=Path, default=PREPARED)
    parser.add_argument("--hard-count", type=int, default=1000)
    parser.add_argument("--hard-seed", type=int, default=0)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--image-size", type=int, choices=(252, 256), default=252)
    args = parser.parse_args()
    prepare(args.board_root, args.mnist_path, args.output, args.hard_count, args.hard_seed, args.verify, args.image_size)


if __name__ == "__main__":
    main()
