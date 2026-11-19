"""Exact benchmark splits and reversible Sudoku symmetry augmentation."""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import torch

BENCH_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("raw_data/visual_sudoku/boards")
SPLITS = {
    "sudoku-ood": {"train": "train", "validation": "id_test", "test": "ood_test"},
}


class SudokuDataset:
    def __init__(self, task, split, root=DEFAULT_DATA):
        self.task, self.split = task, split
        folder = Path(root) / task.replace("-", "_")
        name = SPLITS[task][split]
        path = folder / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}. Run python -m loopsight_benchmarks.visual_sudoku.prepare_boards --download")
        with np.load(path, allow_pickle=False) as data:
            self.inputs = data["inputs"]
            self.labels = data["labels"]
            self.ratings = (data["ratings"] if "ratings" in data else
                            (self.inputs == 0).sum(axis=(1, 2)))
            self.source_rows = (data["source_row_indices"] if "source_row_indices" in data else
                                np.arange(len(self.inputs)))
        manifest = json.loads((folder / "manifest.json").read_text())
        if len(self.inputs) != manifest["splits"][name]["examples"]:
            raise ValueError("Dataset count differs from manifest")

    def __len__(self):
        return len(self.inputs)

    def sample(self, size, rng, augment=True):
        ids = rng.choice(len(self), size, replace=size > len(self))
        x, y = self.inputs[ids].copy(), self.labels[ids].copy()
        if augment:
            for i in range(size):
                transform = sample_transform(rng)
                x[i], y[i] = apply_transform(x[i], transform), apply_transform(y[i], transform)
        return torch.from_numpy(x.astype(np.int64)), torch.from_numpy(y.astype(np.int64))


def sample_transform(rng):
    bands, stacks = rng.permutation(3), rng.permutation(3)
    rows = np.concatenate([b * 3 + rng.permutation(3) for b in bands])
    cols = np.concatenate([b * 3 + rng.permutation(3) for b in stacks])
    digits = np.r_[0, rng.permutation(np.arange(1, 10))]
    return bool(rng.integers(2)), rows, cols, digits


def apply_transform(board, transform):
    transpose, rows, cols, digits = transform
    grid = board.T if transpose else board
    return digits[grid[np.ix_(rows, cols)]].astype(np.uint8)


def invert_transform(board, transform):
    transpose, rows, cols, digits = transform
    restored = np.empty_like(board)
    restored[np.ix_(rows, cols)] = np.argsort(digits)[board]
    return restored.T.copy() if transpose else restored
