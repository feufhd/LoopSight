"""Render shared symbolic Sudoku boards as reproducible MNIST image pairs."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

from .boards import DEFAULT_DATA, SudokuDataset as SymbolicSudokuDataset
from .fetch_assets import MNIST_PATH

PREPARED = Path("raw_data/visual_sudoku/prepared")
SPLITS = ("train", "validation", "easy_test", "hard_test", "hard_test_full")
SYMBOLIC_SPLITS = {"train": "train", "validation": "validation", "easy_test": "validation",
                   "hard_test": "test", "hard_test_full": "test"}

PROTOCOL_FIELDS = (
    "schema_version", "task", "image_size", "digit_size", "padding", "blank_input",
    "target", "mnist_pools", "counts", "hard_seed", "hard_count", "hard_selection",
    "validation_note",
)


def load_protocol(prepared_root):
    record = json.loads((Path(prepared_root) / "manifest.json").read_text())
    return {key: record[key] for key in PROTOCOL_FIELDS if key in record}



def render_board(board, images, by_digit, rng, *, blank_noise=False, image_size=256):
    """Render 28px cells, with optional bottom/right padding for the 256px protocol."""
    if image_size not in (252, 256):
        raise ValueError("Visual-Sudoku image_size must be 252 or 256")
    image = (rng.uniform(-0.1, 0.1, (252, 252)).astype(np.float32)
             if blank_noise else np.zeros((252, 252), dtype=np.float32))
    patches = image.reshape(9, 28, 9, 28).transpose(0, 2, 1, 3)
    indices = np.full((9, 9), -1, dtype=np.int64)
    for digit in range(1, 10):
        rows, cols = np.where(board == digit)
        selected = rng.choice(by_digit[digit], size=len(rows), replace=True)
        patches[rows, cols] = images[selected].astype(np.float32) / 255.0
        indices[rows, cols] = selected
    padding = image_size - 252
    return np.pad(image, ((0, padding), (0, padding)))[None], indices


class VisualSudokuDataset(Dataset):
    def __init__(self, split, board_root=DEFAULT_DATA, mnist_path=MNIST_PATH,
                 prepared_root=PREPARED, seed=0, image_size=None):
        if split not in SPLITS or seed < 0:
            raise ValueError(f"Expected split in {SPLITS} and a nonnegative seed")
        self.split, self.seed, self.epoch = split, int(seed), 0
        self.prepared_root = Path(prepared_root)
        manifest_path = self.prepared_root / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError("Run python -m loopsight_benchmarks.visual_sudoku.prepare first")
        self.protocol = load_protocol(self.prepared_root)
        self.image_size = self.protocol["image_size"] if image_size is None else image_size
        if self.image_size not in (252, 256) or self.image_size != self.protocol["image_size"]:
            raise ValueError("Image size differs from the prepared protocol; select matching prepared_root")
        shared = SymbolicSudokuDataset("sudoku-ood", SYMBOLIC_SPLITS[split], board_root)
        if split == "hard_test":
            rows_path = self.prepared_root / "hard_test_rows.npy"
            indices = np.load(rows_path, allow_pickle=False)
        else:
            indices = np.arange(len(shared))
        self.inputs, self.labels = shared.inputs[indices], shared.labels[indices]
        self.source_rows = shared.source_rows[indices]
        self.ratings = (self.inputs == 0).sum(axis=(1, 2))
        self.mnist_pool = "test" if split == "train" else "train"
        with np.load(mnist_path, allow_pickle=False) as source:
            self.digits = source[f"x_{self.mnist_pool}"]
            self.digit_labels = source[f"y_{self.mnist_pool}"]
        self.by_digit = {d: np.flatnonzero(self.digit_labels == d) for d in range(1, 10)}

    def __len__(self):
        return len(self.inputs)

    def set_epoch(self, epoch):
        if epoch < 0 or (self.split != "train" and epoch != 0):
            raise ValueError("Only training may change rendering epoch")
        self.epoch = int(epoch)

    def _item(self, index, seed):
        # Separate input/target RNG streams prevent target glyphs affecting the input.
        input_seed, target_seed = np.random.SeedSequence(seed).spawn(2)
        image, _ = render_board(self.inputs[index], self.digits, self.by_digit,
                                np.random.default_rng(input_seed), blank_noise=True, image_size=self.image_size)
        target, _ = render_board(self.labels[index], self.digits, self.by_digit,
                                 np.random.default_rng(target_seed), image_size=self.image_size)
        return {"image": torch.from_numpy(image), "target": torch.from_numpy(target),
                "puzzle": torch.from_numpy(self.inputs[index].astype(np.int64)),
                "solution": torch.from_numpy(self.labels[index].astype(np.int64)),
                "source_row": int(self.source_rows[index])}

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        # Validation and easy_test intentionally refer to the same shared ID split.
        split_code = {"train": 0, "validation": 1, "easy_test": 1,
                      "hard_test": 2, "hard_test_full": 2}[self.split]
        return self._item(index, [self.seed, self.epoch, split_code, int(self.source_rows[index])])

    def sample(self, count, rng):
        """Fresh glyphs per optimizer update, controlled by the checkpointed RNG."""
        indices = rng.choice(len(self), size=count, replace=count > len(self))
        return self.sample_indices(indices, rng)

    def sample_indices(self, indices, rng):
        """Render explicit board rows; epoch training visits each row exactly once."""
        indices = np.asarray(indices, dtype=np.int64)
        if indices.ndim != 1 or not len(indices) or np.any(indices < 0) or np.any(indices >= len(self)):
            raise ValueError("Expected a nonempty vector of valid dataset indices")
        seeds = rng.integers(0, 2**63 - 1, size=len(indices))
        items = [self._item(int(i), int(s)) for i, s in zip(indices, seeds)]
        return torch.utils.data.default_collate(items)


class PrototypeDecoder(torch.nn.Module):
    """Parameter-free nearest MNIST-test-mean readout, including class 0=blank."""
    def __init__(self, prepared_root=PREPARED):
        super().__init__()
        root = Path(prepared_root)
        path = root / "prototypes.npz"
        with np.load(path, allow_pickle=False) as source:
            self.register_buffer("prototypes", torch.from_numpy(source["images"]))

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1 or tuple(images.shape[-2:]) not in ((252, 252), (256, 256)):
            raise ValueError("Expected floating-point [B,1,252,252] or [B,1,256,256] images")
        cells = images[:, 0, :252, :252].float().reshape(-1, 9, 28, 9, 28)
        cells = cells.permute(0, 1, 3, 2, 4).reshape(-1, 9, 9, 784)
        prototypes = self.prototypes.float().flatten(1)
        distances = (cells.square().sum(-1, keepdim=True) + prototypes.square().sum(-1)
                     - 2 * (cells @ prototypes.T))
        return distances.argmin(-1)
