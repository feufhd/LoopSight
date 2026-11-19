"""MNIST board rendering and nearest-prototype digit decoding."""

import numpy as np
import torch
from torch import nn


def render_board(board, images, by_digit, rng, *, blank_noise=False, image_size=252):
    """Render a 9x9 board using pools of uint8 MNIST glyphs indexed by digit."""
    if image_size not in (252, 256):
        raise ValueError("Visual Sudoku image_size must be 252 or 256")
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


def mnist_prototypes(test_images, test_labels):
    """Compute MNIST test-split class means in [0, 1], with a zero blank prototype."""
    images, labels = np.asarray(test_images), np.asarray(test_labels)
    if images.ndim != 3 or images.shape[1:] != (28, 28) or images.dtype != np.uint8:
        raise ValueError("Expected uint8 MNIST images with shape [N, 28, 28]")
    if labels.shape != (len(images),) or set(labels.tolist()) != set(range(10)):
        raise ValueError("Expected one label per image with all ten MNIST classes present")
    prototypes = np.zeros((10, 28, 28), dtype=np.float32)
    for digit in range(1, 10):
        prototypes[digit] = images[labels == digit].mean(axis=0) / 255.0
    return torch.from_numpy(prototypes)


class PrototypeDecoder(nn.Module):
    """Map a grayscale board to [B, 9, 9] digits by squared pixel distance."""

    def __init__(self, prototypes):
        super().__init__()
        prototypes = torch.as_tensor(prototypes, dtype=torch.float32)
        if prototypes.shape != (10, 28, 28):
            raise ValueError("Expected ten [28, 28] prototypes, including zero for blank")
        self.register_buffer("prototypes", prototypes.detach().clone())

    def forward(self, images):
        if images.ndim != 4 or images.shape[1] != 1 or images.shape[-2:] not in ((252, 252), (256, 256)):
            raise ValueError("Expected [B,1,252,252] or [B,1,256,256] images")
        cells = images[:, 0, :252, :252].float().reshape(-1, 9, 28, 9, 28)
        cells = cells.permute(0, 1, 3, 2, 4).reshape(-1, 9, 9, 784)
        prototypes = self.prototypes.float().flatten(1)
        distances = (cells.square().sum(-1, keepdim=True) + prototypes.square().sum(-1)
                     - 2 * (cells @ prototypes.T))
        return distances.argmin(-1)
