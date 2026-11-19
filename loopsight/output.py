"""Decode a single predicted ARC canvas into a color grid."""

from numbers import Integral

import torch


PAD_INDEX = 11


def decode_arc_grid(logits: torch.Tensor, offset=(0, 0), scale_factor=1) -> list[list[int]]:
    """Take argmax over [C, H, W], crop at (x, y), and undo integer upscaling."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape [C, H, W]")
    if isinstance(scale_factor, bool) or not isinstance(scale_factor, Integral) or scale_factor < 1:
        raise ValueError("scale_factor must be a positive integer")
    x, y = offset
    if not 0 <= x < logits.size(2) or not 0 <= y < logits.size(1):
        raise ValueError("offset must be inside the output canvas")

    canvas = logits.argmax(dim=0)[y:, x:].cpu()
    end_x = (canvas[0] == PAD_INDEX).nonzero().flatten()
    end_y = (canvas[:, 0] == PAD_INDEX).nonzero().flatten()
    width = end_x[0].item() if end_x.numel() else canvas.size(1)
    height = end_y[0].item() if end_y.numel() else canvas.size(0)
    grid = canvas[:height, :width]
    if scale_factor == 1:
        return grid.tolist()

    result = []
    for row_start in range(0, height, scale_factor):
        row = []
        for column_start in range(0, width, scale_factor):
            block = grid[row_start:row_start + scale_factor, column_start:column_start + scale_factor]
            row.append(torch.bincount(block.reshape(-1)).argmax().item())
        result.append(row)
    return result
