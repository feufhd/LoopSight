"""Configuration adapter for the current LoopSight visual Sudoku model."""
import math
from loopsight_benchmarks.visual_sudoku import LoopSightVisualSudoku
from loopsight.checkpoint import ARCHITECTURE

def resolve_config(config):
    c = dict(config)
    if c.get("model") != "loopsight" or c.get("architecture", ARCHITECTURE) != ARCHITECTURE:
        raise ValueError("This release contains only the latest LoopSight model")
    if c.get("use_checkpoint", False):
        raise ValueError("Visual Sudoku uses full activations")
    defaults = dict(image_size=252, patch_size=14, embed_dim=256, heads=8,
                    core_depth=8, dropout=0.3, step_table_size=18,
                    fusion_heads=2, fusion_logit_bound=math.log(16), use_checkpoint=False)
    for key, value in defaults.items():
        c.setdefault(key, value)
    c.setdefault("mlp_dim", 2 * c["embed_dim"])
    rank = c["embed_dim"] // 16
    if c.get("fusion_rank") not in (None, rank):
        raise ValueError("Write gate rank must equal D/16")
    if c["image_size"] not in (252, 256) or c["patch_size"] < 1 or c["image_size"] % c["patch_size"]:
        raise ValueError("Patch size must divide a 252- or 256-pixel image")
    if c["heads"] < 1 or c["embed_dim"] % c["heads"] or (c["embed_dim"] // c["heads"]) % 4:
        raise ValueError("2D RoPE requires head width divisible by four")
    if c["embed_dim"] < 16 or c["embed_dim"] % 16 or c["fusion_heads"] < 1 or rank % c["fusion_heads"]:
        raise ValueError("Gate rank D/16 must be divisible by positive gate heads")
    if min(c["core_depth"], c["step_table_size"], c["mlp_dim"]) < 1 or not math.isfinite(c["dropout"]) or not 0 <= c["dropout"] < 1:
        raise ValueError("Invalid depth, MLP width, or dropout")
    c.update(architecture=ARCHITECTURE, fusion_rank=rank,
             input_representation="mnist_grayscale_image",
             output_representation="unconstrained_grayscale_pixels")
    return c

class VisualSudokuModel(LoopSightVisualSudoku):
    def __init__(self, config):
        c = resolve_config(config)
        super().__init__(**{k: c[k] for k in ("image_size", "patch_size", "embed_dim",
            "heads", "mlp_dim", "core_depth", "dropout", "fusion_heads", "fusion_logit_bound")})
        self.config = c
