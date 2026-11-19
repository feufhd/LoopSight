"""Configuration adapter for the current LoopSight ImageNet-100 model."""
from dataclasses import asdict, dataclass
from loopsight_benchmarks.imagenet100 import LoopSightImageNet100

LOOP_MODELS = MODEL_NAMES = ("loopsight_small",)

@dataclass(frozen=True)
class ModelConfig:
    name: str = "loopsight_small"
    image_size: int = 224
    patch_size: int = 16
    num_classes: int = 100
    embed_dim: int = 256
    core_depth: int = 6
    num_heads: int = 8
    mlp_dim: int = 512
    loops: int = 6
    step_table_size: int = 18
    dropout: float = 0.2
    drop_path_rate: float = 0.0
    checkpoint_blocks: bool = False

    def __post_init__(self):
        if self.name not in MODEL_NAMES or self.num_classes != 100:
            raise ValueError("This release contains only LoopSight ImageNet-100")

    def to_dict(self):
        return asdict(self)

def build_model(config):
    c = ModelConfig(**config) if isinstance(config, dict) else config
    model = LoopSightImageNet100(**{k: v for k, v in c.to_dict().items()
                                   if k not in {"name", "num_classes", "step_table_size"}})
    model.config = c
    return model

def forward_logits(model, images, depth):
    plain = getattr(model, "module", model)
    return model(images, depth=depth) if isinstance(plain, LoopSightImageNet100) else model(images)
