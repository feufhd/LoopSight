"""Frozen ImageNet class selections and their dataset identities (stdlib only)."""
import json
from pathlib import Path

DATASETS = ("imagenet100",)
CLASS_LIST = Path(__file__).with_name("imagenet100_classes.txt")


def dataset_spec(name):
    if name != "imagenet100":
        raise ValueError(f"Unknown dataset: {name}")
    content = CLASS_LIST.read_bytes()
    selected = content.decode().split()
    if len(selected) != 100 or len(set(selected)) != 100:
        raise ValueError("ImageNet-100 requires exactly 100 unique classes")
    provenance = json.loads(CLASS_LIST.with_name("imagenet100_source.json").read_text())
    return dict(name=name, classes=100, class_ids=sorted(selected),
                source_url=provenance["source_url"],
                expected_rows={"train": 126689, "val": 5000})
