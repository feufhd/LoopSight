"""Folder-based ImageNet data and exact, unpadded distributed validation."""
import json
from pathlib import Path
from PIL import ImageOps

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

DEFAULT_DATA = Path("raw_data/imagenet")


def posterize_compat(image, bits_to_keep, **kwargs):
    """Keep timm 1.0.12's zero-bit result on Pillow versions that reject bits=0."""
    if bits_to_keep >= 8:
        return image
    if bits_to_keep == 0 and image.mode in ('RGB', 'L'):
        # This is the exact all-zero lookup table used by historical Pillow,
        # rather than clamping the augmentation to a different bit depth.
        return image.point([0] * (256 * len(image.getbands())))
    return ImageOps.posterize(image, bits_to_keep)


class DistributedEvalSampler(Sampler):
    """Every validation example occurs exactly once, even for uneven partitions."""
    def __init__(self, dataset, rank=0, world_size=1):
        if world_size < 1 or not 0 <= rank < world_size:
            raise ValueError("Invalid distributed evaluation partition")
        self.size, self.rank, self.world_size = len(dataset), rank, world_size

    def __iter__(self):
        return iter(range(self.rank, self.size, self.world_size))

    def __len__(self):
        return len(range(self.rank, self.size, self.world_size))


def image_transform(training, image_size=224, augment=True):
    if training:
        transform = create_transform(input_size=image_size, is_training=True,
            auto_augment="rand-m9-mstd0.5-inc1" if augment else None,
            color_jitter=0.3 if augment else 0, re_prob=0.25 if augment else 0,
            re_mode="pixel", re_count=1,
            interpolation="bicubic", mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
        for component in transform.transforms:
            for operation in getattr(component, 'ops', []):
                if operation.name in ('Posterize', 'PosterizeOriginal', 'PosterizeIncreasing'):
                    operation.aug_fn = posterize_compat
        return transform
    return transforms.Compose([
        transforms.Resize(int(image_size / 0.875), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)])


class SelectedImageFolder(datasets.ImageFolder):
    """Scan only selected synsets and assign the same contiguous labels in each split."""
    def __init__(self, root, class_ids, **kwargs):
        self.selected_classes = sorted(class_ids)
        if len(set(self.selected_classes)) != len(self.selected_classes):
            raise ValueError("Duplicate class IDs in subset")
        super().__init__(root, **kwargs)

    def find_classes(self, directory):
        missing = [name for name in self.selected_classes if not (Path(directory) / name).is_dir()]
        if missing:
            raise ValueError(f"Missing selected classes in {directory}: {missing}")
        return self.selected_classes, {name: index for index, name in enumerate(self.selected_classes)}


def image_folder(root, split, classes=1000, augment=True, class_ids=None, expected_rows=None):
    root = Path(root)
    if split not in ("train", "val"):
        raise ValueError("ImageNet classification uses the labeled train/val splits")
    transform = image_transform(split == "train", augment=augment)
    dataset = (SelectedImageFolder(root / split, class_ids, transform=transform) if class_ids is not None
               else datasets.ImageFolder(root / split, transform=transform))
    if len(dataset.classes) != classes:
        raise ValueError(f"Expected {classes} classes, found {len(dataset.classes)} in {root / split}")
    if expected_rows is None and classes == 1000:
        expected_rows = {"train": 1281167, "val": 50000}[split]
    if expected_rows is not None and len(dataset) != expected_rows:
        raise ValueError(f"Incomplete ImageNet {split}: expected {expected_rows}, found {len(dataset)} images")
    mapping_file = root / "class_mapping.json"
    if mapping_file.is_file():
        mapping = json.loads(mapping_file.read_text())
        original = {entry["synset"]: entry["label"] for entry in mapping}
        expected = ({name: index for index, name in enumerate(sorted(class_ids))}
                    if class_ids is not None else original)
        if not set(dataset.classes).issubset(original) or dataset.class_to_idx != expected:
            raise ValueError(f"ImageFolder labels differ from class_mapping.json: {split}")
        dataset.original_class_to_idx = {name: original[name] for name in dataset.classes}
    return dataset


def make_loader(dataset, batch_size, workers, device, sampler=None, training=False, generator=None):
    options = dict(dataset=dataset, batch_size=batch_size, sampler=sampler,
                   shuffle=training and sampler is None, num_workers=workers,
                   pin_memory=torch.device(device).type != "cpu", drop_last=training, generator=generator)
    if workers:
        # Workers restart each epoch so epoch-boundary resume reproduces augmentations.
        options.update(prefetch_factor=2, persistent_workers=False)
    return DataLoader(**options)


@torch.no_grad()
def evaluate(model, loader, device, depth=12, amp="bf16"):
    # Unwrap DDP: unequal eval-loader lengths must not trigger forward collectives.
    plain = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    was_training = plain.training
    plain.eval()
    from .config import forward_logits
    totals = torch.zeros(4, dtype=torch.float64, device=device)
    try:
        for images, targets in loader:
            images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
            with torch.autocast(device_type=torch.device(device).type,
                                dtype=torch.bfloat16 if amp == "bf16" else torch.float16,
                                enabled=amp != "none"):
                logits = forward_logits(plain, images, depth)
            loss = torch.nn.functional.cross_entropy(logits.float(), targets, reduction="sum")
            prediction = logits.topk(min(5, logits.shape[1]), dim=1).indices
            correct = prediction.eq(targets[:, None])
            totals[0] += loss.double()
            totals[1] += correct[:, :1].sum()
            totals[2] += correct.any(1).sum()
            totals[3] += len(targets)
        if dist.is_initialized():
            dist.all_reduce(totals)
        loss, top1, top5, count = totals.tolist()
        if count == 0:
            raise ValueError("Empty validation set")
        return {"loss": loss / count, "top1": 100 * top1 / count,
                "top5": 100 * top5 / count, "examples": int(count), "depth": depth}
    finally:
        plain.train(was_training)
