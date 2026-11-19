"""Fixed CMC synsets and ImageNet-100 preprocessing."""

from pathlib import Path

from PIL import ImageOps
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


def class_ids(class_list):
    """Read a caller-supplied CMC class list in classifier output order."""
    content = Path(class_list).read_bytes()
    selected = content.decode().split()
    if len(selected) != 100 or len(set(selected)) != 100:
        raise ValueError("ImageNet-100 requires exactly 100 unique classes")
    return sorted(selected)


def _posterize(image, bits_to_keep, **kwargs):
    if bits_to_keep >= 8:
        return image
    if bits_to_keep == 0 and image.mode in ("RGB", "L"):
        return image.point([0] * (256 * len(image.getbands())))
    return ImageOps.posterize(image, bits_to_keep)


def image_transform(training=False, image_size=224, augment=True):
    """Normalize RGB PIL images using the original training or validation transform."""
    if training:
        transform = create_transform(
            input_size=image_size, is_training=True,
            auto_augment="rand-m9-mstd0.5-inc1" if augment else None,
            color_jitter=0.3 if augment else 0, re_prob=0.25 if augment else 0,
            re_mode="pixel", re_count=1, interpolation="bicubic",
            mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD,
        )
        for component in transform.transforms:
            for operation in getattr(component, "ops", []):
                if operation.name in ("Posterize", "PosterizeOriginal", "PosterizeIncreasing"):
                    operation.aug_fn = _posterize
        return transform
    return transforms.Compose([
        transforms.Resize(int(image_size / 0.875), interpolation=InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
    ])


class _SelectedImageFolder(datasets.ImageFolder):
    def __init__(self, root, selected, **kwargs):
        self._selected = selected
        super().__init__(root, **kwargs)

    def find_classes(self, directory):
        missing = [synset for synset in self._selected if not (Path(directory) / synset).is_dir()]
        if missing:
            raise ValueError(f"Missing selected class folders: {missing}")
        return self._selected, {synset: index for index, synset in enumerate(self._selected)}


def image_folder(root, split, *, class_list, augment=True):
    """Load the CMC subset from a dataset root and an external class list."""
    if split not in ("train", "val"):
        raise ValueError("split must be train or val")
    transform = image_transform(split == "train", augment=augment)
    return _SelectedImageFolder(Path(root) / split, class_ids(class_list), transform=transform)
