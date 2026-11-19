"""Verify the selected ImageNet-100 folders, class mapping and image counts."""

import argparse
import json

from .dataset import image_folder
from .subsets import dataset_spec


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="raw_data/imagenet")
    args = parser.parse_args(argv)
    spec = dataset_spec("imagenet100")
    counts = {}
    for split in ("train", "val"):
        dataset = image_folder(args.data_root, split, classes=100,
                               class_ids=spec["class_ids"], expected_rows=spec["expected_rows"][split])
        counts[split] = len(dataset)
    print(json.dumps(dict(dataset="imagenet100", root=args.data_root, classes=100,
                          images=counts), indent=2))


if __name__ == "__main__":
    main()
