"""Download the MNIST glyphs used for Visual Sudoku."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parent
MNIST_PATH = Path("raw_data/visual_sudoku/mnist.npz")
REPO = "paradigms-of-intelligence/reasoning-nca"
MNIST_URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def fetch(url: str, dest: Path, *, max_bytes: int = 64 * 1024**2) -> dict:
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_name(dest.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "Visual-Sudoku-data-preparation"})
        try:
            with urllib.request.urlopen(request, timeout=45) as response, partial.open("wb") as handle:
                total = 0
                while chunk := response.read(1024**2):
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError(f"Unexpected download size: {url}")
                    handle.write(chunk)
            partial.replace(dest)
        finally:
            partial.unlink(missing_ok=True)
    print(f"Ready: {dest.name} ({dest.stat().st_size:,} bytes)", flush=True)
    return {"url": url, "bytes": dest.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description="Prepare MNIST glyphs for Visual Sudoku")
    parser.add_argument("--mnist-path", type=Path, default=Path("raw_data/visual_sudoku/mnist.npz"))
    args = parser.parse_args()
    metadata = fetch(MNIST_URL, args.mnist_path)
    print(json.dumps(metadata, indent=2))

if __name__ == "__main__":
    main()
