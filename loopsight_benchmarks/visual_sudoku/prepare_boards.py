#!/usr/bin/env python3
"""Prepare the Sudoku-OOD boards used for Visual Sudoku.

This module only prepares the symbolic board inputs.
See this benchmark's README.md for the rendering protocol.
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
from pathlib import Path
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
AKORN_REVISION = "eabbe2728c5b83836ce44f6d780b7cc37eea2a31"
NCA_REVISION = "a34414f2a05f6ae6c08930e735b7b8788805ff29"
SATNET_URL = "https://powei.tw/sudoku.zip"
RRN_URL = "https://www.dropbox.com/s/rp3hbjs91xiqdgc/sudoku-hard.zip?dl=1"
SATNET_MEMBERS = ("features.pt", "labels.pt", "perm.pt")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def fetch(url, path, limit=32 * 1024**2):
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
        total = 0
        while chunk := response.read(1024**2):
            total += len(chunk)
            if total > limit:
                raise ValueError(f"Download exceeded {limit} bytes: {url}")
            out.write(chunk)
    partial.rename(path)
    print(f"Cached {path.name}: {total:,} bytes", flush=True)

class HTTPRangeFile(io.RawIOBase):
    """Expose a seekable remote ZIP without fetching its image payload."""

    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60) as r:
            self.size = int(r.headers["Content-Length"])
            self.etag = r.headers.get("ETag")
        self.position = 0
        self.transferred = 0

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = (offset if whence == 0 else self.position + offset
                         if whence == 1 else self.size + offset)
        if self.position < 0:
            raise ValueError("Negative seek")
        return self.position

    def read(self, size=-1):
        end = self.size - 1 if size < 0 else min(self.position + size, self.size) - 1
        if end < self.position:
            return b""
        # All selected compressed members are below 1 MiB. Refuse accidental
        # reads of features_img.pt or an ignored Range request.
        if end - self.position + 1 > 2 * 1024**2:
            raise ValueError("Unexpected large ZIP range")
        headers = {"Range": f"bytes={self.position}-{end}"}
        if self.etag:
            headers["If-Match"] = self.etag
        with urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=60) as r:
            expected = f"bytes {self.position}-{end}/{self.size}"
            if r.status != 206 or r.headers.get("Content-Range") != expected:
                raise ValueError("Server did not honor the requested ZIP range")
            data = r.read(end - self.position + 2)
        if len(data) != end - self.position + 1:
            raise ValueError("Truncated ZIP range")
        self.position += len(data)
        self.transferred += len(data)
        return data

def download_ood_sources(source):
    satnet = source / "satnet"
    if not all((satnet / name).exists() for name in SATNET_MEMBERS):
        remote = HTTPRangeFile(SATNET_URL)
        with zipfile.ZipFile(remote) as archive:
            catalog = [dict(name=x.filename, bytes=x.file_size, compressed_bytes=x.compress_size)
                       for x in archive.infolist()]
            for name in SATNET_MEMBERS:
                dest = satnet / name
                if dest.exists():
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                partial = dest.with_suffix(".part")
                # One read per compressed member avoids hundreds of tiny
                # HTTP requests while inflating these highly sparse tensors.
                partial.write_bytes(archive.read("sudoku/" + name))
                partial.rename(dest)
        write_json(source / "satnet_zip_catalog.json", dict(
            url=SATNET_URL, archive_bytes=remote.size, etag=remote.etag,
            transferred_bytes=remote.transferred, members=catalog,
            omitted=["sudoku/features_img.pt"]))
        print(f"SATNet selected download: {remote.transferred:,} bytes", flush=True)
    fetch(RRN_URL, source / "sudoku-hard.zip")
    fetch(f"https://codeload.github.com/autonomousvision/akorn/zip/{AKORN_REVISION}",
          source / "akorn.zip")

def validate_boards(x, y):
    import numpy as np
    assert x.dtype == y.dtype == np.uint8
    assert x.shape == y.shape and x.ndim == 3 and x.shape[1:] == (9, 9)
    assert np.all(x <= 9) and np.all((y >= 1) & (y <= 9))
    assert np.all((x == 0) | (x == y)), "A solution changed a clue"
    digits = np.arange(1, 10, dtype=np.uint8)
    assert np.all(np.sort(y, axis=2) == digits), "Invalid solution row"
    assert np.all(np.sort(y, axis=1) == digits[None, :, None]), "Invalid solution column"
    blocks = y.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9)
    assert np.all(np.sort(blocks, axis=1) == digits), "Invalid solution block"

def prepare_ood(base):
    import numpy as np
    import torch
    source = base / "sources"
    dest = base / "sudoku_ood"
    work = base / "sudoku_ood.building"
    if dest.exists() or work.exists():
        raise FileExistsError(f"Output already exists: {dest} or {work}; use --verify")
    work.mkdir(parents=True)
    arrays = []
    for name in ("features.pt", "labels.pt"):
        onehot = torch.load(source / "satnet" / name, map_location="cpu", weights_only=True)
        assert tuple(onehot.shape) == (10000, 9, 9, 9)
        assert torch.all((onehot == 0) | (onehot == 1))
        assert torch.all(onehot.sum(dim=-1) <= 1)
        digits = ((onehot.argmax(dim=-1) + 1) * onehot.sum(dim=-1)).to(torch.uint8).numpy()
        arrays.append(digits)
    x, y = arrays
    validate_boards(x, y)
    # Match AKOrN exactly. Its loader reads perm.pt but does not apply it.
    splits = {"train": (x[:9000], y[:9000]), "id_test": (x[9000:], y[9000:])}
    with zipfile.ZipFile(source / "sudoku-hard.zip") as archive:
        catalog = [dict(name=m.filename, bytes=m.file_size, compressed_bytes=m.compress_size)
                   for m in archive.infolist()]
        member = [m for m in archive.namelist()
                  if m.endswith("/test.csv") and not m.startswith("__MACOSX/")]
        assert len(member) == 1, member
        raw = archive.read(member[0])
    test_path = source / "rrn" / "test.csv"
    test_path.parent.mkdir(exist_ok=True)
    if test_path.exists():
        assert test_path.read_bytes() == raw
    else:
        test_path.write_bytes(raw)
    write_json(source / "rrn_zip_catalog.json", catalog)
    puzzles, solutions = [], []
    for row in csv.reader(io.StringIO(raw.decode("utf-8"))):
        assert len(row) == 2 and all(len(s) == 81 and s.isdigit() for s in row)
        puzzles.append(np.frombuffer(row[0].encode(), dtype=np.uint8) - ord("0"))
        solutions.append(np.frombuffer(row[1].encode(), dtype=np.uint8) - ord("0"))
    ood_x = np.array(puzzles, dtype=np.uint8).reshape(-1, 9, 9)
    ood_y = np.array(solutions, dtype=np.uint8).reshape(-1, 9, 9)
    validate_boards(ood_x, ood_y)
    histogram = dict(sorted(collections.Counter((ood_x == 0).sum(axis=(1, 2)).tolist()).items()))
    assert histogram == {n: 1000 for n in range(47, 65)}, histogram
    splits["ood_test"] = ood_x, ood_y
    stats = {}
    for split, (inp, target) in splits.items():
        path = work / f"{split}.npz"
        np.savez_compressed(path, inputs=inp, labels=target)
        stats[split] = dict(examples=len(inp), shape=list(inp.shape), dtype="uint8",
                           empty_cell_histogram=dict(sorted(collections.Counter(
                               (inp == 0).sum(axis=(1, 2)).tolist()).items())),
                           file=path.name, bytes=path.stat().st_size)
    train_keys = {board.tobytes() for board in x[:9000]}
    overlap = dict(train_vs_id=sum(b.tobytes() in train_keys for b in x[9000:]),
                   train_vs_ood=sum(b.tobytes() in train_keys for b in ood_x))
    references = source / "references" / "akorn"
    with zipfile.ZipFile(source / "akorn.zip") as archive:
        prefix = f"akorn-{AKORN_REVISION}/"
        for member in archive.infolist():
            assert member.filename.startswith(prefix)
            relative = Path(member.filename[len(prefix):])
            assert not relative.is_absolute() and ".." not in relative.parts
            if member.is_dir():
                continue
            target = references / relative
            if target.exists():
                assert target.read_bytes() == archive.read(member)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(archive.read(member))
    records = [dict(file=str(p.relative_to(base)), bytes=p.stat().st_size)
               for p in sorted(source.rglob("*")) if p.is_file()]
    manifest = dict(schema_version=1, benchmark="Sudoku-OOD", akorn_revision=AKORN_REVISION,
                    satnet_url=SATNET_URL, rrn_url=RRN_URL,
                    split_rule="SATNet original order: [0:9000] train, [9000:10000] ID test; RRN test.csv OOD only",
                    perm_applied=False, splits=stats, exact_puzzle_overlap=overlap,
                    validation="All solutions satisfy rows/columns/blocks and preserve clues",
                    numpy_version=np.__version__, torch_version=torch.__version__,
                    source_files=records, source_bytes=sum(r["bytes"] for r in records),
                    prepared_array_bytes=sum(s["bytes"] for s in stats.values()))
    write_json(work / "manifest.json", manifest)
    work.rename(dest)
    print(json.dumps({k: manifest[k] for k in ("splits", "exact_puzzle_overlap", "source_bytes", "prepared_array_bytes")}, indent=2))

def verify(base, benchmark):
    import numpy as np
    dest = base / benchmark.replace("-", "_")
    manifest = json.loads((dest / "manifest.json").read_text())
    sources_checked = 0
    for record in manifest["source_files"]:
        path = base / record["file"]
        if path.exists():
            assert path.stat().st_size == record["bytes"], path
            sources_checked += 1
    for split in manifest["splits"].values():
        path = dest / split["file"]
        assert path.stat().st_size == split["bytes"], path
        with np.load(path, allow_pickle=False) as data:
            validate_boards(data["inputs"], data["labels"])
            assert len(data["inputs"]) == split["examples"]
    count = sum(s["examples"] for s in manifest["splits"].values())
    print(f"Verified all {count:,} boards for {benchmark}; {sources_checked} optional source files checked.")

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("raw_data/visual_sudoku/boards"))
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--download", action="store_true")
    action.add_argument("--verify", action="store_true")
    parser.add_argument("--benchmarks", nargs="+", choices=["sudoku-ood"], default=["sudoku-ood"])
    args = parser.parse_args()
    if args.verify:
        verify(args.root, "sudoku-ood")
    else:
        if args.download:
            download_ood_sources(args.root / "sources")
        prepare_ood(args.root)

if __name__ == "__main__":
    main()
