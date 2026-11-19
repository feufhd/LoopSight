"""Evaluate generated images with the fixed MNIST-prototype Sudoku readout."""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
import time
import numpy as np
import torch
from loopsight.runtime import resolve_device, synchronize
from torch.utils.data import DataLoader, Subset

from .boards import DEFAULT_DATA
from .dataset import PREPARED, SPLITS, PrototypeDecoder, VisualSudokuDataset
from .fetch_assets import MNIST_PATH, write_json
from .config import VisualSudokuModel


@torch.inference_mode()
def evaluate(model, dataset, depth, batch_size=128, device="cpu", max_examples=None,
             save_predictions=None):
    if depth < 1 or batch_size < 1 or (max_examples is not None and max_examples < 1):
        raise ValueError("Depth, batch size and max_examples must be positive")
    device = resolve_device(device)
    model.eval()
    decoder = PrototypeDecoder(dataset.prepared_root).to(device)
    count = len(dataset) if max_examples is None else min(max_examples, len(dataset))
    # A private generator avoids altering the training/dropout RNG at validation.
    loader = DataLoader(Subset(dataset, range(count)), batch_size=batch_size, shuffle=False,
                        generator=torch.Generator().manual_seed(0))
    totals = collections.Counter()
    per_difficulty = collections.defaultdict(collections.Counter)
    pixel_error = 0.0
    pixel_count = 0
    stored, rows = [], []
    synchronize(device)
    started = time.perf_counter()
    digits = torch.arange(1, 10, device=device)
    for batch in loader:
        images, targets = batch["image"].to(device), batch["target"].to(device)
        output = model(images, depth).float()
        if output.shape != targets.shape:
            raise ValueError("Generated image shape differs from the target")
        if not torch.isfinite(output).all():
            raise FloatingPointError("Non-finite generated pixels during evaluation")
        pixel_error += (output - targets).square().sum().item()
        pixel_count += targets.numel()
        pred = decoder(output)
        labels, puzzles = batch["solution"].to(device), batch["puzzle"].to(device)
        equal, blanks = pred == labels, puzzles == 0
        board_correct = equal.flatten(1).all(1)
        rows_ok = (pred.sort(2).values == digits).flatten(1).all(1)
        cols_ok = (pred.sort(1).values == digits[None, :, None]).flatten(1).all(1)
        boxes = pred.reshape(-1, 3, 3, 3, 3).permute(0, 1, 3, 2, 4).reshape(-1, 9, 9)
        boxes_ok = (boxes.sort(2).values == digits).flatten(1).all(1)
        clues_ok = (equal | blanks).flatten(1).all(1)
        totals.update(examples=len(images), correct_boards=int(board_correct.sum()),
                      correct_cells=int(equal.sum()), blank_cells=int(blanks.sum()),
                      correct_blank_cells=int((equal & blanks).sum()),
                      given_cells=int((~blanks).sum()), correct_given_cells=int((equal & ~blanks).sum()),
                      valid_boards=int((rows_ok & cols_ok & boxes_ok & clues_ok).sum()))
        for holes, correct, cells in zip(blanks.sum((1, 2)).tolist(), board_correct.tolist(), equal.sum((1, 2)).tolist()):
            per_difficulty[str(holes)].update(examples=1, correct_boards=int(correct), correct_cells=cells)
        if save_predictions:
            stored.append(pred.cpu().numpy().astype(np.uint8))
            rows.append(batch["source_row"].numpy())
    synchronize(device)
    elapsed = time.perf_counter() - started
    if save_predictions:
        path = Path(save_predictions)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, predictions=np.concatenate(stored), source_rows=np.concatenate(rows))
    return dict(split=dataset.split, examples=count, total_split_examples=len(dataset),
                hard_subset_seed=dataset.protocol["hard_seed"] if dataset.split == "hard_test" else None,
                paper_hard_subset_verified=False if dataset.split == "hard_test" else None,
                full_split=count == len(dataset), depth=depth, candidates=1, batch_size=batch_size,
                image_size=dataset.image_size,
                render_seed=dataset.seed,
                mnist_pool=dataset.mnist_pool, board_accuracy=totals["correct_boards"] / count,
                digit_accuracy=totals["correct_cells"] / (count * 81),
                blank_digit_accuracy=totals["correct_blank_cells"] / max(1, totals["blank_cells"]),
                given_digit_accuracy=totals["correct_given_cells"] / max(1, totals["given_cells"]),
                valid_board_rate=totals["valid_boards"] / count,
                pixel_mse=pixel_error / pixel_count,
                seconds=elapsed, boards_per_second=count / max(elapsed, 1e-9),
                decoder="nearest MNIST-test class mean, 0=blank; no clue correction",
                data_protocol=dataset.protocol,
                reused_validation_split=dataset.split in ("validation", "easy_test"),
                correct_boards=totals["correct_boards"],
                per_difficulty={k: {**v, "board_accuracy": v["correct_boards"] / v["examples"]}
                                for k, v in sorted(per_difficulty.items(), key=lambda item: int(item[0]))})


def load_checkpoint(path, device="cpu", raw=False):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = VisualSudokuModel(checkpoint["model_config"])
    if checkpoint.get("weights") == "final_step_ema":
        if raw:
            raise ValueError("ema_final.pt contains final EMA only; use last.pt for raw weights")
        state, label = checkpoint["model_state"], "final_step_ema"
    elif not raw and checkpoint.get("selection", {}).get("metric") == "easy_board_accuracy":
        selected = checkpoint["selection"]["weights"]
        if selected == "raw":
            state = checkpoint["model_state"]
        elif selected == "ema":
            state = checkpoint["ema_state"]
        else:
            state = checkpoint["ema_states"][selected]
        label = f"best_{selected}"
    else:
        state = checkpoint["model_state"] if raw else checkpoint["ema_state"]
        label = "raw" if raw else "ema"
    model.load_state_dict(state)
    return model.to(device), checkpoint, label


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", choices=[s for s in SPLITS if s != "train"], nargs="+", default=["hard_test"])
    parser.add_argument("--depths", type=int, nargs="+", default=[18])
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--board-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--mnist-path", type=Path, default=MNIST_PATH)
    parser.add_argument("--prepared-root", type=Path, default=PREPARED)
    parser.set_defaults(device="auto")
    parser.add_argument("--seed", type=int, help="Rendering seed; defaults to checkpoint eval_seed")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--raw-weights", action="store_true")
    parser.add_argument("--save-predictions", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    args.device = str(resolve_device(args.device))
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    model, checkpoint, weights = load_checkpoint(args.checkpoint, args.device, args.raw_weights)
    seed = checkpoint["training_config"]["eval_seed"] if args.seed is None else args.seed
    results = []
    for split in args.splits:
        dataset = VisualSudokuDataset(split, args.board_root, args.mnist_path, args.prepared_root,
                                      seed, model.config["image_size"])
        saved_protocol = checkpoint["training_config"].get("data_protocol")
        if saved_protocol is not None and dataset.protocol != saved_protocol:
            raise ValueError("Evaluation protocol differs from the training checkpoint")
        for depth in args.depths:
            predictions = (args.output.with_name(f"{args.output.stem}_{split}_depth{depth}.npz")
                           if args.save_predictions else None)
            result = evaluate(model, dataset, depth, args.batch_size, args.device, args.max_examples, predictions)
            results.append(result)
            write_json(args.output, dict(checkpoint=str(args.checkpoint), step=checkpoint["step"], weights=weights,
                       model_config=model.config, parameters=sum(p.numel() for p in model.parameters()), results=results))
            print(json.dumps({k: result[k] for k in ("split", "depth", "examples", "board_accuracy", "pixel_mse")}), flush=True)


if __name__ == "__main__":
    main()
