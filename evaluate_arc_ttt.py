#!/usr/bin/env python3
"""Evaluate ARC prediction directories with majority voting.

Each prediction directory must contain one ``<task>_predictions.json`` file per
ARC task. Multiple directories are combined by concatenating their candidates,
which gives every TTT attempt equal weight in the majority vote.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one or more ARC TTT prediction directories."
    )
    parser.add_argument(
        "--output-root",
        action="append",
        required=True,
        help=(
            "Prediction directory. May be repeated or supplied as a comma-separated "
            "list to ensemble multiple TTT attempts."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("raw_data/ARC-AGI"),
        help="ARC dataset root containing data/evaluation (default: raw_data/ARC-AGI).",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        help="Optional path for a machine-readable evaluation report.",
    )
    parser.add_argument(
        "--tasks", nargs="+",
        help="Evaluate exactly these task IDs; all selected tasks must have predictions.",
    )
    parser.add_argument(
        "--allow-missing",
        action="store_true",
        help="Evaluate the intersection of available tasks instead of failing on missing files.",
    )
    return parser.parse_args()


def expand_output_roots(values: list[str]) -> list[Path]:
    roots = [Path(item.strip()) for value in values for item in value.split(",") if item.strip()]
    if not roots:
        raise ValueError("At least one non-empty --output-root is required.")
    return roots


def load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Failed to read JSON file {path}: {exc}") from exc


def rank_predictions(predictions: list[Any]) -> list[tuple[Any, int]]:
    """Return predictions ordered by votes, preserving first-seen order on ties."""
    counts: dict[str, int] = {}
    grids: dict[str, Any] = {}
    for prediction in predictions:
        key = json.dumps(prediction, separators=(",", ":"))
        if key not in counts:
            counts[key] = 0
            grids[key] = prediction
        counts[key] += 1
    return [(grids[key], count) for key, count in sorted(counts.items(), key=lambda x: x[1], reverse=True)]


def evaluate(
    output_roots: list[Path],
    evaluation_dir: Path,
    allow_missing: bool,
    task_names: list[str] | None = None,
) -> dict[str, Any]:
    task_paths = sorted(evaluation_dir.glob("*.json"))
    if not task_paths:
        raise RuntimeError(f"No ARC evaluation tasks found under {evaluation_dir}.")
    if task_names is not None:
        if not task_names or len(set(task_names)) != len(task_names):
            raise ValueError("Selected task IDs must be nonempty and unique.")
        available = {path.stem: path for path in task_paths}
        absent = sorted(set(task_names) - available.keys())
        if absent:
            raise RuntimeError(f"Selected tasks are missing ground truth: {', '.join(absent)}")
        task_paths = [available[name] for name in sorted(task_names)]

    missing: dict[str, list[str]] = {}
    for root in output_roots:
        absent = [path.stem for path in task_paths if not (root / f"{path.stem}_predictions.json").is_file()]
        if absent:
            missing[str(root)] = absent

    if missing and not allow_missing:
        details = "; ".join(f"{root}: {len(tasks)} missing" for root, tasks in missing.items())
        raise RuntimeError(
            f"Prediction set is incomplete ({details}). Use --allow-missing only for partial evaluation."
        )

    task_results: dict[str, Any] = {}
    pass_at_1_sum = 0.0
    pass_at_2_sum = 0.0
    oracle_sum = 0.0
    evaluated_examples = 0

    for task_path in task_paths:
        task_name = task_path.stem
        prediction_paths = [root / f"{task_name}_predictions.json" for root in output_roots]
        if any(not path.is_file() for path in prediction_paths):
            continue

        task_data = load_json(task_path)
        test_examples = task_data.get("test", [])
        prediction_sets = [load_json(path) for path in prediction_paths]
        expected_keys = {str(index) for index in range(len(test_examples))}

        for path, predictions in zip(prediction_paths, prediction_sets):
            actual_keys = set(predictions)
            if actual_keys != expected_keys:
                raise RuntimeError(
                    f"Test-example keys in {path} are {sorted(actual_keys)}, expected {sorted(expected_keys)}."
                )

        example_results: dict[str, Any] = {}
        task_pass_1 = 0.0
        task_pass_2 = 0.0
        task_oracle = 0.0

        for index, example in enumerate(test_examples):
            key = str(index)
            candidates: list[Any] = []
            candidate_counts: list[int] = []
            for predictions in prediction_sets:
                current = predictions[key]
                if not isinstance(current, list):
                    raise RuntimeError(f"Predictions for {task_name}[{key}] must be a list.")
                candidates.extend(current)
                candidate_counts.append(len(current))

            ranked = rank_predictions(candidates)
            ground_truth = example["output"]
            pass_at_1 = bool(ranked and ranked[0][0] == ground_truth)
            pass_at_2 = any(prediction == ground_truth for prediction, _ in ranked[:2])
            oracle_rank = next(
                (rank for rank, (prediction, _) in enumerate(ranked, start=1) if prediction == ground_truth),
                None,
            )

            task_pass_1 += float(pass_at_1)
            task_pass_2 += float(pass_at_2)
            task_oracle += float(oracle_rank is not None)
            evaluated_examples += 1
            example_results[key] = {
                "pass_at_1": pass_at_1,
                "pass_at_2": pass_at_2,
                "oracle_rank": oracle_rank,
                "candidate_counts": candidate_counts,
                "unique_candidates": len(ranked),
                "top_vote_counts": [votes for _, votes in ranked[:2]],
            }

        if not test_examples:
            raise RuntimeError(f"Task {task_name} contains no test examples.")

        denominator = len(test_examples)
        task_pass_1 /= denominator
        task_pass_2 /= denominator
        task_oracle /= denominator
        pass_at_1_sum += task_pass_1
        pass_at_2_sum += task_pass_2
        oracle_sum += task_oracle
        task_results[task_name] = {
            "pass_at_1": task_pass_1,
            "pass_at_2": task_pass_2,
            "oracle": task_oracle,
            "examples": example_results,
        }

    evaluated_tasks = len(task_results)
    if evaluated_tasks == 0:
        raise RuntimeError("No tasks could be evaluated.")

    return {
        "output_roots": [str(path) for path in output_roots],
        "evaluation_dir": str(evaluation_dir),
        "expected_tasks": len(task_paths),
        "evaluated_tasks": evaluated_tasks,
        "evaluated_examples": evaluated_examples,
        "missing_predictions": missing,
        "pass_at_1": pass_at_1_sum / evaluated_tasks,
        "pass_at_2": pass_at_2_sum / evaluated_tasks,
        "oracle": oracle_sum / evaluated_tasks,
        "tasks": task_results,
    }


def main() -> int:
    args = parse_args()
    try:
        roots = expand_output_roots(args.output_root)
        for root in roots:
            if not root.is_dir():
                raise RuntimeError(f"Prediction directory does not exist: {root}")
        evaluation_dir = args.data_root / "data" / "evaluation"
        report = evaluate(roots, evaluation_dir, args.allow_missing, args.tasks)
    except (RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print(f"Prediction roots: {', '.join(report['output_roots'])}")
    print(f"Evaluated tasks: {report['evaluated_tasks']} / {report['expected_tasks']}")
    print(f"Evaluated test examples: {report['evaluated_examples']}")
    print(f"Final Oracle Score: {report['oracle']:.4f} ({report['oracle'] * 100:.3f}%)")
    print(f"Final Pass@1 Score: {report['pass_at_1']:.4f} ({report['pass_at_1'] * 100:.3f}%)")
    print(f"Final Pass@2 Score: {report['pass_at_2']:.4f} ({report['pass_at_2'] * 100:.3f}%)")

    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        with args.json_output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"JSON report: {args.json_output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
