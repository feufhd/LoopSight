"""Export an ARC input, target, and majority-vote prediction as one PNG."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from utils.eval_utils import get_majority_vote


COLORS = (
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
)


def export_image(task, predictions, output, *, example_index=0, cell_size=24):
    example = task["test"][example_index]
    candidates = predictions.get(str(example_index), [])
    if not candidates:
        raise ValueError(f"No predictions for test example {example_index}")
    prediction = get_majority_vote(candidates)[0]["prediction"]
    panels = [("Input", example["input"])]
    if "output" in example:
        panels.append(("Target", example["output"]))
    panels.append(("Prediction", prediction))

    if cell_size < 2:
        raise ValueError("Cell size must be at least 2")
    for _, grid in panels:
        if (not grid or not grid[0] or any(len(row) != len(grid[0]) for row in grid)
                or any(not isinstance(value, int) or not 0 <= value < len(COLORS)
                       for row in grid for value in row)):
            raise ValueError("Each grid must be a nonempty rectangle with colors 0 through 9")

    padding, label_height = 16, 24
    widths = [max(96, len(grid[0]) * cell_size) for _, grid in panels]
    width = sum(widths) + padding * (len(panels) + 1)
    height = max(len(grid) for _, grid in panels) * cell_size + label_height + 2 * padding
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left = padding
    for (label, grid), panel_width in zip(panels, widths):
        draw.text((left, padding), label, fill="black")
        for row_index, row in enumerate(grid):
            for column_index, value in enumerate(row):
                x = left + column_index * cell_size
                y = padding + label_height + row_index * cell_size
                draw.rectangle((x, y, x + cell_size - 1, y + cell_size - 1),
                               fill=COLORS[value], outline="#555555")
        left += panel_width + padding
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True, help="Original ARC task JSON")
    parser.add_argument("--predictions", type=Path, required=True, help="One attempt's prediction JSON")
    parser.add_argument("--output", type=Path, default=Path("outputs/visualization.png"))
    parser.add_argument("--example-index", type=int, default=0)
    parser.add_argument("--cell-size", type=int, default=24)
    args = parser.parse_args()
    if args.example_index < 0:
        parser.error("--example-index must be nonnegative")
    export_image(json.loads(args.task.read_text()), json.loads(args.predictions.read_text()),
                 args.output, example_index=args.example_index, cell_size=args.cell_size)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
