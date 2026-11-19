# Visual Sudoku

Run the commands below from the repository root. All data and output paths
are relative to that directory. Data and weights are prepared separately.

Prepare the Sudoku-OOD boards and download the MNIST glyphs separately.
The board preparation tool retrieves the SATNet and RRN sources and writes
the train, Easy, and Hard board arrays. The MNIST helper downloads the official
Keras distribution.

```bash
python -m loopsight_benchmarks.visual_sudoku.prepare_boards --download --benchmarks sudoku-ood
python -m loopsight_benchmarks.visual_sudoku.fetch_assets \
  --mnist-path raw_data/visual_sudoku/mnist.npz
```

The current Visual Sudoku experiment uses 252 × 252 images. Build its own
rendering manifest, fixed Hard subset indices, and prototype digit decoder.

```bash
python -m loopsight_benchmarks.visual_sudoku.prepare \
  --board-root raw_data/visual_sudoku/boards \
  --mnist-path raw_data/visual_sudoku/mnist.npz \
  --image-size 252 \
  --output raw_data/visual_sudoku/prepared
```

| Input | Relative location |
| --- | --- |
| Training boards | `raw_data/visual_sudoku/boards/sudoku_ood/train.npz` |
| Easy boards | `raw_data/visual_sudoku/boards/sudoku_ood/id_test.npz` |
| Full Hard board pool | `raw_data/visual_sudoku/boards/sudoku_ood/ood_test.npz` |
| MNIST glyphs | `raw_data/visual_sudoku/mnist.npz` |
| 252-pixel protocol and prototypes | `raw_data/visual_sudoku/prepared/` |

Training uses 9000 SATNet boards and the MNIST test glyph pool. Easy uses 1000
held-out SATNet boards. Hard is a fixed 1000-board subset of the 18000-board
RRN test pool, sampled without replacement using seed 0. Easy and Hard use
the MNIST training glyph pool. Input and target glyphs are independently
sampled. Training resamples glyphs, and evaluation rendering is deterministic.
Easy also supplies validation for checkpoint selection.

Images are rendered from 28 × 28 glyphs arranged into a 252 × 252 board.
The 252-pixel protocol uses no additional border. The model receives the
grayscale image and reconstructs a complete grayscale board. Symbolic solutions
and clue masks provide rendering/evaluation labels. Pixel MSE supplies training
supervision. MNIST test-class means decode predicted images into board digits.

The preparation tools check split sizes, prototype values,
and fixed Hard-subset indices. Prepared boards, MNIST, and rendering arrays
are generated locally and are excluded from Git.

## Training

```bash
python -m loopsight_benchmarks.visual_sudoku.train
```

The latest recipe uses 252 × 252 grayscale images, 14 × 14 patches, fixed
12-loop computation, pixel MSE supervision, and EMA decay 0.9995. The default
run directory is `outputs/visual_sudoku/loopsight/`. The model, shared core,
and prediction memory are the current LoopSight implementation.

Change the microbatch and accumulation count to fit the available device.

```bash
python -m loopsight_benchmarks.visual_sudoku.train --batch-size 16 --accumulation-steps 4
```

### Multi-GPU training

```bash
torchrun --standalone --nproc_per_node=4 \
  -m loopsight_benchmarks.visual_sudoku.train \
  --batch-size 16 --accumulation-steps 1
```

Set `--nproc_per_node` to the number of GPUs. Batch size is per process, so this
example keeps effective batch at `4 × 16 × 1 = 64`. Each process renders its own
board shard; DistributedDataParallel synchronizes gradients. All processes use
the same sampled depth when Gaussian loop sampling is selected. The primary
process evaluates Easy and Hard and writes checkpoints and final EMA exports.
Resume with the same process count and batch settings; the checkpoint stores
each process's rendering and random state.

For a distributed run, append `--resume outputs/visual_sudoku/loopsight/last.pt`
to the same `torchrun` command. For an ordinary run:

```bash
python -m loopsight_benchmarks.visual_sudoku.train --resume outputs/visual_sudoku/loopsight/last.pt
```

Training exports `outputs/visual_sudoku/loopsight/ema_final.pt` from the final
training update and reports raw and final-EMA results on Easy and Hard.

## Evaluation

```bash
python -m loopsight_benchmarks.visual_sudoku.evaluate \
  --checkpoint outputs/visual_sudoku/loopsight/ema_final.pt \
  --board-root raw_data/visual_sudoku/boards \
  --mnist-path raw_data/visual_sudoku/mnist.npz \
  --prepared-root raw_data/visual_sudoku/prepared \
  --splits easy_test hard_test --depths 12 \
  --output outputs/visual_sudoku/loopsight/final_ema_evaluation.json
```
