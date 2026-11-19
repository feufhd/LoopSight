# ImageNet-100

Run the commands below from the repository root. All data and output paths
are relative to that directory. Image files and weights are supplied separately.

Obtain the ImageNet-1K ILSVRC 2012 training images and labeled validation images
through the official ImageNet access process under its terms. Extract them and
arrange both splits into synset directories. Validation images must be assigned
to synsets using their official labels.

```text
raw_data/imagenet/
  train/
    n01440764/
      image.JPEG
    ...
  val/
    n01440764/
      image.JPEG
    ...
```

The full 1000-class ImageNet tree or a tree containing the selected 100 synsets
can be used. `loopsight_benchmarks/imagenet100/imagenet100_classes.txt` stores the frozen
CMC selection.
Both splits use the same sorted synset-to-label mapping from 0 to 99. The loader
scans selected synsets directly, so a duplicate image copy is unnecessary.

```bash
python -m loopsight_benchmarks.imagenet100.prepare --data-root raw_data/imagenet
```

The fixed subset contains 100 classes, 126689 training images, and 5000
validation images. An optional `raw_data/imagenet/class_mapping.json` is validated when
present. ImageNet images are supplied separately from this repository.

Training uses random resized 224-pixel crops, horizontal flips, RandAugment,
random erasing, Mixup, and CutMix. Validation resizes the short edge to 256 and
center-crops to 224. Both use RGB ImageNet channel normalization. The
classification model mean-pools the final prediction memory and produces 100
logits. Evaluation reports Top-1 and Top-5 accuracy.

## Training

```bash
python -m loopsight_benchmarks.imagenet100.train
```

The current LoopSight classifier uses 224 × 224 RGB inputs, 16 × 16 patches,
six shared blocks, fixed six-loop computation, mean-pooled prediction memory,
and a 100-class head. The recipe uses 100 epochs, effective batch 1024,
AdamW at peak learning rate 5e-4, weight decay 0.05, five warmup epochs,
dropout 0.2, and EMA decay 0.9999.

Adjust the microbatch and accumulation count as needed.

```bash
python -m loopsight_benchmarks.imagenet100.train --batch-size 64 --accumulation-steps 16
```

### Multi-GPU training

```bash
torchrun --standalone --nproc_per_node=4 \
  -m loopsight_benchmarks.imagenet100.train \
  --batch-size 64 --accumulation-steps 4
```

Set `--nproc_per_node` to the number of GPUs. Batch size is per process, so this
example keeps effective batch at `4 × 64 × 4 = 1024`. Each process receives a
different training-data shard and gradients are synchronized with
DistributedDataParallel. Raw and EMA evaluation covers each validation image
once; the primary process writes the checkpoints and final EMA export.

For a distributed run, append `--resume outputs/imagenet100/loopsight/last.pt`
to the same `torchrun` command, preserving process count and batch settings.
For an ordinary run:

```bash
python -m loopsight_benchmarks.imagenet100.train --resume outputs/imagenet100/loopsight/last.pt
```

Training exports the final-update EMA to
`outputs/imagenet100/loopsight/ema_final.pt` and reports raw and final-EMA results.

## Evaluation

```bash
python -m loopsight_benchmarks.imagenet100.evaluate \
  --checkpoint outputs/imagenet100/loopsight/ema_final.pt \
  --data-root raw_data/imagenet --dataset imagenet100 --depths 6 \
  --output outputs/imagenet100/loopsight/final_ema_evaluation.json
```
