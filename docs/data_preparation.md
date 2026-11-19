# ARC data preparation

Run these commands from the repository root. Data locations below are relative
to that directory. Generated datasets and checkpoints stay in ignored folders.

## ARC-AGI and RE-ARC

Datasets and model weights are distributed separately. Obtain the public
ARC-AGI-1 data from the official `fchollet/ARC-AGI` source and ARC-AGI-2 from
`arcprize/ARC-AGI-2`. Stage the source trees as `downloads/ARC-AGI/` and
`downloads/ARC-AGI-2/`. Keep each task's original JSON identifier and preserve
its `train` and `test` examples.

For RE-ARC, obtain the `re_arc.zip` data release from `michaelhodel/re-arc` and
save it as `downloads/re_arc.zip`. That upstream release contains 1000 verified
synthetic examples for each of the 400 ARC-AGI-1 training tasks. Its archive
contains the `re_arc/tasks/` directory. All downloaded and generated files stay
in ignored directories.

```bash
mkdir -p raw_data/ARC-AGI/data raw_data/ARC-AGI-2/data
cp -R downloads/ARC-AGI/data/training downloads/ARC-AGI/data/evaluation raw_data/ARC-AGI/data/
cp -R downloads/ARC-AGI-2/data/training downloads/ARC-AGI-2/data/evaluation raw_data/ARC-AGI-2/data/
python -m zipfile -e downloads/re_arc.zip raw_data
```

| Data | Relative directory | Tasks |
| --- | --- | ---: |
| ARC-AGI-1 training | `raw_data/ARC-AGI/data/training/` | 400 |
| ARC-AGI-1 evaluation | `raw_data/ARC-AGI/data/evaluation/` | 400 |
| ARC-AGI-1 TTT augmentations | `raw_data/ARC-AGI/data/eval_color_permute_ttt_9/` | 400 task directories |
| ARC-AGI-2 training | `raw_data/ARC-AGI-2/data/training/` | 1000 |
| ARC-AGI-2 evaluation | `raw_data/ARC-AGI-2/data/evaluation/` | 120 |
| RE-ARC synthetic examples | `raw_data/re_arc/tasks/` | 400 |

Offline training uses the ARC-AGI-1 training demonstrations plus RE-ARC. The
held-out examples of those training tasks provide offline validation. TTT
optimizes each evaluation task's demonstrations. Predictions are compared with
that task's held-out query outputs after adaptation.

Prepare TTT augmentations separately for each ARC evaluation split. This
writes an independent directory of per-task transformations and nine
additional color permutations per transformation.

```bash
python prepare_arc_ttt.py --data-root raw_data/ARC-AGI
python prepare_arc_ttt.py --data-root raw_data/ARC-AGI-2
```

The results are each dataset's `data/eval_color_permute_ttt_9/` directories. Both ARC TTT
launchers select tasks by their original evaluation JSON IDs. Input translation
and resolution augmentation are applied by the existing data loader.
