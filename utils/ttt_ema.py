"""Paired raw/EMA prediction RNG and per-attempt TTT reports."""

from pathlib import Path
import json
import math
import os
import random

import numpy as np
import torch
from loopsight.runtime import capture_device_rng, restore_device_rng
from utils.model_ema import ModelEMA


def ema_variant_decays(decay=0.99, decays=None):
    values = list(decays) if decays is not None else [decay]
    if not values or len(values) != len(set(values)):
        raise ValueError('TTT EMA decays must be nonempty and unique')
    if any(not math.isfinite(value) or not 0 <= value < 1 for value in values):
        raise ValueError('TTT EMA decays must be finite and in [0, 1)')
    return {('ema' if index == len(values)-1 else 'ema_' + str(value).replace('.', 'p')): value
            for index, value in enumerate(values)}


class TTTEMA:
    """Multiple averages share successful optimizer steps and resume identity."""
    def __init__(self, model, decay=.99, decays=None):
        self.variants = {name: ModelEMA(model, value)
                         for name, value in ema_variant_decays(decay, decays).items()}

    @property
    def decay(self):
        return self.variants['ema'].decay

    @property
    def num_updates(self):
        return self.variants['ema'].num_updates

    def update(self, model):
        for average in self.variants.values():
            average.update(model)

    def state_dict(self):
        result = self.variants['ema'].state_dict()
        result['variants'] = {name: average.state_dict()
                              for name, average in self.variants.items() if name != 'ema'}
        return result

    def load_state_dict(self, state):
        variants = state.get('variants', {})
        if set(variants) != set(self.variants) - {'ema'}:
            raise ValueError('TTT resume EMA variants do not match')
        self.variants['ema'].load_state_dict(state)
        for name, saved in variants.items():
            self.variants[name].load_state_dict(saved)


def capture_rng(loader, device):
    generator = getattr(loader, 'generator', None)
    dataset_rng = getattr(getattr(loader, 'dataset', None), 'rng', None)
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                device=capture_device_rng(device),
                loader=generator.get_state() if generator is not None else None,
                dataset=dataset_rng.getstate() if dataset_rng is not None else None)


def restore_rng(state, loader, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    restore_device_rng(state['device'], device)
    if state['loader'] is not None:
        loader.generator.set_state(state['loader'])
    if state['dataset'] is not None:
        loader.dataset.rng.setstate(state['dataset'])


def record_attempt(args, attempt, ema, prediction_names):
    from evaluate_arc_ttt import evaluate

    task = Path(args.train_split).name
    reports = {
        kind: evaluate([Path('outputs') / name],
                       Path(args.data_root) / 'data/evaluation', False, [task])
        for kind, name in prediction_names.items()
    }
    report = dict(task=task, attempt=attempt, epoch=args.epochs,
                  source_checkpoint=args.resume_checkpoint, ema_decay=ema.decay,
                  ema_num_updates=ema.num_updates,
                  ema_decays={name: average.decay for name, average in ema.variants.items()},
                  scores={kind: {metric: result[metric]
                                 for metric in ('pass_at_1', 'pass_at_2', 'oracle')}
                          for kind, result in reports.items()})
    destination = Path(args.ttt_checkpoint_dir).parent / 'task_metrics' / task / f'attempt_{attempt}.json'
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f'.{destination.name}.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(report, indent=2) + '\n')
    temporary.replace(destination)
    print(f'TTT paired attempt {attempt}: {json.dumps(report["scores"])} '
          f'ema_decay={ema.decay} ema_updates={ema.num_updates}', flush=True)
