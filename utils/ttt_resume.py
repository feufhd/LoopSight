"""Atomic epoch checkpoints for independent TTT processes, including RNG state."""

import os
from pathlib import Path
import random

import numpy as np
import torch
from loopsight.runtime import capture_device_rng, restore_device_rng


SETTINGS = ('architecture', 'embed_dim', 'mlp_dim', 'num_heads', 'loop_core_depth', 'image_size',
            'patch_size', 'fusion_gate_rank', 'fusion_gate_logit_bound', 'epochs', 'batch_size',
            'learning_rate', 'weight_decay', 'max_grad_norm', 'lr_scheduler', 'warmup_epochs',
            'seed', 'no_amp', 'max_loop_steps', 'runtime_max_steps', 'num_attempts',
            'ttt_num_each', 'use_checkpoint', 'checkpoint_blocks', 'checkpoint_gate',
            'data_root', 'train_split', 'disable_translation', 'disable_resolution_augmentation', 'fix_scale_factor')


def progress_path(args, attempt):
    return Path(args.ttt_resume_dir) / Path(args.train_split).name / f'attempt_{attempt}.last.pt'


def identity(args, attempt, loader):
    settings = {name: getattr(args, name, None) for name in SETTINGS}
    if getattr(args, 'ttt_ema', False):
        settings.update(ttt_ema=True, ttt_ema_decay=args.ttt_ema_decay)
        if getattr(args, 'ttt_ema_decays', None) is not None:
            settings['ttt_ema_decays'] = args.ttt_ema_decays
    return dict(source_checkpoint=str(Path(args.resume_checkpoint).resolve()),
                attempt=attempt,
                task_lookup=dict(loader.dataset.task_lookup),
                settings=settings)


def restore(model, optimizer, scaler, scheduler, args, attempt, loader, device, *, ema=None):
    if args.distributed or loader.num_workers != 0:
        raise ValueError('Epoch-resumable TTT requires an independent process with num-workers=0')
    path = progress_path(args, attempt)
    if not path.is_file():
        return 1
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)

    expected = identity(args, attempt, loader)
    saved_identity = {key: value for key, value in checkpoint['identity'].items() if key in expected}
    if saved_identity != expected:
        raise ValueError(f'TTT resume checkpoint has a different source/task/budget: {path}')
    if not 1 <= checkpoint['epoch'] <= args.epochs:
        raise ValueError(f'TTT resume checkpoint has an invalid epoch: {path}')
    model.load_state_dict(checkpoint['model_state'], strict=True)
    if ema is not None:
        if 'ema_state' not in checkpoint:
            raise ValueError('TTT resume checkpoint lacks EMA history')
        ema.load_state_dict(checkpoint['ema_state'])
    optimizer.load_state_dict(checkpoint['optimizer_state'])
    if scaler.is_enabled():
        scaler.load_state_dict(checkpoint['scaler_state'])
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state'])
    rng = checkpoint['rng']
    random.setstate(rng['python'])
    np.random.set_state(rng['numpy'])
    torch.set_rng_state(rng['torch'])
    restore_device_rng(rng['device'], device)
    if loader.generator is not None:
        loader.generator.set_state(rng['loader'])
    if rng.get('dataset') is not None:
        loader.dataset.rng.setstate(rng['dataset'])
    print(f'Resuming TTT {Path(args.train_split).name} attempt {attempt}: '
          f'completed epoch {checkpoint["epoch"]}/{args.epochs}', flush=True)
    return checkpoint['epoch'] + 1


def save(model, optimizer, scaler, scheduler, args, attempt, loader, device, epoch, *, ema=None):
    path = progress_path(args, attempt)
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = dict(format_version=1, identity=identity(args, attempt, loader), epoch=epoch,
                      model_state={name: value.detach().cpu() for name, value in model.state_dict().items()},
                      optimizer_state=optimizer.state_dict(),
                      scaler_state=scaler.state_dict() if scaler.is_enabled() else None,
                      scheduler_state=scheduler.state_dict() if scheduler is not None else None,
                      rng=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                               device=capture_device_rng(device),
                               loader=loader.generator.get_state() if loader.generator is not None else None,
                               dataset=loader.dataset.rng.getstate() if hasattr(loader.dataset, 'rng') else None))
    if ema is not None:
        checkpoint['ema_state'] = ema.state_dict()
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    try:
        torch.save(checkpoint, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
