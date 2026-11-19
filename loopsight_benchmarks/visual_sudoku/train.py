"""Train grayscale LoopSight with pixel MSE and final-EMA evaluation."""
from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time
import numpy as np
import torch
from loopsight.runtime import resolve_device, capture_device_rng, restore_device_rng
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.distribution import setup_execution

from .boards import DEFAULT_DATA
from utils.loop_depth_sampling import LoopDepthSampler
from .dataset import PREPARED, VisualSudokuDataset
from .evaluate import evaluate
from .fetch_assets import MNIST_PATH, write_json
from .config import VisualSudokuModel, resolve_config


def _evaluate_raw_and_ema(model, emas, easy, hard, args, step):
    results = {}
    for weights, evaluated in (("raw", model), *emas.items()):
        results[weights] = {}
        for split, dataset, limit in (("easy", easy, args.validation_max_examples),
                                      ("hard", hard, args.test_max_examples)):
            result = evaluate(evaluated, dataset, args.eval_depth, args.eval_batch_size, args.device, limit)
            results[weights][split] = result
            print(json.dumps(dict(event="evaluation", step=step, weights=weights, split=split,
                epoch=math.ceil(step / args.steps_per_epoch) if args.epochs is not None else None,
                depth=args.eval_depth, batch_size=args.eval_batch_size, examples=result["examples"],
                board_accuracy=result["board_accuracy"], digit_accuracy=result["digit_accuracy"],
                pixel_mse=result["pixel_mse"], seconds=result["seconds"])), flush=True)
    return results


def evaluate_raw_and_ema(model, emas, easy, hard, args, step):
    primary = not dist.is_initialized() or dist.get_rank() == 0
    results = _evaluate_raw_and_ema(model, emas, easy, hard, args, step) if primary else None
    if dist.is_initialized():
        payload = [results]
        dist.broadcast_object_list(payload, src=0)
        results = payload[0]
    return results


def _save_checkpoint(path, model, emas, primary_name, optimizer, scaler, rng, step, best,
                    model_config, training_config, selection=None, rank_rng_states=None):
    payload = dict(model_state=model.state_dict(), ema_state=emas[primary_name].state_dict(),
                   optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                   numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                   device_rng=capture_device_rng(),
                   step=step, best=best, model_config=model_config, training_config=training_config)
    payload.update(world_size=len(rank_rng_states), rank_rng_states=rank_rng_states)
    if primary_name != "ema":
        payload.update(primary_ema=primary_name,
                       ema_states={name: tracker.state_dict() for name, tracker in emas.items()})
    if selection is not None:
        payload["selection"] = selection
    partial = path.with_suffix(".part")
    torch.save(payload, partial)
    partial.replace(path)


def save_checkpoint(path, model, emas, primary_name, optimizer, scaler, rng, step, best,
                    model_config, training_config, selection=None):
    local = dict(numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                 device_rng=capture_device_rng(next(model.parameters()).device))
    states = [local]
    if dist.is_initialized():
        states = [None] * dist.get_world_size()
        dist.all_gather_object(states, local)
        if dist.get_rank() != 0:
            return
    _save_checkpoint(path, model, emas, primary_name, optimizer, scaler, rng, step, best,
                     model_config, training_config, selection, rank_rng_states=states)


def export_final_ema(path, tracker, model_config, training_config, step, name, decay):
    final = dict(model_config=model_config, training_config=training_config, step=step,
                 model_state=tracker.state_dict(), ema_state=tracker.state_dict(),
                 weights="final_step_ema", selected_weights=name, ema_decay=decay)
    partial = path.with_suffix(".part")
    torch.save(final, partial)
    partial.replace(path)
    exported = torch.load(path, map_location="cpu", weights_only=True)
    if exported["step"] != step or any(
            not torch.equal(value.cpu(), exported["model_state"][key])
            or not torch.equal(value.cpu(), exported["ema_state"][key])
            for key, value in tracker.state_dict().items()):
        raise ValueError(f"Final EMA export differs from the last training step: {name}")


def initialize_from_export(model, path, model_config, data_protocol):
    """Start a new optimizer/EMA stage from an explicitly named inference export."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("weights") != "final_step_ema":
        raise ValueError("Initialization requires a named final EMA export")
    source_config = resolve_config(checkpoint["model_config"])
    changes = {key: [source_config.get(key), model_config.get(key)]
               for key in source_config.keys() | model_config.keys()
               if source_config.get(key) != model_config.get(key)}
    if set(changes) - {"dropout"}:
        raise ValueError(f"Initialization architecture differs: {changes}")
    saved_protocol = checkpoint["training_config"].get("data_protocol")
    if saved_protocol is not None and saved_protocol != data_protocol:
        raise ValueError("Initialization data protocol differs")
    state = checkpoint["model_state"]
    model.load_state_dict(state, strict=True)
    if any(not torch.equal(value.detach().cpu(), state[key].cpu())
           for key, value in model.state_dict().items()):
        raise ValueError("Initialized tensors differ from the source export")
    return dict(checkpoint=str(path),
                source_step=checkpoint["step"], source_weights=checkpoint["weights"],
                selected_weights=checkpoint.get("selected_weights", "ema"),
                model_changes=changes, tensors_verified=len(state),
                all_tensors_equal=True, optimizer_reset=True, ema_reset=True)


def select_checkpoint(results, metric, primary_name):
    """Easy is validation; Hard never participates in selection or tie breaking."""
    if metric == "easy_pixel_mse":
        return primary_name, [-results[primary_name]["easy"]["pixel_mse"]]
    def score(name):
        easy = results[name]["easy"]
        return [easy["board_accuracy"], easy["digit_accuracy"], -easy["pixel_mse"]]
    name = max(results, key=score)
    return name, score(name)


def _train(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["loopsight"], default="loopsight")
    p.add_argument("--run-dir", type=Path, default=Path("outputs/visual_sudoku/loopsight"))
    p.add_argument("--board-root", type=Path, default=DEFAULT_DATA)
    p.add_argument("--mnist-path", type=Path, default=MNIST_PATH)
    p.add_argument("--prepared-root", type=Path, default=PREPARED)
    p.set_defaults(device="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-seed", type=int, default=0)
    length = p.add_mutually_exclusive_group()
    length.add_argument("--epochs", type=int, help="Shuffled full passes over training boards; default: 200")
    length.add_argument("--steps", type=int, help="Legacy random step sampling instead of epoch training")
    p.add_argument("--batch-size", type=int, default=64, help="Maximum microbatch size")
    p.add_argument("--accumulation-steps", type=int, default=1,
                   help="Microbatches per optimizer update; effective batch = batch-size * accumulation-steps")
    p.add_argument("--image-size", type=int, choices=(252, 256), default=256)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--mlp-dim", type=int, default=512)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--core-depth", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--train-depth", type=int, default=18)
    p.add_argument("--sampling", choices=["fixed", "gaussian"])
    p.add_argument("--train-min", type=int, default=12)
    p.add_argument("--train-max", type=int, default=24)
    p.add_argument("--train-std", type=float, default=3.0)
    p.add_argument("--eval-depth", type=int, default=18)
    p.add_argument("--loss-steps", choices=["last", "all"], default="last")
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.02)
    ema_options = p.add_mutually_exclusive_group()
    ema_options.add_argument("--ema-decay", type=float, default=0.999)
    ema_options.add_argument("--ema-decays", type=float, nargs="+",
                             help="Track/evaluate several EMAs; the last decay is the legacy primary alias")
    p.add_argument("--amp", choices=["none", "bfloat16", "float16"], default="none")
    p.add_argument("--eval-every-epochs", type=int, default=10)
    p.add_argument("--eval-every", type=int, help="Optional evaluation interval in steps, overriding the epoch interval")
    p.add_argument("--save-every", type=int, default=1000)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--eval-batch-size", type=int, default=128)
    p.add_argument("--validation-max-examples", type=int, help="Limit easy evaluation for smoke checks")
    p.add_argument("--test-max-examples", type=int, help="Limit periodic/final hard evaluation for smoke checks")
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--resume", type=Path)
    initialization.add_argument("--init-checkpoint", type=Path,
                                help="Named final EMA export for a new optimizer/EMA stage")
    p.add_argument("--selection-metric", choices=["easy_pixel_mse", "easy_board_accuracy"],
                   default="easy_pixel_mse")
    p.add_argument("--stop-after", type=int, help="Save and stop at this step without declaring training complete")
    p.add_argument("--cpu-threads", type=int, default=2)
    p.set_defaults(image_size=252, patch_size=14, embed_dim=256, mlp_dim=512,
        heads=8, core_depth=8, dropout=0.3, train_depth=12, eval_depth=12,
        sampling="fixed", learning_rate=2e-4, weight_decay=1e-4,
        warmup_ratio=0.05, ema_decay=0.9995, amp="bfloat16")
    args = p.parse_args(argv)
    if args.epochs is None and args.steps is None:
        args.epochs = 200
    args.sampling = args.sampling or ("gaussian" if args.model == "loopsight" else "fixed")
    decays = args.ema_decays if args.ema_decays is not None else [args.ema_decay]
    if (any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in decays)
            or len({f"{beta:g}" for beta in decays}) != len(decays)):
        p.error("EMA decays must be distinct finite values in [0,1)")
    args.ema_decay = decays[-1]
    positive = [args.batch_size, args.accumulation_steps, args.train_depth, args.eval_depth,
                args.eval_every_epochs, args.log_every, args.eval_batch_size, args.cpu_threads]
    positive += [v for v in (args.epochs, args.steps, args.eval_every) if v is not None]
    if min(positive) < 1 or min(args.seed, args.eval_seed, args.save_every) < 0:
        p.error("Counts/depths must be positive; seeds and save-every must be nonnegative")
    if args.train_min < 1 or args.train_max < args.train_min or not math.isfinite(args.train_std) or args.train_std <= 0:
        p.error("Invalid Gaussian depth settings")
    if (not 0 <= args.warmup_ratio < 1 or not 0 <= args.ema_decay < 1 or
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0 or args.weight_decay < 0):
        p.error("Invalid optimizer/schedule settings")
    if args.stop_after is not None and args.stop_after < 1:
        p.error("stop-after must be positive")
    for limit in (args.validation_max_examples, args.test_max_examples):
        if limit is not None and limit < 1:
            p.error("Example limits must be positive")
    device, rank, world = setup_execution(args.device)
    distributed = world > 1
    is_main_process = rank == 0
    args.device = str(device)
    if args.amp == "float16" and device.type == "cpu":
        p.error("float16 requires an accelerator")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed + rank)
    train = VisualSudokuDataset("train", args.board_root, args.mnist_path, args.prepared_root, args.seed, args.image_size)
    effective_batch_size = args.batch_size * args.accumulation_steps
    global_batch_size = effective_batch_size * world
    padded_size = math.ceil(len(train) / world) * world
    args.steps_per_epoch = math.ceil(padded_size / global_batch_size)
    if args.epochs is not None:
        args.steps = args.epochs * args.steps_per_epoch
    if args.eval_every is None:
        args.eval_every = args.eval_every_epochs * args.steps_per_epoch if args.epochs is not None else 3000
    if args.stop_after is not None and args.stop_after > args.steps:
        p.error("stop-after exceeds the requested training horizon")
    easy = VisualSudokuDataset("easy_test", args.board_root, args.mnist_path, args.prepared_root, args.eval_seed, args.image_size)
    hard = VisualSudokuDataset("hard_test", args.board_root, args.mnist_path, args.prepared_root, args.eval_seed, args.image_size)
    model_config = resolve_config(dict(model=args.model, image_size=args.image_size, patch_size=args.patch_size,
        embed_dim=args.embed_dim, mlp_dim=args.mlp_dim, heads=args.heads, core_depth=args.core_depth,
        dropout=args.dropout, step_table_size=args.train_max if args.sampling == "gaussian" else args.train_depth))
    training_config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    training_config.update(data_protocol=train.protocol, loss="pixel_mse",
        effective_batch_size=global_batch_size, world_size=world, use_checkpoint=False)
    model = VisualSudokuModel(model_config).to(device)
    if args.init_checkpoint:
        training_config["initialization"] = initialize_from_export(
            model, args.init_checkpoint, model_config, train.protocol)
    emas = {(f"ema{beta:g}" if args.ema_decays is not None else "ema"):
            copy.deepcopy(model).eval().requires_grad_(False) for beta in decays}
    primary_name = next(reversed(emas))
    if args.ema_decays is not None:
        training_config["primary_ema"] = primary_name
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp == "float16")
    amp_dtype = torch.bfloat16 if args.amp == "bfloat16" else torch.float16
    first_step, best = 0, [-float("inf")]
    resume_rng = None
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=True)
        if "optimizer" not in checkpoint:
            raise ValueError("Resume requires last.pt/step_*.pt, not an inference-only EMA export")
        if checkpoint["model_config"] != model_config:
            raise ValueError("Resume model configuration differs")
        ignored = {"run_dir", "resume", "device", "board_root", "mnist_path", "prepared_root",
                   "stop_after", "cpu_threads", "log_every", "save_every", "eval_every",
                   "eval_every_epochs", "eval_batch_size", "validation_max_examples", "test_max_examples",
                   "steps_per_epoch"}
        previous = dict(checkpoint["training_config"])
        previous.setdefault("world_size", 1)
        previous.setdefault("data_protocol", train.protocol)
        previous.setdefault("epochs", None)
        previous.setdefault("image_size", 256)
        previous.setdefault("ema_decays", None)
        previous.setdefault("init_checkpoint", None)
        previous.setdefault("selection_metric", "easy_pixel_mse")
        training_config["init_checkpoint"] = previous["init_checkpoint"]
        if "initialization" in previous:
            fields = ("checkpoint", "source_step", "source_weights", "selected_weights",
                      "model_changes", "tensors_verified", "all_tensors_equal",
                      "optimizer_reset", "ema_reset")
            previous["initialization"] = {key: previous["initialization"][key]
                                          for key in fields if key in previous["initialization"]}
            training_config["initialization"] = previous["initialization"]
        old = {k: v for k, v in previous.items() if k in training_config and k not in ignored}
        new = {k: v for k, v in training_config.items() if k not in ignored}
        if old != new:
            raise ValueError("Resume training/data settings differ; preserve original epochs/steps and batch settings")
        model.load_state_dict(checkpoint["model_state"])
        if args.ema_decays is not None:
            if (checkpoint.get("primary_ema") != primary_name
                    or set(checkpoint.get("ema_states", {})) != set(emas)):
                raise ValueError("Resume requires all requested EMA histories")
            for name, tracker in emas.items():
                tracker.load_state_dict(checkpoint["ema_states"][name])
        else:
            emas[primary_name].load_state_dict(checkpoint["ema_state"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        states = checkpoint.get("rank_rng_states")
        if checkpoint.get("world_size", 1) != world or (states is not None and len(states) != world):
            raise ValueError("Resume requires the same number of training processes")
        resume_rng = states[rank] if states is not None else checkpoint
        first_step, best = checkpoint["step"], checkpoint["best"]
    if first_step > args.steps:
        raise ValueError("Checkpoint exceeds the requested training horizon")
    end_step = args.stop_after if args.stop_after is not None else args.steps
    if end_step < first_step:
        raise ValueError("stop-after precedes the resume checkpoint")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any((args.run_dir / f).exists() for f in ("last.pt", "ema_final.pt", "metrics.jsonl")):
        raise FileExistsError("Run already has results; choose another --run-dir or --resume")
    if is_main_process:
        write_json(args.run_dir / "config.json", dict(model=model_config, training=training_config,
                   parameters=sum(p.numel() for p in model.parameters())))
    if args.init_checkpoint:
        if is_main_process:
            write_json(args.run_dir / "initialization.json", training_config["initialization"])
        if is_main_process:
            print(json.dumps(dict(event="initialization", **training_config["initialization"])), flush=True)
    forward_model = DDP(
        model, device_ids=[device.index] if device.type != "cpu" else None,
        broadcast_buffers=False,
        find_unused_parameters=(args.train_min == 1 if args.sampling == "gaussian" else args.train_depth == 1),
    ) if distributed else model
    if distributed:
        torch.manual_seed(args.seed + rank)
    if resume_rng is not None:
        rng.bit_generator.state = resume_rng["numpy_rng"]
        torch.set_rng_state(resume_rng["torch_rng"].cpu())
        restore_device_rng(resume_rng["device_rng"], device)
    sampler = LoopDepthSampler(args.train_min, args.train_max, args.train_depth, args.train_std, args.seed)
    warmup = int(args.steps * args.warmup_ratio)
    started = time.perf_counter()
    final_results = None
    cached_epoch, order = None, None
    for step in range(first_step + 1, end_step + 1):
        forward_model.train()
        depth = args.train_depth if args.sampling == "fixed" else sampler.sample_update(step - 1, device, distributed)
        epoch, batch_in_epoch = None, None
        if args.epochs is not None:
            epoch_index, batch_index = divmod(step - 1, args.steps_per_epoch)
            if cached_epoch != epoch_index:
                # Indexable private RNG reconstructs the same order after a mid-epoch resume.
                order = np.random.default_rng(np.random.SeedSequence(
                    [args.seed, epoch_index, 1907])).permutation(len(train))
                if len(order) < padded_size:
                    order = np.resize(order, padded_size)
                cached_epoch = epoch_index
            start = batch_index * global_batch_size
            indices = order[start:start + global_batch_size][rank::world]
            batch = train.sample_indices(indices, rng)
            epoch, batch_in_epoch = epoch_index + 1, batch_index + 1
        else:
            batch = train.sample(effective_batch_size, rng)
        multiplier = (step / warmup if warmup and step <= warmup else
                      0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, args.steps - warmup))))
        for group in optimizer.param_groups:
            group["lr"] = args.learning_rate * multiplier
        optimizer.zero_grad(set_to_none=True)
        # Render one effective batch before splitting: board/glyph RNG and the sampled
        # depth belong to an optimizer update, as do scheduling, clipping and EMA.
        batch_count = len(batch["image"])
        loss_value = 0.0
        for micro_start in range(0, batch_count, args.batch_size):
            micro_end = min(micro_start + args.batch_size, batch_count)
            images = batch["image"][micro_start:micro_end].to(device)
            targets = batch["target"][micro_start:micro_end].to(device)
            weight = (micro_end - micro_start) / batch_count
            synchronization = (forward_model.no_sync()
                               if distributed and micro_end < batch_count else nullcontext())
            with synchronization:
                with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=args.amp != "none"):
                    if args.loss_steps == "all":
                        _, states = forward_model(images, depth, return_intermediates=True)
                        loss = sum(F.mse_loss(state.float(), targets) for state in states) / len(states)
                    else:
                        loss = F.mse_loss(forward_model(images, depth).float(), targets)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"Non-finite pixel MSE at step {step}")
                # Weight by actual sample count, including uneven final microbatches.
                loss_value += loss.detach().item() * weight
                scaler.scale(loss * weight).backward()
            if args.loss_steps == "all":
                del states, _
            del loss, images, targets
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() >= scale_before:
            with torch.no_grad():
                state = model.state_dict()
                for tracker, decay in zip(emas.values(), decays):
                    for key, value in tracker.state_dict().items():
                        if value.is_floating_point():
                            value.lerp_(state[key], 1 - decay)
                        else:
                            value.copy_(state[key])
        if distributed:
            total_loss = torch.tensor(loss_value, dtype=torch.float64, device=device)
            dist.all_reduce(total_loss)
            loss_value = float(total_loss / world)
        if step == 1 or step % args.log_every == 0 or step == end_step:
            if is_main_process:
                print(json.dumps(dict(event="train", step=step, depth=depth, pixel_mse=loss_value,
                    epoch=epoch, batch_in_epoch=batch_in_epoch, batch_size=batch_count * world,
                    micro_batch_size=args.batch_size, accumulation_steps=args.accumulation_steps,
                    learning_rate=optimizer.param_groups[0]["lr"],
                    elapsed_seconds=time.perf_counter() - started)), flush=True)
        should_evaluate = step % args.eval_every == 0 or step == end_step
        if should_evaluate:
            results = evaluate_raw_and_ema(model, emas, easy, hard, args, step)
            selected_name, score = select_checkpoint(results, args.selection_metric, primary_name)
            selection = dict(metric=args.selection_metric, weights=selected_name, score=score,
                             step=step, epoch=epoch, easy=results[selected_name]["easy"])
            improved = score > best
            if improved:
                best = score
            record = dict(step=step, epoch=epoch, depth=depth, train_pixel_mse=loss_value,
                          evaluation=results, validation=results[selected_name]["easy"], selection=selection,
                          elapsed_seconds=time.perf_counter() - started)
            if is_main_process:
                write_json(args.run_dir / "eval" / f"step_{step:06d}.json", record)
            if is_main_process:
                with (args.run_dir / "metrics.jsonl").open("a") as handle:
                    handle.write(json.dumps(record) + "\n")
            if improved:
                save_checkpoint(args.run_dir / "best.pt", model, emas, primary_name, optimizer, scaler, rng,
                                step, best, model_config, training_config, selection=selection)
                if is_main_process:
                    write_json(args.run_dir / "best_selection.json", selection)
            if step == args.steps:
                final_results = results
        if should_evaluate or (args.save_every and step % args.save_every == 0):
            save_checkpoint(args.run_dir / "last.pt", model, emas, primary_name, optimizer, scaler, rng,
                            step, best, model_config, training_config)
        if args.save_every and step % args.save_every == 0:
            save_checkpoint(args.run_dir / f"step_{step:06d}.pt", model, emas, primary_name, optimizer, scaler, rng,
                            step, best, model_config, training_config)
    if end_step < args.steps:
        if distributed:
            dist.barrier()
        length_option = f"--epochs={args.epochs}" if args.epochs is not None else f"--steps={args.steps}"
        if is_main_process:
            print(f"Stopped at {end_step}; resume last.pt with the same {length_option}", flush=True)
        return
    final_paths = {}
    for (name, tracker), decay in zip(emas.items(), decays):
        filename = f"{name}_final.pt" if args.ema_decays is not None else "ema_final.pt"
        path = args.run_dir / filename
        if is_main_process:
            export_final_ema(path, tracker, model_config, training_config, args.steps, name, decay)
        if args.ema_decays is not None and name == primary_name:
            if is_main_process:
                export_final_ema(args.run_dir / "ema_final.pt", tracker, model_config,
                                 training_config, args.steps, name, decay)
            filename = "ema_final.pt"
        final_paths[name] = filename
    if distributed:
        dist.barrier()
    if final_results is None:
        # Resuming an already completed run must evaluate the verified final EMA states.
        final_results = evaluate_raw_and_ema(model, emas, easy, hard, args, args.steps)
    reports = []
    for split in ("easy", "hard"):
        reports.append(dict(weights="raw_last", checkpoint="last.pt", step=args.steps,
                            result=final_results["raw"][split]))
    for name, decay in zip(emas, decays):
        label = "final_step_ema" if name == primary_name else f"final_step_{name}"
        for split in ("easy", "hard"):
            report = dict(weights=label, checkpoint=final_paths[name], step=args.steps,
                          result=final_results[name][split])
            if args.ema_decays is not None:
                report.update(selected_weights=name, ema_decay=decay,
                              named_checkpoint=f"{name}_final.pt")
            reports.append(report)
    if is_main_process:
        write_json(args.run_dir / "final_evaluation.json", reports)


def main(argv=None):
    try:
        _train(argv)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
