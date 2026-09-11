"""Explicit-path, single-device training with fixed validation draws and resume."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
import time
import numpy as np
import torch
from config import load_config, config_dict
from data import Cohort, collate, example, eligible_indices, choose_landmark
from model import CDHnet
from token_roles import medication
from utils import seed_everything, sha256, source_manifest, write_json, atomic_checkpoint, load_checkpoint, SCHEMA_VERSION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--train-data', required=True)
    p.add_argument('--val-data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--time-family', choices=['exponential', 'weibull', 'discrete'])
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--resume')
    p.add_argument('--stop-after', type=int, help='Stop early without changing the configured LR schedule; useful for resume tests')
    p.add_argument('--seed', type=int, help='Override training seed for a paired replicate')
    p.add_argument('--batch-size', type=int, help='Override micro-batch size for throughput tuning')
    p.add_argument('--accumulation-steps', type=int, help='Override gradient accumulation for throughput tuning')
    p.add_argument('--val-batches', type=int, help='Override validation batches; keep validation examples approximately fixed')
    args = p.parse_args()
    mc, tc = load_config(args.config)
    if args.seed is not None:
        tc.seed = args.seed
    if args.batch_size is not None:
        tc.batch_size = args.batch_size
    if args.accumulation_steps is not None:
        tc.accumulation_steps = args.accumulation_steps
    if args.val_batches is not None:
        tc.val_batches = args.val_batches
    if args.time_family:
        mc.time_family = args.time_family
    mc.validate()
    if min(tc.max_steps, tc.batch_size, tc.accumulation_steps, tc.eval_interval, tc.val_batches) < 1:
        raise ValueError('Step/batch settings must be positive')
    if tc.precision not in ('fp32', 'bf16'):
        raise ValueError('precision must be fp32 or bf16')
    if tc.landmark_sampling not in ('left', 'random_day'):
        raise ValueError('landmark_sampling must be left or random_day')
    if tc.warmup_steps < 0 or tc.learning_rate <= 0 or not 0 <= tc.min_learning_rate <= tc.learning_rate:
        raise ValueError('Invalid learning-rate schedule')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not args.resume:
        raise ValueError('Output already contains a run; choose a new directory or --resume')
    train, val = Cohort(args.train_data), Cohort(args.val_data)
    if np.intersect1d(train.ids, val.ids).size:
        raise ValueError('Training and validation patient IDs overlap')
    if not mc.window_tokens:
        mc.window_tokens = train.select_endpoints(mc.window_top_k)
    train_ix = eligible_indices(train, mc, tc.landmark_sampling)
    val_ix = eligible_indices(val, mc, tc.landmark_sampling)
    data_hashes = {'train': sha256(args.train_data), 'val': sha256(args.val_data)}
    seed_everything(tc.seed)
    patient_rng = np.random.default_rng(tc.seed)
    landmark_rng = np.random.default_rng(tc.seed + 10_000)
    fixed_rng = np.random.default_rng(tc.seed + 1)
    fixed_val = fixed_rng.choice(val_ix, (tc.val_batches, tc.batch_size), replace=True)
    fixed_val_end = [[choose_landmark(val.patient(int(i)), fixed_rng, tc.landmark_sampling) for i in draw]
                     for draw in fixed_val]
    model = CDHnet(mc).to(args.device)
    hist = np.zeros(mc.duration_max + 1, dtype=np.int64)
    # Training-only empirical initialization; reject values outside the head support.
    for i in range(len(train)):
        rows = train.patient(i)
        duration = rows['DUR'][medication(rows['EVENT'])].astype(int)
        if len(duration) and duration.max() > mc.duration_max:
            raise ValueError('Training duration exceeds configured support')
        hist += np.bincount(duration, minlength=len(hist))
    model.duration_head.initialize_from_histogram(torch.as_tensor(hist, device=args.device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)
    step, best, best_step = 0, float('inf'), 0
    if args.resume:
        restored, ckpt = load_checkpoint(args.resume, args.device)
        if ckpt['model_config'] != asdict(mc) or ckpt['training_config'] != asdict(tc) or ckpt['data_hashes'] != data_hashes:
            raise ValueError('Resume requires identical resolved configuration and data')
        if ckpt['sources'] != source_manifest():
            raise ValueError('Source files changed since checkpoint; exact resume requires identical sources')
        model.load_state_dict(restored.state_dict())
        optimizer.load_state_dict(ckpt['optimizer'])
        step, best, best_step = ckpt['step'], ckpt['best_val'], ckpt['best_step']
        patient_rng.bit_generator.state = ckpt['rng']['patient_generator']
        landmark_rng.bit_generator.state = ckpt['rng']['landmark_generator']
        np.random.set_state(ckpt['rng']['numpy'])
        random.setstate(ckpt['rng']['python'])
        torch.set_rng_state(ckpt['rng']['torch'])
        if args.device.startswith('cuda') and ckpt['rng']['cuda'] is not None:
            torch.cuda.set_rng_state_all(ckpt['rng']['cuda'])
    write_json(out / 'config.resolved.json', config_dict(mc, tc))
    sources = source_manifest()
    wall_started = time.monotonic()
    write_json(out / 'manifest.json', dict(data_hashes=data_hashes, sources=sources, device=args.device,
               torch=torch.__version__, train_patients=len(train), val_patients=len(val),
               eligible_train=len(train_ix), eligible_val=len(val_ix),
               post_death=dict(train_patients=train.post_death_patients, train_rows=train.post_death_clinical_rows,
                               val_patients=val.post_death_patients, val_rows=val.post_death_clinical_rows,
                               handling='in-memory truncation through first Death; source file unchanged'),
               validation='fixed patient and landmark draws; configured loss reduction'))
    def amp():
        return torch.autocast('cuda', dtype=torch.bfloat16) if args.device.startswith('cuda') and tc.precision == 'bf16' else nullcontext()

    def save(name):
        atomic_checkpoint(out / name, dict(schema_version=SCHEMA_VERSION, format='composite-delphi-0909',
            model_config=asdict(mc), training_config=asdict(tc), model=model.state_dict(), optimizer=optimizer.state_dict(),
            step=step, best_val=best, best_step=best_step, data_hashes=data_hashes, sources=sources,
            rng=dict(patient_generator=patient_rng.bit_generator.state, landmark_generator=landmark_rng.bit_generator.state,
                     numpy=np.random.get_state(), python=random.getstate(),
                     torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)))

    def log(kind, values):
        record = dict(kind=kind, step=step, elapsed_seconds=time.monotonic() - wall_started, **values)
        with (out / 'metrics.jsonl').open('a') as f:
            f.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps(record), flush=True)

    def validate():
        model.eval()
        sums = {}
        with torch.inference_mode():
            for draw, ends in zip(fixed_val, fixed_val_end):
                batch = collate([example(val.patient(int(i)), mc, end=end) for i, end in zip(draw, ends)], args.device)
                with amp():
                    loss, pieces = model(batch)
                for key, value in dict(total=loss, **pieces).items():
                    sums[key] = sums.get(key, 0.) + float(value) / len(fixed_val)
        if not all(math.isfinite(v) for v in sums.values()):
            raise FloatingPointError('Non-finite validation loss')
        log('validation', sums)
        return sums['total']

    if not args.resume:
        best = validate()
        save('best.pt')
    while step < tc.max_steps and (args.stop_after is None or step < args.stop_after):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        if step < tc.warmup_steps:
            lr = tc.learning_rate * (step + 1) / max(tc.warmup_steps, 1)
        else:
            progress = (step - tc.warmup_steps) / max(tc.max_steps - tc.warmup_steps, 1)
            lr = tc.min_learning_rate + .5 * (tc.learning_rate - tc.min_learning_rate) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = lr
        totals = {}
        for _ in range(tc.accumulation_steps):
            draw = patient_rng.choice(train_ix, tc.batch_size, replace=True)
            ends = [choose_landmark(train.patient(int(i)), landmark_rng, tc.landmark_sampling) for i in draw]
            batch = collate([example(train.patient(int(i)), mc, end=end) for i, end in zip(draw, ends)], args.device)
            with amp():
                loss, pieces = model(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            (loss / tc.accumulation_steps).backward()
            for key, value in dict(total=loss, **pieces).items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) / tc.accumulation_steps
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        log('train', dict(**totals, learning_rate=lr, grad_norm=float(norm)))
        if step % tc.eval_interval == 0 or step == tc.max_steps:
            value = validate()
            if value < best:
                best, best_step = value, step
                save('best.pt')
        stopping = tc.early_stopping_steps > 0 and step - best_step >= tc.early_stopping_steps
        if step % tc.eval_interval == 0 or stopping:
            save('last.pt')
        if stopping:
            break
    save('last.pt')


if __name__ == '__main__':
    main()
