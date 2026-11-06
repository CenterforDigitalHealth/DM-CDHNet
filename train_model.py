"""JMDC continuation from step 40,000 with 2,000-step early stopping."""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from config import load_config, config_dict
from data import Cohort, collate, example, eligible_indices, choose_landmark
from model import CDHnet
from token_roles import medication
from utils import seed_everything, sha256, source_manifest, write_json, atomic_checkpoint, SCHEMA_VERSION


def main():
    parser = argparse.ArgumentParser()
    for name in ('config', 'train-data', 'val-data', 'out'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--resume')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    rank, local_rank, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if world != 4:
        raise ValueError('Expected four GPU processes')
    torch.cuda.set_device(local_rank)
    device = f'cuda:{local_rank}'
    dist.init_process_group('nccl')
    try:
        run(args, rank, local_rank, world, device)
    finally:
        dist.destroy_process_group()


def run(args, rank, local_rank, world, device):
    mc, tc = load_config(args.config)
    mc.validate()
    if (tc.batch_size != 64 or tc.accumulation_steps != 2
            or tc.max_steps != 50000 or tc.early_stopping_steps != 2000
            or tc.learning_rate != 1e-5 or tc.min_learning_rate != 3e-6
            or tc.warmup_steps != 0 or tc.eval_interval != 500
            or tc.val_batches != 200):
        raise ValueError('Unexpected training configuration')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'last.pt').exists() and not args.resume:
        raise ValueError('Existing run needs --resume')
    train, val = Cohort(args.train_data), Cohort(args.val_data)
    if np.intersect1d(train.ids, val.ids).size:
        raise ValueError('Train and validation patients overlap')
    if not mc.window_tokens:
        mc.window_tokens = train.select_endpoints(mc.window_top_k)
    train_ix = eligible_indices(train, mc, tc.landmark_sampling)
    val_ix = eligible_indices(val, mc, tc.landmark_sampling)
    hashes = {'train': sha256(args.train_data), 'val': sha256(args.val_data)}
    sources = source_manifest()
    seed_everything(tc.seed)
    model = CDHnet(mc).to(device)
    if not args.resume:
        histogram = np.zeros(mc.duration_max + 1, dtype=np.int64)
        for i in range(len(train)):
            rows = train.patient(i)
            duration = rows['DUR'][medication(rows['EVENT'])].astype(int)
            if len(duration) and duration.max() > mc.duration_max:
                raise ValueError('Duration exceeds model support')
            histogram += np.bincount(duration, minlength=len(histogram))
        model.duration_head.initialize_from_histogram(torch.as_tensor(histogram, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=tc.learning_rate, weight_decay=tc.weight_decay)
    rng = np.random.default_rng(tc.seed + 20000 * rank)
    landmark_rng = np.random.default_rng(tc.seed + 10000 + 20000 * rank)
    seed_everything(tc.seed + 20000 * rank)
    fixed_rng = np.random.default_rng(tc.seed + 1)
    fixed_val = fixed_rng.choice(val_ix, (tc.val_batches, tc.batch_size), replace=True) if rank == 0 else None
    fixed_ends = ([[choose_landmark(val.patient(int(i)), fixed_rng, tc.landmark_sampling) for i in draw]
                   for draw in fixed_val] if rank == 0 else None)
    step, best, best_step = 0, float('inf'), 0
    parent_checkpoint = False
    if args.resume:
        ckpt = torch.load(args.resume, map_location='cpu', weights_only=False)
        stable_sources = {name: digest for name, digest in sources.items() if name != 'train_model.py'}
        saved_stable_sources = {name: digest for name, digest in ckpt['sources'].items()
                                if name != 'train_model.py'}
        saved_training = ckpt['training_config']
        parent_checkpoint = (
            ckpt['step'] == 40000 and ckpt['best_step'] <= 40000
            and ckpt['world_size'] == 4
            and saved_training['batch_size'] == 64
            and saved_training['accumulation_steps'] == 2
            and saved_training['max_steps'] == 50000
            and saved_training['learning_rate'] == 1e-5
            and saved_training['min_learning_rate'] == 3e-6
            and saved_training['early_stopping_steps'] == 0
            and ckpt['sources'].get('train_model.py') ==
            '26ffeb167f55f854f13bdffef3d8441d146f6eefe9b072ac1033946e18980649'
        )
        same_run = (ckpt['world_size'] == world and saved_training == asdict(tc)
                    and ckpt['sources'] == sources)
        if (ckpt['model_config'] != asdict(mc) or ckpt['data_hashes'] != hashes
                or saved_stable_sources != stable_sources or not (parent_checkpoint or same_run)):
            raise ValueError('Resume configuration, data, code, or world size differs')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        step, best, best_step = ckpt['step'], ckpt['best_val'], ckpt['best_step']
        state = ckpt['rng_by_rank'][rank]
        rng.bit_generator.state = state['patient']
        landmark_rng.bit_generator.state = state['landmark']
        np.random.set_state(state['numpy'])
        random.setstate(state['python'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state(state['cuda'], device=device)
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    if rank == 0:
        write_json(out / 'config.resolved.json', config_dict(mc, tc))
        write_json(out / 'manifest.json', dict(data_hashes=hashes, sources=sources,
            train_patients=len(train), val_patients=len(val), world_size=world,
            batch_size_per_gpu=tc.batch_size, accumulation_steps=tc.accumulation_steps,
            effective_batch_size=tc.batch_size * world * tc.accumulation_steps,
            eligible_train=len(train_ix), eligible_val=len(val_ix),
            parent_checkpoint=str(args.resume) if parent_checkpoint else None,
            parent_step=40000 if parent_checkpoint else None))
    dist.barrier()
    started = time.monotonic()

    def log(kind, values):
        if rank == 0:
            record = dict(kind=kind, step=step, elapsed_seconds=time.monotonic() - started, **values)
            with (out / 'metrics.jsonl').open('a') as file:
                file.write(json.dumps(record, allow_nan=False) + '\n')
            print(json.dumps(record), flush=True)

    def save(name):
        state = dict(patient=rng.bit_generator.state, landmark=landmark_rng.bit_generator.state,
            numpy=np.random.get_state(), python=random.getstate(),
            torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state(device))
        all_states = [None] * world
        dist.all_gather_object(all_states, state)
        if rank == 0:
            atomic_checkpoint(out / name, dict(schema_version=SCHEMA_VERSION,
                format='composite-delphi-0909', model_config=asdict(mc), training_config=asdict(tc),
                model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                best_val=best, best_step=best_step, data_hashes=hashes, sources=sources,
                world_size=world, rng_by_rank=all_states))
        dist.barrier()

    def validate():
        dist.barrier()
        score = torch.zeros(1, device=device)
        if rank == 0:
            model.eval()
            sums = {}
            with torch.inference_mode():
                for draw, ends in zip(fixed_val, fixed_ends):
                    batch = collate([example(val.patient(int(i)), mc, end=end)
                                     for i, end in zip(draw, ends)], device)
                    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tc.precision == 'bf16'):
                        loss, pieces = model(batch)
                    for key, value in dict(total=loss, **pieces).items():
                        sums[key] = sums.get(key, 0.) + float(value) / len(fixed_val)
            if not all(math.isfinite(x) for x in sums.values()):
                raise FloatingPointError('Non-finite validation loss')
            log('validation', sums)
            score[0] = sums['total']
        dist.broadcast(score, src=0)
        dist.barrier()
        return float(score)

    if parent_checkpoint and rank == 0 and not (out / 'best.pt').exists():
        raise ValueError('Copy the parent best.pt before continuation')
    while step < tc.max_steps and (args.stop_after is None or step < args.stop_after):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        progress = (step - 39500) / (tc.max_steps - 39500)
        lr = tc.min_learning_rate + .5 * (tc.learning_rate - tc.min_learning_rate) * (1 + math.cos(math.pi * progress))
        for group in optimizer.param_groups:
            group['lr'] = lr
        totals = {}
        for _ in range(tc.accumulation_steps):
            draw = rng.choice(train_ix, tc.batch_size, replace=True)
            ends = [choose_landmark(train.patient(int(i)), landmark_rng, tc.landmark_sampling) for i in draw]
            batch = collate([example(train.patient(int(i)), mc, end=end) for i, end in zip(draw, ends)], device)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=tc.precision == 'bf16'):
                loss, pieces = ddp(batch)
            if not torch.isfinite(loss):
                raise FloatingPointError('Non-finite training loss')
            (loss / tc.accumulation_steps).backward()
            for key, value in dict(total=loss, **pieces).items():
                totals[key] = totals.get(key, 0.) + float(value.detach()) / tc.accumulation_steps
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        keys = tuple(totals)
        values = torch.tensor([totals[k] for k in keys] + [float(norm)], device=device, dtype=torch.float64)
        dist.all_reduce(values)
        values /= world
        log('train', dict(zip(keys, values[:-1].tolist())) | dict(learning_rate=lr, grad_norm=float(values[-1])))
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
