#!/usr/bin/env python3
"""Batched full-cohort 3-year generation from corrected history into KR test."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import features
from token_roles import DTYPE, clinical, medication, MED_MIN, N_MED, DEATH
from utils import load_checkpoint, sha256, write_json


REASON_CODE = {'horizon': 0, 'death': 1, 'max_events': 2}
MODEL_KEYS = ('token', 'age', 'shift', 'duration', 'input_action', 'history_gap', 'before', 'through')


def patient_order_hash(ids):
    return hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()


def categorical(logits, uniforms):
    probability = logits.float().softmax(-1)
    cdf = probability.cumsum(-1)
    index = (uniforms[:, None] > cdf).sum(-1)
    return index.clamp_max(logits.shape[-1] - 1)


class GenerationState:
    """Incremental equivalent of data.features for an autoregressive trajectory."""
    def __init__(self, prefix, recent_days, context_length):
        full = features(prefix, recent_days)
        self.values = {key: value[-context_length:].copy() for key, value in full.items()}
        self.patient_id = int(prefix['ID'][0])
        self.last_row_age = int(prefix['AGE'][-1])
        clinical_index = np.flatnonzero(clinical(prefix['EVENT']))
        if not len(clinical_index):
            raise ValueError('Generation prefix has no supported clinical event')
        self.previous_clinical_age = int(prefix['AGE'][clinical_index[-1]])
        self.last = full['through'][-1].copy()
        self.day_before = full['before'][-1].copy()
        self.current_day = self.last_row_age
        self.recent_days = recent_days
        self.context_length = context_length

    def append(self, row):
        _, age, token, shift, duration = row
        age, token = int(age), int(token)
        if age != self.current_day:
            self.day_before = self.last.copy()
            self.current_day = age
        before = self.day_before.copy()
        action = 0
        if medication(token):
            med = token - MED_MIN
            action = 1 if before[med] < 0 else (2 if age - before[med] <= self.recent_days else 3)
            self.last[med] = age
        through = self.last.copy()
        additions = {
            'token': np.int64(token), 'age': np.float32(age), 'shift': np.int64(shift),
            'duration': np.float32(duration), 'input_action': np.int64(action),
            'history_gap': np.float32(age - self.previous_clinical_age),
            'before': before.astype(np.float32), 'through': through.astype(np.float32),
        }
        for key, value in additions.items():
            self.values[key] = np.concatenate((self.values[key], np.asarray(value)[None]))[-self.context_length:]
        self.previous_clinical_age = age
        self.last_row_age = age


def collate_states(states, device):
    length = max(len(state.values['token']) for state in states)
    result = {}
    for key in MODEL_KEYS:
        sample = states[0].values[key]
        shape = (len(states), length) + sample.shape[1:]
        array = np.zeros(shape, dtype=sample.dtype)
        for row, state in enumerate(states):
            value = state.values[key]
            array[row, :len(value)] = value
        result[key] = torch.as_tensor(array, device=device)
    padding = np.zeros((len(states), length), dtype=bool)
    for row, state in enumerate(states):
        padding[row, :len(state.values['token'])] = True
    result['padding'] = torch.as_tensor(padding, device=device)
    return result


@torch.inference_mode()
def rollout_batch(model, prefixes, end_ages, seeds, max_events, precision='bf16',
                  forbidden_med_tokens=(), stop_when_all_token_groups=(), minimum_ages=None,
                  forbidden_policy='medication_only', event_trace=None, event_limits=None,
                  rng_skip_counts=None):
    """Generate one trajectory per prefix with an independent NumPy RNG."""
    device = next(model.parameters()).device
    forbidden_meds = sorted({int(token) - MED_MIN for token in forbidden_med_tokens})
    if any(med < 0 or med >= N_MED for med in forbidden_meds):
        raise ValueError('forbidden_med_tokens contains an unsupported medication')
    if forbidden_policy not in ('medication_only', 'whole_event'):
        raise ValueError('Unknown forbidden_policy')
    if event_trace is not None and (len(event_trace) != len(prefixes) or any(event_trace)):
        raise ValueError('event_trace must contain one empty list per prefix')
    states = [GenerationState(prefix, model.config.recent_days, model.config.context_length)
              for prefix in prefixes]
    generated = [[] for _ in states]
    actions = [[] for _ in states]
    reasons = [None] * len(states)
    stop_groups = tuple(frozenset(map(int, group)) for group in stop_when_all_token_groups)
    group_hits = np.zeros((len(states), len(stop_groups)), dtype=bool)
    generators = [np.random.default_rng(int(seed)) for seed in seeds]
    if rng_skip_counts is not None:
        if len(rng_skip_counts) != len(generators):
            raise ValueError('rng_skip_counts must match prefixes')
        for generator, skip in zip(generators, rng_skip_counts):
            if int(skip) != skip or skip < 0:
                raise ValueError('rng_skip_counts must contain non-negative integers')
            generator.random(int(skip))
    if minimum_ages is not None and len(minimum_ages) != len(states):
        raise ValueError('minimum_ages must have one entry per prefix')
    limits = np.full(len(states), max_events, dtype=np.int64) if event_limits is None else np.asarray(event_limits)
    if limits.shape != (len(states),) or np.any(limits < 0) or np.any(limits > max_events) or np.any(limits != limits.astype('int64')):
        raise ValueError('event_limits must be integer budgets between zero and max_events')
    active = [i for i in range(len(states)) if limits[i] > 0]
    for i in range(len(states)):
        if limits[i] == 0:
            reasons[i] = 'max_events'
    for _ in range(max_events):
        if not active:
            break
        active_states = [states[i] for i in active]
        lengths = torch.tensor([len(state.values['token']) for state in active_states], device=device) - 1
        batch = collate_states(active_states, device)
        amp = precision == 'bf16' and device.type == 'cuda'
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            hidden, raw, _ = model.encode(batch)
            row = torch.arange(len(active), device=device)
            h, raw = hidden[row, lengths], raw[row, lengths]
            u_time = torch.tensor([generators[i].random() for i in active], device=device, dtype=torch.float32)
            if minimum_ages is not None:
                lower = torch.tensor([
                    max(0, int(minimum_ages[i]) - states[i].last_row_age)
                    if not generated[i] else 0 for i in active
                ], device=device, dtype=torch.float32)
                gap = model.time_distribution.conditional_quantile(raw, u_time, lower)
            else:
                gap = model.time_distribution.quantile(raw, u_time)
        age = np.asarray([float(states[i].last_row_age) for i in active]) + gap.cpu().numpy()
        if minimum_ages is not None:
            assert all(age[j] >= minimum_ages[i] for j, i in enumerate(active))
        within = np.asarray([age[j] <= end_ages[i] for j, i in enumerate(active)])
        next_active = []
        if within.any():
            local = np.flatnonzero(within)
            take = torch.as_tensor(local, device=device)
            source_row = row[take]
            source_pos = lengths[take]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                condition = model.condition(
                    h[take], gap[take], batch['age'][source_row, source_pos],
                    batch['before'][source_row, source_pos], batch['through'][source_row, source_pos])
            global_indices = [active[j] for j in local]
            gate_u = torch.tensor([generators[i].random() for i in global_indices], device=device)
            med_probability = condition['gate'].sigmoid()
            if forbidden_meds and forbidden_policy == 'whole_event':
                # Condition the complete next-EVENT distribution on not emitting
                # a forbidden medication. TIME remains the previously drawn TIME.
                allowed = condition['joint_logp'].float().exp().clone()
                allowed[:, :, forbidden_meds] = 0
                mass = allowed.sum((-2, -1))
                med_mass = med_probability.float() * mass
                med_probability = med_mass / (1 - med_probability.float() + med_mass).clamp_min(1e-12)
            is_med = gate_u < med_probability
            med_local = torch.nonzero(is_med, as_tuple=True)[0]
            nonmed_local = torch.nonzero(~is_med, as_tuple=True)[0]
            tokens = torch.empty(len(global_indices), dtype=torch.long, device=device)
            shifts = torch.zeros(len(global_indices), dtype=torch.long, device=device)
            totals = torch.zeros(len(global_indices), dtype=torch.long, device=device)
            sampled_actions = torch.full((len(global_indices),), -1, dtype=torch.long, device=device)
            if len(med_local):
                med_global = [global_indices[j] for j in med_local.cpu().tolist()]
                joint_u = torch.tensor([generators[i].random() for i in med_global], device=device)
                joint_logits = condition['joint_logp'][med_local].flatten(-2)
                if forbidden_meds:
                    joint_logits = joint_logits.clone()
                    blocked = [action * N_MED + med for action in range(3) for med in forbidden_meds]
                    joint_logits[:, blocked] = -torch.inf
                    if torch.isneginf(joint_logits).all(-1).any():
                        raise ValueError('Medication policy masked every joint medication/action outcome')
                joint = categorical(joint_logits, joint_u)
                action, med = joint // N_MED, joint % N_MED
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    dose, duration = model.attributes(
                        {'hidden': condition['hidden'][med_local]}, med, action, gap[take][med_local])
                dose_u = torch.tensor([generators[i].random() for i in med_global], device=device)
                shifts[med_local] = torch.where(dose_u < dose.sigmoid(), 1, 2)
                mix, mu, scale = duration
                component_u = torch.tensor([generators[i].random() for i in med_global], device=device)
                component = categorical(mix, component_u)[:, None]
                selected_mu = mu.gather(-1, component).squeeze(-1)
                selected_scale = scale.gather(-1, component).squeeze(-1)
                duration_u = torch.tensor([generators[i].random() for i in med_global], device=device).clamp(1e-7, 1 - 1e-7)
                totals[med_local] = (selected_mu + selected_scale * torch.logit(duration_u)).round().clamp(
                    0, model.config.duration_max).long()
                tokens[med_local] = 1278 + med
                sampled_actions[med_local] = action
            if len(nonmed_local):
                nonmed_global = [global_indices[j] for j in nonmed_local.cpu().tolist()]
                token_u = torch.tensor([generators[i].random() for i in nonmed_global], device=device)
                selected = categorical(
                    condition['event_logp'][nonmed_local][:, model.nonmed_tokens], token_u)
                tokens[nonmed_local] = model.nonmed_tokens[selected]
            tokens_np = tokens.cpu().numpy()
            shifts_np = shifts.cpu().numpy()
            totals_np = totals.cpu().numpy()
            actions_np = sampled_actions.cpu().numpy()
            if event_trace is not None:
                gaps_np = gap[take].float().cpu().numpy()
                insulin_probability = (condition['gate'].float().sigmoid() *
                                       condition['joint_logp'].float().exp()[:, :, 3].sum(-1)).cpu().numpy()
            for j, i in enumerate(global_indices):
                new_row = (states[i].patient_id, int(age[local[j]]), int(tokens_np[j]),
                           int(shifts_np[j]), int(totals_np[j]))
                generated[i].append(new_row)
                actions[i].append(int(actions_np[j]))
                if event_trace is not None:
                    event_trace[i].append({'gap': float(gaps_np[j]),
                                           'insulin_probability': float(insulin_probability[j])})
                states[i].append(new_row)
                for group_number, group in enumerate(stop_groups):
                    group_hits[i, group_number] |= int(tokens_np[j]) in group
                if int(tokens_np[j]) == DEATH:
                    reasons[i] = 'death'
                elif len(stop_groups) and group_hits[i].all():
                    # Once every requested occurrence outcome is known positive,
                    # simulating the rest of this trajectory cannot change them.
                    reasons[i] = 'targets_complete'
                elif len(generated[i]) >= limits[i]:
                    reasons[i] = 'max_events'
                else:
                    next_active.append(i)
        for j, i in enumerate(active):
            if not within[j]:
                reasons[i] = 'horizon'
        active = next_active
    for i in active:
        reasons[i] = 'max_events'
    return ([np.asarray(rows, dtype=DTYPE) for rows in generated], actions, reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--cohort-manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--patients', type=int, help='Optional random subset; omit for every eligible patient')
    parser.add_argument('--replicates', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--precision', choices=('fp32', 'bf16'), default='bf16')
    parser.add_argument('--horizon-days', type=int, default=1096)
    parser.add_argument('--minimum-history-days', type=int, default=0,
                        help='Minimum corrected-history span; zero includes every full-horizon/death-eligible test patient')
    parser.add_argument('--max-events', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20260914)
    parser.add_argument('--selection-seed', type=int,
                        help='Patient-subset seed; defaults to --seed for backward compatibility')
    parser.add_argument('--replicate-offset', type=int, default=0,
                        help='Global replicate index added to local indices, for resumable sharding')
    args = parser.parse_args()
    if min(args.replicates, args.batch_size, args.horizon_days, args.max_events) < 1:
        raise ValueError('Positive replicate, batch, horizon, and cap arguments are required')
    if args.replicate_offset < 0:
        raise ValueError('replicate-offset must be non-negative')
    model, _ = load_checkpoint(args.checkpoint, args.device)
    rows = np.memmap(args.cohort, dtype=DTYPE, mode='r')
    index = np.load(args.index)
    eligible = []
    for i, (start, boundary, end) in enumerate(zip(index['start'], index['boundary'], index['end'])):
        history = rows[int(start):int(boundary)]
        future = rows[int(boundary):int(end)]
        hix = np.flatnonzero(clinical(history['EVENT']))
        observed = future[clinical(future['EVENT'])]
        if not len(hix) or not len(observed):
            continue
        anchor = int(history['AGE'][hix[-1]])
        death = observed['AGE'][observed['EVENT'] == DEATH]
        complete = int(observed['AGE'][-1]) >= anchor + args.horizon_days
        terminated = len(death) and int(death[0]) <= anchor + args.horizon_days
        history_days = anchor - int(history['AGE'][hix[0]])
        if history_days >= args.minimum_history_days and (complete or terminated):
            eligible.append(i)
    chosen = np.asarray(eligible, dtype=np.int64)
    if args.patients is not None:
        if args.patients < 1 or args.patients > len(chosen):
            raise ValueError(f'patients must be in [1, {len(chosen)}]')
        selection_seed = args.seed if args.selection_seed is None else args.selection_seed
        chosen = np.random.default_rng(selection_seed).choice(chosen, args.patients, replace=False)
    jobs = [(int(i), replicate) for i in chosen
            for replicate in range(args.replicate_offset, args.replicate_offset + args.replicates)]
    patient_metrics, intervals = [], []
    generated_sequences, generated_actions = [], []
    trajectory_patients, trajectory_replicates, trajectory_seeds, reason_codes = [], [], [], []
    for start_job in range(0, len(jobs), args.batch_size):
        batch_jobs = jobs[start_job:start_job + args.batch_size]
        prefixes, observed_rows, end_ages, seeds = [], [], [], []
        for i, replicate in batch_jobs:
            start, boundary, end = (int(index[key][i]) for key in ('start', 'boundary', 'end'))
            prefix = np.asarray(rows[start:boundary]).copy()
            future = np.asarray(rows[boundary:end])
            anchor = int(index['anchor_age'][i])
            end_age = anchor + args.horizon_days
            prefixes.append(prefix)
            observed_rows.append(future[(future['AGE'] <= end_age) & clinical(future['EVENT'])])
            end_ages.append(end_age)
            seeds.append(int(np.random.SeedSequence(
                [args.seed, int(index['patient_id'][i]), replicate]).generate_state(1)[0]))
        generated, actions, reasons = rollout_batch(
            model, prefixes, end_ages, seeds, args.max_events, args.precision)
        for (i, replicate), prefix, observed, end_age, seed, sequence, action, reason in zip(
                batch_jobs, prefixes, observed_rows, end_ages, seeds, generated, actions, reasons):
            metrics, refill = compare_patient(prefix, observed, sequence, end_age)
            if reason == 'max_events':
                cutoff = float(sequence['AGE'][-1]) if len(sequence) else float(index['anchor_age'][i])
                for item in refill['generated']['censored_or_terminated']:
                    item['end_age'] = cutoff
                    item['lower_bound_days'] = max(0., cutoff - item['start_age'])
                    item['termination'] = 'generation_cap'
            patient_id = int(index['patient_id'][i])
            metrics.update(patient=patient_id, replicate=replicate, seed=seed,
                           stop_reason=reason, anchor_age=int(index['anchor_age'][i]),
                           end_age=end_age, followup_days=args.horizon_days)
            patient_metrics.append(metrics)
            refill.update(patient=patient_id, replicate=replicate)
            intervals.append(refill)
            generated_sequences.append(sequence)
            generated_actions.append(np.asarray(action, dtype=np.int8))
            trajectory_patients.append(patient_id)
            trajectory_replicates.append(replicate)
            trajectory_seeds.append(seed)
            reason_codes.append(REASON_CODE[reason])
        if len(patient_metrics) % 1024 < len(batch_jobs):
            print(f'completed={len(patient_metrics)}/{len(jobs)}', flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    lengths = np.asarray([len(sequence) for sequence in generated_sequences], dtype=np.int64)
    offsets = np.r_[0, np.cumsum(lengths)]
    trajectory_path = out / 'trajectories.npz'
    np.savez_compressed(
        trajectory_path,
        patient=np.asarray(trajectory_patients, dtype=np.int64),
        replicate=np.asarray(trajectory_replicates, dtype=np.int16),
        seed=np.asarray(trajectory_seeds, dtype=np.uint32),
        reason=np.asarray(reason_codes, dtype=np.int8), offsets=offsets,
        rows=np.concatenate(generated_sequences) if generated_sequences else np.empty(0, dtype=DTYPE),
        actions=np.concatenate(generated_actions) if generated_actions else np.empty(0, dtype=np.int8),
    )
    write_json(out / 'patient_metrics.json', patient_metrics)
    write_json(out / 'refill_intervals.json', intervals)
    summary = aggregate(patient_metrics, intervals)
    summary.update(unique_patients=len(chosen), replicates=args.replicates,
                   cap_rate=sum(row['stop_reason'] == 'max_events' for row in patient_metrics) / len(patient_metrics))
    write_json(out / 'metrics.json', summary)
    write_json(out / 'manifest.json', {
        'status': 'PASS', 'checkpoint_sha256': sha256(args.checkpoint),
        'cohort_sha256': sha256(args.cohort), 'index_sha256': sha256(args.index),
        'cohort_manifest': json.loads(Path(args.cohort_manifest).read_text()),
        'arguments': vars(args), 'eligible_patients': len(eligible),
        'selected_patients': len(chosen),
        'selected_patient_order_sha256': patient_order_hash(index['patient_id'][chosen]),
        'trajectories_sha256': sha256(trajectory_path),
        'boundary': 'Corrected train/val prefix; observed events are strictly-later kr_test rows.',
        'horizon': 'Every selected patient has complete 1096-day follow-up proxy or Death within horizon.',
        'sampling': ('Independent patient-level RNG. Patient selection uses selection_seed '
                     '(or seed when omitted); saved replicate values are global indices including replicate_offset.'),
        'calibration': 'No rollout Platt transformation; validation attribute calibration has a different estimand.',
    })
    print(f'Generated {len(patient_metrics)} trajectories for {len(chosen)} patients: {out}')


if __name__ == '__main__':
    main()
