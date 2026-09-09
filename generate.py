"""Sample time first, then marks and attributes; matched follow-up comparisons."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from data import Cohort, example, collate, landmark
from token_roles import DTYPE, medication, clinical, DEATH
from utils import load_checkpoint, seed_everything, write_json, sha256
from evaluate_generation import compare_patient, aggregate


@torch.inference_mode()
def rollout(model, prefix, end_age, max_events=1024, seed=1):
    seed_everything(seed)
    history = prefix.copy()
    generated, actions = [], []
    reason = 'max_events'
    for _ in range(max_events):
        batch = collate([example(history, model.config, end=len(history) - 1, supervised=False)], next(model.parameters()).device)
        hidden, raw, _ = model.encode(batch)
        gap = model.time_distribution.sample(raw[:, -1])
        age = float(history['AGE'][-1]) + float(gap.item())
        if not np.isfinite(age):
            raise FloatingPointError('TIME sampler produced a non-finite age')
        if age > end_age:
            reason = 'horizon'
            break
        c = model.condition(hidden[:, -1], gap, batch['age'][:, -1], batch['before'][:, -1], batch['through'][:, -1])
        is_med = bool(torch.bernoulli(c['gate'].sigmoid()).item())
        if is_med:
            joint = int(torch.distributions.Categorical(logits=c['joint_logp'].flatten(-2)).sample().item())
            action, med = joint // 7, joint % 7
            dose, dur = model.attributes(c, torch.tensor([med], device=gap.device), torch.tensor([action], device=gap.device), gap)
            shift = 1 if torch.bernoulli(dose.sigmoid()).item() else 2
            total = int(model.duration_head.sample(dur).item())
            token = 1278 + med
        else:
            logits = c['event_logp'][:, model.nonmed_tokens]
            token = int(model.nonmed_tokens[torch.distributions.Categorical(logits=logits).sample()].item())
            action, shift, total = -1, 0, 0
        row = (int(history['ID'][0]), int(age), token, shift, total)
        generated.append(row)
        actions.append(action)
        history = np.concatenate((history, np.array([row], dtype=DTYPE)))
        if token == DEATH:
            reason = 'death'
            break
    return np.array(generated, dtype=DTYPE), actions, reason


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--patients', type=int, default=128)
    p.add_argument('--replicates', type=int, default=3)
    p.add_argument('--horizon-days', type=int, default=1826)
    p.add_argument('--minimum-history-days', type=int, default=365)
    p.add_argument('--max-events', type=int, default=1024)
    p.add_argument('--seed', type=int, default=20260909)
    args = p.parse_args()
    if min(args.patients, args.replicates, args.horizon_days, args.max_events) < 1:
        raise ValueError('Patient/replicate/horizon/cap arguments must be positive')
    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    cohort = Cohort(args.data)
    eligible = []
    for i in range(len(cohort)):
        r = cohort.patient(i)
        end = landmark(r, args.minimum_history_days)
        if end is not None and int(r['AGE'][-1]) > int(r['AGE'][end]):
            eligible.append((i, end))
    if not eligible:
        raise ValueError('No landmark with later observed follow-up')
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(len(eligible), min(args.patients, len(eligible)), replace=False)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    patient_metrics, intervals, records = [], [], []
    for j in chosen:
        i, end = eligible[j]
        rows = cohort.patient(i)
        prefix = rows[:end + 1].copy()
        anchor = int(prefix['AGE'][-1])
        end_age = min(anchor + args.horizon_days, int(rows['AGE'][-1]))
        observed = rows[(rows['AGE'] > anchor) & (rows['AGE'] <= end_age) & clinical(rows['DATA'])]
        for rep in range(args.replicates):
            seed = int(np.random.SeedSequence([args.seed, int(cohort.ids[i]), rep]).generate_state(1)[0])
            generated, actions, reason = rollout(model, prefix, end_age, args.max_events, seed)
            # Generated same-day additions remain scored: the observed landmark consumed that day's records.
            metrics, refill = compare_patient(prefix, observed, generated, end_age)
            if reason == 'max_events':
                # The ungenerated future cannot be treated as observed event-free time.
                cutoff = float(generated['AGE'][-1]) if len(generated) else float(anchor)
                for item in refill['generated']['censored_or_terminated']:
                    item['end_age'] = cutoff
                    item['lower_bound_days'] = max(0., cutoff - item['start_age'])
                    item['termination'] = 'generation_cap'
            metrics.update(patient=int(cohort.ids[i]), replicate=rep, seed=seed, stop_reason=reason,
                           anchor_age=anchor, end_age=end_age, followup_days=end_age - anchor)
            patient_metrics.append(metrics)
            refill.update(patient=int(cohort.ids[i]), replicate=rep)
            intervals.append(refill)
            for source, sequence in [('observed', observed), ('generated', generated)]:
                for ri, row in enumerate(sequence):
                    records.append(dict(patient=int(row['ID']), replicate=rep, source=source, age=int(row['AGE']),
                        token=int(row['DATA']), shift=int(row['SHIFT']), duration=int(row['TOTAL']),
                        action=actions[ri] if source == 'generated' else None))
    with (out / 'records.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=['patient', 'replicate', 'source', 'age', 'token', 'shift', 'duration', 'action'])
        writer.writeheader()
        writer.writerows(records)
    write_json(out / 'patient_metrics.json', patient_metrics)
    write_json(out / 'refill_intervals.json', intervals)
    summary = aggregate(patient_metrics, intervals)
    summary['unique_patients'] = len(chosen)
    summary['replicates'] = args.replicates
    summary['cap_rate'] = sum(p['stop_reason'] == 'max_events' for p in patient_metrics) / len(patient_metrics)
    write_json(out / 'metrics.json', summary)
    write_json(out / 'manifest.json', dict(checkpoint_sha256=sha256(args.checkpoint), data_sha256=sha256(args.data),
        arguments=vars(args), eligible_patients=len(eligible), comparison='Observed follow-up proxy matched; actual horizon can be shorter than requested.',
        calibration='No rollout Platt transformation: evaluation calibration has a different conditioning estimand.',
        censoring='Cap-limited runs retained and flagged; terminal refill intervals in those runs are incomplete, not reliable censoring times.'))
    print(f'Generated {len(patient_metrics)} trajectories: {out}')


if __name__ == '__main__':
    main()
