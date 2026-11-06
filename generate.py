"""Native or explicitly corrected JMDC trajectories with matched-follow-up metrics."""
import argparse
import csv
from pathlib import Path

import numpy as np
from data import Cohort, landmark
from token_roles import clinical, DEATH
from utils import write_json, sha256, source_manifest
from evaluate_generation import compare_patient, aggregate, policy_receipt
from simulation.rollout import rollout_batch


def rollout(model, prefix, end_age, max_events=1024, seed=1):
    """Frozen native sampler uses the active ten-medication joint vocabulary."""
    if len(prefix) == 0 or not np.any(clinical(prefix['EVENT'])):
        raise ValueError('Generation requires a nonempty clinical history')
    if np.any(prefix['AGE'][1:] < prefix['AGE'][:-1]):
        raise ValueError('Generation history must be chronological')
    if int(prefix['AGE'][-1]) > end_age or max_events < 0:
        raise ValueError('Invalid generation horizon or event cap')
    if np.any(prefix['EVENT'] == DEATH):
        from token_roles import DTYPE
        return np.empty(0, dtype=DTYPE), [], 'death'
    generated, actions, reasons = rollout_batch(model, [prefix], [end_age], [seed], max_events, precision='fp32')
    return generated[0], actions[0], reasons[0]


def add_arguments(parser, temporal=False):
    parser.add_argument('--checkpoint', required=True)
    if not temporal:
        parser.add_argument('--data', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--policy')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--correction', choices=('native', 'final'), default='native')
    parser.add_argument('--patients', type=int, default=128)
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--horizon-days', type=int, default=1096 if temporal else 1826)
    parser.add_argument('--minimum-history-days', type=int, default=365)
    parser.add_argument('--max-events', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=1)


def validate_arguments(args):
    if min(args.patients, args.replicates, args.horizon_days, args.max_events) < 1:
        raise ValueError('Patient, replicate, horizon and cap arguments must be positive')
    if args.minimum_history_days < 0 or args.seed < 0:
        raise ValueError('Minimum history and seed must be nonnegative')
    if Path(args.out).exists():
        raise FileExistsError(args.out)


def run_generation(model, trajectories, args, receipt, provenance, excluded, policy):
    """Each trajectory is (patient_id, full_prefix, raw_observed, end_age)."""
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'manifest.json', dict(status='RUNNING', arguments=vars(args), **provenance))
    patients, intervals, prefixes = [], [], {}
    with (out / 'records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('patient', 'replicate', 'source', 'age', 'token', 'shift', 'duration', 'action'))
        writer.writeheader()
        for patient, prefix, observed, end_age in trajectories:
            prefixes[str(patient)] = prefix.copy()
            anchor = int(prefix['AGE'][-1])
            for rep in range(args.replicates):
                seed = int(np.random.SeedSequence([args.seed, patient, rep]).generate_state(1)[0])
                generated, actions, reason = rollout(model, prefix, end_age, args.max_events, seed)
                metric, refill = compare_patient(prefix, observed, generated, end_age,
                                                  excluded_tokens=excluded, stop_reason=reason)
                metric.update(patient=patient, replicate=rep, seed=seed, stop_reason=reason,
                              anchor_age=anchor, end_age=end_age, followup_days=end_age - anchor)
                patients.append(metric)
                refill.update(patient=patient, replicate=rep)
                intervals.append(refill)
                for source, sequence in (('observed', observed), ('generated', generated)):
                    for i, row in enumerate(sequence):
                        writer.writerow(dict(patient=int(row['ID']), replicate=rep, source=source,
                            age=int(row['AGE']), token=int(row['EVENT']), shift=int(row['DOSE']),
                            duration=int(row['DUR']), action=actions[i] if source == 'generated' else None))
    np.savez(out / 'prefixes.npz', **prefixes)
    write_json(out / 'patient_metrics.json', patients)
    write_json(out / 'refill_intervals.json', intervals)
    summary = aggregate(patients, intervals)
    summary.update(unique_patients=len(prefixes), replicates=args.replicates,
                   cap_rate=sum(p['stop_reason'] == 'max_events' for p in patients) / len(patients))
    write_json(out / 'metrics.json', summary)
    write_json(out / 'manifest.json', dict(status='PASS', arguments=vars(args), policy=policy,
        checkpoint_sha256=sha256(args.checkpoint), inference=receipt, source_hashes=source_manifest(), **provenance,
        selected_patient_ids=[int(patient) for patient in prefixes],
        artifacts={name: sha256(out / name) for name in ('records.csv', 'prefixes.npz', 'patient_metrics.json', 'refill_intervals.json', 'metrics.json')},
        calibration='No rollout Platt transformation; attribute calibration has a different conditioning estimand.',
        censoring='Cap-limited trajectories retained and flagged; terminal refill intervals end at the last generated record.'))
    print(f'Generated {len(patients)} trajectories: {out}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    args = parser.parse_args()
    validate_arguments(args)
    excluded, policy = policy_receipt(args.labels, args.policy)
    cohort = Cohort(args.data)
    eligible = []
    for i in range(len(cohort)):
        rows = cohort.patient(i)
        end = landmark(rows, args.minimum_history_days)
        if end is not None and int(rows['AGE'][-1]) > int(rows['AGE'][end]):
            eligible.append((i, end))
    if not eligible:
        raise ValueError('No landmark with later observed follow-up')
    chosen = np.random.default_rng(args.seed).choice(len(eligible), min(args.patients, len(eligible)), replace=False)

    def trajectories():
        for j in chosen:
            i, end = eligible[j]
            rows = cohort.patient(i)
            prefix = rows[:end + 1].copy()
            anchor = int(prefix['AGE'][-1])
            end_age = min(anchor + args.horizon_days, int(rows['AGE'][-1]))
            observed = rows[(rows['AGE'] > anchor) & (rows['AGE'] <= end_age)].copy()
            yield int(cohort.ids[i]), prefix, observed, end_age

    from inference import load_model
    model, _, receipt = load_model(args.checkpoint, args.device, correction=args.correction)
    run_generation(model, trajectories(), args, receipt,
                   dict(data_sha256=sha256(args.data), eligible_patients=len(eligible),
                        comparison='Last-record follow-up proxy matched; actual horizon can be shorter than requested. Same-day generated additions remain scored.'),
                   excluded, policy)


if __name__ == '__main__':
    main()
