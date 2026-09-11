#!/usr/bin/env python3
"""Generate from corrected history and compare only with strictly-future KR test records."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from evaluate_generation import aggregate, compare_patient  # noqa: E402
from generate import rollout  # noqa: E402
from token_roles import DTYPE, clinical, DEATH  # noqa: E402
from utils import load_checkpoint, sha256, write_json  # noqa: E402


def patient_order_hash(ids):
    return hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--cohort-manifest', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--patients', type=int, default=256)
    parser.add_argument('--replicates', type=int, default=3)
    parser.add_argument('--horizon-days', type=int, default=1096)
    parser.add_argument('--minimum-history-days', type=int, default=365)
    parser.add_argument('--max-events', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20260914)
    args = parser.parse_args()
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
        history_days = anchor - int(history['AGE'][hix[0]])
        complete = int(observed['AGE'][-1]) >= anchor + args.horizon_days
        terminated = len(death) and int(death[0]) <= anchor + args.horizon_days
        if history_days >= args.minimum_history_days and (complete or terminated):
            eligible.append(i)
    if len(eligible) < args.patients:
        raise ValueError(f'Only {len(eligible)} eligible full-horizon patients')
    rng = np.random.default_rng(args.seed)
    chosen = rng.choice(eligible, args.patients, replace=False)
    patient_metrics, intervals, records = [], [], []
    for i in chosen:
        start, boundary, end = (int(index[key][i]) for key in ('start', 'boundary', 'end'))
        prefix = np.asarray(rows[start:boundary]).copy()
        future = np.asarray(rows[boundary:end])
        anchor = int(index['anchor_age'][i])
        end_age = anchor + args.horizon_days
        observed = future[(future['AGE'] <= end_age) & clinical(future['EVENT'])]
        for replicate in range(args.replicates):
            seed = int(np.random.SeedSequence([args.seed, int(index['patient_id'][i]), replicate]).generate_state(1)[0])
            generated, actions, reason = rollout(model, prefix, end_age, args.max_events, seed)
            metrics, refill = compare_patient(prefix, observed, generated, end_age)
            if reason == 'max_events':
                cutoff = float(generated['AGE'][-1]) if len(generated) else float(anchor)
                for item in refill['generated']['censored_or_terminated']:
                    item['end_age'] = cutoff
                    item['lower_bound_days'] = max(0., cutoff - item['start_age'])
                    item['termination'] = 'generation_cap'
            metrics.update(patient=int(index['patient_id'][i]), replicate=replicate, seed=seed,
                           stop_reason=reason, anchor_age=anchor, end_age=end_age,
                           followup_days=args.horizon_days)
            patient_metrics.append(metrics)
            refill.update(patient=int(index['patient_id'][i]), replicate=replicate)
            intervals.append(refill)
            for source, sequence in (('observed', observed), ('generated', generated)):
                for row_number, row in enumerate(sequence):
                    records.append(dict(patient=int(row['ID']), replicate=replicate, source=source,
                        age=int(row['AGE']), token=int(row['EVENT']), shift=int(row['DOSE']),
                        duration=int(row['DUR']),
                        action=actions[row_number] if source == 'generated' else None))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    with (out / 'records.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('patient', 'replicate', 'source', 'age', 'token',
                                                    'shift', 'duration', 'action'))
        writer.writeheader(); writer.writerows(records)
    write_json(out / 'patient_metrics.json', patient_metrics)
    write_json(out / 'refill_intervals.json', intervals)
    summary = aggregate(patient_metrics, intervals)
    summary.update(unique_patients=len(chosen), replicates=args.replicates,
                   cap_rate=sum(row['stop_reason'] == 'max_events' for row in patient_metrics) / len(patient_metrics))
    write_json(out / 'metrics.json', summary)
    write_json(out / 'manifest.json', dict(
        status='PASS', checkpoint_sha256=sha256(args.checkpoint), cohort_sha256=sha256(args.cohort),
        index_sha256=sha256(args.index), cohort_manifest=json.loads(Path(args.cohort_manifest).read_text()),
        arguments=vars(args), eligible_patients=len(eligible),
        selected_patient_order_sha256=patient_order_hash(index['patient_id'][chosen]),
        boundary='Generation prefix is corrected train/val history; observed events come only from strictly-later kr_test.',
        horizon='Every selected patient has complete 1096-day follow-up proxy or Death within horizon.',
        calibration='No rollout Platt transformation; validation attribute calibration has a different conditioning estimand.',
    ))
    print(f'Generated {len(patient_metrics)} temporal-test trajectories: {out}')


if __name__ == '__main__':
    main()
