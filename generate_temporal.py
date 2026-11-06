"""Generate from validated histories and compare with strictly future full-horizon records."""
import argparse

import numpy as np
from generate import add_arguments, validate_arguments, run_generation
from evaluate_generation import policy_receipt
from prepare_temporal import validate_temporal
from token_roles import clinical, DEATH
from utils import sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser, temporal=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--cohort-manifest', required=True)
    args = parser.parse_args()
    validate_arguments(args)
    excluded, policy = policy_receipt(args.labels, args.policy)
    rows, index, manifest = validate_temporal(args.cohort, args.index, args.cohort_manifest)
    eligible = []
    for i, (start, boundary, end) in enumerate(zip(index['start'], index['boundary'], index['end'])):
        history, future = rows[int(start):int(boundary)], rows[int(boundary):int(end)]
        history_clinical = history[clinical(history['EVENT'])]
        observed = future[clinical(future['EVENT'])]
        if not len(observed) or np.any(history['EVENT'] == DEATH):
            continue
        anchor = int(index['anchor_age'][i])
        death = observed['AGE'][observed['EVENT'] == DEATH]
        history_days = anchor - int(history_clinical['AGE'][0])
        complete = int(observed['AGE'][-1]) >= anchor + args.horizon_days
        terminated = len(death) and int(death[0]) <= anchor + args.horizon_days
        if history_days >= args.minimum_history_days and (complete or terminated):
            eligible.append(i)
    if len(eligible) < args.patients:
        raise ValueError(f'Only {len(eligible)} eligible full-horizon patients; requested {args.patients}')
    chosen = np.random.default_rng(args.seed).choice(eligible, args.patients, replace=False)

    def trajectories():
        for i in chosen:
            start, boundary, end = (int(index[key][i]) for key in ('start', 'boundary', 'end'))
            prefix = rows[start:boundary].copy()
            end_age = int(index['anchor_age'][i]) + args.horizon_days
            future = rows[boundary:end]
            observed = future[future['AGE'] <= end_age].copy()
            yield int(index['patient_id'][i]), prefix, observed, end_age

    from inference import load_model
    model, _, receipt = load_model(args.checkpoint, args.device, correction=args.correction)
    run_generation(model, trajectories(), args, receipt,
                   dict(cohort_sha256=sha256(args.cohort), index_sha256=sha256(args.index),
                        cohort_manifest_sha256=sha256(args.cohort_manifest), cohort_manifest=manifest,
                        eligible_patients=len(eligible),
                        comparison='Complete requested-horizon last-record follow-up proxy or death within horizon. Future records are strictly after history boundary; generated same-day additions remain scored.'),
                   excluded, policy)


if __name__ == '__main__':
    main()
