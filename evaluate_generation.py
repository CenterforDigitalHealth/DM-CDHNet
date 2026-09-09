"""Matched-horizon counts and completed/censored refill intervals.

Refill means another recorded prescription of the same class, not adherence.
Same-day duplicates count as records but are collapsed for refill-day intervals.
"""
import argparse
import csv
from collections import Counter
from pathlib import Path
import numpy as np
from token_roles import medication, disease, MED_MIN, MED_MAX, DEATH
from utils import write_json


def summarize(values):
    return dict(n=len(values), mean=float(np.mean(values)) if len(values) else None,
                median=float(np.median(values)) if len(values) else None,
                p90=float(np.quantile(values, .9)) if len(values) else None)


def refill_intervals(prefix, future, end_age):
    completed, censored = [], []
    for token in range(MED_MIN, MED_MAX + 1):
        prior = prefix['AGE'][prefix['DATA'] == token]
        days = np.unique(future['AGE'][future['DATA'] == token]).astype(float)
        last = float(prior[-1]) if len(prior) else None
        for day in days:
            if last is not None and day > last:
                completed.append(dict(token=token, start_age=last, end_age=day, gap_days=day - last))
            last = day
        if last is not None and end_age >= last:
            censored.append(dict(token=token, start_age=last, end_age=float(end_age), lower_bound_days=float(end_age - last)))
    return completed, censored


def compare_patient(prefix, observed, generated, end_age):
    result = {}
    for name, selector in [('medication', medication), ('disease', disease)]:
        a, b = observed['DATA'][selector(observed['DATA'])], generated['DATA'][selector(generated['DATA'])]
        ca, cb = Counter(a.tolist()), Counter(b.tolist())
        keys = set(ca) | set(cb)
        result[name] = dict(observed=len(a), generated=len(b), total_count_absolute_error=abs(len(a) - len(b)),
                            class_count_l1=sum(abs(ca[k] - cb[k]) for k in keys),
                            set_precision=len(set(a) & set(b)) / len(set(b)) if len(set(b)) else None,
                            set_recall=len(set(a) & set(b)) / len(set(a)) if len(set(a)) else None)
    result['observed_death'] = bool(np.any(observed['DATA'] == DEATH))
    result['generated_death'] = bool(np.any(generated['DATA'] == DEATH))
    intervals = {}
    for name, rows in [('observed', observed), ('generated', generated)]:
        death = rows['AGE'][rows['DATA'] == DEATH]
        endpoint = min(float(end_age), float(death[0])) if len(death) else float(end_age)
        complete, censored = refill_intervals(prefix, rows, endpoint)
        # Death terminates exposure; these are competing termination, not ordinary censoring.
        for item in censored:
            item['termination'] = 'death' if len(death) and endpoint == death[0] else 'horizon_or_followup'
        intervals[name] = dict(completed=complete, censored_or_terminated=censored)
    return result, intervals


def aggregate(patients, intervals):
    result = dict(patients=len(patients))
    for group in ('medication', 'disease'):
        obs = sum(p[group]['observed'] for p in patients)
        gen = sum(p[group]['generated'] for p in patients)
        result[group] = dict(observed=obs, generated=gen, generated_observed_ratio=gen / obs if obs else None,
            count_mae=float(np.mean([p[group]['total_count_absolute_error'] for p in patients])) if patients else None)
    for source in ('observed', 'generated'):
        complete = [v['gap_days'] for r in intervals for v in r[source]['completed']]
        censored = [v['lower_bound_days'] for r in intervals for v in r[source]['censored_or_terminated']]
        result[source + '_refill'] = dict(completed=summarize(complete), censored_or_terminated=summarize(censored))
        result[source + '_deaths'] = sum(p[source + '_death'] for p in patients)
    result['refill_interpretation'] = 'Descriptive completed-gap summaries; censoring and generated event counts affect their comparison. Not a censor-adjusted survival estimate.'
    return result


def main():
    import json
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    args = p.parse_args()
    root = Path(args.run)
    patients = json.loads((root / 'patient_metrics.json').read_text())
    intervals = json.loads((root / 'refill_intervals.json').read_text())
    write_json(root / 'aggregate_recomputed.json', aggregate(patients, intervals))


if __name__ == '__main__':
    main()
