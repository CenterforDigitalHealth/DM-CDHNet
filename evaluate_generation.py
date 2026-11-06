"""Policy-filtered matched-follow-up counts and descriptive refill intervals.

Raw records and input histories are preserved. Refill means another recorded
prescription of the same class, not adherence or a censor-adjusted survival outcome.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from token_roles import DTYPE, medication, disease, clinical, MED_MIN, MED_MAX, DEATH
from utils import write_json, sha256, source_manifest


def summarize(values):
    return dict(n=len(values), mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,
                p90=float(np.quantile(values, .9)) if values else None)


def refill_intervals(prefix, future, end_age):
    completed, censored = [], []
    for token in range(MED_MIN, MED_MAX + 1):
        prior = prefix['AGE'][prefix['EVENT'] == token]
        days = np.unique(future['AGE'][future['EVENT'] == token]).astype(float)
        last = float(prior[-1]) if len(prior) else None
        for day in days:
            if last is not None and day > last:
                completed.append(dict(token=token, start_age=last, end_age=day, gap_days=day - last))
            last = day
        if last is not None and end_age >= last:
            censored.append(dict(token=token, start_age=last, end_age=float(end_age),
                                 lower_bound_days=float(end_age - last)))
    return completed, censored


def compare_patient(prefix, observed, generated, end_age, *, excluded_tokens, stop_reason=None):
    """Score copies only; cap affects terminal refill follow-up, never input rows."""
    excluded = tuple(sorted(excluded_tokens))
    raw = {'observed': observed, 'generated': generated}
    scored = {key: value[~np.isin(value['EVENT'], excluded)] for key, value in raw.items()}
    result = {'exclusions': {key: int(np.isin(value['EVENT'], excluded).sum())
                             for key, value in raw.items()}}
    result['denominators'] = {key + '_raw_records': len(value) for key, value in raw.items()}
    result['denominators'].update({key + '_clinical_records': int(clinical(value['EVENT']).sum())
                                   for key, value in scored.items()})
    for name, selector in [('medication', medication), ('disease', disease), ('clinical', clinical)]:
        a = scored['observed']['EVENT'][selector(scored['observed']['EVENT'])]
        b = scored['generated']['EVENT'][selector(scored['generated']['EVENT'])]
        ca, cb = Counter(a.tolist()), Counter(b.tolist())
        keys = set(ca) | set(cb)
        result[name] = dict(observed=len(a), generated=len(b), total_count_absolute_error=abs(len(a) - len(b)),
                            class_count_l1=sum(abs(ca[k] - cb[k]) for k in keys),
                            set_precision=len(set(a) & set(b)) / len(set(b)) if len(set(b)) else None,
                            set_recall=len(set(a) & set(b)) / len(set(a)) if len(set(a)) else None)
    intervals = {}
    for name, sequence in raw.items():
        death = sequence['AGE'][sequence['EVENT'] == DEATH]
        result[name + '_death'] = bool(len(death))
        endpoint = min(float(end_age), float(death[0])) if len(death) else float(end_age)
        termination = 'death' if len(death) and endpoint == death[0] else 'horizon_or_followup'
        if name == 'generated' and stop_reason == 'max_events' and termination != 'death':
            endpoint = min(endpoint, float(sequence['AGE'][-1]) if len(sequence) else float(prefix['AGE'][-1]))
            termination = 'generation_cap'
        # Histories are unmodified. Policy excludes only disease rows, so medication intervals are intact.
        complete, censored = refill_intervals(prefix, sequence[sequence['AGE'] <= endpoint], endpoint)
        for item in censored:
            item['termination'] = termination
        intervals[name] = dict(completed=complete, censored_or_terminated=censored)
    return result, intervals


def aggregate(patients, intervals):
    result = dict(patients=len(patients), trajectories=len(patients),
                  unit='patient-replicate trajectory; observed records repeat once per replicate')
    for group in ('medication', 'disease', 'clinical'):
        obs = sum(p[group]['observed'] for p in patients)
        gen = sum(p[group]['generated'] for p in patients)
        result[group] = dict(observed=obs, generated=gen, generated_observed_ratio=gen / obs if obs else None,
                            count_mae=float(np.mean([p[group]['total_count_absolute_error'] for p in patients])) if patients else None)
    result['exclusions'] = {source: sum(p['exclusions'][source] for p in patients)
                            for source in ('observed', 'generated')}
    result['denominators'] = {key: sum(p['denominators'][key] for p in patients)
                               for key in ('observed_raw_records', 'generated_raw_records',
                                           'observed_clinical_records', 'generated_clinical_records')}
    for source in ('observed', 'generated'):
        complete = [v['gap_days'] for r in intervals for v in r[source]['completed']]
        censored = [v['lower_bound_days'] for r in intervals for v in r[source]['censored_or_terminated']]
        result[source + '_refill'] = dict(completed=summarize(complete), censored_or_terminated=summarize(censored))
        result[source + '_deaths'] = sum(p[source + '_death'] for p in patients)
    result['refill_interpretation'] = 'Descriptive completed-gap summaries; censoring and generated event counts affect comparison. Not a censor-adjusted survival estimate.'
    return result


def policy_receipt(labels, policy=None):
    from evaluation_policy import load_excluded_tokens
    excluded = load_excluded_tokens(labels, policy)
    policy_path = Path(policy) if policy else Path(__file__).with_name('evaluation_policy.json')
    return excluded, dict(labels_path=str(Path(labels).resolve()), labels_sha256=sha256(labels),
                          policy_path=str(policy_path.resolve()), policy_sha256=sha256(policy_path),
                          excluded_tokens=sorted(excluded), input_history='preserved', raw_records='preserved')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--labels', required=True)
    parser.add_argument('--policy')
    args = parser.parse_args()
    out, root = Path(args.out), Path(args.run)
    if out.exists():
        raise FileExistsError(out)
    excluded, policy = policy_receipt(args.labels, args.policy)
    metadata = json.loads((root / 'patient_metrics.json').read_text())
    with np.load(root / 'prefixes.npz', allow_pickle=False) as archive:
        prefixes = {int(key): archive[key].copy() for key in archive.files}
    records = {}
    with (root / 'records.csv').open(newline='') as handle:
        for row in csv.DictReader(handle):
            key = (int(row['patient']), int(row['replicate']), row['source'])
            records.setdefault(key, []).append(tuple(int(row[k]) for k in ('patient', 'age', 'token', 'shift', 'duration')))
    patients, intervals = [], []
    for old in metadata:
        patient, replicate = old['patient'], old['replicate']
        sequences = {source: np.array(records.get((patient, replicate, source), []), dtype=DTYPE)
                     for source in ('observed', 'generated')}
        metric, refill = compare_patient(prefixes[patient], sequences['observed'], sequences['generated'],
                                         old['end_age'], excluded_tokens=excluded, stop_reason=old['stop_reason'])
        metric.update({key: old[key] for key in ('patient', 'replicate', 'seed', 'stop_reason', 'anchor_age', 'end_age', 'followup_days')})
        refill.update(patient=patient, replicate=replicate)
        patients.append(metric)
        intervals.append(refill)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'patient_metrics.json', patients)
    write_json(out / 'refill_intervals.json', intervals)
    summary = aggregate(patients, intervals)
    summary.update(unique_patients=len(prefixes), cap_rate=sum(p['stop_reason'] == 'max_events' for p in patients) / len(patients) if patients else None)
    write_json(out / 'metrics.json', summary)
    write_json(out / 'manifest.json', dict(status='PASS', arguments=vars(args), policy=policy,
        source_code_hashes=source_manifest(), source_hashes={name: sha256(root / name) for name in ('records.csv', 'prefixes.npz', 'patient_metrics.json', 'manifest.json')},
        source_manifest=json.loads((root / 'manifest.json').read_text())))


if __name__ == '__main__':
    main()
