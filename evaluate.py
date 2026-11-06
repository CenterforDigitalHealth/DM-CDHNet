"""Policy-filtered next-EVENT and separate recorded-occurrence window evaluation.

Uses frozen prefix/landmark sampling and TIME-quantile marginalization. Raw
probabilities retain the full native distribution; excluded targets are never
replaced by a later event. Window labels retain the frozen follow-up proxy.

The default 100,000 raw-transition limit bounds retained EVENT arrays to about
1 GB at concatenation (excluding model/batch memory). Split larger prefix
cohorts or reduce --max-patients; raising the limit requires sufficient RAM.
"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch
from data import Cohort, example, collate, landmark
from evaluate_auc import predict, binary_metrics, dose_metrics, duration_metrics, apply_calibration, load_calibration
from evaluation_policy import DEFAULT_POLICY, load_excluded_tokens, load_label_dictionary, filter_target_rows
from token_roles import medication, supported_tokens
from utils import sha256, write_json


def summarize_predictions(arrays, excluded_tokens, candidates=None):
    candidates = supported_tokens() if candidates is None else list(candidates)
    mask = filter_target_rows(arrays['target'], excluded_tokens)
    target = arrays['target'][mask]
    metrics = [dict(token=int(token), **binary_metrics(target == token, arrays['event'][mask, token]))
               for token in candidates if token not in excluded_tokens]
    meds = mask & medication(arrays['target'])
    dose = dose_metrics(arrays['dose_y'][meds], arrays['dose'][meds])
    duration = duration_metrics(arrays['duration_y'][meds], arrays['duration'][meds])
    aucs = [row['auc'] for row in metrics if row['auc'] is not None]
    summary = dict(patients=int(len(np.unique(arrays['patient'][mask]))),
        transitions=int(mask.sum()), raw_target_observations=len(mask),
        excluded_target_observations=int((~mask).sum()),
        candidate_endpoint_rows=len(candidates), retained_endpoint_rows=len(metrics),
        excluded_endpoint_rows=len(candidates) - len(metrics),
        event_macro_auc=float(np.mean(aucs)) if aucs else None,
        evaluable_event_codes=len(aucs), dose=dose, duration=duration,
        duration_mae=duration['mae'],
        time_mae_days=float(np.abs(arrays['time'][mask] - arrays['time_y'][mask]).mean()) if mask.any() else None,
        attribute_estimand='Observed drug class; predicted time/action marginalized. Not joint generation accuracy.')
    return metrics, summary, mask


def summarize_windows(y, valid, probability, horizons, tokens, excluded_tokens):
    keep = filter_target_rows(tokens, excluded_tokens)
    mask = np.asarray(valid, dtype=bool) & keep[None, None, :]
    metrics = []
    for hi, horizon in enumerate(horizons):
        for ti, token in enumerate(tokens):
            if keep[ti]:
                rows = mask[:, hi, ti]
                metrics.append(dict(horizon_days=horizon, token=int(token),
                                    **binary_metrics(y[rows, hi, ti], probability[rows, hi, ti])))
    return metrics, dict(candidate_window_endpoint_rows=len(horizons) * len(tokens),
        excluded_window_endpoint_rows=int((~keep).sum()) * len(horizons),
        retained_window_endpoint_rows=len(metrics),
        raw_window_observations=int(np.asarray(valid).sum()),
        excluded_window_observations=int(np.asarray(valid).sum() - mask.sum()),
        retained_window_observations=int(mask.sum())), mask


def predict_batch(model, examples, device, quantiles, raw_transitions, max_transitions):
    # Count original valid targets, including excluded targets retained in raw output.
    total = raw_transitions + sum(int(e['valid'].sum()) for e in examples)
    if total > max_transitions:
        raise ValueError(f'Raw transition count {total:,} exceeds --max-transitions {max_transitions:,}. '
                         'Split the cohort or reduce --max-patients; no targets were silently truncated.')
    batch = collate(examples, device)
    return batch, predict(model, batch, quantiles), total


def validate_calibration_context(calibration, args, correction_receipt):
    checks = dict(quantiles=args.quantiles, evaluation_mode=args.mode,
                  minimum_history_days=args.minimum_history_days,
                  labels_sha256=sha256(args.labels), policy_sha256=sha256(args.policy),
                  correction=correction_receipt)
    for key, value in checks.items():
        if calibration.get(key) != value:
            raise ValueError(f'Calibration differs from evaluation context: {key}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'data', 'out', 'labels'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--policy', default=str(DEFAULT_POLICY))
    p.add_argument('--device', default='cpu')
    p.add_argument('--correction', choices=['native', 'final'], default='native')
    p.add_argument('--quantiles', type=int, default=16)
    p.add_argument('--mode', choices=['landmark', 'prefix'], default='landmark')
    p.add_argument('--minimum-history-days', type=int, default=365)
    p.add_argument('--max-patients', type=int)
    p.add_argument('--max-transitions', type=int, default=100000,
                   help='Fail before exceeding this raw-target count (default: 100000); split larger cohorts')
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--calibration')
    args = p.parse_args()
    if args.quantiles < 1 or args.batch_size < 1 or args.minimum_history_days < 0:
        p.error('quantiles/batch-size must be positive and minimum-history-days nonnegative')
    if args.max_patients is not None and args.max_patients < 1:
        p.error('max-patients must be positive')
    if args.max_transitions < 1:
        p.error('max-transitions must be positive')
    out = Path(args.out)
    if out.exists():
        raise FileExistsError(f'Choose a new output directory: {out}')
    excluded = load_excluded_tokens(args.labels, args.policy)
    labels = load_label_dictionary(args.labels)
    missing = set(supported_tokens()) - labels.keys()
    if missing:
        raise ValueError(f'Label dictionary is missing {len(missing)} supported EVENT token IDs')
    from inference import load_model
    model, ckpt, correction = load_model(args.checkpoint, args.device, args.correction)
    calibration = load_calibration(args.calibration, args.checkpoint) if args.calibration else None
    if calibration:
        validate_calibration_context(calibration, args, correction)
    cohort = Cohort(args.data)
    collected = {k: [] for k in ['patient', 'age', 'target', 'event', 'dose', 'dose_y', 'duration', 'duration_y', 'time', 'time_y']}
    windows, pending, pending_ids = [], [], []
    included = input_rows = perinatal_input_rows = raw_transitions = 0

    def evaluate_pending():
        nonlocal raw_transitions
        if not pending:
            return
        b, pred, raw_transitions = predict_batch(
            model, pending, args.device, args.quantiles, raw_transitions, args.max_transitions)
        mask = b['valid']
        patient = np.concatenate([np.full(int(mask[j].sum()), patient_id, dtype=np.int64)
                                  for j, patient_id in enumerate(pending_ids)])
        values = dict(patient=patient, age=b['age'][mask], target=b['target'][mask],
                      event=pred['event'], dose=pred['dose'], dose_y=b['dose'][mask],
                      duration=pred['duration'], duration_y=b['dur'][mask],
                      time=pred['time'], time_y=b['gap'][mask])
        for key, value in values.items():
            collected[key].append(value.cpu().numpy() if torch.is_tensor(value) else value)
        if pred['window'] is not None:
            windows.extend((e['window_y'], e['window_valid'], pred['window'][j].cpu().numpy())
                           for j, e in enumerate(pending))
        pending.clear()
        pending_ids.clear()

    for i in range(len(cohort)):
        if args.max_patients is not None and included >= args.max_patients:
            break
        rows = cohort.patient(i)
        end = landmark(rows, args.minimum_history_days) if args.mode == 'landmark' else None
        if (args.mode == 'landmark' and end is None) or len(rows) < 2:
            continue
        e = example(rows, model.config, end=end)
        if args.mode == 'landmark':
            keep = bool(e['valid'][-1])
            e['valid'][:] = False
            e['valid'][-1] = keep
        input_rows += len(e['token'])
        perinatal_input_rows += int((~filter_target_rows(e['token'], excluded)).sum())
        pending.append(e)
        pending_ids.append(int(cohort.ids[i]))
        included += 1
        if len(pending) == args.batch_size:
            evaluate_pending()
    evaluate_pending()
    if not included:
        raise ValueError('No evaluable patients; inspect landmark/follow-up eligibility')
    arrays = {key: np.concatenate(value) for key, value in collected.items()}
    metrics, summary, mask = summarize_predictions(arrays, excluded)
    arrays['evaluation_mask'] = mask
    summary.update(sampled_patients=included, input_rows=input_rows,
                   perinatal_input_rows_preserved=perinatal_input_rows, excluded_input_rows=0)
    if calibration:
        meds = mask & medication(arrays['target'])
        logits = torch.logit(torch.tensor(arrays['dose'][meds]).clamp(1e-7, 1 - 1e-7))
        summary['dose']['calibrated'] = dose_metrics(arrays['dose_y'][meds], apply_calibration(logits, calibration).numpy())
    window_metrics = []
    window_counts = dict(candidate_window_endpoint_rows=0, excluded_window_endpoint_rows=0,
        retained_window_endpoint_rows=0, raw_window_observations=0,
        excluded_window_observations=0, retained_window_observations=0)
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'predictions.npz', **arrays)
    if windows:
        wy, wv, wp = (np.stack([w[j] for w in windows]) for j in range(3))
        window_metrics, window_counts, wm = summarize_windows(wy, wv, wp, model.config.horizons_days, model.config.window_tokens, excluded)
        np.savez_compressed(out / 'window_predictions.npz', target=wy, valid=wv, probability=wp,
                            evaluation_mask=wm, tokens=model.config.window_tokens, horizons_days=model.config.horizons_days)
    summary.update(window_counts)
    with (out / 'event_auc.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['token', 'n', 'positives', 'auc', 'auprc', 'brier'])
        writer.writeheader()
        writer.writerows(metrics)
    write_json(out / 'window_metrics.json', window_metrics)
    write_json(out / 'metrics.json', summary)
    write_json(out / 'manifest.json', dict(checkpoint_sha256=sha256(args.checkpoint), data_sha256=sha256(args.data),
        checkpoint_path=str(Path(args.checkpoint).resolve()), data_path=str(Path(args.data).resolve()),
        predictions_sha256=sha256(out / 'predictions.npz'), checkpoint_data_hashes=ckpt['data_hashes'],
        labels_sha256=sha256(args.labels), policy_sha256=sha256(args.policy),
        labels_path=str(Path(args.labels).resolve()), policy_path=str(Path(args.policy).resolve()),
        policy=json.loads(Path(args.policy).read_text()), excluded_tokens=sorted(excluded),
        mode=args.mode, minimum_history_days=args.minimum_history_days, quantiles=args.quantiles,
        max_patients=args.max_patients, max_transitions=args.max_transitions,
        batch_size=args.batch_size, correction=correction,
        calibration_sha256=sha256(args.calibration) if args.calibration else None,
        prediction_layout='All original target rows and full EVENT probabilities; use evaluation_mask for reporting',
        sources={name: sha256(Path(__file__).with_name(name)) for name in
                 ('evaluate.py', 'evaluation_policy.py', 'evaluate_auc.py', 'data.py', 'inference.py')},
        estimand='Next clinical EVENT with deterministic TIME quantile marginalization; separate recorded-occurrence window head with last-record follow-up proxy'))
    print(f'Evaluated {summary["transitions"]} retained transitions from {included} sampled patients: {out}')


if __name__ == '__main__':
    main()
