"""Next clinical EVENT AUC and observed-medication attribute evaluation.

TIME is marginalized by deterministic quantiles; no true future time is input.
Separate window head assesses recorded occurrence within each horizon.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, r2_score
from data import Cohort, example, collate, landmark
from token_roles import medication, supported_tokens
from utils import load_checkpoint, sha256, write_json


def binary_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return dict(n=len(y), positives=int(y.sum()), auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
                auprc=float(average_precision_score(y, p)) if y.sum() else None,
                brier=float(np.mean((y - p) ** 2)) if len(y) else None)


def dose_metrics(y, p):
    result = binary_metrics(y, p)
    result['balanced_accuracy'] = (float(balanced_accuracy_score(y, np.asarray(p) >= .5))
                                   if len(np.unique(y)) == 2 else None)
    return result


def duration_metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    error = prediction - y
    return dict(
        n=len(y),
        r2=float(r2_score(y, prediction)) if len(y) >= 2 and np.var(y) > 0 else None,
        rmse=float(np.sqrt(np.mean(error ** 2))) if len(y) else None,
        mae=float(np.mean(np.abs(error))) if len(y) else None,
    )


def load_calibration(path, checkpoint):
    import json
    calibration = json.loads(Path(path).read_text())
    if calibration['checkpoint_sha256'] != sha256(checkpoint):
        raise ValueError('Calibration belongs to a different checkpoint')
    return calibration


def apply_calibration(logits, calibration):
    return torch.sigmoid(logits * calibration['slope'] + calibration['intercept'])


@torch.inference_mode()
def predict(model, batch, quantiles=16):
    hidden, raw, _ = model.encode(batch)
    valid = batch['valid']
    h, raw = hidden[valid], raw[valid]
    probs = torch.zeros((len(h), model.config.vocab_size), device=h.device)
    dose_prob, duration = torch.zeros(len(h), device=h.device), torch.zeros(len(h), device=h.device)
    target = batch['target'][valid]
    med = (target - 1278).clamp(0, 6)
    meds = (target >= 1278) & (target <= 1284)
    attribute_mass = torch.zeros(int(meds.sum()), device=h.device)
    for q in range(quantiles):
        gap = model.time_distribution.quantile(raw, raw.new_full((len(raw),), (q + .5) / quantiles))
        c = model.condition(h, gap, batch['age'][valid], batch['before'][valid], batch['through'][valid])
        probs += c['event_logp'].exp() / quantiles
        # Bayes weights p(time, action, observed medication | H), normalized below.
        selected = med[meds]
        weights = c['joint_logp'][meds].gather(-1, selected[:, None, None].expand(-1, 3, 1)).squeeze(-1).exp()
        weights = weights * c['gate'][meds].sigmoid()[:, None] / quantiles
        attribute_mass += weights.sum(-1)
        for a in range(3):
            if len(selected):
                dose, params = model.attributes({'hidden': c['hidden'][meds]}, selected, torch.full_like(selected, a), gap[meds])
                dose_prob[meds] += weights[:, a] * dose.sigmoid()
                duration[meds] += weights[:, a] * model.duration_head.mean(params)
    dose_prob[meds] /= attribute_mass.clamp_min(1e-30)
    duration[meds] /= attribute_mass.clamp_min(1e-30)
    window = None
    if model.window_head is not None:
        anchors = hidden[torch.arange(len(hidden), device=h.device), batch['anchor']]
        window = model.window_head(anchors).reshape_as(batch['window_y']).sigmoid()
    return dict(event=probs, dose=dose_prob, duration=duration, time=model.time_distribution.mean(raw), window=window)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--quantiles', type=int, default=16)
    p.add_argument('--mode', choices=['landmark', 'prefix'], default='landmark')
    p.add_argument('--minimum-history-days', type=int, default=365)
    p.add_argument('--max-patients', type=int)
    p.add_argument('--batch-size', type=int, default=1)
    p.add_argument('--calibration')
    args = p.parse_args()
    if args.quantiles < 1:
        raise ValueError('quantiles must be positive')
    if args.batch_size < 1:
        raise ValueError('batch-size must be positive')
    model, ckpt = load_checkpoint(args.checkpoint, args.device)
    cohort = Cohort(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    collected = {k: [] for k in ['patient', 'age', 'target', 'event', 'dose', 'dose_y', 'duration', 'duration_y', 'time', 'time_y']}
    windows = []
    included = 0
    pending, pending_ids = [], []

    def evaluate_pending():
        nonlocal pending, pending_ids
        if not pending:
            return
        b = collate(pending, args.device)
        pred = predict(model, b, args.quantiles)
        mask = b['valid']
        patient = np.concatenate([np.full(int(mask[j].sum()), patient_id)
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
        pending, pending_ids = [], []

    for i in range(len(cohort)):
        if args.max_patients is not None and included >= args.max_patients:
            break
        rows = cohort.patient(i)
        end = landmark(rows, args.minimum_history_days) if args.mode == 'landmark' else None
        if args.mode == 'landmark' and end is None or len(rows) < 2:
            continue
        e = example(rows, model.config, end=end)
        if args.mode == 'landmark':
            keep = bool(e['valid'][-1])
            e['valid'][:] = False
            e['valid'][-1] = keep
        pending.append(e)
        pending_ids.append(int(cohort.ids[i]))
        included += 1
        if len(pending) == args.batch_size:
            evaluate_pending()
    evaluate_pending()
    if not included:
        raise ValueError('No evaluable patients; inspect landmark/follow-up eligibility')
    arrays = {k: np.concatenate(v) for k, v in collected.items()}
    np.savez_compressed(out / 'predictions.npz', **arrays)
    metrics = []
    for token in supported_tokens():
        metrics.append(dict(token=token, **binary_metrics(arrays['target'] == token, arrays['event'][:, token])))
    with (out / 'event_auc.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    meds = medication(arrays['target'])
    dose_result = dose_metrics(arrays['dose_y'][meds], arrays['dose'][meds])
    if args.calibration:
        calibration = load_calibration(args.calibration, args.checkpoint)
        if calibration['quantiles'] != args.quantiles or calibration['evaluation_mode'] != args.mode:
            raise ValueError('Use the calibration evaluation mode and quantile count')
        logits = torch.logit(torch.tensor(arrays['dose'][meds]).clamp(1e-7, 1 - 1e-7))
        dose_result['calibrated'] = dose_metrics(
            arrays['dose_y'][meds], apply_calibration(logits, calibration).numpy())
    aucs = [m['auc'] for m in metrics if m['auc'] is not None]
    window_metrics = []
    if windows:
        wy, wv, wp = (np.stack([w[j] for w in windows]) for j in range(3))
        np.savez_compressed(out / 'window_predictions.npz', target=wy, valid=wv, probability=wp)
        for hi, horizon in enumerate(model.config.horizons_days):
            for ti, token in enumerate(model.config.window_tokens):
                valid = wv[:, hi, ti]
                window_metrics.append(dict(horizon_days=horizon, token=token, **binary_metrics(wy[valid, hi, ti], wp[valid, hi, ti])))
    write_json(out / 'window_metrics.json', window_metrics)
    duration_result = duration_metrics(arrays['duration_y'][meds], arrays['duration'][meds])
    write_json(out / 'metrics.json', dict(patients=included, transitions=len(arrays['target']), event_macro_auc=float(np.mean(aucs)) if aucs else None,
        evaluable_event_codes=len(aucs), dose=dose_result, duration=duration_result,
        duration_mae=duration_result['mae'],
        time_mae_days=float(np.abs(arrays['time'] - arrays['time_y']).mean()) if len(arrays['time']) else None,
        attribute_estimand='Observed drug class; predicted time/action marginalized. Not joint generation accuracy.'))
    write_json(out / 'manifest.json', dict(checkpoint_sha256=sha256(args.checkpoint), data_sha256=sha256(args.data),
        predictions_sha256=sha256(out / 'predictions.npz'),
        checkpoint_data_hashes=ckpt['data_hashes'], mode=args.mode, minimum_history_days=args.minimum_history_days,
        quantiles=args.quantiles, max_patients=args.max_patients, batch_size=args.batch_size,
        sources='0909 daily-mass TIME; no true future time in prediction'))
    print(f'Evaluated {included} patients, {len(arrays["target"])} clinical transitions: {out}')


if __name__ == '__main__':
    main()
