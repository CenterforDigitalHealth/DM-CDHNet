"""Fit DOSE Platt calibration on verified checkpoint-validation predictions only.

Example: python calibrate.py --evaluation outputs/validation --out outputs/dose_calibration.json
The output is a new file; existing artifacts are never overwritten.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from token_roles import medication
from utils import sha256


def fit_calibration(evaluation, out):
    root, out = Path(evaluation), Path(out)
    if out.exists():
        raise FileExistsError(f'Choose a new calibration output: {out}')
    manifest = json.loads((root / 'manifest.json').read_text())
    hashes = manifest['checkpoint_data_hashes']
    if not hashes.get('val') or manifest['data_sha256'] != hashes['val'] or manifest['data_sha256'] == hashes.get('test'):
        raise ValueError('Platt calibration requires the checkpoint validation cohort, never test data')
    predictions = root / 'predictions.npz'
    if sha256(predictions) != manifest['predictions_sha256']:
        raise ValueError('Predictions changed after evaluation')
    with np.load(predictions, allow_pickle=False) as data:
        if 'evaluation_mask' not in data or data['evaluation_mask'].dtype != np.bool_ or data['evaluation_mask'].shape != data['target'].shape:
            raise ValueError('Calibration requires policy-filtered evaluation with a boolean target mask')
        mask = medication(data['target']) & data['evaluation_mask']
        excluded = int((~data['evaluation_mask']).sum())
        y = data['dose_y'][mask]
        prob = data['dose'][mask]
    if not np.isfinite(prob).all() or np.any((prob < 0) | (prob > 1)) or set(np.unique(y)) != {0, 1}:
        raise ValueError('Calibration requires finite probabilities and both binary validation classes')
    prob = prob.clip(1e-7, 1 - 1e-7)
    logits = np.log(prob / (1 - prob))[:, None]
    fit = LogisticRegression(C=1e6, solver='lbfgs', max_iter=2000).fit(logits, y)
    result = dict(method='Platt logistic regression', slope=float(fit.coef_[0, 0]),
        intercept=float(fit.intercept_[0]), n=len(y), positives=int(y.sum()),
        excluded_prediction_rows=excluded, checkpoint_sha256=manifest['checkpoint_sha256'],
        validation_data_sha256=manifest['data_sha256'], evaluation_mode=manifest['mode'],
        quantiles=manifest['quantiles'], minimum_history_days=manifest['minimum_history_days'],
        labels_sha256=manifest['labels_sha256'], policy_sha256=manifest['policy_sha256'],
        correction=manifest['correction'], predictions_sha256=manifest['predictions_sha256'],
        evaluation_manifest_sha256=sha256(root / 'manifest.json'),
        source_sha256=sha256(Path(__file__)),
        applies_to='DOSE probability marginalized over predicted time/action given observed medication class; not rollout conditional logits')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        handle.write(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()
    fit_calibration(args.evaluation, args.out)
    print(args.out)


if __name__ == '__main__':
    main()
