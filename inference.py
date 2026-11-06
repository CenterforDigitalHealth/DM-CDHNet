"""Load the final architecture and explicitly select native or published correction."""
import json
from pathlib import Path

from simulation.correction import install_correction
from utils import load_checkpoint, sha256


ROOT = Path(__file__).resolve().parent


def load_model(checkpoint, device='cpu', correction='native'):
    if correction not in {'native', 'final'}:
        raise ValueError('correction must be native or final')
    digest = sha256(checkpoint)
    provenance = json.loads((ROOT / 'docs/checkpoint_provenance.json').read_text())
    if correction == 'final' and digest != provenance['checkpoint_sha256']:
        raise ValueError('The published correction requires the documented final checkpoint')
    model, saved = load_checkpoint(checkpoint, device)
    receipt = dict(checkpoint_sha256=digest, correction=correction)
    if correction == 'final':
        parameters = ROOT / 'simulation/first_parameters.json'
        if sha256(parameters) != provenance['first_parameters_sha256']:
            raise ValueError('Final correction parameters differ from checkpoint provenance')
        if sha256(ROOT / 'simulation/correction.py') != provenance['correction_sha256']:
            raise ValueError('Final correction code differs from checkpoint provenance')
        install_correction(model, 'gate_unweighted_first_same_day')
        receipt['correction_parameters_sha256'] = sha256(parameters)
    return model, saved, receipt
