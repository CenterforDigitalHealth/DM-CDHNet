"""Run provenance, deterministic setup and checkpoint I/O."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch

SCHEMA_VERSION = 1


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    os.replace(tmp, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_checkpoint(path, payload):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device='cpu'):
    from config import ModelConfig
    from model import CompositeDelphi
    # Checkpoints are local/trusted training artifacts, including optimizer/RNG state.
    ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if ckpt.get('schema_version') != SCHEMA_VERSION or ckpt.get('format') != 'composite-delphi-0909':
        raise ValueError('Expected a 0909 checkpoint; legacy checkpoint conversion is not implicit')
    model = CompositeDelphi(ModelConfig(**ckpt['model_config'])).to(device)
    model.load_state_dict(ckpt['model'], strict=True)
    model.eval()
    return model, ckpt


def source_manifest():
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root)): sha256(p) for p in sorted(root.rglob('*.py')) if 'outputs' not in p.parts}
