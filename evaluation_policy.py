"""Resolve reporting exclusions from the active dictionary; never alter inference."""
import csv
import json
from pathlib import Path
import re
import numpy as np

DEFAULT_POLICY = Path(__file__).with_name('evaluation_policy.json')


def load_label_dictionary(labels_path):
    """Read a CSV with unique nonnegative token_id and name and/or icd10."""
    labels = {}
    with Path(labels_path).open(newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or 'token_id' not in reader.fieldnames or not {'name', 'icd10'}.intersection(reader.fieldnames):
            raise ValueError('Labels require token_id and name or icd10 columns')
        for line, row in enumerate(reader, 2):
            raw_id = (row.get('token_id') or '').strip()
            if not re.fullmatch(r'\d+', raw_id):
                raise ValueError(f'Invalid token_id on dictionary line {line}')
            token = int(raw_id)
            if token in labels:
                raise ValueError(f'Duplicate token_id {token} in dictionary')
            name = (row.get('name') or '').strip()
            explicit = (row.get('icd10') or '').strip().upper()
            if not name and not explicit:
                raise ValueError(f'Missing label for token {token}')
            if explicit and not re.fullmatch(r'[A-Z]\d{2}(?:\.?[A-Z0-9]+)?', explicit):
                raise ValueError(f'Invalid icd10 for token {token}: {explicit}')
            inferred = name.split()[0].upper() if name else ''
            code = explicit or (inferred if re.fullmatch(r'[A-Z]\d{2}(?:\.?[A-Z0-9]+)?', inferred) else '')
            if explicit and re.fullmatch(r'[A-Z]\d{2}(?:\.?[A-Z0-9]+)?', inferred) and inferred[:3] != explicit[:3]:
                raise ValueError(f'Conflicting name and icd10 for token {token}')
            labels[token] = code
    if not labels:
        raise ValueError('Empty label dictionary')
    return labels


def load_excluded_tokens(labels_path, policy_path=None):
    """Use the active token mapping and exactly P00–P96, including subcodes."""
    policy = json.loads(Path(policy_path or DEFAULT_POLICY).read_text())
    if policy.get('excluded_icd10_chapters') != ['XVI. Perinatal Conditions']:
        raise ValueError('Expected the supported P00-P96 perinatal exclusion policy')
    labels = load_label_dictionary(labels_path)
    return frozenset(token for token, code in labels.items()
                     if code.startswith('P') and 0 <= int(code[1:3]) <= 96)


def filter_target_rows(targets, excluded_tokens):
    """Mask the original targets, without changing targets or histories."""
    return ~np.isin(np.asarray(targets), list(excluded_tokens))
