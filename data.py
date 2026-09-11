"""uint32 (ID, AGE in days, EVENT, DOSE, DUR); no synthetic events."""
import numpy as np
import torch
from token_roles import DTYPE, clinical, disease, medication, MED_MIN, DEATH


class Cohort:
    def __init__(self, path):
        self.path = str(path)
        self.rows = np.memmap(path, dtype=DTYPE, mode='r')
        if not len(self.rows):
            raise ValueError('Empty cohort')
        ids = self.rows['ID']
        self.starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1]
        self.raw_ends = np.r_[self.starts[1:], len(ids)]
        self.ends = self.raw_ends.copy()
        self.ids = ids[self.starts].copy()
        if len(np.unique(self.ids)) != len(self.ids):
            raise ValueError('Patient records must be contiguous')
        if np.any(self.rows['EVENT'] >= 1289) or np.any(self.rows['EVENT'] == 0):
            raise ValueError('Invalid vocabulary or stored padding token')
        self.post_death_patients = self.post_death_clinical_rows = 0
        for i, (start, raw_end) in enumerate(zip(self.starts, self.raw_ends)):
            r = self.rows[start:raw_end]
            if np.any(r['AGE'][1:] < r['AGE'][:-1]):
                raise ValueError(f'AGE is not chronological for ID {self.ids[i]}')
            meds = medication(r['EVENT'])
            if np.any((r['DOSE'][meds] < 1) | (r['DOSE'][meds] > 3)):
                raise ValueError('Medication DOSE must be 1=increase, 2=maintain, 3=decrease')
            deaths = np.flatnonzero(r['EVENT'] == DEATH)
            if len(deaths):
                after = clinical(r['EVENT'][deaths[0] + 1:])
                self.post_death_patients += int(after.any())
                self.post_death_clinical_rows += int(after.sum())
                # Death is terminal in this model. Preserve the source file and
                # expose only the prefix through the first Death to all callers.
                self.ends[i] = start + int(deaths[0]) + 1

    def __len__(self):
        return len(self.starts)

    def patient(self, i):
        return self.rows[self.starts[i]:self.ends[i]]

    def select_endpoints(self, k):
        counts = np.zeros(1289, dtype=np.int64)
        for i in range(len(self)):
            t = self.patient(i)['EVENT']
            counts += np.bincount(t[disease(t)].astype(int), minlength=1289)
        return sorted(np.argsort(-counts, kind='stable')[:min(k, np.count_nonzero(counts))].tolist())


def features(rows, recent_days=90):
    """State at each input position, computed before cropping the history."""
    n = len(rows)
    before = np.full((n, 7), -1, dtype=np.float32)
    through = before.copy()
    last = np.full(7, -1, dtype=np.float32)
    day_last = last.copy()
    day = -1
    action = np.zeros(n, dtype=np.int64)
    gaps = np.zeros(n, dtype=np.float32)
    prev_clinical = None
    for j, r in enumerate(rows):
        age, token = int(r['AGE']), int(r['EVENT'])
        if age != day:
            day_last, day = last.copy(), age
        before[j] = day_last
        if medication(token):
            m = token - MED_MIN
            action[j] = 1 if day_last[m] < 0 else (2 if age - day_last[m] <= recent_days else 3)
            last[m] = age
        through[j] = last
        if clinical(token):
            gaps[j] = 0 if prev_clinical is None else age - prev_clinical
            prev_clinical = age
    return dict(token=rows['EVENT'].astype(np.int64), age=rows['AGE'].astype(np.float32),
                shift=rows['DOSE'].astype(np.int64), duration=rows['DUR'].astype(np.float32),
                input_action=action, history_gap=gaps, before=before, through=through)


def window_labels(rows, age, tokens, horizons):
    y = np.zeros((len(horizons), len(tokens)), dtype=np.float32)
    valid = np.zeros_like(y, dtype=bool)
    future = rows[rows['AGE'] > age]
    death = future['AGE'][future['EVENT'] == DEATH]
    for h, days in enumerate(horizons):
        in_window = future[future['AGE'] <= age + days]
        y[h] = np.isin(tokens, in_window['EVENT'])
        # Last record is a follow-up PROXY, not a verified enrollment end date.
        complete = int(rows['AGE'][-1]) >= age + days or (len(death) and death[0] <= age + days)
        valid[h] = (y[h] == 1) | bool(complete)
    return y, valid


def example(rows, config, end=None, supervised=True):
    """Training uses a left prefix; explicit end supports landmark inference."""
    stop = min(len(rows) - 1, config.context_length) if end is None else end + 1
    if stop < 1:
        raise ValueError('At least two records required for training')
    start = max(0, stop - config.context_length)
    f = {k: v[start:stop] for k, v in features(rows[:stop], config.recent_days).items()}
    n = stop - start
    f['padding'] = np.ones(n, dtype=bool)
    for key, dtype in [('target', np.int64), ('gap', np.float32), ('dose', np.float32),
                       ('dur', np.float32), ('valid', bool)]:
        f[key] = np.zeros(n, dtype=dtype)
    indices = np.flatnonzero(clinical(rows['EVENT']))
    if supervised:
        for a, b in zip(indices[:-1], indices[1:]):
            if start <= a < stop and rows['EVENT'][a] != DEATH:
                j = a - start
                f['target'][j], f['gap'][j] = rows['EVENT'][b], int(rows['AGE'][b]) - int(rows['AGE'][a])
                f['dose'][j], f['dur'][j] = rows['DOSE'][b] == 1, rows['DUR'][b]
                f['valid'][j] = True
                if medication(rows['EVENT'][b]) and rows['DUR'][b] > config.duration_max:
                    raise ValueError('Medication duration exceeds duration_max; do not silently clip labels')
    anchors = np.flatnonzero(clinical(f['token']))
    anchor = int(anchors[-1]) if len(anchors) else n - 1
    f['anchor'] = np.int64(anchor)
    f['window_y'], f['window_valid'] = window_labels(rows, f['age'][anchor], config.window_tokens, config.horizons_days)
    return f


def collate(examples, device='cpu'):
    length = max(len(e['token']) for e in examples)
    result = {}
    for key in examples[0]:
        if key in ('anchor', 'window_y', 'window_valid'):
            arr = np.stack([e[key] for e in examples])
        else:
            arr = np.stack([np.pad(e[key], [(0, length - len(e[key]))] + [(0, 0)] * (e[key].ndim - 1)) for e in examples])
        result[key] = torch.as_tensor(arr, device=device)
    return result


def eligible_indices(cohort, config, landmark_sampling='left'):
    result = []
    for i in range(len(cohort)):
        rows = cohort.patient(i)
        idx = np.flatnonzero(clinical(rows['EVENT']))
        has_landmark = landmark_sampling == 'left' or len(landmark_candidates(rows)) > 0
        if len(idx) >= 2 and idx[0] < min(len(rows) - 1, config.context_length) and has_landmark:
            result.append(i)
    if not result:
        raise ValueError('No patient has a supervised clinical transition')
    return np.asarray(result)


def landmark_candidates(rows):
    """Last clinical row of each date that has a later clinical event.

    Day-end cutoffs match the generation landmark and never expose part of a
    same-day record set as future window data. Same-day transitions inside the
    retained history remain ordinary supervised targets.
    """
    idx = np.flatnonzero(clinical(rows['EVENT']))
    if len(idx) < 2:
        return np.empty(0, dtype=np.int64)
    ages = rows['AGE'][idx]
    last_of_day = idx[np.r_[ages[1:] != ages[:-1], True]]
    candidates = last_of_day[:-1]
    return candidates[rows['EVENT'][candidates] != DEATH]


def choose_landmark(rows, rng, mode):
    if mode == 'left':
        return None
    if mode != 'random_day':
        raise ValueError(f'Unknown landmark_sampling: {mode}')
    candidates = landmark_candidates(rows)
    if not len(candidates):
        raise ValueError('Patient has no eligible longitudinal day landmark')
    return int(candidates[int(rng.integers(len(candidates)))])


def landmark(rows, minimum_history_days=365):
    idx = np.flatnonzero(clinical(rows['EVENT']))
    if not len(idx):
        return None
    candidates = idx[rows['AGE'][idx] >= int(rows['AGE'][idx[0]]) + minimum_history_days]
    if not len(candidates):
        return None
    age = rows['AGE'][candidates[0]]
    same_day = idx[rows['AGE'][idx] == age]
    end = int(same_day[-1])
    return None if DEATH in rows['EVENT'][:end + 1] else end
