#!/usr/bin/env python3
"""Batched full-cohort 3-year generation from corrected history into KR test."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from data import features
from token_roles import DTYPE, clinical, medication, MED_MIN, N_MED, DEATH
from utils import load_checkpoint, sha256, write_json


REASON_CODE = {'horizon': 0, 'death': 1, 'max_events': 2}
MODEL_KEYS = ('token', 'age', 'shift', 'duration', 'input_action', 'history_gap', 'before', 'through')


def patient_order_hash(ids):
    return hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest()


def categorical(logits, uniforms):
    probability = logits.float().softmax(-1)
    cdf = probability.cumsum(-1)
    index = (uniforms[:, None] > cdf).sum(-1)
    return index.clamp_max(logits.shape[-1] - 1)


class GenerationState:
    """Incremental equivalent of data.features for an autoregressive trajectory."""
    def __init__(self, prefix, recent_days, context_length):
        full = features(prefix, recent_days)
        self.values = {key: value[-context_length:].copy() for key, value in full.items()}
        self.patient_id = int(prefix['ID'][0])
        self.last_row_age = int(prefix['AGE'][-1])
        clinical_index = np.flatnonzero(clinical(prefix['EVENT']))
        if not len(clinical_index):
            raise ValueError('Generation prefix has no supported clinical event')
        self.previous_clinical_age = int(prefix['AGE'][clinical_index[-1]])
        self.last = full['through'][-1].copy()
        self.day_before = full['before'][-1].copy()
        self.current_day = self.last_row_age
        self.recent_days = recent_days
        self.context_length = context_length

    def append(self, row):
        _, age, token, shift, duration = row
        age, token = int(age), int(token)
        if age != self.current_day:
            self.day_before = self.last.copy()
            self.current_day = age
        before = self.day_before.copy()
        action = 0
        if medication(token):
            med = token - MED_MIN
            action = 1 if before[med] < 0 else (2 if age - before[med] <= self.recent_days else 3)
            self.last[med] = age
        through = self.last.copy()
        additions = {
            'token': np.int64(token), 'age': np.float32(age), 'shift': np.int64(shift),
            'duration': np.float32(duration), 'input_action': np.int64(action),
            'history_gap': np.float32(age - self.previous_clinical_age),
            'before': before.astype(np.float32), 'through': through.astype(np.float32),
        }
        for key, value in additions.items():
            self.values[key] = np.concatenate((self.values[key], np.asarray(value)[None]))[-self.context_length:]
        self.previous_clinical_age = age
        self.last_row_age = age


def collate_states(states, device, static_width=None):
    length = static_width or max(len(state.values['token']) for state in states)
    result = {}
    for key in MODEL_KEYS:
        sample = states[0].values[key]
        shape = (len(states), length) + sample.shape[1:]
        array = np.zeros(shape, dtype=sample.dtype)
        for row, state in enumerate(states):
            value = state.values[key]
            array[row, :len(value)] = value
        result[key] = torch.as_tensor(array, device=device)
    padding = np.zeros((len(states), length), dtype=bool)
    for row, state in enumerate(states):
        padding[row, :len(state.values['token'])] = True
    result['padding'] = torch.as_tensor(padding, device=device)
    return result


@torch.inference_mode()
def rollout_batch(model, prefixes, end_ages, seeds, max_events, precision='bf16',
                  forbidden_med_tokens=(), stop_when_all_token_groups=(), minimum_ages=None,
                  forbidden_policy='medication_only', event_trace=None, event_limits=None,
                  rng_skip_counts=None):
    """Independent path streams, fixed TIME/gate/joint/DOSE/mixture/DUR/nonmed channels."""
    device = next(model.parameters()).device
    forbidden_meds = sorted({int(token) - MED_MIN for token in forbidden_med_tokens})
    if any(med < 0 or med >= N_MED for med in forbidden_meds):
        raise ValueError('forbidden_med_tokens contains an unsupported medication')
    if forbidden_policy not in ('medication_only', 'whole_event'):
        raise ValueError('Unknown forbidden_policy')
    if event_trace is not None and (len(event_trace) != len(prefixes) or any(event_trace)):
        raise ValueError('event_trace must contain one empty list per prefix')
    states = [GenerationState(prefix, model.config.recent_days, model.config.context_length)
              for prefix in prefixes]
    generated = [[] for _ in states]
    actions = [[] for _ in states]
    reasons = [None] * len(states)
    stop_groups = tuple(frozenset(map(int, group)) for group in stop_when_all_token_groups)
    group_hits = np.zeros((len(states), len(stop_groups)), dtype=bool)
    generators = [np.random.default_rng(seed) for seed in seeds]
    if rng_skip_counts is not None:
        if len(rng_skip_counts) != len(generators):
            raise ValueError('rng_skip_counts must match prefixes')
        for generator, skip in zip(generators, rng_skip_counts):
            if int(skip) != skip or skip < 0:
                raise ValueError('rng_skip_counts must contain non-negative integers')
            generator.random(int(skip))
    if minimum_ages is not None and len(minimum_ages) != len(states):
        raise ValueError('minimum_ages must have one entry per prefix')
    limits = np.full(len(states), max_events, dtype=np.int64) if event_limits is None else np.asarray(event_limits)
    if limits.shape != (len(states),) or np.any(limits < 0) or np.any(limits > max_events) or np.any(limits != limits.astype('int64')):
        raise ValueError('event_limits must be integer budgets between zero and max_events')
    active = [i for i in range(len(states)) if limits[i] > 0]
    for i in range(len(states)):
        if limits[i] == 0:
            reasons[i] = 'max_events'
    for _ in range(max_events):
        if not active:
            break
        # Seven fixed channels consumed at every step, including unused branches.
        step_u = {i: generators[i].random(7) for i in active}
        active_states = [states[i] for i in active]
        lengths = torch.tensor([len(state.values['token']) for state in active_states], device=device) - 1
        width = None
        if device.type == 'mps':
            needed = max(len(state.values['token']) for state in active_states)
            width = min(model.config.context_length, max(32, 1 << (needed - 1).bit_length()))
        batch = collate_states(active_states, device, width)
        amp = precision == 'bf16' and device.type == 'cuda'
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            hidden, raw, _ = model.encode(batch)
            row = torch.arange(len(active), device=device)
            h, raw = hidden[row, lengths], raw[row, lengths]
            u_time = torch.tensor([step_u[i][0] for i in active], device=device, dtype=torch.float32)
            if minimum_ages is not None:
                lower = torch.tensor([
                    max(0, int(minimum_ages[i]) - states[i].last_row_age)
                    if not generated[i] else 0 for i in active
                ], device=device, dtype=torch.float32)
                gap = model.time_distribution.conditional_quantile(raw, u_time, lower)
            else:
                gap = model.time_distribution.quantile(raw, u_time)
        age = np.asarray([float(states[i].last_row_age) for i in active]) + gap.cpu().numpy()
        if minimum_ages is not None:
            assert all(age[j] >= minimum_ages[i] for j, i in enumerate(active))
        within = np.asarray([age[j] <= end_ages[i] for j, i in enumerate(active)])
        next_active = []
        if within.any():
            local = np.flatnonzero(within)
            take = torch.as_tensor(local, device=device)
            source_row = row[take]
            source_pos = lengths[take]
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                condition = model.condition(
                    h[take], gap[take], batch['age'][source_row, source_pos],
                    batch['before'][source_row, source_pos], batch['through'][source_row, source_pos])
            global_indices = [active[j] for j in local]
            gate_u = torch.tensor([step_u[i][1] for i in global_indices], device=device, dtype=torch.float32)
            med_probability = condition['gate'].sigmoid()
            if forbidden_meds and forbidden_policy == 'whole_event':
                # Condition the complete next-EVENT distribution on not emitting
                # a forbidden medication. TIME remains the previously drawn TIME.
                allowed = condition['joint_logp'].float().exp().clone()
                allowed[:, :, forbidden_meds] = 0
                mass = allowed.sum((-2, -1))
                med_mass = med_probability.float() * mass
                med_probability = med_mass / (1 - med_probability.float() + med_mass).clamp_min(1e-12)
            is_med = gate_u < med_probability
            med_local = torch.nonzero(is_med, as_tuple=True)[0]
            nonmed_local = torch.nonzero(~is_med, as_tuple=True)[0]
            tokens = torch.empty(len(global_indices), dtype=torch.long, device=device)
            shifts = torch.zeros(len(global_indices), dtype=torch.long, device=device)
            totals = torch.zeros(len(global_indices), dtype=torch.long, device=device)
            sampled_actions = torch.full((len(global_indices),), -1, dtype=torch.long, device=device)
            if len(med_local):
                med_global = [global_indices[j] for j in med_local.cpu().tolist()]
                joint_u = torch.tensor([step_u[i][2] for i in med_global], device=device, dtype=torch.float32)
                joint_logits = condition['joint_logp'][med_local].flatten(-2)
                if forbidden_meds:
                    joint_logits = joint_logits.clone()
                    blocked = [action * N_MED + med for action in range(3) for med in forbidden_meds]
                    joint_logits[:, blocked] = -torch.inf
                    if torch.isneginf(joint_logits).all(-1).any():
                        raise ValueError('Medication policy masked every joint medication/action outcome')
                joint = categorical(joint_logits, joint_u)
                action, med = joint // N_MED, joint % N_MED
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                    dose, duration = model.attributes(
                        {'hidden': condition['hidden'][med_local]}, med, action, gap[take][med_local])
                dose_u = torch.tensor([step_u[i][3] for i in med_global], device=device, dtype=torch.float32)
                shifts[med_local] = torch.where(dose_u < dose.sigmoid(), 1, 2)
                mix, mu, scale = duration
                component_u = torch.tensor([step_u[i][4] for i in med_global], device=device, dtype=torch.float32)
                component = categorical(mix, component_u)[:, None]
                selected_mu = mu.gather(-1, component).squeeze(-1)
                selected_scale = scale.gather(-1, component).squeeze(-1)
                duration_u = torch.tensor([step_u[i][5] for i in med_global], device=device, dtype=torch.float32).clamp(1e-7, 1 - 1e-7)
                totals[med_local] = (selected_mu + selected_scale * torch.logit(duration_u)).round().clamp(
                    0, model.config.duration_max).long()
                tokens[med_local] = 1278 + med
                sampled_actions[med_local] = action
            if len(nonmed_local):
                nonmed_global = [global_indices[j] for j in nonmed_local.cpu().tolist()]
                token_u = torch.tensor([step_u[i][6] for i in nonmed_global], device=device, dtype=torch.float32)
                selected = categorical(
                    condition['event_logp'][nonmed_local][:, model.nonmed_tokens], token_u)
                tokens[nonmed_local] = model.nonmed_tokens[selected]
            tokens_np = tokens.cpu().numpy()
            shifts_np = shifts.cpu().numpy()
            totals_np = totals.cpu().numpy()
            actions_np = sampled_actions.cpu().numpy()
            if event_trace is not None:
                gaps_np = gap[take].float().cpu().numpy()
                insulin_probability = (condition['gate'].float().sigmoid() *
                                       condition['joint_logp'].float().exp()[:, :, 3].sum(-1)).cpu().numpy()
            for j, i in enumerate(global_indices):
                new_row = (states[i].patient_id, int(age[local[j]]), int(tokens_np[j]),
                           int(shifts_np[j]), int(totals_np[j]))
                generated[i].append(new_row)
                actions[i].append(int(actions_np[j]))
                if event_trace is not None:
                    event_trace[i].append({'gap': float(gaps_np[j]),
                                           'insulin_probability': float(insulin_probability[j])})
                states[i].append(new_row)
                for group_number, group in enumerate(stop_groups):
                    group_hits[i, group_number] |= int(tokens_np[j]) in group
                if int(tokens_np[j]) == DEATH:
                    reasons[i] = 'death'
                elif len(stop_groups) and group_hits[i].all():
                    # Once every requested occurrence outcome is known positive,
                    # simulating the rest of this trajectory cannot change them.
                    reasons[i] = 'targets_complete'
                elif len(generated[i]) >= limits[i]:
                    reasons[i] = 'max_events'
                else:
                    next_active.append(i)
        for j, i in enumerate(active):
            if not within[j]:
                reasons[i] = 'horizon'
        active = next_active
    for i in active:
        reasons[i] = 'max_events'
    return ([np.asarray(rows, dtype=DTYPE) for rows in generated], actions, reasons)
