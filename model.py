"""History -> TIME -> EVENT -> medication class/action -> DOSE/DUR.

Independent implementation: old experiment checkpoints are not compatible.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from time_distributions import make_time_distribution
from token_roles import MED_MIN, MED_MAX, DEATH


class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, c, local):
        super().__init__()
        self.h, self.kv, self.d = c.n_head, c.n_kv_head, c.n_embd // c.n_head
        self.qkv = nn.Linear(c.n_embd, (self.h + 2 * self.kv) * self.d, bias=False)
        self.out = nn.Linear(c.n_embd, c.n_embd, bias=False)
        self.local, self.drop = local, c.dropout
        self.register_buffer('freq', 1 / (10000 ** (torch.arange(0, self.d, 2).float() / self.d)))

    def rope(self, x):
        angle = torch.arange(x.shape[-2], device=x.device).float()[:, None] * self.freq[None, :]
        a, b = x[..., 0::2], x[..., 1::2]
        return torch.stack((a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1).flatten(-2).to(x.dtype)

    def forward(self, x, padding):
        b, t, _ = x.shape
        q, k, v = self.qkv(x).split([self.h * self.d, self.kv * self.d, self.kv * self.d], -1)
        reshape = lambda z, h: z.view(b, t, h, self.d).transpose(1, 2)
        q, k, v = self.rope(reshape(q, self.h)), self.rope(reshape(k, self.kv)), reshape(v, self.kv)
        k, v = k.repeat_interleave(self.h // self.kv, 1), v.repeat_interleave(self.h // self.kv, 1)
        pos = torch.arange(t, device=x.device)
        allowed = pos[:, None] >= pos[None, :]
        if self.local:
            allowed &= pos[:, None] - pos[None, :] < self.local
        mask = allowed[None, None] & padding[:, None, None, :]
        z = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop if self.training else 0.)
        return self.out(z.transpose(1, 2).reshape(b, t, -1)) * padding[..., None]


class Expert(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.up = nn.Linear(d, 8 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x):
        a, b = self.up(x).chunk(2, -1)
        return self.down(F.silu(a) * b)


class MoE(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.router = nn.Linear(c.n_embd, c.n_experts, bias=False)
        self.experts = nn.ModuleList([Expert(c.n_embd) for _ in range(c.n_experts)])
        self.k = c.experts_per_token

    def forward(self, x, padding):
        flat = x.reshape(-1, x.shape[-1])
        probs = self.router(flat).float().softmax(-1)
        weights, indices = probs.topk(self.k, -1)
        weights = weights / weights.sum(-1, keepdim=True)
        out = torch.zeros_like(flat)
        for i, expert in enumerate(self.experts):
            row, slot = torch.where((indices == i) & padding.reshape(-1, 1))
            if len(row):
                out.index_add_(0, row, (expert(flat[row]) * weights[row, slot, None]).to(out.dtype))
        valid = padding.flatten()
        importance = probs[valid].mean(0)
        load = F.one_hot(indices[valid], len(self.experts)).float().mean((0, 1))
        aux = len(self.experts) * (importance * load).sum()
        return out.view_as(x), aux


class Block(nn.Module):
    def __init__(self, c, layer):
        super().__init__()
        self.norm1, self.norm2 = RMSNorm(c.n_embd), RMSNorm(c.n_embd)
        self.attn = Attention(c, c.local_window if layer % 2 == 0 else None)
        self.moe = MoE(c)
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x, padding):
        x = x + self.drop(self.attn(self.norm1(x), padding))
        z, aux = self.moe(self.norm2(x), padding)
        return x + self.drop(z), aux


class GapConditioner(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, hidden, gap):
        return hidden + self.net(torch.log1p(gap.float().clamp_min(0))[..., None])


class AttributeFiLM(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.med = nn.Embedding(7, d)
        self.action = nn.Embedding(3, d)
        self.gap = nn.Linear(1, d)
        self.film = nn.Linear(d, 2 * d)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, hidden, med, action, gap):
        context = self.med(med) + self.action(action) + self.gap(torch.log1p(gap.float())[..., None])
        scale, shift = self.film(F.silu(context)).chunk(2, -1)
        return hidden * (1 + scale) + shift


class DurationMixture(nn.Module):
    def __init__(self, d, components=16, maximum=550):
        super().__init__()
        self.head = nn.Linear(d, 3 * components)
        self.maximum = maximum

    def params(self, hidden):
        mix, mu, scale = self.head(hidden).float().chunk(3, -1)
        return mix.log_softmax(-1), mu.sigmoid() * self.maximum, F.softplus(scale) + .1

    def log_prob(self, params, days):
        mix, mu, scale = params
        d = days.float()[..., None]
        upper, lower = (d + .5 - mu) / scale, (d - .5 - mu) / scale
        interior = F.logsigmoid(upper) + F.logsigmoid(-lower) + torch.log(-torch.expm1(-1 / scale))
        mass = torch.where(d == 0, F.logsigmoid(upper), torch.where(d == self.maximum, F.logsigmoid(-lower), interior))
        return torch.logsumexp(mix + mass, -1)

    def mean(self, params):
        mix, mu, scale = params
        # E[D] = sum_{d=1}^{max} P(D >= d), for the bounded integer law.
        result = torch.zeros_like(mu)
        for start in range(1, self.maximum + 1, 64):
            d = torch.arange(start, min(start + 64, self.maximum + 1), device=mu.device)
            result += ((mu[..., None] - d + .5) / scale[..., None]).sigmoid().sum(-1)
        return (mix.exp() * result).sum(-1)

    def sample(self, params):
        mix, mu, scale = params
        component = torch.distributions.Categorical(logits=mix).sample()[..., None]
        m, s = mu.gather(-1, component).squeeze(-1), scale.gather(-1, component).squeeze(-1)
        u = torch.rand_like(m).clamp(1e-7, 1 - 1e-7)
        return (m + s * torch.logit(u)).round().clamp(0, self.maximum)

    @torch.no_grad()
    def initialize_from_histogram(self, counts):
        k = self.head.out_features // 3
        counts = counts.float()
        if counts.sum() == 0:
            return
        cdf = counts.cumsum(0) / counts.sum()
        mu = torch.searchsorted(cdf, (torch.arange(k, device=counts.device) + .5) / k).float()
        self.head.weight.zero_()
        self.head.bias[:k].zero_()
        self.head.bias[k:2 * k].copy_(torch.logit((mu / self.maximum).clamp(.001, .999)))
        self.head.bias[2 * k:].fill_(math.log(math.expm1(7.9)))


class CDHnet(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = c = config
        d = c.n_embd
        self.token = nn.Embedding(c.vocab_size, d, padding_idx=0)
        self.shift = nn.Embedding(4, d, padding_idx=0)
        self.input_action = nn.Embedding(4, d, padding_idx=0)
        self.numeric = nn.Linear(18, d, bias=False)
        self.register_buffer('age_freq', 2 * math.pi / torch.logspace(1, 5, 8))
        self.blocks = nn.ModuleList([Block(c, i) for i in range(c.n_layer)])
        self.norm = RMSNorm(d)
        self.time_distribution = make_time_distribution(c.time_family)
        self.time_head = nn.Linear(d, self.time_distribution.output_dim)
        self.event_time, self.med_time = GapConditioner(d), GapConditioner(d)
        self.med_gate = nn.Linear(d, 1)
        self.med_joint = nn.Linear(d, 21)
        self.dose_film, self.duration_film = AttributeFiLM(d), AttributeFiLM(d)
        self.dose_head = nn.Linear(d, 1)
        self.duration_head = DurationMixture(d, c.duration_components, c.duration_max)
        self.window_head = nn.Linear(d, len(c.window_tokens) * len(c.horizons_days)) if c.window_tokens else None
        self.register_buffer('nonmed_tokens', torch.tensor(list(range(22, 1278)) + [DEATH]))
        self.apply(self._initialize)
        for module in self.modules():
            if isinstance(module, GapConditioner):
                nn.init.zeros_(module.net[-1].weight)
                nn.init.zeros_(module.net[-1].bias)
            elif isinstance(module, AttributeFiLM):
                nn.init.zeros_(module.film.weight)
                nn.init.zeros_(module.film.bias)
        self.time_distribution.initialize(self.time_head)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0., std=.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
            if isinstance(module, nn.Embedding) and module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def encode(self, batch):
        phase = batch['age'][..., None] * self.age_freq
        is_med = (batch['token'] >= MED_MIN) & (batch['token'] <= MED_MAX)
        numeric = torch.cat((phase.sin(), phase.cos(), torch.log1p(batch['duration'] * is_med)[..., None],
                             torch.log1p(batch['history_gap'])[..., None]), -1)
        hidden = self.token(batch['token']) + self.shift((batch['shift'] * is_med).clamp(0, 3))
        hidden = hidden + self.input_action(batch['input_action']) + self.numeric(numeric)
        auxiliary = hidden.new_zeros(())
        for block in self.blocks:
            hidden, aux = block(hidden, batch['padding'])
            auxiliary = auxiliary + aux / len(self.blocks)
        hidden = self.norm(hidden)
        return hidden, self.time_head(hidden), auxiliary

    def condition(self, hidden, gap, age, before, through):
        """No future labels consulted: state depends on the supplied (observed/sampled) gap."""
        hidden = self.event_time(hidden, gap)
        med_hidden = self.med_time(hidden, gap)
        last = torch.where((gap == 0)[..., None], before, through)
        seen = last >= 0
        recent = seen & ((age[..., None] + gap[..., None] - last) <= self.config.recent_days)
        allowed = torch.stack((~seen, seen, seen), -2)
        penalties = torch.stack((torch.zeros_like(last), (~recent).float(), recent.float()), -2)
        joint = self.med_joint(med_hidden).float().reshape(*hidden.shape[:-1], 3, 7)
        joint = (joint - penalties * self.config.action_soft_penalty).masked_fill(~allowed, -torch.inf)
        joint = joint.flatten(-2).log_softmax(-1).reshape_as(joint)
        gate = self.med_gate(hidden).float().squeeze(-1)
        nonmed = F.linear(hidden, self.token.weight[self.nonmed_tokens]).float().log_softmax(-1)
        event = hidden.new_full((*hidden.shape[:-1], self.config.vocab_size), -torch.inf, dtype=torch.float32)
        event[..., self.nonmed_tokens] = F.logsigmoid(-gate)[..., None] + nonmed
        event[..., MED_MIN:MED_MAX + 1] = F.logsigmoid(gate)[..., None] + joint.logsumexp(-2)
        return dict(hidden=hidden, event_logp=event, joint_logp=joint, gate=gate, last=last, recent=recent)

    def attributes(self, conditioned, med, action, gap):
        h = conditioned['hidden']
        dose = self.dose_head(self.dose_film(h, med, action, gap)).float().squeeze(-1)
        dur = self.duration_head.params(self.duration_film(h, med, action, gap))
        return dose, dur

    def _reduce(self, values, patients, weights=None):
        """Configured reduction; empty branches remain differentiable zeros."""
        if not len(values):
            return values.sum()
        weights = torch.ones_like(values) if weights is None else weights.to(values)
        if self.config.loss_reduction == 'event':
            return (values * weights).sum() / weights.sum().clamp_min(1e-12)
        unique = patients.unique()
        per_patient = []
        for patient in unique:
            take = patients == patient
            per_patient.append((values[take] * weights[take]).sum() / weights[take].sum().clamp_min(1e-12))
        return torch.stack(per_patient).mean()

    def forward(self, batch):
        hidden, raw_time, aux = self.encode(batch)
        mask = batch['valid'] & batch['padding']
        if not mask.any():
            raise ValueError('Batch contains no clinical targets')
        patient = mask.nonzero(as_tuple=False)[:, 0]
        h, gap, target = hidden[mask], batch['gap'][mask], batch['target'][mask]
        c = self.condition(h, gap, batch['age'][mask], batch['before'][mask], batch['through'][mask])
        med_mask = (target >= MED_MIN) & (target <= MED_MAX)
        med = (target - MED_MIN).clamp(0, 6)
        last = c['last'].gather(-1, med[:, None]).squeeze(-1)
        recent = c['recent'].gather(-1, med[:, None]).squeeze(-1)
        action = torch.where(last < 0, 0, torch.where(recent, 1, 2))
        gate_loss = torch.where(med_mask, -F.logsigmoid(c['gate']), -F.logsigmoid(-c['gate']))
        full_event_loss = -c['event_logp'].gather(-1, target[:, None]).squeeze(-1)
        conditional_loss = torch.where(med_mask,
            -c['joint_logp'][torch.arange(len(target), device=target.device), action, med],
            full_event_loss - gate_loss)
        weights = torch.where(med_mask, self.config.medication_weight, 1.)
        if self.config.medication_gate_weighting == 'combined':
            event_gate = self._reduce(gate_loss, patient, weights)
            event_conditional = self._reduce(conditional_loss, patient, weights)
            event = self._reduce(gate_loss + conditional_loss, patient, weights)
        else:
            event_gate = self._reduce(gate_loss, patient)
            event_conditional = self._reduce(conditional_loss, patient, weights)
            event = event_gate + event_conditional
        time = self._reduce(-self.time_distribution.log_prob(raw_time[mask], gap), patient)
        # Select medication positions before computing attributes; avoid empty means.
        dose = dur = hidden.sum() * 0
        if med_mask.any():
            subset = {'hidden': c['hidden'][med_mask]}
            logits, params = self.attributes(subset, med[med_mask], action[med_mask], gap[med_mask])
            y = batch['dose'][mask][med_mask]
            ce = F.binary_cross_entropy_with_logits(logits, y, reduction='none')
            pt = torch.where(y == 1, logits.sigmoid(), (-logits).sigmoid())
            dose = self._reduce((1 - pt) ** self.config.focal_gamma * ce, patient[med_mask])
            dur = self._reduce(-self.duration_head.log_prob(params, batch['dur'][mask][med_mask]), patient[med_mask])
        window = hidden.sum() * 0
        if self.window_head is not None:
            anchors = hidden[torch.arange(len(hidden), device=hidden.device), batch['anchor']]
            logits = self.window_head(anchors).reshape_as(batch['window_y'])
            valid = batch['window_valid']
            if valid.any():
                raw_window = F.binary_cross_entropy_with_logits(logits.float(), batch['window_y'], reduction='none')
                window_patient = torch.arange(len(hidden), device=hidden.device)[:, None, None].expand_as(valid)[valid]
                window = self._reduce(raw_window[valid], window_patient)
        losses = dict(event=event, event_gate=event_gate, event_conditional=event_conditional,
                      time=time, dose=dose, duration=dur, window=window, moe=aux)
        total = event + self.config.time_weight * time + self.config.dose_weight * dose
        total = total + self.config.duration_weight * dur + self.config.window_weight * window + self.config.moe_weight * aux
        return total, losses
