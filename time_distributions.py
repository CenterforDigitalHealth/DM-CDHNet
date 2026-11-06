"""Distribution interface. Reference adapters are not a final TIME selection.

All adapters describe INTEGER DAYS: separate zero mass, positive daily mass,
unbounded support. Likelihood, mean and sampling refer to the same law.
Future mixture implementations register here without changing EVENT heads.
"""
from abc import ABC, abstractmethod
import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeDistribution(nn.Module, ABC):
    @property
    @abstractmethod
    def output_dim(self): ...

    @abstractmethod
    def initialize(self, head): ...

    @abstractmethod
    def log_prob(self, raw, days): ...

    @abstractmethod
    def quantile(self, raw, u): ...

    @abstractmethod
    def mean(self, raw): ...

    @abstractmethod
    def cdf_before(self, raw, minimum):
        """P(integer gap < minimum), used for left-truncated rollout starts."""
        ...

    def sample(self, raw, generator=None):
        u = torch.rand(raw.shape[:-1], device=raw.device, generator=generator)
        return self.quantile(raw, u)

    def conditional_quantile(self, raw, u, minimum):
        """Integer quantile conditional on the gap being at least minimum."""
        below = self.cdf_before(raw, minimum)
        return self.quantile(raw, below + (1 - below) * u)


class ReferenceDailyTime(TimeDistribution):
    """Adapted from the matched 0907 daily-mass experiment; local-only dependencies."""
    def __init__(self, family):
        super().__init__()
        self.family = family
        upper = torch.logspace(0, math.log10(29220.), 62).round()
        for i in range(1, len(upper)):
            upper[i] = max(upper[i], upper[i - 1] + 1)
        self.register_buffer('upper', upper)
        self.register_buffer('lower', torch.cat([torch.ones(1), upper[:-1] + 1]))
        self.register_buffer('width', upper - torch.cat([torch.zeros(1), upper[:-1]]))

    @property
    def output_dim(self):
        return {'exponential': 2, 'weibull': 3, 'discrete': 64}[self.family]

    def initialize(self, head):
        nn.init.zeros_(head.weight)
        with torch.no_grad():
            head.bias.zero_()
            head.bias[0] = math.log(.25 / .75)
            if self.family == 'discrete':
                hazard = -torch.expm1(-self.width / 22.15)
                head.bias[1:-1].copy_(torch.logit(hazard.clamp(1e-6, 1 - 1e-6)))
                head.bias[-1] = math.log(math.expm1(1 / 22.15))
            else:
                head.bias[1] = math.log(22.15)

    def parameters_from(self, raw):
        raw = raw.float()
        return (raw[..., 0].sigmoid(), raw[..., 1].clamp(math.log(.05), math.log(1e5)),
                raw[..., 2].clamp(math.log(.15), math.log(5.)).exp()
                if self.family == 'weibull' else torch.ones_like(raw[..., 0]))

    def log_prob(self, raw, days):
        raw, days = raw.float(), days.float()
        if (days < 0).any() or (days != days.floor()).any():
            raise ValueError('TIME labels must be nonnegative integer days')
        d = days.clamp_min(1)
        if self.family != 'discrete':
            _, ls, k = self.parameters_from(raw)
            lo = torch.where(d > 1, (k * ((d - 1).clamp_min(1).log() - ls)).exp(), torch.zeros_like(d))
            inc = torch.where(d > 1, lo * torch.expm1(k * torch.log1p(1 / (d - 1).clamp_min(1))), (-k * ls).exp())
            positive = -lo + torch.log(-torch.expm1(-inc.clamp_min(1e-30)))
        else:
            before = torch.cat([torch.zeros_like(raw[..., :1]), F.logsigmoid(-raw[..., 1:-1]).cumsum(-1)], -1)
            ix = torch.bucketize(d.contiguous(), self.upper).clamp_max(62)
            safe = ix.clamp_max(61)
            gather = lambda x, idx: x.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            finite = gather(before, ix) + gather(F.logsigmoid(raw[..., 1:-1]), safe) - self.width[safe].log()
            tail = before[..., -1] + F.logsigmoid(raw[..., -1]) + (d - self.upper[-1] - 1) * F.logsigmoid(-raw[..., -1])
            positive = torch.where(ix < 62, finite, tail)
        return torch.where(days == 0, F.logsigmoid(raw[..., 0]), F.logsigmoid(-raw[..., 0]) + positive)

    def quantile(self, raw, u):
        raw, u = raw.float(), u.float().clamp(1e-7, 1 - 1e-7)
        rho = raw[..., 0].sigmoid()
        v = ((u - rho) / (1 - rho).clamp_min(1e-7)).clamp(1e-7, 1 - 1e-7)
        if self.family != 'discrete':
            _, ls, k = self.parameters_from(raw)
            positive = (ls + torch.log(-torch.log1p(-v)) / k).exp().ceil().clamp_min(1)
        else:
            before = torch.cat([torch.zeros_like(raw[..., :1]), F.logsigmoid(-raw[..., 1:-1]).cumsum(-1)], -1)
            mass = (before[..., :-1] + F.logsigmoid(raw[..., 1:-1])).exp()
            cdf = mass.cumsum(-1)
            ix = (v.unsqueeze(-1) > cdf).sum(-1).clamp_max(62)
            safe = ix.clamp_max(61)
            gather = lambda x, idx: x.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            previous = torch.cat([torch.zeros_like(cdf[..., :1]), cdf[..., :-1]], -1)
            frac = ((v - gather(previous, safe)) / gather(mass, safe).clamp_min(1e-30)).clamp(0, 1)
            finite = self.lower[safe] + (frac * self.width[safe]).ceil().clamp_min(1) - 1
            tail_u = ((v - cdf[..., -1]) / before[..., -1].exp().clamp_min(1e-30)).clamp(1e-7, 1 - 1e-7)
            tail = self.upper[-1] + (torch.log1p(-tail_u) / F.logsigmoid(-raw[..., -1]).clamp_max(-1e-12)).ceil().clamp_min(1)
            positive = torch.where(ix < 62, finite, tail)
        return torch.where(u <= rho, torch.zeros_like(positive), positive)

    def cdf_before(self, raw, minimum):
        """CDF immediately below an integer lower bound."""
        raw, minimum = raw.float(), minimum.float().clamp_min(0)
        rho = raw[..., 0].sigmoid()
        threshold = (minimum - 1).clamp_min(0)
        if self.family != 'discrete':
            _, ls, k = self.parameters_from(raw)
            positive = -torch.expm1(-torch.exp(
                k * (threshold.clamp_min(1).log() - ls)))
            positive = torch.where(threshold > 0, positive, torch.zeros_like(positive))
        else:
            before = torch.cat(
                [torch.zeros_like(raw[..., :1]), F.logsigmoid(-raw[..., 1:-1]).cumsum(-1)], -1)
            mass = (before[..., :-1] + F.logsigmoid(raw[..., 1:-1])).exp()
            cumulative = mass.cumsum(-1)
            ix = torch.bucketize(threshold.contiguous(), self.upper).clamp_max(62)
            safe = ix.clamp_max(61)
            gather = lambda x, idx: x.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
            previous = torch.cat([torch.zeros_like(cumulative[..., :1]), cumulative[..., :-1]], -1)
            fraction = ((threshold - self.lower[safe] + 1) / self.width[safe]).clamp(0, 1)
            finite = gather(previous, safe) + fraction * gather(mass, safe)
            tail_survival = before[..., -1].exp() * torch.exp(
                (threshold - self.upper[-1]).clamp_min(0) * F.logsigmoid(-raw[..., -1]))
            positive = torch.where(ix < 62, finite, 1 - tail_survival)
            positive = torch.where(threshold > 0, positive, torch.zeros_like(positive))
        result = rho + (1 - rho) * positive
        return torch.where(minimum <= 0, torch.zeros_like(result), result).clamp(0, 1 - 1e-7)

    def conditional_quantile(self, raw, u, minimum):
        raw, u = raw.float(), u.float().clamp(1e-7, 1 - 1e-7)
        minimum = minimum.float().clamp_min(0)
        if self.family != 'discrete':
            _, ls, k = self.parameters_from(raw)
            threshold = (minimum - 1).clamp_min(0)
            cumulative_hazard = torch.pow(threshold * torch.exp(-ls), k)
            continuous = torch.exp(ls) * torch.pow(
                cumulative_hazard - torch.log1p(-u), 1 / k)
            positive = continuous.ceil().clamp_min(1)
            # Roundoff at large lower bounds can place ceil(continuous) one day
            # below the requested integer truncation point.
            positive = torch.maximum(positive, minimum)
            return torch.where(minimum > 0, positive, self.quantile(raw, u))
        # Work in log survival space so long left truncation does not round its
        # CDF to one in float32. Each finite hazard bin is uniform over days.
        before = torch.cat(
            [torch.zeros_like(raw[..., :1]), F.logsigmoid(-raw[..., 1:-1]).cumsum(-1)], -1)
        lower = torch.maximum(minimum[..., None], self.lower)
        allowed = (self.upper - lower + 1).clamp_min(0)
        finite_log_weight = (before[..., :-1] + F.logsigmoid(raw[..., 1:-1]) -
                             self.width.log() + allowed.clamp_min(1).log())
        finite_log_weight = finite_log_weight.masked_fill(allowed == 0, -torch.inf)
        tail_start = torch.maximum(minimum, self.upper[-1] + 1)
        tail_log_weight = (before[..., -1] +
                           (tail_start - self.upper[-1] - 1) * F.logsigmoid(-raw[..., -1]))
        weights = torch.cat([finite_log_weight, tail_log_weight[..., None]], -1).softmax(-1)
        cdf = weights.cumsum(-1)
        component = (u[..., None] > cdf).sum(-1).clamp_max(62)
        previous = torch.cat([torch.zeros_like(cdf[..., :1]), cdf[..., :-1]], -1)
        gather = lambda x, idx: x.gather(-1, idx.unsqueeze(-1)).squeeze(-1)
        mass = gather(weights, component).clamp_min(1e-30)
        local = ((u - gather(previous, component)) / mass).clamp(0, 1 - 1e-7)
        safe = component.clamp_max(61)
        finite_count = gather(allowed, safe).clamp_min(1)
        finite = gather(lower, safe) + torch.floor(local * finite_count)
        tail = tail_start + torch.floor(
            torch.log1p(-local) / F.logsigmoid(-raw[..., -1]).clamp_max(-1e-12))
        conditioned = torch.where(component < 62, finite, tail)
        conditioned = torch.maximum(conditioned, minimum)
        return torch.where(minimum > 0, conditioned, self.quantile(raw, u))

    def mean(self, raw):
        raw = raw.float()
        rho = raw[..., 0].sigmoid()
        if self.family == 'exponential':
            _, ls, _ = self.parameters_from(raw)
            return (1 - rho) / (-torch.expm1(-torch.exp(-ls)))
        if self.family == 'discrete':
            before = torch.cat([torch.zeros_like(raw[..., :1]), F.logsigmoid(-raw[..., 1:-1]).cumsum(-1)], -1)
            mass = (before[..., :-1] + F.logsigmoid(raw[..., 1:-1])).exp()
            finite = (mass * (self.lower + self.upper) / 2).sum(-1)
            tail = before[..., -1].exp() * (self.upper[-1] + torch.exp(-F.logsigmoid(raw[..., -1])))
            return (1 - rho) * (finite + tail)
        # Weibull daily mean uses fixed quantile quadrature, explicitly approximate.
        return torch.stack([self.quantile(raw, torch.full_like(rho, (i + .5) / 128)) for i in range(128)]).mean(0)


REGISTRY = {name: (lambda name=name: ReferenceDailyTime(name)) for name in ('exponential', 'weibull', 'discrete')}


def register_time_distribution(name, factory):
    if name in REGISTRY:
        raise ValueError(f'TIME adapter already registered: {name}')
    REGISTRY[name] = factory


def make_time_distribution(name):
    if name not in REGISTRY:
        raise ValueError(f'Choose an explicit implemented TIME adapter from {sorted(REGISTRY)}; got {name!r}')
    return REGISTRY[name]()
