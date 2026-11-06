"""Portable, strict JSON configuration. No experiment directory is imported."""
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path


@dataclass
class ModelConfig:
    vocab_size: int = 1289
    context_length: int = 512
    n_layer: int = 12
    n_embd: int = 384
    n_head: int = 12
    n_kv_head: int = 4
    n_experts: int = 8
    experts_per_token: int = 2
    local_window: int = 128
    dropout: float = 0.2
    # Deliberately unresolved: choose explicitly for each experiment.
    time_family: str | None = None
    duration_components: int = 16
    duration_max: int = 550
    recent_days: float = 90.0
    medication_weight: float = 1.5
    dose_weight: float = 10.0
    duration_weight: float = 1.0
    time_weight: float = 1.0
    window_weight: float = 0.1
    moe_weight: float = 0.01
    focal_gamma: float = 2.0
    action_soft_penalty: float = 3.0
    # `combined` reproduces the initial 0909 weighted joint EVENT loss.
    # `conditional_only` keeps medication occurrence calibration unweighted.
    medication_gate_weighting: str = 'combined'
    # `event` averages eligible events; `patient` first averages within patient.
    loss_reduction: str = 'event'
    window_top_k: int = 128
    window_tokens: list[int] = field(default_factory=list)
    horizons_days: list[float] = field(default_factory=lambda: [30., 90., 365.25, 1095.75, 1826.25])

    def validate(self):
        if self.vocab_size != 1289:
            raise ValueError('0909 uses the explicit 1289-token contract')
        if min(self.n_embd, self.n_head, self.n_kv_head, self.n_experts) < 1:
            raise ValueError('Attention/expert dimensions must be positive')
        if self.n_embd % self.n_head or self.n_head % self.n_kv_head or (self.n_embd // self.n_head) % 2:
            raise ValueError('GQA requires width/heads integral, even head dimension, Q heads divisible by KV heads')
        if not 1 <= self.experts_per_token <= self.n_experts:
            raise ValueError('Invalid MoE routing sizes')
        if min(self.n_layer, self.context_length, self.local_window, self.duration_components, self.duration_max) < 1:
            raise ValueError('Model dimensions must be positive')
        if not 0 <= self.dropout < 1 or self.recent_days <= 0:
            raise ValueError('Invalid dropout/recent_days')
        if self.window_top_k < 0 or min(self.medication_weight, self.dose_weight, self.duration_weight,
                self.time_weight, self.window_weight, self.moe_weight, self.focal_gamma, self.action_soft_penalty) < 0:
            raise ValueError('Loss weights and window_top_k must be nonnegative')
        if self.medication_gate_weighting not in ('combined', 'conditional_only'):
            raise ValueError('medication_gate_weighting must be combined or conditional_only')
        if self.loss_reduction not in ('event', 'patient'):
            raise ValueError('loss_reduction must be event or patient')
        if any(not 22 <= t <= 1277 for t in self.window_tokens) or len(set(self.window_tokens)) != len(self.window_tokens):
            raise ValueError('Window endpoints must be unique disease tokens')
        if sorted(set(self.horizons_days)) != self.horizons_days or any(h <= 0 for h in self.horizons_days):
            raise ValueError('Horizons must be positive, unique and increasing')
        if self.time_family is None:
            raise ValueError('TIME distribution is not selected. Pass --time-family explicitly; see docs/TIME.md')


@dataclass
class TrainConfig:
    max_steps: int = 20000
    batch_size: int = 16
    accumulation_steps: int = 8
    learning_rate: float = 2e-4
    min_learning_rate: float = 2e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 250
    val_batches: int = 40
    early_stopping_steps: int = 1500
    seed: int = 20260909
    precision: str = 'bf16'
    # `left` is the initial prefix; `random_day` samples a longitudinal cutoff.
    landmark_sampling: str = 'left'


def load_config(path):
    payload = json.loads(Path(path).read_text())
    if set(payload) - {'model', 'training'}:
        raise ValueError('Config sections are model and training only')
    result = []
    for section, cls in [('model', ModelConfig), ('training', TrainConfig)]:
        values = payload.get(section, {})
        unknown = set(values) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'Unknown {section} settings: {sorted(unknown)}')
        result.append(cls(**values))
    return tuple(result)


def config_dict(model, training):
    return {'model': asdict(model), 'training': asdict(training)}
