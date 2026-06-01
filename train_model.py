"""
Composite Delphi Training Script
Multi-GPU: Auto-detects GPUs and uses DDP (DistributedDataParallel)
"""

import os
import sys
import time
import math
import pickle
import fnmatch
import json
import subprocess
from datetime import datetime
from contextlib import nullcontext

import numpy as np
import torch
import torch.nn.functional as F

# =============================================================================
# Model Size Presets
# =============================================================================
MODEL_SIZE_PRESETS = {
    "small": {
        "n_layer": 8,
        "n_head": 8,
        "n_kv_head": 4,
        "n_embd": 256,
        "dropout": 0.2,
        "batch_size": 128,
        "gradient_accumulation_steps": 1,
    },
    "medium": {
        "n_layer": 12,
        "n_head": 12,
        "n_kv_head": 4,
        "n_embd": 384,
        "dropout": 0.2,
        "batch_size": 96,
        "gradient_accumulation_steps": 1,
    },
    "large": {
        "n_layer": 16,
        "n_head": 16,
        "n_kv_head": 4,
        "n_embd": 512,
        "dropout": 0.2,
        "batch_size": 48,
        "gradient_accumulation_steps": 2,
    },
}
MODEL_SIZE_KEYS = tuple(next(iter(MODEL_SIZE_PRESETS.values())).keys())
MODEL_ARCH_KEYS = ("n_layer", "n_head", "n_kv_head", "n_embd")


def _get_cli_arg_value(key: str):
    prefix = f"--{key}="
    for arg in sys.argv[1:]:
        if arg.startswith(prefix):
            return arg[len(prefix):]
    return None


def _apply_model_size_preset(size_name: str) -> str:
    size_name = str(size_name).lower()
    if size_name == "custom":
        return size_name
    if size_name not in MODEL_SIZE_PRESETS:
        valid = ", ".join(sorted(list(MODEL_SIZE_PRESETS.keys()) + ["custom"]))
        raise ValueError(f"Unknown model_size '{size_name}'. Valid options: {valid}")
    preset = MODEL_SIZE_PRESETS[size_name]
    for key, value in preset.items():
        globals()[key] = value
    return size_name


# =============================================================================
# Auto Multi-GPU Detection & DDP Launch
# =============================================================================
def _auto_ddp():
    """Auto-detect multiple GPUs and re-launch with torchrun for DDP training.
    
    If multiple GPUs are available and we're not already running under torchrun,
    re-launches the script via torchrun with --nproc_per_node=<num_gpus>.
    Skipped if user specifies --gpu_id (explicit single-GPU mode).
    """
    if 'RANK' in os.environ:
        return  # Already running under torchrun/DDP
    
    # If user explicitly specified a single GPU, skip DDP
    for arg in sys.argv[1:]:
        if arg.startswith('--gpu_id='):
            return
    
    n_gpus = torch.cuda.device_count()
    if n_gpus <= 1:
        return
    
    import subprocess
    import random
    port = random.randint(29500, 29999)
    script = os.path.abspath(__file__)
    
    print(f"\n{'='*60}")
    print(f"  Auto-detected {n_gpus} GPUs → launching DDP training")
    for i in range(n_gpus):
        print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Master port: {port}")
    print(f"{'='*60}\n")
    
    cmd = [
        sys.executable, '-m', 'torch.distributed.run',
        f'--nproc_per_node={n_gpus}',
        f'--master_port={port}',
        script,
    ] + sys.argv[1:]
    
    sys.exit(subprocess.run(cmd).returncode)

_auto_ddp()

from model import CompositeDelphi, CompositeDelphiConfig
from utils import get_p2i_composite, get_batch_composite

# =============================================================================
# Default Configuration
# =============================================================================

out_dir = 'out'
out_dir_use_timestamp = True  # when out_dir=='out' and scratch, save to out/MMDD/HHMM
eval_interval = 500
log_interval = 100
eval_iters = 200
eval_only = False
always_save_checkpoint = False
save_eval_checkpoints = False
save_eval_checkpoint_interval = 1000
save_initial_checkpoint = False
init_from = 'scratch'  # 'scratch', 'resume', 'finetune', or 'legacy_partial'
finetune_ckpt_path = ''  # warm-start weights from this checkpoint when init_from='finetune'
seed = 42

# wandb logging
wandb_log = False
wandb_project = 'composite-delphi'
wandb_run_name = 'run' + str(time.time())

# data
gradient_accumulation_steps = 1
batch_size = 96
block_size = 512

# model selection (composite only)
model_type = 'composite'
model_size = 'small'  # 'small' | 'medium' | 'large' | 'custom'

# Model config
n_layer = 12
n_head = 12
n_kv_head = 4  # GQA (must divide n_head evenly: 12/4=3 heads per group)
n_embd = 384
dropout = 0.2
bias = False

# Allow one-line architecture preset from CLI, e.g. --model_size=small
# Manual architecture args (e.g. --n_layer=10) still override this later via configurator.
_cli_model_size = _get_cli_arg_value('model_size')
if _cli_model_size is not None:
    model_size = _cli_model_size
model_size = _apply_model_size_preset(model_size)
_pre_config_model_size = model_size
_arch_before_configurator = {k: globals()[k] for k in MODEL_ARCH_KEYS}

# Composite Delphi model config (5-column data)
data_vocab_size = 1289   # DATA: 약품/질병 코드 수 (Classification) - raw token range 2~1288
shift_vocab_size = 4     # SHIFT raw space: 0=N/A/no-event/pad, 1=decrease, 2=maintain, 3=increase
total_vocab_size = 551   # TOTAL: Embedding vocab - raw range 0~550

# SHIFT imbalance handling
shift_loss_type = 'dice_focal'      # 'dice_focal', 'focal', 'ce'
shift_dice_weight = 0.5
shift_ignore_index = -1
shift_focal_gamma = 2.0  # Reduced from 5.0 to standard value to prevent hallucinations
shift_class_weights = []  # Empty list = auto-computed
shift_maintain_idx = 2
shift_change_weight_max = 10.0
shift_class_weight_cap = 8.0
num_shift_classes = 2   # 2-class: label1(0) / label2or3(1)
drug_token_only_shift = True  # compute SHIFT loss only at drug token positions
drug_token_only_total = True  # compute TOTAL loss only at drug token positions

# TOTAL MDN settings
mdn_n_components = 8
mdn_log_s_min = -1.0
total_min_value = 0.0
total_max_value = 550.0

# Loss weights for composite model
loss_weight_data = 1.0
loss_weight_shift = 20.0
loss_weight_total = 5.0
loss_weight_time = 1.0

# architecture features
use_moe = True
num_experts = 8
experts_per_token = 2
moe_expert_type = 'swiglu'
sliding_window = 128

# Drug-conditioning
use_drug_conditioning = True
film_dropout = 0.0
film_in_backbone = False
use_teacher_forcing_drug_cond = False
data_label_smoothing = 0.0
data_loss_static_weight = 1.0
data_loss_disease_weight = 1.0
data_loss_drug_weight = 1.0
horizon_disease_aux_enabled = False
horizon_disease_aux_weight = 0.0
horizon_disease_aux_offset_days = 365.25
horizon_disease_aux_pos_weight = 20.0
horizon_disease_aux_max_tokens = 128
tie_data_head = True
rope_theta = 10000.0

# Uncertainty-weighted multi-task learning (Kendall 2018)
use_uncertainty_weighting = False

# TOTAL regression (legacy fallback only; MDN path does not use this)
total_log_transform = False

# adamw optimizer
learning_rate = 6e-4
max_iters = 20000        # Increased from 10000
# early stopping
# Stop when validation loss has not improved for this many iterations.
# Set <= 0 to disable early stopping.
early_stop_patience_iters = 1000
# max_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0
trainable_parameter_patterns = []  # fnmatch patterns; when set, only matching params train
frozen_parameter_patterns = []     # fnmatch patterns; applied after trainable selection

# learning rate decay settings
decay_lr = True
warmup_iters = 1000
lr_decay_iters = 19000   # Adjusted for 20000 max_iters
min_lr = 3e-5

# system
gpu_id = 0  # GPU device ID (e.g., 0, 1, 2, ...)
device = 'cpu'  # Will be set after config parsing
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float32'
compile = False  # torch.compile (requires PyTorch 2.0+)

# delphi training
token_dropout = 0.0
t_min = 0.1  # Prevent log(0) numerical instability
mask_ties = True
# Raw DATA tokens 0-21 are technical/static covariates (padding, no-event,
# sex, lifestyle/vitals); the first disease code starts at raw index 22.
ignore_tokens = list(range(22))
eot_token = None  # set via config; end-of-trajectory censoring token id
data_fraction = 1.0
no_event_token_rate = 5
apply_token_shift = False
separate_shift_na_from_padding = False
shift_na_raw_token = 4

# Time-to-Event distribution: 'exponential' or 'weibull'
time_distribution = 'exponential'

use_inner_train_val_split = True
INNER_SPLIT_SOURCE_PATH = '../data/kr_train.bin'
TRAIN_DATA_PATH = '../data/train_inner.bin'
VAL_DATA_PATH = '../data/val_inner.bin'
INTERNAL_TEST_DATA_PATH = '../data/kr_val.bin'
inner_val_fraction = 0.2
inner_split_seed = 42
rebuild_inner_split = False
run_internal_test_after_training = True
internal_test_iters = 200
# JMDC path for domain generalization (mixing)
JMDC_DATA_PATH = '../data/JMDC_extval.bin'

# Disease-AUC checkpoint selection on held-out KR validation.
# This is a lightweight proxy for the full evaluate_auc.py run: it samples a
# fixed patient subset and a fixed set of frequent disease tokens so training
# can track ranking quality without running the full external evaluator.
auc_eval_enabled = False
auc_eval_interval = 1000
auc_eval_data_path = ''
auc_eval_subset_patients = 1024
auc_eval_batch_size = 128
auc_eval_max_tokens = 64
auc_eval_min_cases = 5
auc_eval_min_controls = 20
auc_eval_select = 'random'
auc_eval_padding = 'regular'
auc_eval_seed = 42
auc_eval_exclude_control_tokens = []
checkpoint_selection_metric = 'val_loss'  # 'val_loss', 'auc_mean', or 'auc_median'
save_loss_best_checkpoint = True

# Optional full evaluator hook. Unlike auc_eval_enabled, this calls
# evaluate_auc.py on saved ckpt_eval_N.pt files and can be used for checkpoint
# selection against the same kr_val_pp disease AUC reported after training.
full_auc_eval_enabled = False
full_auc_eval_interval = 1000
full_auc_eval_input_path = '../data'
full_auc_eval_data_file = 'kr_val_pp.bin'
full_auc_eval_train_data_file = 'train_inner_pp.bin'
full_auc_eval_output_root = ''
full_auc_eval_dataset_subset_size = 10000
full_auc_eval_batch_size = 128
full_auc_eval_exclude_eot_from_controls = True
full_auc_eval_extra_args = []

# -----------------------------------------------------------------------------
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, list))]
exec(open('configurator.py').read())

# Support model_size inside config files when architecture keys are not explicitly set there.
# If CLI already provided --model_size, pre-config application is sufficient.
if _cli_model_size is None:
    _post_config_model_size = str(model_size).lower()
    if _post_config_model_size != _pre_config_model_size:
        _arch_after_configurator = {k: globals()[k] for k in MODEL_ARCH_KEYS}
        if _arch_after_configurator == _arch_before_configurator:
            model_size = _apply_model_size_preset(_post_config_model_size)

def _required_shift_vocab_size(apply_token_shift: bool, separate_shift_na_from_padding: bool) -> int:
    if not separate_shift_na_from_padding:
        return 4
    return 6 if apply_token_shift else 5


required_shift_vocab_size = _required_shift_vocab_size(
    apply_token_shift=apply_token_shift,
    separate_shift_na_from_padding=separate_shift_na_from_padding,
)
if shift_vocab_size < required_shift_vocab_size:
    shift_vocab_size = required_shift_vocab_size

config = {k: globals()[k] for k in config_keys}
# -----------------------------------------------------------------------------

if model_type != 'composite':
    raise ValueError("Only composite model_type is supported.")

# =============================================================================
# DDP Setup & Device Configuration
# =============================================================================
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(ddp_local_rank)
    master_process = (ddp_rank == 0)
    seed_offset = ddp_rank
    if master_process:
        print(f"DDP training: {ddp_world_size} GPUs")
        for i in range(ddp_world_size):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    ddp_world_size = 1
    master_process = True
    seed_offset = 0
    if torch.cuda.is_available():
        device = f'cuda:{gpu_id}'
        torch.cuda.set_device(gpu_id)
        if master_process:
            print(f"Using GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("Using MPS (Apple Silicon)")
    else:
        device = 'cpu'
        print("Using CPU")

# Resolve final checkpoint directory
# - default behavior: out -> out/MMDD/HHMM for scratch runs
# - keep explicit out_dir values unchanged (important for ablation scripts)
if (
    init_from == 'scratch'
    and out_dir_use_timestamp
    and os.path.normpath(out_dir) == 'out'
):
    run_timestamp = os.environ.get('TRAIN_RUN_TIMESTAMP')
    if run_timestamp is None:
        if ddp:
            now = datetime.now()
            obj = [now.strftime('%m%d/%H%M') if master_process else None]
            dist.broadcast_object_list(obj, src=0)
            run_timestamp = obj[0]
        else:
            run_timestamp = datetime.now().strftime('%m%d/%H%M')
        os.environ['TRAIN_RUN_TIMESTAMP'] = run_timestamp
    out_dir = os.path.join(out_dir, run_timestamp)

# Keep saved config aligned with the effective checkpoint directory
config['out_dir'] = out_dir

tokens_per_iter = gradient_accumulation_steps * batch_size * block_size * ddp_world_size
if master_process:
    print(
        f"Model size: {model_size} | "
        f"layers={n_layer}, heads={n_head}, kv_heads={n_kv_head}, embd={n_embd}, "
        f"batch_size={batch_size}, grad_acc={gradient_accumulation_steps}"
    )
    print(f"Checkpoint directory: {out_dir}")
    print(f"Tokens per iteration: {tokens_per_iter:,} ({ddp_world_size} GPU(s))")

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else ('mps' if 'mps' in device else 'cpu')
ptdtype = {'float32': torch.float32, 'float64': torch.float64,
           'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

torch.set_default_dtype(ptdtype)

# =============================================================================
# Data Loading
# =============================================================================

# data_dir = '../data'

def _compute_shift_class_weights(shift_values, n_classes, shift_ignore_index, beta=0.9999):
    """Effective number of samples class weighting (Cui et al. CVPR 2019).

    weight_i = (1 - beta) / (1 - beta^n_i), normalized so min weight = 1.0.
    """
    counts = np.bincount(shift_values, minlength=n_classes).astype(np.float64)
    if shift_ignore_index is not None and 0 <= shift_ignore_index < n_classes:
        counts[shift_ignore_index] = 0.0
    nonzero = counts > 0
    weights = np.zeros(n_classes, dtype=np.float32)
    if nonzero.any():
        effective_num = 1.0 - np.power(beta, counts[nonzero])
        w = (1.0 - beta) / effective_num
        # Normalize so minimum weight = 1.0
        w = w / w.min()
        cap = float(globals().get('shift_class_weight_cap', 8.0))
        weights[nonzero] = np.clip(w, 1.0, cap).astype(np.float32)
    return weights.tolist()


def _remap_shift_to_change_np(shift_values: np.ndarray, shifted: bool, num_classes: int = 3) -> np.ndarray:
    """
    Remap SHIFT labels to class indices.

    num_classes=3: decrease(0) / maintain(1) / increase(2)
      raw labels: 1->0, 2->1, 3->2
    num_classes=2 (binary fallback): label1->0, label2/3->1
    """
    out = np.full(shift_values.shape, -1, dtype=np.int64)
    if shifted:
        is_label1 = shift_values == 2
        is_label2 = shift_values == 3
        is_label3 = shift_values == 4
    else:
        is_label1 = shift_values == 1
        is_label2 = shift_values == 2
        is_label3 = shift_values == 3
    if num_classes == 3:
        out[is_label1] = 0  # decrease
        out[is_label2] = 1  # maintain
        out[is_label3] = 2  # increase
    else:
        out[is_label1] = 0
        out[is_label2 | is_label3] = 1
    return out

# 6-column structured data: (ID, AGE, DATA, DOSE, TOTAL, UNIT)
# composite_dtype = np.dtype([
#     ('ID', '<u4'),
#     ('AGE', '<u4'),
#     ('DATA', '<u4'),
#     ('DOSE', '<f4'),
#     ('TOTAL', '<u4'),
#     ('UNIT', '<u4')
# ])
composite_dtype = np.dtype([
    ('ID', np.uint32),
    ('AGE', np.uint32),
    ('DATA', np.uint32),
    ('SHIFT', np.uint32),
    ('TOTAL', np.uint32)
])


def _file_signature(path):
    st = os.stat(path)
    return {
        'path': os.path.abspath(path),
        'size_bytes': st.st_size,
        'mtime_ns': st.st_mtime_ns,
    }


def _inner_split_meta_path(train_path):
    return f"{train_path}.split_meta.pkl"


def _expected_inner_split_meta(source_path, train_path, val_path, dtype, val_fraction, split_seed):
    return {
        'source': _file_signature(source_path),
        'train_path': os.path.abspath(train_path),
        'val_path': os.path.abspath(val_path),
        'dtype_descr': dtype.descr,
        'val_fraction': float(val_fraction),
        'split_seed': int(split_seed),
    }


def _inner_split_is_current(source_path, train_path, val_path, dtype, val_fraction, split_seed):
    if not (os.path.exists(train_path) and os.path.exists(val_path)):
        return False
    meta_path = _inner_split_meta_path(train_path)
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, 'rb') as f:
            actual = pickle.load(f)
        expected = _expected_inner_split_meta(
            source_path, train_path, val_path, dtype, val_fraction, split_seed
        )
    except Exception:
        return False
    for key, expected_value in expected.items():
        if actual.get(key) != expected_value:
            return False
    return True


def _write_patient_subset(data, p2i, patient_indices, output_path):
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp_path = f"{output_path}.tmp.{os.getpid()}"
    rows_written = 0
    with open(tmp_path, 'wb') as f:
        for patient_idx in patient_indices:
            start_idx, length = p2i[int(patient_idx)]
            chunk = data[int(start_idx):int(start_idx + length)]
            chunk.tofile(f)
            rows_written += int(length)
    os.replace(tmp_path, output_path)
    return rows_written


def _prepare_inner_train_val_split(source_path, train_path, val_path, dtype,
                                   val_fraction, split_seed, force_rebuild=False):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Inner split source not found: {source_path}")
    source_abs = os.path.abspath(source_path)
    train_abs = os.path.abspath(train_path)
    val_abs = os.path.abspath(val_path)
    if train_abs == val_abs:
        raise ValueError("TRAIN_DATA_PATH and VAL_DATA_PATH must be different for inner split")
    if source_abs in (train_abs, val_abs):
        raise ValueError("Inner split outputs must not overwrite INNER_SPLIT_SOURCE_PATH")
    if not (0.0 < float(val_fraction) < 1.0):
        raise ValueError(f"inner_val_fraction must be between 0 and 1, got {val_fraction}")
    if not force_rebuild and _inner_split_is_current(
        source_path, train_path, val_path, dtype, val_fraction, split_seed
    ):
        return False

    source_data = np.fromfile(source_path, dtype=dtype)
    source_p2i = get_p2i_composite(source_data)
    num_patients = len(source_p2i)
    if num_patients < 2:
        raise ValueError(f"Need at least 2 patients for inner split, got {num_patients}")

    rng = np.random.default_rng(int(split_seed))
    patient_perm = rng.permutation(num_patients)
    num_val = int(round(num_patients * float(val_fraction)))
    num_val = min(max(1, num_val), num_patients - 1)
    val_patient_indices = np.sort(patient_perm[:num_val])
    train_patient_indices = np.sort(patient_perm[num_val:])

    train_rows = _write_patient_subset(source_data, source_p2i, train_patient_indices, train_path)
    val_rows = _write_patient_subset(source_data, source_p2i, val_patient_indices, val_path)

    meta = _expected_inner_split_meta(source_path, train_path, val_path, dtype, val_fraction, split_seed)
    meta.update({
        'source_patients': int(num_patients),
        'train_patients': int(len(train_patient_indices)),
        'val_patients': int(len(val_patient_indices)),
        'train_rows': int(train_rows),
        'val_rows': int(val_rows),
    })
    with open(_inner_split_meta_path(train_path), 'wb') as f:
        pickle.dump(meta, f)
    return True

# train_data = np.memmap(TRAIN_DATA_PATH, dtype=composite_dtype, mode='r')
# val_data = np.memmap(VAL_DATA_PATH, dtype=composite_dtype, mode='r')
if use_inner_train_val_split:
    if master_process:
        rebuilt_inner_split = _prepare_inner_train_val_split(
            INNER_SPLIT_SOURCE_PATH,
            TRAIN_DATA_PATH,
            VAL_DATA_PATH,
            composite_dtype,
            inner_val_fraction,
            inner_split_seed,
            force_rebuild=rebuild_inner_split,
        )
        action = "rebuilt" if rebuilt_inner_split else "reused"
        print(
            f"Inner patient split {action}: "
            f"{INNER_SPLIT_SOURCE_PATH} -> {TRAIN_DATA_PATH}, {VAL_DATA_PATH} "
            f"(train/val={1.0 - inner_val_fraction:.1f}/{inner_val_fraction:.1f}, "
            f"seed={inner_split_seed})"
        )
        print(f"Internal test reserved (not used for early stopping): {INTERNAL_TEST_DATA_PATH}")
    if ddp:
        dist.barrier()

train_data = np.fromfile(TRAIN_DATA_PATH, dtype=composite_dtype)
val_data = np.fromfile(VAL_DATA_PATH, dtype=composite_dtype)

train_p2i = get_p2i_composite(train_data)
val_p2i = get_p2i_composite(val_data)

auc_eval_data = None
auc_eval_p2i = None
auc_eval_patient_indices = None
auc_eval_token_ids = None
auc_eval_patient_token_sets = None
horizon_aux_token_ids = None
if auc_eval_enabled:
    if not auc_eval_data_path:
        auc_eval_data_path = INTERNAL_TEST_DATA_PATH
        config['auc_eval_data_path'] = auc_eval_data_path
    if os.path.exists(auc_eval_data_path):
        auc_eval_data = np.fromfile(auc_eval_data_path, dtype=composite_dtype)
        auc_eval_p2i = get_p2i_composite(auc_eval_data)

        rng = np.random.default_rng(int(auc_eval_seed))
        n_auc_patients = min(int(auc_eval_subset_patients), len(auc_eval_p2i))
        if n_auc_patients > 0:
            auc_eval_patient_indices = np.sort(
                rng.choice(len(auc_eval_p2i), size=n_auc_patients, replace=False)
            )

        offset = 1 if apply_token_shift else 0
        disease_min = 22 + offset
        disease_max_exclusive = 1279 if apply_token_shift else 1278
        raw_tokens = auc_eval_data['DATA'].astype(np.int64)
        if apply_token_shift:
            raw_tokens = raw_tokens + 1
        values, counts = np.unique(raw_tokens, return_counts=True)
        disease_mask = (values >= disease_min) & (values < disease_max_exclusive)
        token_counts = sorted(
            zip(values[disease_mask].tolist(), counts[disease_mask].tolist()),
            key=lambda x: x[1],
            reverse=True,
        )
        auc_eval_token_ids = np.array(
            [int(tok) for tok, _ in token_counts[:max(1, int(auc_eval_max_tokens))]],
            dtype=np.int64,
        )
        horizon_aux_token_ids = np.array(
            [int(tok) for tok, _ in token_counts[:max(1, int(horizon_disease_aux_max_tokens))]],
            dtype=np.int64,
        )

        candidate_set = set(auc_eval_token_ids.tolist())
        auc_eval_patient_token_sets = {int(tok): set() for tok in auc_eval_token_ids.tolist()}
        if auc_eval_patient_indices is not None:
            for patient_idx in auc_eval_patient_indices:
                start_idx, length = auc_eval_p2i[int(patient_idx)]
                patient_tokens = auc_eval_data['DATA'][int(start_idx):int(start_idx + length)].astype(np.int64)
                if apply_token_shift:
                    patient_tokens = patient_tokens + 1
                for tok in np.unique(patient_tokens):
                    tok = int(tok)
                    if tok in candidate_set:
                        auc_eval_patient_token_sets[tok].add(int(patient_idx))
    elif master_process:
        print(f"[auc-proxy] disabled: file not found: {auc_eval_data_path}")
        auc_eval_enabled = False
        config['auc_eval_enabled'] = False

if horizon_disease_aux_enabled and horizon_aux_token_ids is None:
    raw_tokens_for_aux = train_data['DATA'].astype(np.int64)
    if apply_token_shift:
        raw_tokens_for_aux = raw_tokens_for_aux + 1
    offset = 1 if apply_token_shift else 0
    disease_min = 22 + offset
    disease_max_exclusive = 1279 if apply_token_shift else 1278
    values, counts = np.unique(raw_tokens_for_aux, return_counts=True)
    disease_mask = (values >= disease_min) & (values < disease_max_exclusive)
    token_counts = sorted(
        zip(values[disease_mask].tolist(), counts[disease_mask].tolist()),
        key=lambda x: x[1],
        reverse=True,
    )
    horizon_aux_token_ids = np.array(
        [int(tok) for tok, _ in token_counts[:max(1, int(horizon_disease_aux_max_tokens))]],
        dtype=np.int64,
    )

if master_process:
    print(f"Loaded composite data: train={len(train_data)} ({TRAIN_DATA_PATH}), val={len(val_data)} ({VAL_DATA_PATH})")
    print(f"Unique patients: train={len(train_p2i)}, val={len(val_p2i)}")
    if use_inner_train_val_split:
        print(f"Original KR validation kept for final internal test: {INTERNAL_TEST_DATA_PATH}")
    print(f"SHIFT N/A separation: {separate_shift_na_from_padding} (shift_na_raw_token={shift_na_raw_token})")
    if auc_eval_enabled:
        print(
            f"[auc-proxy] data={auc_eval_data_path}, patients={len(auc_eval_patient_indices)}, "
            f"tokens={len(auc_eval_token_ids)}, interval={auc_eval_interval}, "
            f"selection={checkpoint_selection_metric}"
        )
    if horizon_disease_aux_enabled:
        print(
            f"[horizon-aux] enabled: tokens={len(horizon_aux_token_ids)}, "
            f"offset_days={horizon_disease_aux_offset_days}, "
            f"weight={horizon_disease_aux_weight}, pos_weight={horizon_disease_aux_pos_weight}"
        )

# Drug token range (used by both class-weighting and patient sampling)
drug_token_min = 1279 if apply_token_shift else 1278
drug_token_max = 1289 if apply_token_shift else 1288

if master_process:
    space = "SHIFTED (+1)" if apply_token_shift else "RAW"
    print(f"Token space: {space} (apply_token_shift={apply_token_shift}) | "
          f"Drug range: {drug_token_min}-{drug_token_max} | Death={drug_token_max}")

# Dynamic Class Weighting (SHIFT)
if not shift_class_weights:
    drug_mask = (train_data['DATA'] >= drug_token_min) & (train_data['DATA'] <= drug_token_max)
    shift_values = train_data['SHIFT'][drug_mask].astype(np.int64)
    if apply_token_shift:
        shift_values = shift_values + 1
    shift_values = _remap_shift_to_change_np(shift_values, shifted=apply_token_shift, num_classes=num_shift_classes)
    shift_values = shift_values[shift_values >= 0]
    shift_class_weights = _compute_shift_class_weights(
        shift_values,
        num_shift_classes,
        shift_ignore_index,
    )
    if master_process:
        print(f"Computed {num_shift_classes}-class weights (effective num samples, drug-token subset): {shift_class_weights}")

# WeightedRandomSampler: Patient-level balanced sampling
if master_process:
    print(f"Computing patient-level sampling weights for {num_shift_classes}-class balancing...")

patient_weights = np.zeros(len(train_p2i), dtype=np.float32)
for pid, (start_idx, length) in enumerate(train_p2i):
    patient_data = train_data[start_idx:start_idx + length]
    drug_mask = (patient_data['DATA'] >= drug_token_min) & (patient_data['DATA'] <= drug_token_max)
    patient_shifts = patient_data['SHIFT'][drug_mask].astype(np.int64)
    if apply_token_shift:
        patient_shifts = patient_shifts + 1
    patient_changes = _remap_shift_to_change_np(patient_shifts, shifted=apply_token_shift, num_classes=num_shift_classes)
    if num_shift_classes == 3:
        # Boost patients with decrease(0) or increase(2) events (non-maintain)
        minority_count = ((patient_changes == 0) | (patient_changes == 2)).sum()
    else:
        minority_count = (patient_changes == 1).sum()
    patient_weights[pid] = 1.0 + minority_count * 0.3

patient_weights = patient_weights / patient_weights.sum()

# Downsample to requested fraction
if data_fraction < 1.0:
    subset_size = max(1, int(data_fraction * len(train_p2i)))
    train_p2i = train_p2i[:subset_size]
    patient_weights = patient_weights[:subset_size]
    patient_weights = patient_weights / patient_weights.sum()
    if master_process:
        print(f"Using {data_fraction*100:.1f}% of training data: {len(train_p2i)} patients")

patient_weights_tensor = torch.from_numpy(patient_weights)
minority_patient_count = (patient_weights > 1.0 / len(train_p2i)).sum()
if master_process:
    print(f"  Patients with non-maintain SHIFT events: {minority_patient_count:,} / {len(train_p2i):,}")
    print(f"  Max sampling weight: {patient_weights.max():.4f}, Min: {patient_weights.min():.6f}")

iter_num = 0
best_val_loss = 1e9
best_val_loss_ema = 1e9
best_auc_metric = -1.0
best_auc_iter = -1
best_auc_metrics = None

# =============================================================================
# Model Initialization
# =============================================================================

# Composite Delphi with multi-head output
model_args = dict(
    n_layer=n_layer,
    n_head=n_head,
    n_kv_head=n_kv_head,
    n_embd=n_embd,
    block_size=block_size,
    bias=bias,
    dropout=dropout,
    token_dropout=token_dropout,
    t_min=t_min,
    mask_ties=mask_ties,
    ignore_tokens=ignore_tokens,
    eot_token=eot_token,
    use_moe=use_moe,
    num_experts=num_experts,
    experts_per_token=experts_per_token,
    moe_expert_type=moe_expert_type,
    sliding_window=sliding_window,
    rope_theta=rope_theta,
    use_drug_conditioning=use_drug_conditioning,
    film_dropout=film_dropout,
    film_in_backbone=film_in_backbone,
    use_teacher_forcing_drug_cond=use_teacher_forcing_drug_cond,
    data_label_smoothing=data_label_smoothing,
    tie_data_head=tie_data_head,
    drug_token_min=drug_token_min,
    drug_token_max=drug_token_max,
    mdn_n_components=mdn_n_components,
    total_min_value=total_min_value,
    total_max_value=total_max_value,
    total_log_transform=total_log_transform,
    mdn_log_s_min=mdn_log_s_min,
    # Composite-specific
    data_vocab_size=data_vocab_size,
    shift_vocab_size=shift_vocab_size,
    total_vocab_size=total_vocab_size,
    # SHIFT head
    num_shift_classes=num_shift_classes,
    drug_token_only_shift=drug_token_only_shift,
    drug_token_only_total=drug_token_only_total,
    # SHIFT loss options
    shift_loss_type=shift_loss_type,
    shift_dice_weight=shift_dice_weight,
    shift_ignore_index=shift_ignore_index,
    shift_focal_gamma=shift_focal_gamma,
    shift_class_weights=shift_class_weights,
    apply_token_shift=apply_token_shift,
    separate_shift_na_from_padding=separate_shift_na_from_padding,
    shift_na_raw_token=shift_na_raw_token,
    # Uncertainty weighting
    use_uncertainty_weighting=use_uncertainty_weighting,
    loss_weight_data=loss_weight_data,
    loss_weight_shift=loss_weight_shift,
    loss_weight_total=loss_weight_total,
    loss_weight_time=loss_weight_time,
    data_loss_static_weight=data_loss_static_weight,
    data_loss_disease_weight=data_loss_disease_weight,
    data_loss_drug_weight=data_loss_drug_weight,
    # Time-to-Event distribution
    time_distribution=time_distribution,
)


def _filtered_model_args(args):
    valid_keys = set(CompositeDelphiConfig.__dataclass_fields__)
    return {k: v for k, v in args.items() if k in valid_keys}


def _log_unsupported_model_args(args, source):
    valid_keys = set(CompositeDelphiConfig.__dataclass_fields__)
    dropped = sorted(k for k in args if k not in valid_keys)
    if dropped and master_process:
        print(f"[config] dropped unsupported {source} model_args: {dropped}")


def _copy_tensor_overlap(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    out = dst.clone()
    slices = tuple(slice(0, min(dst.shape[i], src.shape[i])) for i in range(dst.dim()))
    out[slices].copy_(src[slices].to(dtype=dst.dtype))
    return out


def _load_legacy_partial_weights(target_model, ckpt_path: str, device: str):
    checkpoint = torch.load(ckpt_path, map_location=device)
    legacy_state = checkpoint['model']
    legacy_state = {
        k.replace('module.', '').replace('_orig_mod.', ''): v
        for k, v in legacy_state.items()
    }
    current_state = target_model.state_dict()
    copied = []

    def copy_exact(old_key: str, new_key: str | None = None):
        new_key = old_key if new_key is None else new_key
        if old_key in legacy_state and new_key in current_state:
            src = legacy_state[old_key]
            dst = current_state[new_key]
            if getattr(src, 'shape', None) == getattr(dst, 'shape', None):
                current_state[new_key] = src.to(dtype=dst.dtype)
                copied.append((old_key, new_key))

    def copy_overlap(old_key: str, new_key: str):
        if old_key in legacy_state and new_key in current_state:
            src = legacy_state[old_key]
            dst = current_state[new_key]
            if src.dim() == dst.dim():
                current_state[new_key] = _copy_tensor_overlap(dst, src)
                copied.append((old_key, new_key))

    for old_key in legacy_state:
        copy_exact(old_key)

    copy_overlap('composite_emb.data_emb.weight', 'composite_emb.data_emb.weight')
    copy_overlap('multi_head.data_head.weight', 'multi_head.data_head.weight')
    copy_overlap('multi_head.time_head.weight', 'multi_head.time_head.weight')
    copy_overlap('composite_emb.dose_emb.weight', 'composite_emb.shift_emb.weight')
    copy_overlap('composite_emb.dur_emb.weight', 'composite_emb.total_emb.weight')

    # Smart-init the expanded EOT row: copy the learned "No event" row instead
    # of leaving it at random init. EOT ("observation ends") is semantically
    # close to "no event", so this gives the frozen-backbone polish a sane
    # starting point for the new token in data_emb / data_head / time_head.
    cfg = target_model.config
    eot = getattr(cfg, 'eot_token', None)
    if eot is not None:
        noev = 1 + (1 if getattr(cfg, 'apply_token_shift', False) else 0)
        for key in ('composite_emb.data_emb.weight',
                    'multi_head.data_head.weight',
                    'multi_head.time_head.weight'):
            w = current_state.get(key)
            if w is not None and w.dim() == 2 and w.size(0) > max(eot, noev):
                w = w.clone()
                w[eot] = w[noev]
                current_state[key] = w
                copied.append((f'<eot-init<-row{noev}>', key))

    prefix_maps = (
        ('multi_head.dur_head.', 'multi_head.total_head.'),
        ('multi_head.dur_drug_cond_head.', 'multi_head.total_drug_cond_head.'),
        ('multi_head.dur_film_generator.', 'multi_head.total_film_generator.'),
        ('multi_head.dose_film_generator.', 'multi_head.shift_film_generator.'),
    )
    for old_prefix, new_prefix in prefix_maps:
        for old_key in legacy_state:
            if old_key.startswith(old_prefix):
                copy_exact(old_key, new_prefix + old_key[len(old_prefix):])

    target_model.load_state_dict(current_state, strict=True)
    return copied, checkpoint


if init_from == 'scratch':
    if master_process:
        print("Initializing a new Composite Delphi model from scratch")
    gptconf = CompositeDelphiConfig(**_filtered_model_args(model_args))
    model = CompositeDelphi(gptconf)
elif init_from == 'legacy_partial':
    if not finetune_ckpt_path:
        raise ValueError("finetune_ckpt_path must be set when init_from='legacy_partial'")
    if master_process:
        print(f"Initializing from legacy partial checkpoint: {finetune_ckpt_path}")
    gptconf = CompositeDelphiConfig(**_filtered_model_args(model_args))
    model = CompositeDelphi(gptconf)
    copied, legacy_checkpoint = _load_legacy_partial_weights(model, finetune_ckpt_path, device)
    if master_process:
        print(f"[legacy-partial] copied {len(copied)} tensors from {finetune_ckpt_path}")
elif init_from == 'resume':
    if master_process:
        print(f"Resuming Composite Delphi training from {out_dir}")
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = dict(checkpoint['model_args'])
    model_args.update(checkpoint_model_args)
    # Keep the current ignore-token policy even when resuming older checkpoints.
    model_args['ignore_tokens'] = ignore_tokens
    _log_unsupported_model_args(model_args, "resume")
    gptconf = CompositeDelphiConfig(**_filtered_model_args(model_args))
    model = CompositeDelphi(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
    best_val_loss_ema = checkpoint.get('best_val_loss_ema', best_val_loss)
    best_auc_metric = checkpoint.get('best_auc_metric', best_auc_metric)
    best_auc_iter = checkpoint.get('best_auc_iter', best_auc_iter)
    best_auc_metrics = checkpoint.get('best_auc_metrics', best_auc_metrics)
elif init_from == 'finetune':
    if not finetune_ckpt_path:
        raise ValueError("finetune_ckpt_path must be set when init_from='finetune'")
    if master_process:
        print(f"Fine-tuning Composite Delphi weights from {finetune_ckpt_path}")
    checkpoint = torch.load(finetune_ckpt_path, map_location=device)
    checkpoint_model_args = dict(checkpoint['model_args'])
    current_overrides = {
        key: model_args[key]
        for key in (
            'ignore_tokens',
            'dropout',
            'loss_weight_data',
            'loss_weight_shift',
            'loss_weight_total',
            'loss_weight_time',
            'data_loss_static_weight',
            'data_loss_disease_weight',
            'data_loss_drug_weight',
            'mdn_log_s_min',
            'moe_expert_type',
            'tie_data_head',
            'use_teacher_forcing_drug_cond',
            'use_uncertainty_weighting',
        )
        if key in model_args
    }
    model_args.update(checkpoint_model_args)
    model_args.update(current_overrides)
    _log_unsupported_model_args(model_args, "finetune")
    gptconf = CompositeDelphiConfig(**_filtered_model_args(model_args))
    model = CompositeDelphi(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)

model_args = _filtered_model_args(model_args)
model.to(device)

# raw_model: always points to the unwrapped model (before compile/DDP)
# Use for state_dict, configure_optimizers, get_num_params, etc.
raw_model = model


def _matches_any_pattern(name: str, patterns) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


if trainable_parameter_patterns or frozen_parameter_patterns:
    trainable_patterns = list(trainable_parameter_patterns)
    frozen_patterns = list(frozen_parameter_patterns)
    trainable_count = 0
    frozen_count = 0
    trainable_tensors = 0
    frozen_tensors = 0
    for name, param in raw_model.named_parameters():
        train_this = True
        if trainable_patterns:
            train_this = _matches_any_pattern(name, trainable_patterns)
        if frozen_patterns and _matches_any_pattern(name, frozen_patterns):
            train_this = False
        param.requires_grad_(train_this)
        if train_this:
            trainable_count += param.numel()
            trainable_tensors += 1
        else:
            frozen_count += param.numel()
            frozen_tensors += 1
    if master_process:
        print(
            "[freeze] trainable tensors="
            f"{trainable_tensors} ({trainable_count:,} params), "
            f"frozen tensors={frozen_tensors} ({frozen_count:,} params)"
        )
        if trainable_patterns:
            print(f"[freeze] trainable patterns: {trainable_patterns}")
        if frozen_patterns:
            print(f"[freeze] frozen patterns: {frozen_patterns}")

if master_process:
    print(f"Model type: {model_type}")
    print(f"Model parameters: {raw_model.get_num_params()/1e6:.2f}M")

# =============================================================================
# Optimizer & Scaler
# =============================================================================

scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16' and device_type == 'cuda'))
optimizer = raw_model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])

# Track the iteration where EMA validation loss last improved for early stopping.
best_val_improve_iter = iter_num

# Compile (before DDP wrapping)
if compile:
    if master_process:
        print("Compiling the model... (takes a ~minute)")
    model = torch.compile(model)

# DDP wrapping (after compile)
# find_unused_parameters=True: needed because drug-conditioned heads and MoE experts
# may not participate in every forward pass (conditional activation)
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)

# =============================================================================
# Loss Estimation Functions
# =============================================================================

@torch.no_grad()
def estimate_split_loss(eval_model, data, p2i, num_iters):
    was_training = eval_model.training
    eval_model.eval()
    losses = torch.zeros(num_iters, 5)  # loss, data, shift, total, time
    for k in range(num_iters):
        ix = torch.randint(len(p2i), (batch_size,))
        batch = get_batch_composite(ix, data, p2i, block_size=block_size,
                                    device=device, select='left',
                                    no_event_token_rate=no_event_token_rate,
                                    cut_batch=True,
                                    apply_token_shift=apply_token_shift,
                                    separate_shift_na_from_padding=separate_shift_na_from_padding,
                                    shift_na_raw_token=shift_na_raw_token)
        x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch

        with ctx:
            logits, loss, _ = eval_model(
                x_data, x_shift, x_total, x_ages,
                y_data, y_shift, y_total, y_ages,
                validation_loss_mode=True
            )
        losses[k] = torch.stack([
            loss['loss'],
            loss['loss_data'],
            loss['loss_shift'],
            loss['loss_total'],
            loss['loss_time']
        ]).detach().cpu()
    if was_training:
        eval_model.train()
    return losses.mean(0)


@torch.no_grad()
def estimate_loss():
    """Estimate train and inner-validation loss for Composite Delphi."""
    out = {
        'train': estimate_split_loss(model, train_data, train_p2i, eval_iters),
        'val': estimate_split_loss(model, val_data, val_p2i, eval_iters),
    }
    return out


def _binary_auc(labels, scores):
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    finite = np.isfinite(scores)
    labels = labels[finite]
    scores = scores[finite]
    n_pos = int(labels.sum())
    n = labels.size
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return None

    order = np.argsort(scores, kind='mergesort')
    sorted_scores = scores[order]
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j

    rank_sum_pos = ranks[labels == 1].sum()
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def estimate_auc_proxy(eval_model):
    if (
        not auc_eval_enabled
        or auc_eval_data is None
        or auc_eval_patient_indices is None
        or auc_eval_token_ids is None
        or len(auc_eval_token_ids) == 0
    ):
        return None

    was_training = eval_model.training
    eval_model.eval()

    token_ids = np.asarray(auc_eval_token_ids, dtype=np.int64)
    token_id_list = token_ids.tolist()
    token_col = {int(tok): i for i, tok in enumerate(token_id_list)}
    token_case_scores = {int(tok): [] for tok in token_id_list}
    token_control_scores = {int(tok): [] for tok in token_id_list}
    time_log_abs_errors = []
    time_ratios = []

    exclude_controls = set(int(x) for x in auc_eval_exclude_control_tokens)
    ignored = set(int(x) for x in ignore_tokens)
    if eot_token is not None:
        ignored.add(int(eot_token))
    token_tensor = torch.as_tensor(token_ids, dtype=torch.long, device=device)

    batch_n = max(1, int(auc_eval_batch_size))
    for start in range(0, len(auc_eval_patient_indices), batch_n):
        batch_patient_indices = auc_eval_patient_indices[start:start + batch_n]
        batch = get_batch_composite(
            batch_patient_indices,
            auc_eval_data,
            auc_eval_p2i,
            block_size=block_size,
            device=device,
            select=auc_eval_select,
            padding=auc_eval_padding,
            no_event_token_rate=no_event_token_rate,
            cut_batch=True,
            apply_token_shift=apply_token_shift,
            separate_shift_na_from_padding=separate_shift_na_from_padding,
            shift_na_raw_token=shift_na_raw_token,
        )
        x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch

        with ctx:
            logits, _, _ = eval_model(
                x_data, x_shift, x_total, x_ages,
                return_attention=False,
            )

        data_logits = logits['data']
        selected_scores = data_logits.index_select(-1, token_tensor).float()
        valid_positions = y_data > 0
        for tok in ignored:
            valid_positions &= y_data != tok
        for tok in exclude_controls:
            valid_positions &= y_data != tok

        scores_np = selected_scores.detach().cpu().numpy()
        y_np = y_data.detach().cpu().numpy()
        valid_np = valid_positions.detach().cpu().numpy()
        for col, tok in enumerate(token_id_list):
            case_mask = valid_np & (y_np == int(tok))
            if case_mask.any():
                token_case_scores[int(tok)].append(scores_np[..., col][case_mask])

            patient_has_tok = np.array(
                [int(patient_idx) in auc_eval_patient_token_sets.get(int(tok), set())
                 for patient_idx in batch_patient_indices],
                dtype=bool,
            )
            control_mask = valid_np & (y_np != int(tok)) & (~patient_has_tok[:, None])
            if control_mask.any():
                control_scores = scores_np[..., col][control_mask]
                if control_scores.size > 20000:
                    rng = np.random.default_rng(int(auc_eval_seed) + int(iter_num) + int(tok))
                    control_scores = rng.choice(control_scores, size=20000, replace=False)
                token_control_scores[int(tok)].append(control_scores)

        if 'time_scale' in logits:
            dt = (y_ages - x_ages).float()
            time_valid = valid_positions & torch.isfinite(dt) & (dt > 0)
            if time_valid.any():
                time_logits = logits['time_scale'].float()
                lse = torch.logsumexp(time_logits, -1)
                lse = -torch.log(torch.exp(-lse) + float(t_min))
                pred_dt = torch.exp(-lse)
                ratio = torch.clamp(pred_dt[time_valid] / dt[time_valid], min=1e-6, max=1e6)
                time_ratios.append(ratio.detach().cpu())
                time_log_abs_errors.append(torch.abs(torch.log(ratio)).detach().cpu())

    aucs = []
    token_metrics = []
    for tok in token_id_list:
        if not token_case_scores[int(tok)] or not token_control_scores[int(tok)]:
            continue
        case_scores = np.concatenate(token_case_scores[int(tok)])
        control_scores = np.concatenate(token_control_scores[int(tok)])
        n_pos = int(case_scores.size)
        n_neg = int(control_scores.size)
        if n_pos < int(auc_eval_min_cases) or n_neg < int(auc_eval_min_controls):
            continue
        labels = np.concatenate([
            np.ones(n_pos, dtype=np.int8),
            np.zeros(n_neg, dtype=np.int8),
        ])
        scores = np.concatenate([case_scores, control_scores])
        auc = _binary_auc(labels, scores)
        if auc is None:
            continue
        aucs.append(auc)
        token_metrics.append({'token': int(tok), 'auc': auc, 'cases': n_pos, 'controls': n_neg})

    if len(aucs) == 0:
        metrics = {
            'auc_mean': float('nan'),
            'auc_median': float('nan'),
            'n_tokens': 0,
            'patients': int(len(auc_eval_patient_indices)),
        }
    else:
        metrics = {
            'auc_mean': float(np.mean(aucs)),
            'auc_median': float(np.median(aucs)),
            'auc_min': float(np.min(aucs)),
            'auc_max': float(np.max(aucs)),
            'n_tokens': int(len(aucs)),
            'patients': int(len(auc_eval_patient_indices)),
        }

    if time_log_abs_errors:
        log_err = torch.cat(time_log_abs_errors)
        ratios = torch.cat(time_ratios)
        metrics.update({
            'time_log_mae': float(log_err.mean().item()),
            'time_median_ratio': float(ratios.median().item()),
            'time_mean_ratio': float(ratios.mean().item()),
            'time_n': int(log_err.numel()),
        })

    metrics['tokens'] = token_metrics
    if was_training:
        eval_model.train()
    return metrics


def horizon_disease_aux_loss(data_logits, x_data, x_ages, y_data, y_ages):
    """Multi-label risk loss for diseases that occur at least offset days later."""
    if (
        not horizon_disease_aux_enabled
        or float(horizon_disease_aux_weight) <= 0
        or horizon_aux_token_ids is None
        or len(horizon_aux_token_ids) == 0
    ):
        return None

    token_ids = torch.as_tensor(horizon_aux_token_ids, dtype=torch.long, device=data_logits.device)
    selected_logits = data_logits.index_select(-1, token_ids).float()
    labels = torch.zeros_like(selected_logits, dtype=torch.bool)
    offset_days = float(horizon_disease_aux_offset_days)

    with torch.no_grad():
        valid_event_age = torch.isfinite(y_ages)
        future_threshold = x_ages.float() + offset_days
        neg_inf = torch.full_like(y_ages.float(), -torch.inf)
        for col, tok in enumerate(token_ids):
            event_age = torch.where((y_data == tok) & valid_event_age, y_ages.float(), neg_inf)
            suffix_max_age = torch.flip(torch.cummax(torch.flip(event_age, dims=(1,)), dim=1).values, dims=(1,))
            labels[..., col] = suffix_max_age >= future_threshold

    valid = (x_data > 0).unsqueeze(-1).expand_as(labels)
    if not valid.any() or labels[valid].sum() == 0:
        return None
    return F.binary_cross_entropy_with_logits(
        selected_logits[valid],
        labels[valid].float(),
        pos_weight=torch.tensor(float(horizon_disease_aux_pos_weight), device=data_logits.device),
    )

# =============================================================================
# Learning Rate Scheduler
# =============================================================================

def get_lr(it):
    # Linear warmup
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # After decay, return min_lr
    if it > lr_decay_iters:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

# =============================================================================
# Logging Setup
# =============================================================================

if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)


def _save_checkpoint(checkpoint_obj, filename: str, tag: str):
    ckpt_path = os.path.join(out_dir, filename)
    torch.save(checkpoint_obj, ckpt_path)
    if master_process:
        print(f"[checkpoint:{tag}] saved: {ckpt_path}")
    return ckpt_path


def _run_full_auc_eval(ckpt_path: str, iter_value: int):
    if not master_process or not full_auc_eval_enabled:
        return None
    if int(full_auc_eval_interval) > 0 and iter_value % int(full_auc_eval_interval) != 0:
        return None

    output_root = full_auc_eval_output_root or os.path.join(out_dir, 'full_auc_eval')
    output_path = os.path.join(output_root, f'iter_{iter_value}')
    os.makedirs(output_path, exist_ok=True)

    cmd = [
        sys.executable,
        'evaluate_auc.py',
        '--model_ckpt_path', ckpt_path,
        '--input_path', str(full_auc_eval_input_path),
        '--output_path', output_path,
        '--data_files', str(full_auc_eval_data_file),
        '--train_data_file', str(full_auc_eval_train_data_file),
        '--next_token_data_file', str(full_auc_eval_data_file),
        '--dataset_subset_size', str(int(full_auc_eval_dataset_subset_size)),
        '--eval_batch_size', str(int(full_auc_eval_batch_size)),
        '--skip_next_token_prediction',
        '--skip_composite_fields',
        '--skip_delong',
    ]
    if full_auc_eval_exclude_eot_from_controls:
        cmd.append('--exclude_eot_from_auc_controls')
    cmd.extend(str(x) for x in full_auc_eval_extra_args)

    print(f"[full-auc] running: {' '.join(cmd)}")
    env = os.environ.copy()
    env['PYTHONUNBUFFERED'] = '1'
    result = subprocess.run(cmd, cwd=os.getcwd(), env=env)
    if result.returncode != 0:
        print(f"[full-auc] evaluator failed at iter {iter_value} with code {result.returncode}")
        return None

    metrics_path = os.path.join(output_path, 'val_composite_metrics.json')
    if not os.path.exists(metrics_path):
        print(f"[full-auc] metrics file missing: {metrics_path}")
        return None

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    metrics = dict(metrics)
    metrics['metric_type'] = 'full_auc'
    metrics['output_path'] = output_path
    metrics['checkpoint_path'] = ckpt_path
    print(
        f"[full-auc] iter {iter_value}: "
        f"mean={metrics.get('auc_mean', float('nan')):.6f}, "
        f"median={metrics.get('auc_median', float('nan')):.6f}, "
        f"n={metrics.get('n_diseases_auc', 0)}"
    )
    return metrics


def _build_checkpoint(
    current_val_loss_raw=None,
    current_val_loss_ema=None,
    current_auc_metrics=None,
):
    return {
        'model': raw_model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'model_args': model_args,
        'iter_num': iter_num,
        'best_val_loss': best_val_loss,
        'best_val_loss_raw': best_val_loss,
        'best_val_loss_ema': best_val_loss_ema,
        'best_auc_metric': best_auc_metric,
        'best_auc_iter': best_auc_iter,
        'best_auc_metrics': best_auc_metrics,
        'current_val_loss_raw': current_val_loss_raw,
        'current_val_loss_ema': current_val_loss_ema,
        'current_auc_metrics': current_auc_metrics,
        'config': config,
        'model_type': model_type,
        'train_data_path': TRAIN_DATA_PATH,
        'validation_data_path': VAL_DATA_PATH,
        'inner_split_source_path': INNER_SPLIT_SOURCE_PATH if use_inner_train_val_split else None,
        'internal_test_data_path': INTERNAL_TEST_DATA_PATH,
    }


if master_process and save_initial_checkpoint:
    checkpoint = _build_checkpoint()
    _save_checkpoint(checkpoint, 'ckpt_init.pt', 'init')


def _save_loss_plot(train_steps, train_losses, val_steps, val_losses):
    if not master_process:
        return None
    if len(train_losses) == 0 and len(val_losses) == 0:
        print("[plot] skipped: no loss history")
        return None
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot] skipped: matplotlib unavailable ({e})")
        return None

    fig, ax = plt.subplots(figsize=(10, 6))
    if len(train_losses) > 0:
        ax.plot(train_steps, train_losses, label='train/loss', color='#1f77b4')
    if len(val_losses) > 0:
        ax.plot(val_steps, val_losses, label='val/loss', color='#ff7f0e')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Curve')
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    plot_path = os.path.join(out_dir, 'loss_plot.png')
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[plot] saved: {plot_path}")
    return plot_path

# =============================================================================
# Training Loop
# =============================================================================

if master_process:
    print(f"{'='*60}")
    print(f"  Device: {device} ({'DDP x' + str(ddp_world_size) if ddp else 'single'})")
    print(f"  Batch size: {batch_size} (x{ddp_world_size} GPUs = {batch_size * ddp_world_size} effective)")
    print(f"  Block size: {block_size}")
    print(f"  Max iterations: {max_iters}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Inner validation interval: every {eval_interval} iterations")
    print(f"{'='*60}\n")

# Initial batch (weighted sampling for SHIFT class balance)
ix = torch.multinomial(patient_weights_tensor, batch_size, replacement=True)
batch = get_batch_composite(ix, train_data, train_p2i, block_size=block_size, device=device,
                            padding='random', lifestyle_augmentations=True, select='left',
                            no_event_token_rate=no_event_token_rate,
                            apply_token_shift=apply_token_shift,
                            separate_shift_na_from_padding=separate_shift_na_from_padding,
                            shift_na_raw_token=shift_na_raw_token)
x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch

t0 = time.time()
local_iter_num = 0
val_loss = None
val_loss_ema = None
early_stop_triggered = False
train_loss_steps, train_loss_history = [], []
val_loss_steps, val_loss_history = [], []

while True:
    # Set learning rate
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # Evaluate and checkpoint
    # All processes evaluate independently; only master prints/saves
    if iter_num % eval_interval == 0 and iter_num > 0:
        losses = estimate_loss()

        # DDP: synchronize val losses across GPUs (average) before EMA / improvement check
        if ddp:
            for split in ['train', 'val']:
                sync_tensor = losses[split].clone().to(device)
                dist.all_reduce(sync_tensor, op=dist.ReduceOp.SUM)
                losses[split] = (sync_tensor / ddp_world_size).cpu()

        # Composite model loss components. Checkpoint selection uses raw
        # val_inner loss; EMA is kept for logging and early stopping stability.
        val_loss_raw = losses['val'][0].item()
        if val_loss_ema is None:
            val_loss_unpooled = losses['val']
        else:
            val_loss_unpooled = 0.1 * losses['val'] + 0.9 * val_loss_unpooled
        val_loss = val_loss_raw
        val_loss_ema = val_loss_unpooled[0].item()

        current_auc_metrics = None
        if master_process and auc_eval_enabled and iter_num % int(auc_eval_interval) == 0:
            current_auc_metrics = estimate_auc_proxy(raw_model)
            if current_auc_metrics is not None:
                auc_mean = current_auc_metrics.get('auc_mean', float('nan'))
                auc_median = current_auc_metrics.get('auc_median', float('nan'))
                print(
                    f"  [auc-proxy] mean={auc_mean:.4f}, median={auc_median:.4f}, "
                    f"tokens={current_auc_metrics.get('n_tokens', 0)}, "
                    f"patients={current_auc_metrics.get('patients', 0)}"
                )
                if 'time_log_mae' in current_auc_metrics:
                    print(
                        "  [time-calib] "
                        f"log_mae={current_auc_metrics['time_log_mae']:.4f}, "
                        f"median_pred/obs={current_auc_metrics['time_median_ratio']:.4f}, "
                        f"mean_pred/obs={current_auc_metrics['time_mean_ratio']:.4f}, "
                        f"n={current_auc_metrics['time_n']}"
                    )
                if wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "auc_proxy/mean": auc_mean,
                        "auc_proxy/median": auc_median,
                        "auc_proxy/n_tokens": current_auc_metrics.get('n_tokens', 0),
                        "time_calib/log_mae": current_auc_metrics.get('time_log_mae', float('nan')),
                        "time_calib/median_ratio": current_auc_metrics.get('time_median_ratio', float('nan')),
                        "time_calib/mean_ratio": current_auc_metrics.get('time_mean_ratio', float('nan')),
                    })

        if master_process:
            train_breakdown = losses['train']
            val_breakdown = losses['val']
            print(f"step {iter_num}: train loss {train_breakdown[0].item():.4f}, val_inner loss {val_loss_raw:.4f} (ema {val_loss_ema:.4f})")
            print(
                "  breakdown (train/val) - "
                f"data: {train_breakdown[1].item():.4f}/{val_breakdown[1].item():.4f}, "
                f"shift: {train_breakdown[2].item():.4f}/{val_breakdown[2].item():.4f}, "
                f"total: {train_breakdown[3].item():.4f}/{val_breakdown[3].item():.4f}, "
                f"time: {train_breakdown[4].item():.4f}/{val_breakdown[4].item():.4f}"
            )
            if getattr(raw_model.config, 'use_uncertainty_weighting', False):
                sigmas = {k.replace('log_sigma_', 'σ_'): f"{torch.exp(v).item():.4f}"
                          for k, v in raw_model.named_parameters() if 'log_sigma' in k}
                print(f"  uncertainty σ: {sigmas}")
            val_loss_steps.append(iter_num)
            val_loss_history.append(val_loss_raw)

            if wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "train/loss": losses['train'][0].item(),
                    "val/loss": val_loss_raw,
                    "val/loss_ema": val_loss_ema,
                    "val/loss_data": losses['val'][1].item(),
                    "val/loss_shift": losses['val'][2].item(),
                    "val/loss_total": losses['val'][3].item(),
                    "val/loss_time": losses['val'][4].item(),
                    "val/loss_data_ema": val_loss_unpooled[1].item(),
                    "val/loss_shift_ema": val_loss_unpooled[2].item(),
                    "val/loss_total_ema": val_loss_unpooled[3].item(),
                    "val/loss_time_ema": val_loss_unpooled[4].item(),
                })

        if val_loss_ema < best_val_loss_ema:
            best_val_loss_ema = val_loss_ema
            best_val_improve_iter = iter_num

        loss_improved = val_loss_raw < best_val_loss
        if loss_improved:
            best_val_loss = val_loss_raw
            if master_process and iter_num > 0:
                checkpoint = _build_checkpoint(val_loss_raw, val_loss_ema, current_auc_metrics)
                if save_loss_best_checkpoint:
                    _save_checkpoint(checkpoint, 'ckpt_loss.pt', 'loss-best')
                if checkpoint_selection_metric == 'val_loss':
                    _save_checkpoint(checkpoint, 'ckpt.pt', 'best')
                    print(f"[best] ckpt.pt updated at iter {iter_num} (raw val_inner/loss={val_loss_raw:.6f}, ema={val_loss_ema:.6f})")
                else:
                    print(f"[loss-best] updated at iter {iter_num} (raw val_inner/loss={val_loss_raw:.6f}, ema={val_loss_ema:.6f})")
        elif master_process:
            print(f"[best] not updated at iter {iter_num} (raw val_inner/loss={val_loss_raw:.6f}, best_raw={best_val_loss:.6f}, ema={val_loss_ema:.6f})")

        auc_improved = False
        auc_selection_key = None
        if current_auc_metrics is not None and checkpoint_selection_metric in ('auc_mean', 'auc_median'):
            auc_selection_key = checkpoint_selection_metric
            candidate_auc = current_auc_metrics.get(auc_selection_key, float('nan'))
            auc_improved = np.isfinite(candidate_auc) and candidate_auc > best_auc_metric
            if auc_improved:
                best_auc_metric = float(candidate_auc)
                best_auc_iter = int(iter_num)
                best_auc_metrics = dict(current_auc_metrics)
                if master_process and iter_num > 0:
                    checkpoint = _build_checkpoint(val_loss_raw, val_loss_ema, current_auc_metrics)
                    _save_checkpoint(checkpoint, 'ckpt_auc.pt', 'auc-best')
                    _save_checkpoint(checkpoint, 'ckpt.pt', 'best')
                    print(
                        f"[auc-best] ckpt.pt updated at iter {iter_num} "
                        f"({auc_selection_key}={candidate_auc:.6f}, "
                        f"mean={current_auc_metrics.get('auc_mean', float('nan')):.6f}, "
                        f"median={current_auc_metrics.get('auc_median', float('nan')):.6f})"
                    )
            elif master_process:
                print(
                    f"[auc-best] not updated at iter {iter_num} "
                    f"({auc_selection_key}={candidate_auc:.6f}, best={best_auc_metric:.6f}, iter={best_auc_iter})"
                )

        eval_ckpt_path = None
        if (
            master_process
            and save_eval_checkpoints
            and iter_num > 0
            and (
                int(save_eval_checkpoint_interval) <= 0
                or iter_num % int(save_eval_checkpoint_interval) == 0
            )
        ):
            checkpoint = _build_checkpoint(val_loss, val_loss_ema, current_auc_metrics)
            eval_ckpt_path = _save_checkpoint(checkpoint, f'ckpt_eval_{iter_num}.pt', f'eval@{iter_num}')

        full_auc_metrics = None
        if master_process and full_auc_eval_enabled and iter_num > 0:
            if eval_ckpt_path is None:
                checkpoint = _build_checkpoint(val_loss, val_loss_ema, current_auc_metrics)
                eval_ckpt_path = _save_checkpoint(checkpoint, f'ckpt_eval_{iter_num}.pt', f'eval@{iter_num}')
            full_auc_metrics = _run_full_auc_eval(eval_ckpt_path, iter_num)
            if full_auc_metrics is not None and wandb_log:
                wandb.log({
                    "iter": iter_num,
                    "full_auc/mean": full_auc_metrics.get('auc_mean', float('nan')),
                    "full_auc/median": full_auc_metrics.get('auc_median', float('nan')),
                    "full_auc/n_diseases": full_auc_metrics.get('n_diseases_auc', 0),
                })

        if full_auc_metrics is not None and checkpoint_selection_metric in ('full_auc_mean', 'full_auc_median'):
            full_auc_key = 'auc_mean' if checkpoint_selection_metric == 'full_auc_mean' else 'auc_median'
            candidate_full_auc = full_auc_metrics.get(full_auc_key, float('nan'))
            full_auc_improved = np.isfinite(candidate_full_auc) and candidate_full_auc > best_auc_metric
            if full_auc_improved:
                best_auc_metric = float(candidate_full_auc)
                best_auc_iter = int(iter_num)
                best_auc_metrics = dict(full_auc_metrics)
                checkpoint = _build_checkpoint(val_loss_raw, val_loss_ema, full_auc_metrics)
                _save_checkpoint(checkpoint, 'ckpt_auc_full.pt', 'full-auc-best')
                _save_checkpoint(checkpoint, 'ckpt.pt', 'best')
                print(
                    f"[full-auc-best] ckpt.pt updated at iter {iter_num} "
                    f"({checkpoint_selection_metric}={candidate_full_auc:.6f}, "
                    f"mean={full_auc_metrics.get('auc_mean', float('nan')):.6f}, "
                    f"median={full_auc_metrics.get('auc_median', float('nan')):.6f})"
                )
            else:
                print(
                    f"[full-auc-best] not updated at iter {iter_num} "
                    f"({checkpoint_selection_metric}={candidate_full_auc:.6f}, "
                    f"best={best_auc_metric:.6f}, iter={best_auc_iter})"
                )

        # Early stopping: no EMA validation improvement for N iterations.
        should_early_stop = False
        if early_stop_patience_iters > 0:
            no_improve_iters = iter_num - best_val_improve_iter
            should_early_stop = no_improve_iters >= early_stop_patience_iters

        # Keep stop decision consistent across all DDP workers.
        if ddp:
            stop_tensor = torch.tensor(1 if should_early_stop else 0, device=device)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX)
            should_early_stop = bool(stop_tensor.item())

        if should_early_stop:
            if master_process:
                no_improve_iters = iter_num - best_val_improve_iter
                print(
                    f"[early-stop] no val_inner EMA improvement for {no_improve_iters} iterations "
                    f"(patience={early_stop_patience_iters}, best_ema={best_val_loss_ema:.6f}); "
                    f"stopping at iter {iter_num}."
                )
            early_stop_triggered = True
            break

        # Save periodic checkpoint (master only)
        if master_process and iter_num % 10_000 == 0:
            checkpoint = _build_checkpoint(val_loss, val_loss_ema, current_auc_metrics)
            _save_checkpoint(checkpoint, f'ckpt_{iter_num}.pt', f'periodic@{iter_num}')

    if iter_num == 0 and eval_only:
        break

    # Training step
    for micro_step in range(gradient_accumulation_steps):
        # DDP: only sync gradients on the last micro-step (performance optimization)
        if ddp:
            model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)

        with ctx:
            logits, loss, att = model(
                x_data, x_shift, x_total, x_ages,
                y_data, y_shift, y_total, y_ages
            )
            aux_horizon_loss = horizon_disease_aux_loss(
                logits['data'], x_data, x_ages, y_data, y_ages
            )
            if aux_horizon_loss is not None:
                loss['loss_horizon_disease'] = aux_horizon_loss
                loss['loss'] = loss['loss'] + float(horizon_disease_aux_weight) * aux_horizon_loss

        # Prefetch next batch (weighted sampling for SHIFT class balance)
        ix = torch.multinomial(patient_weights_tensor, batch_size, replacement=True)
        batch = get_batch_composite(ix, train_data, train_p2i, block_size=block_size, device=device,
                                    padding='random', lifestyle_augmentations=True, select='left',
                                    no_event_token_rate=no_event_token_rate, cut_batch=True,
                                    apply_token_shift=apply_token_shift,
                                    separate_shift_na_from_padding=separate_shift_na_from_padding,
                                    shift_na_raw_token=shift_na_raw_token)
        x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch
        total_loss = loss['loss'] / gradient_accumulation_steps

        scaler.scale(total_loss).backward()

    # Gradient clipping
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

    # Optimizer step
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    # Logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    
    if master_process and iter_num % log_interval == 0:
        lossf = total_loss.item()
        train_loss_steps.append(iter_num)
        train_loss_history.append(lossf)
        valf = f"{val_loss:.4f} (ema {val_loss_ema:.4f})" if val_loss is not None else "n/a"
        # Show lr trend
        if iter_num > 0 and iter_num % (log_interval * 10) == 0:
            prev_lr = get_lr(iter_num - log_interval) if decay_lr else learning_rate
            lr_change = "↑" if lr > prev_lr else "↓" if lr < prev_lr else "="
            print(f"iter {iter_num}: loss {lossf:.4f}, val {valf}, time {dt*1000:.2f}ms, lr {lr:.2e} {lr_change} (warmup: {iter_num < warmup_iters}, decay: {iter_num > warmup_iters})")
        else:
            print(f"iter {iter_num}: loss {lossf:.4f}, val {valf}, time {dt*1000:.2f}ms, lr {lr:.2e}")

        if wandb_log:
            log_dict = {
                "iter": iter_num,
                "train/loss": total_loss.item(),
                "train/loss_data": loss['loss_data'].item(),
                "train/loss_shift": loss['loss_shift'].item(),
                "train/loss_total": loss['loss_total'].item(),
                "train/loss_time": loss['loss_time'].item(),
                "train/loss_horizon_disease": loss.get('loss_horizon_disease', torch.tensor(float('nan'))).item(),
                "lr": lr,
            }
            # Log uncertainty sigmas if available
            for k in ('sigma_data', 'sigma_shift', 'sigma_total', 'sigma_time'):
                if k in loss:
                    log_dict[f'train/{k}'] = loss[k]
            wandb.log(log_dict)

    iter_num += 1
    local_iter_num += 1

    # Termination
    if iter_num > max_iters:
        break

internal_test_breakdown = None
if run_internal_test_after_training:
    if master_process:
        if internal_test_iters <= 0:
            print("[internal-test] skipped: internal_test_iters <= 0")
        elif not os.path.exists(INTERNAL_TEST_DATA_PATH):
            print(f"[internal-test] skipped: file not found: {INTERNAL_TEST_DATA_PATH}")
        else:
            best_ckpt_path = os.path.join(out_dir, 'ckpt.pt')
            if os.path.exists(best_ckpt_path):
                best_checkpoint = torch.load(best_ckpt_path, map_location=device)
                state_dict = best_checkpoint['model']
                cleaned_state_dict = {}
                for k, v in state_dict.items():
                    for unwanted_prefix in ('module.', '_orig_mod.'):
                        if k.startswith(unwanted_prefix):
                            k = k[len(unwanted_prefix):]
                    cleaned_state_dict[k] = v
                raw_model.load_state_dict(cleaned_state_dict)
                print(f"[internal-test] loaded best checkpoint: {best_ckpt_path}")
            else:
                print("[internal-test] best checkpoint not found; evaluating current weights")

            internal_test_data = np.fromfile(INTERNAL_TEST_DATA_PATH, dtype=composite_dtype)
            internal_test_p2i = get_p2i_composite(internal_test_data)
            internal_test_breakdown = estimate_split_loss(
                raw_model,
                internal_test_data,
                internal_test_p2i,
                int(internal_test_iters),
            )
            print(
                f"[internal-test] next-token loss on {INTERNAL_TEST_DATA_PATH}: "
                f"loss={internal_test_breakdown[0].item():.4f}, "
                f"data={internal_test_breakdown[1].item():.4f}, "
                f"shift={internal_test_breakdown[2].item():.4f}, "
                f"total={internal_test_breakdown[3].item():.4f}, "
                f"time={internal_test_breakdown[4].item():.4f} "
                f"(iters={internal_test_iters}, patients={len(internal_test_p2i)})"
            )
    if ddp:
        dist.barrier()

if master_process:
    loss_plot_path = _save_loss_plot(
        train_loss_steps, train_loss_history,
        val_loss_steps, val_loss_history,
    )
    if wandb_log and loss_plot_path is not None:
        wandb.log({"train/loss_plot": wandb.Image(loss_plot_path)})

    print(f"\n{'='*60}")
    print(f"Training completed!")
    if early_stop_triggered:
        print(f"Early stopping: triggered (patience={early_stop_patience_iters} iters)")
    print(f"Best raw inner validation loss: {best_val_loss:.4f}")
    print(f"Best EMA inner validation loss: {best_val_loss_ema:.4f}")
    print(f"Total iterations: {iter_num}")
    print(f"{'='*60}")

# DDP cleanup
if ddp:
    dist.destroy_process_group()
