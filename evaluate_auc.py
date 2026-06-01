import scipy.stats
import scipy
import warnings
import torch
import json
import re
# Suppress sklearn warnings about classes not in y_true
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.metrics._classification')
from model import CompositeDelphi, CompositeDelphiConfig
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from utils import get_batch_composite, get_p2i_composite
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    top_k_accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    r2_score,
    confusion_matrix,
    roc_auc_score
)


def remap_shift_to_change_torch(shift_values: torch.Tensor, apply_token_shift: bool, num_classes: int = 3):
    """
    Remap SHIFT labels to class indices.

    num_classes=3: decrease(0) / maintain(1) / increase(2)
    num_classes=2: label1->0, label2or3->1 (binary fallback)

    Returns:
      mapped_targets, valid_mask
    """
    if apply_token_shift:
        is_label1 = shift_values == 2
        is_label2 = shift_values == 3
        is_label3 = shift_values == 4
    else:
        is_label1 = shift_values == 1
        is_label2 = shift_values == 2
        is_label3 = shift_values == 3

    valid_mask = is_label1 | is_label2 | is_label3
    mapped = torch.full_like(shift_values, fill_value=-1)
    if num_classes == 3:
        mapped[is_label1] = 0  # decrease
        mapped[is_label2] = 1  # maintain
        mapped[is_label3] = 2  # increase
    else:
        mapped[is_label1] = 0
        mapped[is_label2 | is_label3] = 1
    return mapped, valid_mask


# Backward-compatible alias
def remap_shift_to_binary_change_torch(shift_values, apply_token_shift):
    return remap_shift_to_change_torch(shift_values, apply_token_shift, num_classes=2)


def auc(x1, x2):
    n1 = len(x1)
    n2 = len(x2)
    if n1 == 0 or n2 == 0:
        return np.nan
    R1 = np.concatenate([x1, x2]).argsort().argsort()[:n1].sum() + n1
    U1 = R1 - 0.5 * n1 * (n1 + 1)
    return U1 / n1 / n2


# Raw DATA token IDs 0–21: padding, no-event, sex, lifestyle / vitals (through Fasting Glucose high).
# First ICD-10 disease code (A00) is raw index 22 (see labels_chapter.csv token_id).
MIN_DISEASE_RAW_TOKEN_INDEX = 22
DATA_ONLY_EXTVAL_PREFIXES = {"extval_ukb", "extval_ckb"}
DIABETES_DIAGNOSIS_RAW_TOKEN_INDICES = {223, 224, 225, 226, 227}  # E10-E14
CKB_DIABETES_CODEBOOK_GROUP_ID = 14
CKB_MALIGNANT_NEOPLASMS_CODEBOOK_GROUP_ID = 20


CKB_CODEBOOK_GROUP_SPECS = [
    {"id": 1, "name": "Hypertension", "codes": ["I10", "I11", "I12", "I13", "I15"]},
    {"id": 2, "name": "Coronary heart disease", "codes": ["I20", "I21", "I22", "I23", "I24", "I25"]},
    {"id": 3, "name": "Acute myocardial infarction", "codes": ["I21", "I22"]},
    {"id": 4, "name": "Angina pectoris", "codes": ["I20"]},
    {"id": 5, "name": "Other ischaemic heart disease", "codes": ["I24", "I25"]},
    {"id": 6, "name": "Rheumatic heart disease", "codes": ["I05", "I06", "I07", "I08", "I09"]},
    {"id": 7, "name": "Pulmonary heart disease", "codes": ["I26", "I27", "I28"]},
    {"id": 8, "name": "Stroke or TIA", "codes": ["I60", "I61", "I63", "I64", "G45"]},
    {"id": 9, "name": "Asthma", "codes": ["J45", "J46"]},
    {"id": 10, "name": "COPD", "codes": ["J41", "J42", "J43", "J44"]},
    {"id": 11, "name": "Emphysema/Bronchitis", "codes": ["J43", "J41", "J42"]},
    {"id": 12, "name": "Emphysema", "codes": ["J43"]},
    {"id": 13, "name": "Chronic bronchitis", "codes": ["J41", "J42"]},
    {"id": 14, "name": "Diabetes mellitus", "codes": ["E10", "E11", "E12", "E13", "E14"]},
    {"id": 15, "name": "Osteoporosis", "codes": ["M80", "M81"]},
    {"id": 16, "name": "Cirrhosis/Chronic hepatitis", "codes": ["K70", "K71", "K72", "K73", "K74"]},
    {"id": 17, "name": "Gallstone/Gallbladder disease", "codes": ["K80", "K81", "K82"]},
    {"id": 18, "name": "Peptic ulcer", "codes": ["K25", "K26", "K27", "K28"]},
    {"id": 19, "name": "Kidney disease", "codes": ["N00", "N03", "N04", "N05", "N07", "N11", "N18"]},
    {"id": 20, "name": "Malignant neoplasms", "codes": ["C00-C97"]},
    {"id": 21, "name": "Rheumatoid arthritis", "codes": ["M05", "M06"]},
    {"id": 22, "name": "Fracture", "codes": ["S02", "S12", "S22", "S32", "S42", "S52", "S62", "S72", "S82", "S92"]},
    {"id": 23, "name": "Head injury", "codes": ["S00", "S01", "S02", "S06", "S09"]},
    {"id": 24, "name": "Tuberculosis", "codes": ["A15", "A16", "A17", "A18", "A19"]},
    {"id": 25, "name": "Neurasthenia", "codes": ["F48"]},
    {"id": 26, "name": "Psychiatric disorder", "codes": ["F20-F29", "F30-F39"]},
    {"id": 27, "name": "Depression", "codes": ["F32", "F33"]},
    {"id": 28, "name": "Anxiety disorder", "codes": ["F40", "F41"]},
    {"id": 29, "name": "Other psychiatric disorders", "codes": ["F99"]},
    {"id": 30, "name": "Suspected suicide & self-harm", "codes": ["T50.9", "T58", "T59", "W13-W19", "W65"]},
    # Codebook row 31 is non-vascular mortality and has no DATA ICD-10 token target.
    {"id": 32, "name": "Any cerebrovascular disease", "codes": ["I60", "I61", "I62", "I63", "I64", "I65", "I66", "I67", "I68", "I69", "G45", "G46"]},
    {"id": 33, "name": "Chronic kidney disease", "codes": ["E10.2", "E11.2", "I12", "I13", "N03", "N07", "N11", "N18"]},
]


def normalize_icd10_level3(code, allow_subcode_to_level3=False):
    """
    Normalize ICD-10 strings to level-3 model tokens.

    Decimal subcodes such as E11.2 are not broadened by default because mapping
    them to E11 changes the phenotype definition.
    """
    if code is None:
        return None
    code = str(code).strip().upper()
    if "." in code and not allow_subcode_to_level3:
        return None
    match = re.match(r"^\s*([A-Z])(\d{2})(?:\.\d+)?\s*$", code)
    if match is None:
        return None
    return f"{match.group(1)}{match.group(2)}"


def expand_icd10_level3_specs(code_specs, allow_subcode_to_level3=False):
    """Expand codebook specs like C00-C97 and F20-F29 to level-3 ICD-10 codes."""
    expanded = []
    skipped = []
    for spec in code_specs:
        spec = str(spec).strip().upper()
        if not spec:
            continue
        if "-" in spec:
            left_raw, right_raw = spec.split("-", 1)
            left = normalize_icd10_level3(left_raw, allow_subcode_to_level3=allow_subcode_to_level3)
            right = normalize_icd10_level3(right_raw, allow_subcode_to_level3=allow_subcode_to_level3)
            if left is None or right is None or left[0] != right[0]:
                skipped.append(spec)
                continue
            for number in range(int(left[1:]), int(right[1:]) + 1):
                expanded.append(f"{left[0]}{number:02d}")
        else:
            code = normalize_icd10_level3(spec, allow_subcode_to_level3=allow_subcode_to_level3)
            if code is not None:
                expanded.append(code)
            else:
                skipped.append(spec)
    return list(dict.fromkeys(expanded)), skipped


def extract_icd10_level3_from_label(label_name):
    """Extract the leading ICD-10 level-3 code from a labels.csv name."""
    if label_name is None:
        return None
    match = re.match(r"^\s*([A-Z]\d{2})(?:\b|\s|\()", str(label_name).upper())
    return match.group(1) if match is not None else None


def build_icd10_level3_token_map(labels_df, token_offset=0):
    """Map ICD-10 level-3 codes to DATA token IDs in the current model token space."""
    if labels_df is None or "index" not in labels_df.columns or "name" not in labels_df.columns:
        return {}

    code_to_token = {}
    for _, row in labels_df.iterrows():
        code = extract_icd10_level3_from_label(row.get("name"))
        if code is None:
            continue
        code_to_token.setdefault(code, int(row["index"]) + int(token_offset))
    return code_to_token


def build_ckb_codebook_group_maps(labels_df, token_offset=0, vocab_size=None):
    """
    Build CKB codebook disease groups.

    A group is positive when any target DATA ICD-10 token overlaps the group's code
    set. Scores are aggregated across the same ICD-10 token set.
    """
    code_to_token = build_icd10_level3_token_map(labels_df, token_offset=token_offset)
    token_sets = {}
    label_map = {}
    icd_code_map = {}
    missing_code_map = {}

    for spec in CKB_CODEBOOK_GROUP_SPECS:
        group_id = int(spec["id"])
        codes, skipped_codes = expand_icd10_level3_specs(spec["codes"])
        tokens = []
        missing_codes = list(skipped_codes)
        for code in codes:
            token = code_to_token.get(code)
            if token is None:
                missing_codes.append(code)
                continue
            if vocab_size is not None and not (0 <= int(token) < int(vocab_size)):
                missing_codes.append(code)
                continue
            tokens.append(int(token))

        if not tokens:
            missing_code_map[group_id] = missing_codes or codes
            continue

        token_sets[group_id] = sorted(set(tokens))
        label_map[group_id] = spec["name"]
        icd_code_map[group_id] = ",".join(codes)
        if missing_codes:
            missing_code_map[group_id] = missing_codes

    return token_sets, label_map, icd_code_map, missing_code_map


def get_common_diseases(labels_df, filter_min_total=100, apply_token_shift=False):
    """
    Get common diseases from labels DataFrame.

    Args:
        labels_df: DataFrame with columns including 'index' and optionally 'count'
        filter_min_total: Minimum count to include a token
        apply_token_shift: If True, get_batch_composite adds +1 to all DATA tokens,
                           so returned token IDs = raw index + 1.
                           If False (default), tokens are raw values (no shift).

    Returns:
        List of token IDs as used by the model.
    """
    idx_col = 'index' if 'index' in labels_df.columns else (
        'token_id' if 'token_id' in labels_df.columns else None
    )
    if idx_col is None:
        raise ValueError("labels_df must contain 'index' or 'token_id' for disease selection")

    if 'count' in labels_df.columns:
        labels_df_filtered = labels_df[labels_df['count'] > filter_min_total]
    else:
        labels_df_filtered = labels_df

    labels_df_filtered = labels_df_filtered[
        labels_df_filtered[idx_col] >= MIN_DISEASE_RAW_TOKEN_INDEX
    ]

    raw_indices = labels_df_filtered[idx_col].tolist()
    if apply_token_shift:
        return [idx + 1 for idx in raw_indices]
    return raw_indices


def get_data_token_offset(apply_token_shift: bool) -> int:
    """Return the DATA-token offset used by the model/tokenizer pipeline."""
    return 1 if apply_token_shift else 0


def get_diabetes_diagnosis_tokens(apply_token_shift: bool):
    """Return E10-E14 DATA token IDs in the current model token space."""
    offset = get_data_token_offset(apply_token_shift)
    return {int(tok + offset) for tok in DIABETES_DIAGNOSIS_RAW_TOKEN_INDICES}


def build_labels_df_for_merge(labels_df, apply_token_shift: bool):
    """Attach the model-space DATA token id used for label merges."""
    labels_df_for_merge = labels_df.copy()
    if 'index' in labels_df_for_merge.columns:
        labels_df_for_merge['shifted_token'] = (
            labels_df_for_merge['index'] + get_data_token_offset(apply_token_shift)
        )
    return labels_df_for_merge


def finalize_token_columns(df):
    """Make raw-vs-shifted token ids explicit in exported evaluation tables."""
    if df is None or df.empty or 'token' not in df.columns:
        return df

    df = df.copy()

    if 'shifted_token' not in df.columns:
        df['shifted_token'] = df['token']
    # Keep `token` as a backward-compatible alias used by plotting code.
    df['token'] = df['shifted_token']

    if 'index' in df.columns and 'raw_token' not in df.columns:
        df['raw_token'] = df['index']

    for col in ('token', 'shifted_token', 'index', 'raw_token'):
        if col in df.columns:
            df[col] = df[col].astype(np.int64)

    preferred = ['raw_token', 'shifted_token', 'token', 'name']
    ordered = [c for c in preferred if c in df.columns]
    ordered += [c for c in df.columns if c not in ordered]
    return df[ordered]


def build_age_stratified_auc_summary(df_unpooled: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize disease-level AUC by prediction-time age stratum (calendar age at the
    token position used for prediction, in years — same binning as get_calibration_auc).

    Each row of df_unpooled is one (token, age bin) from get_calibration_auc. This
    aggregates across diseases to describe the AUC distribution within each age bin.

    Prefers ``auc_delong`` when present; otherwise uses ``auc``. Drops rows with
    status != 'ok' when ``status`` is present. For bootstrap runs, uses ``bootstrap_idx == 0`` only.
    """
    if df_unpooled is None or df_unpooled.empty or "age" not in df_unpooled.columns:
        return pd.DataFrame()

    work = df_unpooled.copy()
    if "bootstrap_idx" in work.columns:
        work = work.loc[work["bootstrap_idx"] == 0]
    if "status" in work.columns:
        work = work.loc[work["status"].eq("ok")]

    if "auc_delong" in work.columns and work["auc_delong"].notna().any():
        auc_col = "auc_delong"
    elif "auc" in work.columns:
        auc_col = "auc"
    else:
        return pd.DataFrame()

    work[auc_col] = pd.to_numeric(work[auc_col], errors="coerce")
    work = work.loc[np.isfinite(work[auc_col])]
    if work.empty:
        return pd.DataFrame()

    def _agg(group: pd.DataFrame) -> pd.Series:
        s = group[auc_col]
        row = {
            "n_evaluations": int(len(group)),
            "n_unique_tokens": int(group["token"].nunique()) if "token" in group.columns else np.nan,
            "auc_mean": float(s.mean()),
            "auc_median": float(s.median()),
            "auc_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "auc_q25": float(s.quantile(0.25)),
            "auc_q75": float(s.quantile(0.75)),
        }
        if "n_healthy" in group.columns:
            row["n_healthy_total"] = int(group["n_healthy"].sum())
        if "n_diseased" in group.columns:
            row["n_diseased_total"] = int(group["n_diseased"].sum())
        return pd.Series(row)

    out = work.groupby("age", sort=True).apply(_agg, include_groups=False).reset_index()
    if out.empty:
        return out
    ages_sorted = np.sort(out["age"].to_numpy())
    if len(ages_sorted) >= 2:
        step = float(ages_sorted[1] - ages_sorted[0])
        out["age_bin_years"] = out["age"].apply(lambda a: f"[{a:g}, {a + step:g})")
    else:
        out["age_bin_years"] = out["age"].apply(lambda a: f"[{a:g}, ?)")
    return out


def _detect_sex_per_patient(x_data_np, apply_token_shift: bool):
    """
    Determine each patient's sex from the input DATA tokens.

    Returns
    -------
    sex : np.ndarray of shape (B,)
        0 = unknown, 1 = female, 2 = male
    """
    offset = 1 if apply_token_shift else 0
    female_tok = 2 + offset
    male_tok = 3 + offset
    is_female = (x_data_np == female_tok).any(axis=1)
    is_male = (x_data_np == male_tok).any(axis=1)
    sex = np.zeros(x_data_np.shape[0], dtype=np.int8)
    sex[is_female] = 1
    sex[is_male] = 2
    return sex


def _filter_data_by_sex(d, p, pred_idx, sex_arr, sex_value):
    """
    Subset d (list of 4 arrays), p, and pred_idx to patients with ``sex_arr == sex_value``.
    Returns (d_sub, p_sub, pred_idx_sub).
    """
    mask = sex_arr == sex_value
    d_sub = [arr[mask] for arr in d]
    p_sub = p[mask] if p is not None else None
    pred_sub = pred_idx[mask] if pred_idx is not None else None
    return d_sub, p_sub, pred_sub


def build_sex_stratified_auc_summary(df_unpooled: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize disease-level AUC by sex stratum.

    Expects ``df_unpooled`` to contain a ``sex`` column (values: ``'all'``,
    ``'female'``, ``'male'``).  Groups by ``sex`` (and optionally ``age``) and
    computes mean / median / std / quartiles of AUC across diseases.

    Returns an empty DataFrame when the ``sex`` column is absent.
    """
    if df_unpooled is None or df_unpooled.empty or "sex" not in df_unpooled.columns:
        return pd.DataFrame()

    work = df_unpooled.copy()
    if "bootstrap_idx" in work.columns:
        work = work.loc[work["bootstrap_idx"] == 0]
    if "status" in work.columns:
        work = work.loc[work["status"].eq("ok")]

    if "auc_delong" in work.columns and work["auc_delong"].notna().any():
        auc_col = "auc_delong"
    elif "auc" in work.columns:
        auc_col = "auc"
    else:
        return pd.DataFrame()

    work[auc_col] = pd.to_numeric(work[auc_col], errors="coerce")
    work = work.loc[np.isfinite(work[auc_col])]
    if work.empty:
        return pd.DataFrame()

    def _agg(group: pd.DataFrame) -> pd.Series:
        s = group[auc_col]
        row = {
            "n_evaluations": int(len(group)),
            "n_unique_tokens": int(group["token"].nunique()) if "token" in group.columns else np.nan,
            "auc_mean": float(s.mean()),
            "auc_median": float(s.median()),
            "auc_std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "auc_q25": float(s.quantile(0.25)),
            "auc_q75": float(s.quantile(0.75)),
        }
        if "n_healthy" in group.columns:
            row["n_healthy_total"] = int(group["n_healthy"].sum())
        if "n_diseased" in group.columns:
            row["n_diseased_total"] = int(group["n_diseased"].sum())
        return pd.Series(row)

    group_cols = ["sex"]
    if "age" in work.columns:
        group_cols.append("age")

    out = work.groupby(group_cols, sort=True).apply(_agg, include_groups=False).reset_index()
    return out


def optimized_bootstrapped_auc_gpu(case, control, n_bootstrap=1000):
    """
    Computes bootstrapped AUC estimates using PyTorch on CUDA.

    Parameters:
        case: 1D tensor of scores for positive cases
        control: 1D tensor of scores for controls
        n_bootstrap: Number of bootstrap replicates

    Returns:
        Tensor of shape (n_bootstrap,) containing AUC for each bootstrap replicate
    """
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This function requires a GPU.")

    # Convert inputs to CUDA tensors
    if not torch.is_tensor(case):
        case = torch.tensor(case, device="cuda", dtype=torch.float32)
    else:
        case = case.to("cuda", dtype=torch.float32)

    if not torch.is_tensor(control):
        control = torch.tensor(control, device="cuda", dtype=torch.float32)
    else:
        control = control.to("cuda", dtype=torch.float32)

    n_case = case.size(0)
    n_control = control.size(0)
    total = n_case + n_control

    # Generate bootstrap samples
    boot_idx_case = torch.randint(0, n_case, (n_bootstrap, n_case), device="cuda")
    boot_idx_control = torch.randint(0, n_control, (n_bootstrap, n_control), device="cuda")

    boot_case = case[boot_idx_case]
    boot_control = control[boot_idx_control]

    combined = torch.cat([boot_case, boot_control], dim=1)

    # Mask to identify case entries
    mask = torch.zeros((n_bootstrap, total), dtype=torch.bool, device="cuda")
    mask[:, :n_case] = True

    # Compute ranks and AUC
    ranks = combined.argsort(dim=1).argsort(dim=1)
    case_ranks_sum = torch.sum(ranks.float() * mask.float(), dim=1)
    min_case_rank_sum = n_case * (n_case - 1) / 2.0
    U = case_ranks_sum - min_case_rank_sum
    aucs = U / (n_case * n_control)
    return aucs.cpu().tolist()


# AUC comparison adapted from
# https://github.com/Netflix/vmaf/
def compute_midrank(x):
    """Computes midranks.
    Args:
       x - a 1D numpy array
    Returns:
       array of midranks
    """
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=np.float32)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=np.float32)
    # Note(kazeevn) +1 is due to Python using 0-based indexing
    # instead of 1-based in the AUC formula in the paper
    T2[J] = T + 1
    return T2


def fastDeLong(predictions_sorted_transposed, label_1_count):
    """
    The fast version of DeLong's method for computing the covariance of
    unadjusted AUC.
    Args:
       predictions_sorted_transposed: a 2D numpy.array[n_classifiers, n_examples]
          sorted such as the examples with label "1" are first
    Returns:
       (AUC value, DeLong covariance)
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }
    """
    # Short variables are named as they are in the paper
    m = label_1_count
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]

    tx = np.empty([k, m], dtype=np.float32)
    ty = np.empty([k, n], dtype=np.float32)
    tz = np.empty([k, m + n], dtype=np.float32)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(predictions_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    
    # Handle cases with insufficient samples for covariance calculation
    # Suppress warnings for small sample sizes
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        # Calculate covariance with error handling
        if m > 1 and v01.shape[1] > 1:
            sx = np.cov(v01)
            # Handle case where covariance might be singular
            if np.any(np.isnan(sx)) or np.any(np.isinf(sx)):
                sx = np.zeros_like(sx)
        else:
            sx = np.zeros((k, k), dtype=np.float32)
        
        if n > 1 and v10.shape[1] > 1:
            sy = np.cov(v10)
            # Handle case where covariance might be singular
            if np.any(np.isnan(sy)) or np.any(np.isinf(sy)):
                sy = np.zeros_like(sy)
        else:
            sy = np.zeros((k, k), dtype=np.float32)
    
    # Calculate delongcov with protection against division by zero
    if m > 0 and n > 0:
        delongcov = sx / m + sy / n
        # Replace any invalid values with 0
        delongcov = np.nan_to_num(delongcov, nan=0.0, posinf=0.0, neginf=0.0)
    else:
        delongcov = np.zeros((k, k), dtype=np.float32)
    
    return aucs, delongcov


def compute_ground_truth_statistics(ground_truth):
    assert np.array_equal(np.unique(ground_truth), [0, 1])
    order = (-ground_truth).argsort()
    label_1_count = int(ground_truth.sum())
    return order, label_1_count


def get_auc_delong_var(healthy_scores, diseased_scores):
    """
    Computes ROC AUC value and variance using DeLong's method

    Args:
        healthy_scores: Values for class 0 (healthy/controls)
        diseased_scores: Values for class 1 (diseased/cases)
    Returns:
        AUC value and variance
    """
    # Create ground truth labels (1 for diseased, 0 for healthy)
    ground_truth = np.array([1] * len(diseased_scores) + [0] * len(healthy_scores))
    predictions = np.concatenate([diseased_scores, healthy_scores])

    # Compute statistics needed for DeLong method
    order, label_1_count = compute_ground_truth_statistics(ground_truth)
    predictions_sorted_transposed = predictions[np.newaxis, order]

    # Calculate AUC and covariance
    aucs, delongcov = fastDeLong(predictions_sorted_transposed, label_1_count)
    assert len(aucs) == 1, "There is a bug in the code, please forward this to the developers"

    # Convert delongcov to scalar if it's an array (for single classifier case)
    if isinstance(delongcov, np.ndarray):
        delongcov = delongcov.item() if delongcov.size == 1 else delongcov[0, 0]
    
    return aucs[0], delongcov


def get_calibration_auc(
    j,
    k,
    d,
    p,
    diseases_chunk,
    offset=365.25,
    age_groups=range(45, 80, 5),
    precomputed_idx=None,
    n_bootstrap=1,
    use_delong=False,
    target_tokens=None,
    exclude_control_tokens=None,
):
    """
    Compute calibration AUC for a specific disease token.
    
    Args:
        j: index of disease in the chunk
        k: disease token ID or synthetic disease-group ID
        d: data tuple [input_tokens, input_ages, target_tokens, target_ages]
        p: predictions (logits) from model, shape (B, T, chunk_size) - only for diseases in chunk
        diseases_chunk: array of disease token IDs in this chunk
        offset: time offset in days
        age_groups: age groups to evaluate
        precomputed_idx: precomputed prediction indices
        n_bootstrap: number of bootstrap samples
        use_delong: whether to use DeLong method
        target_tokens: DATA tokens that count as this disease/group. If None,
                       defaults to exact match on k.
        exclude_control_tokens: DATA tokens to exclude from the control pool
                                without changing case definitions.
    
    Returns:
        list of dictionaries with AUC results (includes N/A results for insufficient data)
    """
    age_step = age_groups[1] - age_groups[0]
    if target_tokens is None:
        target_tokens = [k]
    target_tokens = np.asarray(list(target_tokens), dtype=d[2].dtype)

    # Indexes of cases with disease k, or any DATA token in the disease group.
    # d[2] contains target tokens (disease tokens)
    case_mask = np.isin(d[2], target_tokens)
    wk = np.where(case_mask)
    n_cases = len(wk[0])

    # For controls, we need to exclude cases with disease k/group.
    # Controls are positions where the target token is not in the target set.
    # We also exclude patients who have the disease/group anywhere in their trajectory.
    control_mask = ~case_mask
    if exclude_control_tokens is not None:
        exclude_control_tokens = np.asarray(list(exclude_control_tokens), dtype=d[2].dtype)
        if exclude_control_tokens.size > 0:
            control_mask &= ~np.isin(d[2], exclude_control_tokens)
    patient_has_disease = case_mask.any(axis=1)  # (B,) - True if patient has disease/group
    wc = np.where(control_mask & (~patient_has_disease[:, None]))
    n_controls = len(wc[0])

    # If insufficient data, return N/A results for all age groups
    if n_cases < 2 or n_controls == 0:
        out = []
        reason = "insufficient_cases" if n_cases < 2 else "no_controls"
        for aa in age_groups:
            out_item = {
                "token": k,
                "auc": np.nan,
                "age": aa,
                "n_healthy": n_controls if n_controls > 0 else 0,
                "n_diseased": n_cases,
                "n_target_tokens": len(target_tokens),
                "status": reason,
            }
            if use_delong:
                out_item["auc_delong"] = np.nan
                out_item["auc_variance_delong"] = np.nan
            if n_bootstrap > 1:
                out_item["bootstrap_idx"] = 0  # Only one record per age group for N/A
            out.append(out_item)
        return out

    wall = (np.concatenate([wk[0], wc[0]]), np.concatenate([wk[1], wc[1]]))  # All cases and controls

    # We need to take into account the offset t and use the tokens for prediction that are at least t before the event
    if precomputed_idx is None:
        pred_idx = (d[1][wall[0]] <= d[3][wall].reshape(-1, 1) - offset).sum(1) - 1
    else:
        pred_idx = precomputed_idx[wall]  # It's actually much faster to precompute this

    z = d[1][(wall[0], pred_idx)]  # Times of the tokens for prediction
    z = z[pred_idx != -1]

    zk = d[3][wall]  # Target times
    zk = zk[pred_idx != -1]

    # Extract predictions for disease k
    # p shape: (B, T, chunk_size) - logits only for diseases in chunk
    # j is the index within the chunk, so we use p[..., j]
    x = p[..., j][(wall[0], pred_idx)]
    x = x[pred_idx != -1]

    wk = (wk[0][pred_idx[: len(wk[0])] != -1], wk[1][pred_idx[: len(wk[0])] != -1])
    p_idx = wall[0][pred_idx != -1]

    out = []

    for i, aa in enumerate(age_groups):
        a = np.logical_and(z / 365.25 >= aa, z / 365.25 < aa + age_step)
        # Optionally, add extra filtering on the time difference, for example:
        # a *= (zk - z < 365.25)
        selected_groups = p_idx[a]
        perm = np.random.permutation(len(selected_groups))
        _, indices = np.unique(selected_groups[perm], return_index=True)
        indices = perm[indices]
        selected = np.zeros(np.sum(a), dtype=bool)
        selected[indices] = True
        a[a] = selected

        control = x[len(wk[0]) :][a[len(wk[0]) :]]
        case = x[: len(wk[0])][a[: len(wk[0])]]

        if len(control) == 0 or len(case) == 0:
            continue

        if use_delong:
            auc_value_delong, auc_variance_delong = get_auc_delong_var(control, case)
            # Ensure auc_variance_delong is a scalar for parquet compatibility
            if isinstance(auc_variance_delong, np.ndarray):
                auc_variance_delong = auc_variance_delong.item() if auc_variance_delong.size == 1 else float(auc_variance_delong[0, 0])
            else:
                auc_variance_delong = float(auc_variance_delong)
            auc_delong_dict = {"auc_delong": float(auc_value_delong), "auc_variance_delong": auc_variance_delong}
        else:
            auc_delong_dict = {}

        if n_bootstrap > 1:
            aucs_bootstrapped = optimized_bootstrapped_auc_gpu(case, control, n_bootstrap)

        for bootstrap_idx in range(n_bootstrap):
            if n_bootstrap == 1:
                if use_delong:
                    y = auc_value_delong
                else:
                    y = auc(case, control)
            else:
                y = aucs_bootstrapped[bootstrap_idx]
            
            out_item = {
                "token": k,
                "auc": y,
                "age": aa,
                "n_healthy": len(control),
                "n_diseased": len(case),
                "n_target_tokens": len(target_tokens),
                "status": "ok",
            }
            if n_bootstrap > 1:
                out_item["bootstrap_idx"] = bootstrap_idx
            out.append(out_item | auc_delong_dict)
    return out


def evaluate_composite_fields(
    model,
    d100k,
    batch_size=64,
    device="mps",
    raw_output_path=None,
    raw_output_prefix="composite",
):
    """
    Evaluate binary CHANGE(from SHIFT), TOTAL predictions for CompositeDelphi model.
    
    Args:
        model: CompositeDelphi model
        d100k: Data batch from get_batch_composite
        batch_size: Batch size for inference
        device: Device identifier
        raw_output_path: Optional directory for saving per-event SHIFT/TOTAL
            predictions. These arrays are needed for threshold tuning and
            regression diagnostic plots.
        raw_output_prefix: Prefix used for the raw prediction ``.npz`` file.
    
    Returns:
        dict with evaluation metrics for each field
    """
    model.eval()
    model.to(device)
    all_predictions = {'shift': [], 'total': []}
    all_targets = {'shift': [], 'total': []}
    # For regression, also compute "positive-only" metrics (targets > 0), since zeros dominate in these fields.
    all_predictions_pos = {'total': []}
    all_targets_pos = {'total': []}

    # Softmax probabilities for SHIFT AUC computation
    all_shift_probs = []
    all_shift_probs_drug_cond = []

    # Drug-conditioned predictions (if model uses drug-conditioning)
    # Only evaluated for drug tokens within configured range.
    all_predictions_shift_drug_cond = []
    all_predictions_total_drug_cond = []
    all_targets_shift_drug_cond = []  # Targets for drug-conditioned SHIFT
    all_targets_total_drug_cond = []  # Targets for drug-conditioned TOTAL
    use_drug_conditioning = getattr(model.config, 'use_drug_conditioning', False)
    drug_token_min = int(getattr(model.config, 'drug_token_min', 1278))
    drug_token_max = int(getattr(model.config, 'drug_token_max', 1288))
    eval_apply_token_shift = bool(getattr(model.config, 'apply_token_shift', False))
    drug_token_note = f"Metrics computed only for drug tokens ({drug_token_min}-{drug_token_max})"
    
    x_data, x_shift, x_total, x_ages = d100k[0], d100k[1], d100k[2], d100k[3]
    y_data, y_shift, y_total, y_ages = d100k[4], d100k[5], d100k[6], d100k[7]
    
    num_batches = (x_data.shape[0] + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Evaluating composite fields"):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, x_data.shape[0])
            
            batch_x_data = x_data[start_idx:end_idx].to(device)
            batch_x_shift = x_shift[start_idx:end_idx].to(device)
            batch_x_total = x_total[start_idx:end_idx].to(device)
            batch_x_ages = x_ages[start_idx:end_idx].to(device)
            
            # Get targets on device
            batch_y_data = y_data[start_idx:end_idx].to(device)
            batch_y_shift = y_shift[start_idx:end_idx].to(device)
            batch_y_total = y_total[start_idx:end_idx].to(device)
            
            # Pass targets for drug-conditioning evaluation (uses GT drug embedding)
            outputs = model(
                batch_x_data, batch_x_shift, batch_x_total, batch_x_ages,
                targets_data=batch_y_data, targets_shift=batch_y_shift, 
                targets_total=batch_y_total,
                targets_age=y_ages[start_idx:end_idx].to(device)
            )[0]  # Get logits dict
            
            # Get predictions on device
            # ============================================================
            # - SHIFT: logits (B, T, num_shift_classes), supports 2 or 3 classes
            # - TOTAL: total_head → (B, T) regression output (continuous)
            # ============================================================
            shift_logits = outputs['shift']  # (B, T, C) where C = num_shift_classes
            total_pred = outputs['total']  # (B, T) - regression output

            # Inverse log-transform if model was trained with total_log_transform
            total_log_transform = getattr(model.config, 'total_log_transform', False)
            has_mdn_total = isinstance(outputs.get('total_mdn', None), dict)
            if total_log_transform and not has_mdn_total:
                total_pred = torch.expm1(total_pred)  # inverse of log1p

            # SHIFT: N-class classification
            eval_num_shift_classes = shift_logits.size(-1)
            shift_pred = torch.argmax(shift_logits, dim=-1)  # (B, T)

            shift_target_mapped, shift_valid = remap_shift_to_change_torch(
                batch_y_shift, apply_token_shift=eval_apply_token_shift,
                num_classes=eval_num_shift_classes
            )

            # Use per-field valid masks
            shift_mask = shift_valid
            total_mask = (batch_y_total != -1) & (batch_y_total >= 0)
            
            # Drug token mask: only evaluate drug-conditioned predictions for configured drug range
            drug_token_mask = (batch_y_data >= drug_token_min) & (batch_y_data <= drug_token_max)

            # SHIFT (classification): store predictions/targets/probs for valid shift tokens
            if shift_mask.any():
                all_predictions['shift'].append(shift_pred[shift_mask].cpu().numpy())
                all_targets['shift'].append(shift_target_mapped[shift_mask].cpu().numpy())
                shift_probs = torch.softmax(shift_logits, dim=-1)
                all_shift_probs.append(shift_probs[shift_mask].cpu().numpy())

                # Drug-conditioned SHIFT (if available) - ONLY for drug tokens
                if use_drug_conditioning and 'shift_drug_cond' in outputs:
                    shift_drug_logits = outputs['shift_drug_cond']
                    shift_drug_pred = torch.argmax(shift_drug_logits, dim=-1)
                    shift_drug_probs = torch.softmax(shift_drug_logits, dim=-1)
                    # Filter: only drug tokens AND valid shift tokens
                    drug_shift_mask = shift_mask & drug_token_mask
                    if drug_shift_mask.any():
                        all_predictions_shift_drug_cond.append(shift_drug_pred[drug_shift_mask].cpu().numpy())
                        all_targets_shift_drug_cond.append(shift_target_mapped[drug_shift_mask].cpu().numpy())
                        all_shift_probs_drug_cond.append(shift_drug_probs[drug_shift_mask].cpu().numpy())

            # TOTAL (regression): store clipped predictions to match domain (non-negative)
            if total_mask.any():
                tp = total_pred[total_mask]
                tt = batch_y_total[total_mask].float()
                all_predictions['total'].append(torch.clamp(tp, min=0.0).cpu().numpy())
                all_targets['total'].append(tt.cpu().numpy())
                
                # Drug-conditioned TOTAL (if available) - ONLY for drug tokens
                # Create drug_total_mask BEFORE filtering by total_mask
                if use_drug_conditioning and 'total_drug_cond' in outputs:
                    # Filter: only drug tokens AND valid total tokens
                    drug_total_mask = total_mask & drug_token_mask
                    if drug_total_mask.any():
                        # Apply drug_total_mask directly to outputs (before total_mask filtering)
                        tp_drug = outputs['total_drug_cond'][drug_total_mask]
                        has_mdn_drug_total = isinstance(outputs.get('total_mdn_drug_cond', None), dict)
                        if total_log_transform and not has_mdn_drug_total:
                            tp_drug = torch.expm1(tp_drug)
                        tp_drug = torch.clamp(tp_drug, min=0.0)
                        all_predictions_total_drug_cond.append(tp_drug.cpu().numpy())
                        all_targets_total_drug_cond.append(batch_y_total[drug_total_mask].float().cpu().numpy())

                total_pos = total_mask & (batch_y_total > 0)
                if total_pos.any():
                    tp_pos = total_pred[total_pos]
                    tt_pos = batch_y_total[total_pos].float()
                    all_predictions_pos['total'].append(torch.clamp(tp_pos, min=0.0).cpu().numpy())
                    all_targets_pos['total'].append(tt_pos.cpu().numpy())

    # Concatenate all batches (skip empty)
    for field in ['shift', 'total']:
        if len(all_predictions[field]) > 0:
            all_predictions[field] = np.concatenate(all_predictions[field])
            all_targets[field] = np.concatenate(all_targets[field])
        else:
            all_predictions[field] = np.array([])
            all_targets[field] = np.array([])

    for field in ['total']:
        if len(all_predictions_pos[field]) > 0:
            all_predictions_pos[field] = np.concatenate(all_predictions_pos[field])
            all_targets_pos[field] = np.concatenate(all_targets_pos[field])
        else:
            all_predictions_pos[field] = np.array([])
            all_targets_pos[field] = np.array([])
    
    if len(all_shift_probs) > 0:
        all_shift_probs = np.concatenate(all_shift_probs)
    else:
        all_shift_probs = np.array([])

    if len(all_predictions_shift_drug_cond) > 0:
        all_predictions_shift_drug_cond = np.concatenate(all_predictions_shift_drug_cond)
        all_targets_shift_drug_cond = np.concatenate(all_targets_shift_drug_cond)
        all_shift_probs_drug_cond = np.concatenate(all_shift_probs_drug_cond)
    else:
        all_predictions_shift_drug_cond = np.array([])
        all_targets_shift_drug_cond = np.array([])
        all_shift_probs_drug_cond = np.array([])
    
    if len(all_predictions_total_drug_cond) > 0:
        all_predictions_total_drug_cond = np.concatenate(all_predictions_total_drug_cond)
        all_targets_total_drug_cond = np.concatenate(all_targets_total_drug_cond)
    else:
        all_predictions_total_drug_cond = np.array([])
        all_targets_total_drug_cond = np.array([])

    if raw_output_path is not None:
        raw_output_path = Path(raw_output_path)
        raw_output_path.mkdir(parents=True, exist_ok=True)
        raw_prefix = str(raw_output_prefix or "composite")
        raw_file = raw_output_path / f"{raw_prefix}_composite_raw_predictions.npz"

        raw_payload = {
            "schema_version": np.array("composite_raw_predictions_v1"),
            "shift_target": all_targets["shift"].astype(np.int16, copy=False),
            "shift_pred": all_predictions["shift"].astype(np.int16, copy=False),
            "shift_probs": all_shift_probs.astype(np.float32, copy=False),
            "total_target": all_targets["total"].astype(np.float32, copy=False),
            "total_pred": all_predictions["total"].astype(np.float32, copy=False),
            "total_target_pos": all_targets_pos["total"].astype(np.float32, copy=False),
            "total_pred_pos": all_predictions_pos["total"].astype(np.float32, copy=False),
            "shift_target_drug_cond": all_targets_shift_drug_cond.astype(np.int16, copy=False),
            "shift_pred_drug_cond": all_predictions_shift_drug_cond.astype(np.int16, copy=False),
            "shift_probs_drug_cond": all_shift_probs_drug_cond.astype(np.float32, copy=False),
            "total_target_drug_cond": all_targets_total_drug_cond.astype(np.float32, copy=False),
            "total_pred_drug_cond": all_predictions_total_drug_cond.astype(np.float32, copy=False),
            "drug_token_min": np.array(drug_token_min, dtype=np.int32),
            "drug_token_max": np.array(drug_token_max, dtype=np.int32),
            "apply_token_shift": np.array(eval_apply_token_shift, dtype=np.bool_),
        }
        np.savez_compressed(raw_file, **raw_payload)
        print(f"Composite raw predictions saved to {raw_file}")
    
    # Calculate metrics
    results = {}
    
    # ============================================================
    # SHIFT: CLASSIFICATION METRICS (supports 2 or 3 classes)
    # ============================================================
    def _shift_classification_metrics(targets, preds, n_classes, probs=None):
        """Compute classification metrics for SHIFT. Returns a dict."""
        class_labels = np.arange(n_classes)
        if n_classes == 3:
            class_name_map = {"0": "Decrease", "1": "Maintain", "2": "Increase"}
        else:
            class_name_map = {"0": "Label1", "1": "Label2or3"}

        m = {}
        m['accuracy'] = accuracy_score(targets, preds)
        m['balanced_accuracy'] = balanced_accuracy_score(targets, preds)
        if n_classes == 2:
            m['f1_binary'] = f1_score(targets, preds, average='binary', pos_label=1, zero_division=0)
        m['f1_macro'] = f1_score(targets, preds, average='macro', zero_division=0)
        m['f1_micro'] = f1_score(targets, preds, average='micro', zero_division=0)
        m['f1_weighted'] = f1_score(targets, preds, average='weighted', zero_division=0)
        m['precision_macro'] = precision_score(targets, preds, average='macro', zero_division=0)
        m['recall_macro'] = recall_score(targets, preds, average='macro', zero_division=0)

        # ROC AUC from softmax probabilities
        if probs is not None and len(probs) > 0:
            try:
                if n_classes == 2:
                    m['roc_auc'] = float(roc_auc_score(targets, probs[:, 1]))
                else:
                    m['roc_auc'] = float(roc_auc_score(targets, probs, multi_class='ovr'))
            except (ValueError, IndexError):
                m['roc_auc'] = float('nan')
        m['support'] = int(len(targets))

        cm = confusion_matrix(targets, preds, labels=class_labels)
        m['confusion_matrix'] = cm.tolist()
        m['confusion_matrix_classes'] = class_labels.tolist()
        m['class_name_map'] = class_name_map

        per_class = {}
        for i, cls in enumerate(class_labels):
            tp = cm[i, i]
            fp = cm[:, i].sum() - tp
            fn = cm[i, :].sum() - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_c = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            per_class[int(cls)] = {
                'precision': float(prec), 'recall': float(rec),
                'f1': float(f1_c), 'support': int(tp + fn)
            }
        m['per_class_metrics'] = per_class
        return m

    if len(all_targets['shift']) > 0:
        shift_pred = all_predictions['shift']
        shift_target = all_targets['shift']
        n_classes = int(max(shift_target.max(), shift_pred.max())) + 1
        n_classes = max(n_classes, 2)  # at least binary

        shift_probs_arr = all_shift_probs if len(all_shift_probs) > 0 else None
        shift_metrics = _shift_classification_metrics(shift_target, shift_pred, n_classes, probs=shift_probs_arr)
        for k, v in shift_metrics.items():
            results[f'shift_{k}'] = v

        # Drug-conditioned SHIFT metrics (ONLY for drug tokens)
        if len(all_predictions_shift_drug_cond) > 0:
            n_classes_drug = int(max(all_targets_shift_drug_cond.max(), all_predictions_shift_drug_cond.max())) + 1
            n_classes_drug = max(n_classes_drug, 2)
            drug_probs_arr = all_shift_probs_drug_cond if len(all_shift_probs_drug_cond) > 0 else None
            drug_metrics = _shift_classification_metrics(
                all_targets_shift_drug_cond, all_predictions_shift_drug_cond, n_classes_drug, probs=drug_probs_arr
            )
            for k, v in drug_metrics.items():
                results[f'shift_{k}_drug_cond'] = v
            results['shift_drug_cond_note'] = drug_token_note
    
    # ============================================================
    # TOTAL: REGRESSION METRICS
    # ============================================================
    for field in ['total']:
        if len(all_targets[field]) == 0:
            continue
            
        pred = all_predictions[field]  # continuous regression output
        target = all_targets[field]    # continuous target
        
        # Regression metrics
        mae = mean_absolute_error(target, pred)
        rmse = np.sqrt(mean_squared_error(target, pred))
        median_ae = np.median(np.abs(target - pred))
        
        # R² score
        try:
            r2 = r2_score(target, pred)
        except:
            r2 = np.nan
        
        results[f'{field}_mae'] = mae
        results[f'{field}_rmse'] = rmse
        results[f'{field}_median_ae'] = median_ae
        results[f'{field}_r2'] = r2
        
        # Additional stats
        results[f'{field}_mean_target'] = np.mean(target)
        results[f'{field}_mean_pred'] = np.mean(pred)
        results[f'{field}_std_target'] = np.std(target)
        results[f'{field}_std_pred'] = np.std(pred)

        # Positive-only regression metrics (targets > 0)
        if len(all_targets_pos[field]) > 0:
            pred_pos = all_predictions_pos[field]
            target_pos = all_targets_pos[field]
            results[f'{field}_mae_pos'] = mean_absolute_error(target_pos, pred_pos)
            results[f'{field}_rmse_pos'] = float(np.sqrt(mean_squared_error(target_pos, pred_pos)))
            results[f'{field}_median_ae_pos'] = float(np.median(np.abs(target_pos - pred_pos)))
            try:
                results[f'{field}_r2_pos'] = r2_score(target_pos, pred_pos)
            except:
                results[f'{field}_r2_pos'] = np.nan
            results[f'{field}_support_pos'] = int(len(target_pos))

    # Drug-conditioned TOTAL metrics (ONLY for drug tokens)
    if len(all_predictions_total_drug_cond) > 0:
        pred_drug = all_predictions_total_drug_cond
        tgt_drug = all_targets_total_drug_cond
        results['total_mae_drug_cond'] = mean_absolute_error(tgt_drug, pred_drug)
        results['total_rmse_drug_cond'] = float(np.sqrt(mean_squared_error(tgt_drug, pred_drug)))
        results['total_median_ae_drug_cond'] = float(np.median(np.abs(tgt_drug - pred_drug)))
        try:
            results['total_r2_drug_cond'] = r2_score(tgt_drug, pred_drug)
        except:
            results['total_r2_drug_cond'] = np.nan
        results['total_mean_target_drug_cond'] = float(np.mean(tgt_drug))
        results['total_mean_pred_drug_cond'] = float(np.mean(pred_drug))
        results['total_support_drug_cond'] = int(len(tgt_drug))
        results['total_drug_cond_note'] = drug_token_note
    
    return results


def evaluate_next_token_prediction(
    model,
    data,
    p2i,
    patient_indices,
    block_size=512,
    batch_size=64,
    device="cpu",
    no_event_token_rate=5,
    apply_token_shift=False,
    separate_shift_na_from_padding=False,
    shift_na_raw_token=4,
):
    """
    Teacher-forced next-token prediction loss on a patient cohort.

    This mirrors train_model.py validation/checkpoint selection: targets are
    provided to the model and validation_loss_mode masks technical tokens.
    """
    model.eval()
    model.to(device)

    loss_keys = ["loss", "loss_data", "loss_shift", "loss_total", "loss_time"]
    weighted_loss_sum = np.zeros(len(loss_keys), dtype=np.float64)
    total_weight = 0
    total_patients = 0
    valid_data_targets = 0

    ignored_tokens = set(getattr(model.config, "ignore_tokens", []))
    ignored_tokens.add(1)

    with torch.no_grad():
        for start_idx in tqdm(
            range(0, len(patient_indices), batch_size),
            desc="Next-token prediction",
        ):
            batch_patient_indices = patient_indices[start_idx:start_idx + batch_size]
            if len(batch_patient_indices) == 0:
                continue

            batch = get_batch_composite(
                batch_patient_indices,
                data,
                p2i,
                select="left",
                block_size=block_size,
                device=device,
                padding="random",
                no_event_token_rate=no_event_token_rate,
                cut_batch=True,
                apply_token_shift=apply_token_shift,
                separate_shift_na_from_padding=separate_shift_na_from_padding,
                shift_na_raw_token=shift_na_raw_token,
            )
            x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages = batch

            _, loss, _ = model(
                x_data,
                x_shift,
                x_total,
                x_ages,
                targets_data=y_data,
                targets_shift=y_shift,
                targets_total=y_total,
                targets_age=y_ages,
                validation_loss_mode=True,
                return_attention=False,
            )

            target_mask = y_data != -1
            for token in ignored_tokens:
                target_mask = target_mask & (y_data != int(token))
            batch_valid_targets = int(target_mask.sum().item())
            batch_weight = max(batch_valid_targets, 1)

            weighted_loss_sum += np.array(
                [float(loss[key].detach().cpu().item()) for key in loss_keys],
                dtype=np.float64,
            ) * batch_weight
            total_weight += batch_weight
            total_patients += len(batch_patient_indices)
            valid_data_targets += batch_valid_targets

    if total_weight == 0:
        raise ValueError("No valid targets found for next-token prediction evaluation")

    averaged = weighted_loss_sum / total_weight
    return {
        "next_token_loss": float(averaged[0]),
        "next_token_loss_data": float(averaged[1]),
        "next_token_loss_shift": float(averaged[2]),
        "next_token_loss_total": float(averaged[3]),
        "next_token_loss_time": float(averaged[4]),
        "next_token_patients": int(total_patients),
        "next_token_valid_data_targets": int(valid_data_targets),
    }


# New internal function that performs the AUC evaluation pipeline.
def evaluate_auc_pipeline(
    model,
    d100k,
    output_path,
    labels_df,
    model_type='composite',
    evaluate_composite=True,
    diseases_of_interest=None,
    filter_min_total=100,
    disease_chunk_size=200,
    age_groups=np.arange(40, 80, 5),
    offset=0.1,
    batch_size=64,
    device="cpu",
    seed=1337,
    n_bootstrap=1,
    disease_score_mode="logits",
    meta_info={},
    train_valid_tokens=None,  # Set of tokens present in train data (for filtering)
    composite_model=None,
    auc_sex_slices="all,female,male",
    use_delong=True,
    disease_token_sets=None,
    disease_label_map=None,
    disease_icd_code_map=None,
    exclude_diseases_of_interest=None,
    exclude_auc_control_tokens=None,
    composite_raw_output_path=None,
    composite_raw_output_prefix=None,
):
    """
    Runs the AUC evaluation pipeline.

    Args:
        model (torch.nn.Module): The loaded model set to eval().
        d100k: Data batch from get_batch_composite (CompositeDelphi).
        labels_df (pd.DataFrame): DataFrame with label info (token names, etc.).
        output_path (str | None): Directory where CSV files will be written. If None, files will not be saved.
        model_type (str): must be 'composite'
        diseases_of_interest (np.ndarray or list, optional): If provided, these disease indices are used.
        filter_min_total (int): Minimum total token count to include a token.
        disease_chunk_size (int): Maximum chunk size for processing diseases.
        age_groups (np.ndarray): Age groups to use in calibration.
        offset (float): Offset used in get_calibration_auc.
        batch_size (int): Batch size for model forwarding.
        device (str): Device identifier.
        seed (int): Random seed for reproducibility.
        n_bootstrap (int): Number of bootstrap samples. (1 for no bootstrap)
        disease_score_mode (str): Disease score for AUC: logits, data_prob, risk, or time_rate.
        meta_info (dict): Additional metadata to add to output DataFrames.
        use_delong (bool): Whether to compute DeLong AUC variance columns.
        disease_token_sets (dict[int, list[int]], optional): Maps each evaluation
            token/group ID to DATA tokens that count as positives. Used for CKB
            codebook groups where any overlapping ICD-10 token is a case.
        disease_label_map (dict[int, str], optional): Names for synthetic
            evaluation groups.
        disease_icd_code_map (dict[int, str], optional): ICD-10 code lists for
            synthetic evaluation groups.
        exclude_diseases_of_interest (set[int], optional): Evaluation token/group
            IDs to remove, e.g. cohort-defining diabetes targets in DM cohorts.
        exclude_auc_control_tokens (list[int], optional): DATA tokens to remove
            from the disease AUC control pool. This is used for pp EOT rows.
        composite_raw_output_path (str | Path | None): Optional directory where
            per-event SHIFT/TOTAL raw predictions are written.
        composite_raw_output_prefix (str | None): Prefix for raw prediction
            files. Useful when one output directory contains multiple datasets.
    Returns:
        tuple: (df_auc_unpooled, df_auc_merged, composite_metrics, df_auc_age_stratified).
        ``df_auc_age_stratified`` aggregates unpooled AUC across diseases per age bin.
    """

    assert n_bootstrap > 0, "n_bootstrap must be greater than 0"

    disease_score_mode = str(disease_score_mode).lower()
    valid_score_modes = {"logits", "data_prob", "risk", "time_rate"}
    if disease_score_mode not in valid_score_modes:
        raise ValueError(
            f"Unsupported disease_score_mode={disease_score_mode!r}. "
            f"Expected one of {sorted(valid_score_modes)}"
        )
    if meta_info is not None:
        meta_info["disease_score_mode"] = disease_score_mode
        meta_info["use_delong"] = bool(use_delong)
    if exclude_auc_control_tokens is None:
        exclude_auc_control_tokens = []
    else:
        exclude_auc_control_tokens = sorted({int(tok) for tok in exclude_auc_control_tokens})
    if meta_info is not None:
        meta_info["exclude_auc_control_tokens"] = ",".join(map(str, exclude_auc_control_tokens))
    if exclude_auc_control_tokens:
        print(f"AUC controls: excluding DATA target tokens {exclude_auc_control_tokens}")
    exclude_diseases_of_interest = (
        set() if exclude_diseases_of_interest is None
        else {int(tok) for tok in exclude_diseases_of_interest}
    )

    grouped_disease_targets = disease_token_sets is not None
    if grouped_disease_targets:
        disease_token_sets = {
            int(group_id): sorted({int(tok) for tok in tokens})
            for group_id, tokens in disease_token_sets.items()
        }
        disease_label_map = {} if disease_label_map is None else {
            int(group_id): str(name) for group_id, name in disease_label_map.items()
        }
        disease_icd_code_map = {} if disease_icd_code_map is None else {
            int(group_id): str(codes) for group_id, codes in disease_icd_code_map.items()
        }

    def _base_disease_scores(outputs, token_ids):
        data_logits = outputs["data"]
        if disease_score_mode == "logits":
            return data_logits[:, :, token_ids]
        if disease_score_mode == "data_prob":
            return torch.softmax(data_logits, dim=-1)[:, :, token_ids]

        time_logits = outputs.get("time_scale", outputs.get("time", None))
        if time_logits is None:
            raise RuntimeError(
                f"disease_score_mode={disease_score_mode!r} requires a time_scale/time head"
            )

        t_min = float(getattr(model.config, "t_min", 0.1))
        log_t_min = np.log(max(t_min, 1e-8))
        if disease_score_mode == "time_rate":
            log_lambda = time_logits - torch.nn.functional.softplus(time_logits + log_t_min)
            return torch.exp(torch.clamp(log_lambda[:, :, token_ids], min=-20.0, max=20.0))

        lse = torch.logsumexp(time_logits, dim=-1, keepdim=True)
        log_lambda_total = lse - torch.nn.functional.softplus(lse + log_t_min)
        lambda_total = torch.exp(torch.clamp(log_lambda_total, min=-20.0, max=20.0))
        event_any = 1.0 - torch.exp(-lambda_total * float(offset))
        data_prob = torch.softmax(data_logits, dim=-1)
        return data_prob[:, :, token_ids] * event_any

    def _disease_scores(outputs, diseases_chunk):
        if not grouped_disease_targets:
            return _base_disease_scores(outputs, diseases_chunk)

        chunk_token_sets = []
        for group_id in diseases_chunk:
            tokens = disease_token_sets.get(int(group_id), [])
            tokens = [int(tok) for tok in tokens if 0 <= int(tok) < vocab_size]
            if not tokens:
                raise ValueError(f"No valid DATA target tokens for disease group {group_id}")
            chunk_token_sets.append(tokens)

        unique_tokens = sorted({tok for tokens in chunk_token_sets for tok in tokens})
        token_scores = _base_disease_scores(outputs, unique_tokens)
        token_pos = {tok: i for i, tok in enumerate(unique_tokens)}

        group_scores = []
        for tokens in chunk_token_sets:
            idx = [token_pos[tok] for tok in tokens if tok in token_pos]
            scores = token_scores[:, :, idx]
            if scores.shape[-1] == 1:
                group_scores.append(scores[:, :, 0])
            elif disease_score_mode == "logits":
                group_scores.append(scores.max(dim=-1).values)
            else:
                group_scores.append(scores.sum(dim=-1))
        return torch.stack(group_scores, dim=-1)

    print(f"Disease score mode: {disease_score_mode}")

    # Set random seeds
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Get model vocab size to filter out invalid indices
    config_vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else model.config.data_vocab_size if hasattr(model.config, 'data_vocab_size') else 1290
    
    # Adjust vocab_size if labels indicate more tokens (e.g., Death token at 1289)
    # Note: labels_df indices are 0-based and represent raw data values.
    # If max index is 1288 (Death raw), after +1 shift max token is 1289.
    # We need vocab_size > 1289 (i.e. >= 1290) to include it.
    # Use model's vocab_size (trust the model config)
    vocab_size = config_vocab_size
    
    # Optional: warn if labels suggest a larger vocab_size
    if 'index' in labels_df.columns:
        max_label_index = labels_df['index'].max()
        # If labels contain indices >= vocab_size, they will be filtered out
        if max_label_index >= vocab_size - 1:
            print(f"Warning: labels contain index {max_label_index}, but model vocab_size is {vocab_size}.")
            print(f"         Tokens with index >= {vocab_size} will be excluded from evaluation.")
    
    # Get common diseases
    _apply_token_shift = bool(getattr(model.config, 'apply_token_shift', False))
    if diseases_of_interest is None:
        if grouped_disease_targets:
            raise ValueError("diseases_of_interest must be provided when disease_token_sets is used")
        diseases_of_interest = get_common_diseases(labels_df, filter_min_total, apply_token_shift=_apply_token_shift)
    diseases_of_interest = [int(d) for d in diseases_of_interest]
    if exclude_diseases_of_interest:
        diseases_before_exclusion = len(diseases_of_interest)
        diseases_of_interest = [d for d in diseases_of_interest if d not in exclude_diseases_of_interest]
        diseases_excluded = diseases_before_exclusion - len(diseases_of_interest)
        if diseases_excluded > 0:
            target_label = "disease groups" if grouped_disease_targets else "diseases"
            print(
                f"Excluded {diseases_excluded} requested {target_label}: "
                f"{sorted(exclude_diseases_of_interest)}"
            )
    
    if grouped_disease_targets:
        missing_groups = [d for d in diseases_of_interest if d not in disease_token_sets]
        if missing_groups:
            raise ValueError(f"Missing disease_token_sets entries for disease groups: {missing_groups[:10]}")

        filtered_token_sets = {}
        for group_id in diseases_of_interest:
            valid_tokens = [
                int(tok) for tok in disease_token_sets[group_id]
                if 0 <= int(tok) < vocab_size
            ]
            if valid_tokens:
                filtered_token_sets[group_id] = sorted(set(valid_tokens))
        diseases_before_vocab_filter = len(diseases_of_interest)
        diseases_of_interest = [d for d in diseases_of_interest if d in filtered_token_sets]
        disease_token_sets = filtered_token_sets
        diseases_filtered_vocab = diseases_before_vocab_filter - len(diseases_of_interest)
        if diseases_filtered_vocab > 0:
            print(f"Filtered out {diseases_filtered_vocab} disease groups with no valid model-vocab tokens")
    else:
        # Filter out invalid indices (must be < vocab_size)
        # Note: token indices are 0-based, so valid range is [0, vocab_size)
        diseases_of_interest = [d for d in diseases_of_interest if 0 <= d < vocab_size]
    
    if model_type != 'composite':
        raise ValueError("Only composite model_type is supported.")

    # CRITICAL: Filter to only include tokens that actually exist in the evaluation data
    # This prevents evaluating tokens like SGLT-2 or Other that may not exist in val/test data
    target_data_np = d100k[4].cpu().detach().numpy()  # y_data (target DATA tokens)
    # d100k = (x_data, x_shift, x_total, x_ages, y_data, y_shift, y_total, y_ages)
    
    actual_tokens_in_data = set(np.unique(target_data_np).tolist())
    # Remove invalid tokens like -1 (padding)
    actual_tokens_in_data = {t for t in actual_tokens_in_data if t >= 0}
    
    # Filter diseases to only those present in the evaluation data
    diseases_before_filter = len(diseases_of_interest)
    if grouped_disease_targets:
        diseases_of_interest = [
            d for d in diseases_of_interest
            if any(tok in actual_tokens_in_data for tok in disease_token_sets.get(d, []))
        ]
    else:
        diseases_of_interest = [d for d in diseases_of_interest if d in actual_tokens_in_data]
    diseases_filtered_eval = diseases_before_filter - len(diseases_of_interest)
    
    if diseases_filtered_eval > 0:
        target_label = "disease groups" if grouped_disease_targets else "diseases"
        print(f"Filtered out {diseases_filtered_eval} {target_label} not present in evaluation data")
    
    # CRITICAL: Filter to only include tokens present in train data
    # This ensures we only evaluate tokens the model was trained on
    # (e.g., excludes SGLT-2, Other if not in train)
    if train_valid_tokens is not None:
        diseases_before_train_filter = len(diseases_of_interest)
        if grouped_disease_targets:
            disease_token_sets = {
                group_id: [tok for tok in disease_token_sets[group_id] if tok in train_valid_tokens]
                for group_id in diseases_of_interest
            }
            diseases_of_interest = [
                group_id for group_id in diseases_of_interest
                if len(disease_token_sets.get(group_id, [])) > 0
            ]
        else:
            diseases_of_interest = [d for d in diseases_of_interest if d in train_valid_tokens]
        diseases_filtered_train = diseases_before_train_filter - len(diseases_of_interest)
        
        if diseases_filtered_train > 0:
            target_label = "disease groups" if grouped_disease_targets else "diseases"
            print(f"Filtered out {diseases_filtered_train} {target_label} not present in train data")
    
    if len(diseases_of_interest) == 0:
        raise ValueError(f"No valid diseases found. All indices must be in range [0, {vocab_size}), present in evaluation data, and present in train data")
    
    target_label = "disease groups" if grouped_disease_targets else "diseases"
    print(f"Evaluating {len(diseases_of_interest)} {target_label} (vocab_size={vocab_size}, actual unique tokens in eval data={len(actual_tokens_in_data)})")

    # Split diseases into chunks for processing
    num_chunks = (len(diseases_of_interest) + disease_chunk_size - 1) // disease_chunk_size
    diseases_chunks = np.array_split(diseases_of_interest, num_chunks)

    # Precompute prediction indices for calibration
    data_tokens = d100k[0].cpu().detach().numpy()  # x_data
    ages = d100k[3].cpu().detach().numpy()  # x_ages
    target_data = d100k[4].cpu().detach().numpy()  # y_data
    target_ages = d100k[7].cpu().detach().numpy()  # y_ages
    d = [data_tokens, ages, target_data, target_ages]
    
    # Precompute prediction indices: find positions where input age <= target age - offset
    pred_idx_precompute = (d[1][:, :, np.newaxis] <= d[3][:, np.newaxis, :] - offset).sum(1) - 1

    # --- Sex detection per patient ---
    _apply_shift = bool(getattr(model.config, 'apply_token_shift', False))
    sex_arr = _detect_sex_per_patient(data_tokens, _apply_shift)
    n_female = int((sex_arr == 1).sum())
    n_male = int((sex_arr == 2).sum())
    print(f"Sex distribution: female={n_female}, male={n_male}, unknown={int((sex_arr == 0).sum())}")

    requested_sex_slices = {
        item.strip().lower()
        for item in str(auc_sex_slices).split(",")
        if item.strip()
    }
    if not requested_sex_slices:
        requested_sex_slices = {"all"}
    invalid_sex_slices = requested_sex_slices - {"all", "female", "male"}
    if invalid_sex_slices:
        raise ValueError(
            f"Unsupported auc_sex_slices entries: {sorted(invalid_sex_slices)}. "
            "Expected a comma-separated subset of all,female,male."
        )
    if "all" not in requested_sex_slices:
        requested_sex_slices.add("all")

    sex_slices = [("all", None)]
    if "female" in requested_sex_slices and n_female >= 10:
        sex_slices.append(("female", 1))
    if "male" in requested_sex_slices and n_male >= 10:
        sex_slices.append(("male", 2))
    if meta_info is not None:
        meta_info["auc_sex_slices"] = ",".join(label for label, _ in sex_slices)

    # Pre-filter data / pred_idx per sex (avoid recomputing inside the chunk loop)
    sex_data = {}
    for sex_label, sex_val in sex_slices:
        if sex_val is None:
            sex_data[sex_label] = (d, pred_idx_precompute)
        else:
            d_sub, _, pred_sub = _filter_data_by_sex(d, None, pred_idx_precompute, sex_arr, sex_val)
            sex_data[sex_label] = (d_sub, pred_sub)

    all_aucs = []
    tqdm_options = {"desc": "Processing disease chunks", "total": len(diseases_chunks)}
    for disease_chunk_idx, diseases_chunk in tqdm(enumerate(diseases_chunks), **tqdm_options):
        diseases_chunk = np.array(diseases_chunk)
        if grouped_disease_targets:
            diseases_chunk = diseases_chunk.tolist()
        else:
            # Filter out invalid indices for this chunk
            valid_mask = (diseases_chunk >= 0) & (diseases_chunk < vocab_size)
            diseases_chunk = diseases_chunk[valid_mask].tolist()
        
        if len(diseases_chunk) == 0:
            print(f"Skipping chunk {disease_chunk_idx}: no valid diseases")
            continue
        
        p100k = []
        model.to(device)
        model.eval()
        with torch.no_grad():
            # Process the evaluation data in batches
            x_data, x_shift, x_total, x_ages = d100k[0], d100k[1], d100k[2], d100k[3]
            num_batches = (x_data.shape[0] + batch_size - 1) // batch_size
            for batch_idx in tqdm(range(num_batches), desc=f"Model inference, chunk {disease_chunk_idx}"):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, x_data.shape[0])

                batch_x_data = x_data[start_idx:end_idx].to(device)
                batch_x_shift = x_shift[start_idx:end_idx].to(device)
                batch_x_total = x_total[start_idx:end_idx].to(device)
                batch_x_ages = x_ages[start_idx:end_idx].to(device)

                outputs = model(
                    batch_x_data, batch_x_shift, batch_x_total, batch_x_ages,
                    return_attention=False,
                )[0]  # Get logits dict

                scores = _disease_scores(outputs, diseases_chunk)
                p100k.append(scores.cpu().detach().numpy().astype("float16"))
        
        if len(p100k) == 0:
            print(f"Skipping chunk {disease_chunk_idx}: no predictions generated")
            continue
        
        p100k = np.vstack(p100k)

        # Pre-filter p100k per sex
        sex_p = {}
        for sex_label, sex_val in sex_slices:
            if sex_val is None:
                sex_p[sex_label] = p100k
            else:
                sex_p[sex_label] = p100k[sex_arr == sex_val]

        for j, k in tqdm(
            list(enumerate(diseases_chunk)), desc=f"Processing diseases in chunk {disease_chunk_idx}"
        ):
            target_tokens = disease_token_sets.get(int(k), [int(k)]) if grouped_disease_targets else [int(k)]
            for sex_label, _ in sex_slices:
                d_sex, pred_sex = sex_data[sex_label]
                p_sex = sex_p[sex_label]
                out = get_calibration_auc(
                    j,
                    k,
                    d_sex,
                    p_sex,
                    diseases_chunk,
                    age_groups=age_groups,
                    offset=offset,
                    precomputed_idx=pred_sex,
                    n_bootstrap=n_bootstrap,
                    use_delong=use_delong,
                    target_tokens=target_tokens,
                    exclude_control_tokens=exclude_auc_control_tokens,
                )
                if out is None:
                    continue
                for out_item in out:
                    out_item["sex"] = sex_label
                    all_aucs.append(out_item)

    df_auc_unpooled = pd.DataFrame(all_aucs)

    for key, value in meta_info.items():
        df_auc_unpooled[key] = value

    def _attach_group_metadata(df):
        if df is None or df.empty or not grouped_disease_targets:
            return df
        df = df.copy()
        df["ckb_codebook_id"] = df["token"].astype(int)
        if disease_label_map:
            df["name"] = df["ckb_codebook_id"].map(disease_label_map)
        if disease_icd_code_map:
            df["icd10_codes"] = df["ckb_codebook_id"].map(disease_icd_code_map)
        df["target_tokens"] = df["ckb_codebook_id"].map(
            lambda group_id: ",".join(map(str, disease_token_sets.get(int(group_id), [])))
        )
        df["n_target_tokens"] = df["ckb_codebook_id"].map(
            lambda group_id: len(disease_token_sets.get(int(group_id), []))
        )
        return df

    # Merge with labels if available
    if grouped_disease_targets:
        df_auc_unpooled_merged = _attach_group_metadata(df_auc_unpooled)
    elif 'index' in labels_df.columns:
        labels_df_subset = labels_df[['index']].copy()
        if 'name' in labels_df.columns:
            labels_df_subset['name'] = labels_df['name']
        _apply_shift = bool(getattr(model.config, 'apply_token_shift', False))
        labels_df_subset = build_labels_df_for_merge(labels_df_subset, _apply_shift)
        df_auc_unpooled_merged = df_auc_unpooled.merge(
            labels_df_subset, left_on="token", right_on="shifted_token", how="inner"
        )
    else:
        df_auc_unpooled_merged = df_auc_unpooled.copy()
    df_auc_unpooled_merged = finalize_token_columns(df_auc_unpooled_merged)

    # Age-stratified summary (use "all" sex only for backward-compatible age table)
    _unpooled_all = df_auc_unpooled_merged
    if "sex" in _unpooled_all.columns:
        _unpooled_all = _unpooled_all.loc[_unpooled_all["sex"] == "all"]
    df_auc_age_stratified = build_age_stratified_auc_summary(_unpooled_all)
    if df_auc_age_stratified is not None and not df_auc_age_stratified.empty:
        print("\n[AUC by age stratum] (across diseases; prediction-time age bin, years)")
        print(df_auc_age_stratified.to_string(index=False))

    # Sex-stratified summary
    df_auc_sex_stratified = build_sex_stratified_auc_summary(df_auc_unpooled_merged)
    if df_auc_sex_stratified is not None and not df_auc_sex_stratified.empty:
        print("\n[AUC by sex stratum] (across diseases)")
        print(df_auc_sex_stratified.to_string(index=False))

    def aggregate_age_brackets_delong(group):
        # For normal distributions, when averaging n of them:
        # The variance of the sum is the sum of variances
        # The variance of the average is the sum of variances divided by n^2
        n = len(group)
        auc_col = "auc_delong" if use_delong and "auc_delong" in group.columns else "auc"
        
        # Handle cases where all AUC values are NaN (insufficient data)
        valid_aucs = group[auc_col].dropna()
        if len(valid_aucs) == 0:
            mean = np.nan
            var = np.nan
            status = group['status'].iloc[0] if 'status' in group.columns else 'unknown'
        else:
            mean = valid_aucs.mean()
            # Since we're taking the average, divide combined variance by n^2
            if use_delong and "auc_variance_delong" in group.columns:
                valid_vars = group.loc[valid_aucs.index, 'auc_variance_delong']
                var = valid_vars.sum() / (len(valid_vars)**2) if len(valid_vars) > 0 else np.nan
            else:
                var = np.nan
            status = 'ok'
        
        # Ensure var is a scalar (not array) for parquet compatibility
        if isinstance(var, np.ndarray):
            var = var.item() if var.size == 1 else float(var[0, 0]) if var.ndim > 0 else float(var)
        elif not np.isnan(var):
            var = float(var)
        
        return pd.Series({
            'auc': mean,
            'auc_variance_delong': var,
            'n_samples': n, 
            'n_diseased': group['n_diseased'].sum(),
            'n_healthy': group['n_healthy'].sum(),
            'status': status,
        })

    if use_delong:
        print('Using DeLong method to calculate AUC confidence intervals..')
    
    group_cols = ["token", "sex"] if "sex" in df_auc_unpooled.columns else ["token"]
    df_auc = df_auc_unpooled.groupby(group_cols).apply(aggregate_age_brackets_delong, include_groups=False).reset_index()
    
    if 'index' in labels_df.columns:
        if grouped_disease_targets:
            df_auc_merged = _attach_group_metadata(df_auc)
        else:
            _apply_shift = bool(getattr(model.config, 'apply_token_shift', False))
            labels_df_for_merge = build_labels_df_for_merge(labels_df, _apply_shift)
            df_auc_merged = df_auc.merge(labels_df_for_merge, left_on="token", right_on="shifted_token", how="inner")
    else:
        df_auc_merged = _attach_group_metadata(df_auc) if grouped_disease_targets else df_auc.copy()
    df_auc_merged = finalize_token_columns(df_auc_merged)
    
    # Evaluate composite fields (SHIFT, TOTAL) if composite model and enabled
    composite_metrics = None
    if evaluate_composite:
        print("\nEvaluating composite fields (SHIFT, TOTAL)...")
        aux_eval_model = model if composite_model is None else composite_model
        composite_metrics = evaluate_composite_fields(
            aux_eval_model,
            d100k,
            batch_size=batch_size,
            device=device,
            raw_output_path=composite_raw_output_path,
            raw_output_prefix=composite_raw_output_prefix,
        )
        composite_metrics = {**meta_info, **composite_metrics}
        
        # Print results: drug-token subset only (same metrics as *_drug_cond); no extra section title.
        print("\nComposite Field Evaluation Results:")
        print("=" * 60)

        def _print_shift_summary_dc(cm_metrics: dict, prefix: str):
            """Print SHIFT summary using drug-token subset keys (*_drug_cond)."""
            acc_k = f'{prefix}shift_accuracy_drug_cond'
            if acc_k not in cm_metrics:
                return False
            print("SHIFT (Binary: Label1 vs Label2or3):")
            print(f"  Accuracy: {cm_metrics[acc_k]:.4f}")
            bak = f'{prefix}shift_balanced_accuracy_drug_cond'
            if bak in cm_metrics:
                print(f"  Balanced Accuracy: {cm_metrics[bak]:.4f}")
            auc_k = f'{prefix}shift_roc_auc_drug_cond'
            if auc_k in cm_metrics:
                auc_v = cm_metrics[auc_k]
                if auc_v is not None and not (isinstance(auc_v, float) and np.isnan(auc_v)):
                    print(f"  AUC: {float(auc_v):.4f}")
            f1b_k = f'{prefix}shift_f1_binary_drug_cond'
            if f1b_k in cm_metrics:
                print(f"  F1: {cm_metrics[f1b_k]:.4f}")
            sup_k = f'{prefix}shift_support_drug_cond'
            if sup_k in cm_metrics:
                print(f"  Support: {cm_metrics[sup_k]}")
            cm_key = f'{prefix}shift_confusion_matrix_drug_cond'
            cm_cls_key = f'{prefix}shift_confusion_matrix_drug_cond_classes'
            if cm_key in cm_metrics and cm_cls_key in cm_metrics:
                cm_drug = np.array(cm_metrics[cm_key])
                classes_drug = cm_metrics[cm_cls_key]
                print("\n  Confusion Matrix:")
                print("    NOTE: Classes are 0=Label1, 1=Label2or3")
                print("    Predicted →")
                header = "    Actual ↓   " + "  ".join([f"{int(c):>5}" for c in classes_drug])
                print(header)
                print("    " + "-" * len(header[4:]))
                for i, cls in enumerate(classes_drug):
                    row_str = f"    {int(cls):>5} " + "  ".join(
                        [f"{int(cm_drug[i, j]):>5}" for j in range(len(classes_drug))]
                    )
                    print(row_str)
            pcm_key = f'{prefix}shift_per_class_metrics_drug_cond'
            if pcm_key in cm_metrics:
                print("\n  Per-Class Metrics:")
                for cls, metrics in sorted(cm_metrics[pcm_key].items()):
                    print(
                        f"    Class {cls}: Precision={metrics['precision']:.4f}, "
                        f"Recall={metrics['recall']:.4f}, F1={metrics['f1']:.4f}, "
                        f"Support={metrics['support']}"
                    )
            return True

        def _print_total_summary_dc(cm_metrics: dict, prefix: str):
            mae_k = f'{prefix}total_mae_drug_cond'
            if mae_k not in cm_metrics:
                return False
            print("TOTAL:")
            print(f"  MAE: {cm_metrics[mae_k]:.4f}")
            if f'{prefix}total_rmse_drug_cond' in cm_metrics:
                print(f"  RMSE: {cm_metrics[f'{prefix}total_rmse_drug_cond']:.4f}")
            r2_k = f'{prefix}total_r2_drug_cond'
            if r2_k in cm_metrics and not np.isnan(cm_metrics[r2_k]):
                print(f"  R²: {cm_metrics[r2_k]:.4f}")
            return True

        printed_shift = _print_shift_summary_dc(composite_metrics, "")
        if printed_shift:
            print()
        printed_total = _print_total_summary_dc(composite_metrics, "")
        if not (printed_shift or printed_total):
            print(
                "(No drug-token subset metrics: need use_drug_conditioning and "
                "shift_drug_cond / total_drug_cond outputs with drug tokens in the batch.)"
            )
        print("=" * 60)
        
        # Save composite metrics
        if output_path is not None:
            with open(f"{output_path}/composite_metrics.json", 'w') as f:
                # Convert numpy types to native Python types for JSON
                json_metrics = {}
                for k, v in composite_metrics.items():
                    if isinstance(v, dict):
                        json_metrics[k] = {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating, int, float)) else v2 
                                          for k2, v2 in v.items()}
                    elif isinstance(v, (np.integer, np.floating, np.ndarray)):
                        if isinstance(v, np.ndarray):
                            json_metrics[k] = v.tolist()
                        else:
                            json_metrics[k] = float(v)
                    elif isinstance(v, (int, float, bool, str)):
                        json_metrics[k] = v
                    else:
                        # Try to convert to float, if fails, convert to string
                        try:
                            json_metrics[k] = float(v)
                        except (ValueError, TypeError):
                            json_metrics[k] = str(v)
                json.dump(json_metrics, f, indent=2)
            print(f"Composite metrics saved to {output_path}/composite_metrics.json")
    
    if output_path is not None:
        Path(output_path).mkdir(exist_ok=True, parents=True)
        # Backward-compatible pooled file: keep only sex=="all"
        _merged_all = df_auc_merged
        if "sex" in df_auc_merged.columns:
            _merged_all = df_auc_merged.loc[df_auc_merged["sex"] == "all"]
        _merged_all.to_parquet(f"{output_path}/df_both.parquet", index=False)
        df_auc_unpooled_merged.to_parquet(f"{output_path}/df_auc_unpooled.parquet", index=False)
        if df_auc_age_stratified is not None and not df_auc_age_stratified.empty:
            df_auc_age_stratified.to_parquet(f"{output_path}/df_auc_age_stratified.parquet", index=False)
            df_auc_age_stratified.to_csv(f"{output_path}/df_auc_age_stratified.csv", index=False)
        if df_auc_sex_stratified is not None and not df_auc_sex_stratified.empty:
            df_auc_sex_stratified.to_parquet(f"{output_path}/df_auc_sex_stratified.parquet", index=False)
            df_auc_sex_stratified.to_csv(f"{output_path}/df_auc_sex_stratified.csv", index=False)
        # Per-sex pooled AUC (one row per token × sex)
        if "sex" in df_auc_merged.columns:
            df_auc_merged.to_parquet(f"{output_path}/df_both_by_sex.parquet", index=False)
            df_auc_merged.to_csv(f"{output_path}/df_both_by_sex.csv", index=False)

    return df_auc_unpooled_merged, df_auc_merged, composite_metrics, df_auc_age_stratified, df_auc_sex_stratified


def main():
    parser = argparse.ArgumentParser(description="Evaluate AUC")
    parser.add_argument("--input_path", type=str, default="../data", help="Path to the dataset")
    parser.add_argument("--output_path", type=str, default="results", help="Path to the output")
    parser.add_argument("--model_ckpt_path", type=str, required=True, help="Path to the model weights")
    parser.add_argument(
        "--aux_model_ckpt_path",
        type=str,
        default=None,
        help="Optional checkpoint used only for composite SHIFT/TOTAL evaluation.",
    )
    parser.add_argument("--model_type", type=str, default='composite', choices=['composite'],
                        help="Model type (composite only)")
    parser.add_argument("--no_event_token_rate", type=int, default=5, help="No event token rate")
    parser.add_argument(
        "--health_token_replacement_prob", default=0.0, type=float, help="Health token replacement probability"
    )
    parser.add_argument("--dataset_subset_size", type=int, default=10000, help="Dataset subset size for evaluation (-1 for all)")
    parser.add_argument("--n_bootstrap", type=int, default=1, help="Number of bootstrap samples")
    # Optional filtering/chunking parameters:
    parser.add_argument("--filter_min_total", type=int, default=0, help="Minimum total count to filter tokens (0=include all)")
    parser.add_argument("--disease_chunk_size", type=int, default=200, help="Chunk size for processing diseases")
    parser.add_argument("--labels_path", type=str, default=None, help="Path to labels CSV file")
    parser.add_argument("--block_size", type=int, default=512, help="Block size for data loading")
    parser.add_argument("--eval_batch_size", type=int, default=64, help="Batch size for model inference during evaluation")
    parser.add_argument(
        "--keep_batch_on_cpu",
        action="store_true",
        help=(
            "Keep the large prepared evaluation batch on CPU RAM and move only "
            "inference mini-batches to the selected device. Useful for full "
            "external cohorts that do not fit in GPU memory."
        ),
    )
    parser.add_argument("--offset", type=float, default=0.1,
                        help="Prediction lead-time offset in days for disease AUC evaluation")
    parser.add_argument(
        "--disease_score_mode",
        type=str,
        default="logits",
        choices=["logits", "data_prob", "risk", "time_rate"],
        help=(
            "Score used for disease AUC. logits preserves the historical behavior; "
            "risk uses softmax(DATA) times the time-head probability of any event by --offset."
        ),
    )
    parser.add_argument("--age_group_min", type=int, default=40,
                        help="Min age (years) for AUC age-stratification bins (prediction-time age)")
    parser.add_argument("--age_group_max", type=int, default=80,
                        help="Max age (years, exclusive) for AUC age-stratification bins")
    parser.add_argument("--age_group_step", type=int, default=5,
                        help="Width of each age bin in years (must be uniform; same as get_calibration_auc)")
    parser.add_argument("--auc_sex_slices", type=str, default="all,female,male",
                        help="Comma-separated sex strata for AUC computation. Always includes all.")
    parser.add_argument("--data_files", type=str, default=None, 
                        help="Comma-separated list of data files to evaluate (e.g., 'kr_val.bin,kr_test.bin'). If None, evaluates all: kr_val.bin, kr_test.bin, JMDC_extval.bin, UKB_extval.bin, ckb_extval.bin")
    parser.add_argument("--ckb_eval_mode", type=str, default="codebook_icd_overlap",
                        choices=["codebook_icd_overlap", "exact_data_tokens"],
                        help=(
                            "CKB DATA AUC mode. codebook_icd_overlap evaluates the attached CKB "
                            "disease groups and treats any overlapping ICD-10 level-3 token as a case; "
                            "exact_data_tokens preserves the old exact-token evaluation."
                        ))
    parser.add_argument("--ckb_include_drug_death", action="store_true",
                        help="For ckb_extval.bin exact_data_tokens mode, include drug tokens and Death in DATA AUC targets. Default evaluates disease tokens only.")
    parser.add_argument("--include_cohort_diabetes_targets", action="store_true",
                        help=(
                            "Deprecated compatibility flag. Diabetes diagnosis targets are included by default."
                        ))
    parser.add_argument("--train_data_file", type=str, default="train_inner.bin",
                        help="Inner train data file used for diagnostics/token filtering.")
    parser.add_argument("--next_token_data_file", type=str, default="kr_val.bin",
                        help="Data file for final internal next-token prediction evaluation.")
    parser.add_argument("--next_token_subset_size", type=int, default=None,
                        help="Patient subset size for next-token prediction (-1 for all). Defaults to dataset_subset_size.")
    parser.add_argument("--skip_next_token_prediction", action="store_true",
                        help="Skip final internal next-token prediction evaluation.")
    parser.add_argument("--skip_composite_fields", action="store_true",
                        help="Skip SHIFT/TOTAL composite field metrics and compute Disease AUC only.")
    parser.add_argument("--save_composite_raw_predictions", action="store_true",
                        help="Save per-event SHIFT probabilities and TOTAL predictions for plotting and threshold tuning.")
    parser.add_argument("--skip_delong", action="store_true",
                        help="Skip DeLong variance/CI columns. Mean/median AUC values are still computed.")
    parser.add_argument("--exclude_eot_from_auc_controls", action="store_true",
                        help="For pp data, exclude EOT target rows from disease AUC control positions.")
    parser.add_argument("--auc_eot_token", type=int, default=None,
                        help="DATA token ID to treat as EOT for --exclude_eot_from_auc_controls. Defaults to checkpoint model_args['eot_token'].")
    args = parser.parse_args()

    input_path = args.input_path
    output_path = args.output_path
    model_type = args.model_type
    no_event_token_rate = args.no_event_token_rate
    dataset_subset_size = args.dataset_subset_size

    # Auto-derive output_path from ckpt path: out/MMDD/HHMM/ckpt.pt -> results/MMDD/HHMM
    if output_path == "results" and args.model_ckpt_path:
        ckpt_dir = str(Path(args.model_ckpt_path).parent)
        parts = ckpt_dir.replace("\\", "/").split("/")
        try:
            out_idx = parts.index("out")
            sub_parts = parts[out_idx + 1:]
            if sub_parts:
                output_path = str(Path("results") / Path(*sub_parts))
        except ValueError:
            pass

    # Create output folder if it doesn't exist.
    if output_path is not None:
        Path(output_path).mkdir(exist_ok=True, parents=True)

    device = "cuda" if torch.cuda.is_available() else ("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
    print(device)
    seed = 1337

    age_groups = np.arange(args.age_group_min, args.age_group_max, args.age_group_step)
    if age_groups.size < 2:
        raise ValueError(
            "Need at least two age bin starts for AUC evaluation. "
            "Adjust --age_group_min, --age_group_max, or --age_group_step."
        )

    # Load model checkpoint and initialize model.
    ckpt_path = args.model_ckpt_path
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = dict(checkpoint["model_args"])
    eval_eot_token = args.auc_eot_token
    if eval_eot_token is None and model_args.get('eot_token') is not None:
        eval_eot_token = int(model_args.get('eot_token'))
    eval_apply_token_shift = bool(model_args.get('apply_token_shift', False))
    eval_separate_shift_na = bool(model_args.get('separate_shift_na_from_padding', False))
    eval_shift_na_raw_token = int(model_args.get('shift_na_raw_token', 4))
    _eval_drug_min = int(model_args.get('drug_token_min', 1279 if eval_apply_token_shift else 1278))
    _eval_drug_max = int(model_args.get('drug_token_max', 1289 if eval_apply_token_shift else 1288))
    _space = "SHIFTED (+1)" if eval_apply_token_shift else "RAW"
    print(f"Token space: {_space} (apply_token_shift={eval_apply_token_shift}) | "
          f"Drug range: {_eval_drug_min}-{_eval_drug_max} | Death={_eval_drug_max}")
    print(f"separate_shift_na_from_padding from checkpoint: {eval_separate_shift_na}")
    if 'drug_token_min' not in model_args or 'drug_token_max' not in model_args:
        model_args['drug_token_min'] = 1279 if eval_apply_token_shift else 1278
        model_args['drug_token_max'] = 1289 if eval_apply_token_shift else 1288
        print(
            f"Checkpoint missing drug token range; using fallback "
            f"[{model_args['drug_token_min']}, {model_args['drug_token_max']}] "
            f"(apply_token_shift={eval_apply_token_shift})."
        )

    state_dict = checkpoint["model"]
    # Strip DDP 'module.' or torch.compile '_orig_mod.' prefixes if present
    cleaned = {}
    for k, v in state_dict.items():
        k = k.replace('module.', '').replace('_orig_mod.', '')
        cleaned[k] = v

    use_moe = bool(model_args.get('use_moe', False))
    # Extract MoE and other architecture info for metadata
    num_experts = model_args.get('num_experts', 0)
    experts_per_token = model_args.get('experts_per_token', 0)
    
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(CompositeDelphiConfig)}
    model_args = {k: v for k, v in model_args.items() if k in valid_fields}
    conf = CompositeDelphiConfig(**model_args)
    model = CompositeDelphi(conf)
    model.load_state_dict(cleaned)
    model.eval()
    model = model.to(device)

    aux_model = None
    if args.aux_model_ckpt_path:
        aux_checkpoint = torch.load(args.aux_model_ckpt_path, map_location=device)
        aux_model_args = dict(aux_checkpoint["model_args"])
        aux_apply_token_shift = bool(aux_model_args.get('apply_token_shift', False))
        aux_separate_shift_na = bool(aux_model_args.get('separate_shift_na_from_padding', False))
        if aux_apply_token_shift != eval_apply_token_shift:
            raise ValueError(
                "aux_model_ckpt_path apply_token_shift does not match primary model: "
                f"{aux_apply_token_shift} != {eval_apply_token_shift}"
            )
        if aux_separate_shift_na != eval_separate_shift_na:
            raise ValueError(
                "aux_model_ckpt_path separate_shift_na_from_padding does not match primary model: "
                f"{aux_separate_shift_na} != {eval_separate_shift_na}"
            )
        aux_state_dict = aux_checkpoint["model"]
        aux_cleaned = {}
        for k, v in aux_state_dict.items():
            k = k.replace('module.', '').replace('_orig_mod.', '')
            aux_cleaned[k] = v
        aux_model_args = {k: v for k, v in aux_model_args.items() if k in valid_fields}
        aux_conf = CompositeDelphiConfig(**aux_model_args)
        aux_model = CompositeDelphi(aux_conf)
        aux_model.load_state_dict(aux_cleaned)
        aux_model.eval()
        aux_model = aux_model.to(device)
        print(f"Auxiliary model for SHIFT/TOTAL: {args.aux_model_ckpt_path}")
    
    # Print model architecture info
    print(f"\n{'='*60}")
    print(f"Model Architecture Info:")
    print(f"  Model type: {model_type}")
    print(f"  Use MoE: {use_moe}")
    if use_moe:
        print(f"  Number of experts: {num_experts}")
        print(f"  Experts per token: {experts_per_token}")
    print(f"  Parameters: {model.get_num_params()/1e6:.2f}M")
    print(f"{'='*60}\n")

    # Load labels (external) to be passed in.
    # IMPORTANT: Use header=None to avoid treating first line as header!
    # labels.csv format: each line is "name," or "name" (may have trailing comma)
    # Line number = index (0-based), so line 0 = index 0, line 1288 = index 1288 (Death)
    if args.labels_path:
        labels_df = pd.read_csv(args.labels_path, header=None, usecols=[0], names=['name'])
        labels_df['index'] = range(len(labels_df))
    else:
        # Try to load from default location
        labels_path = f"{input_path}/labels.csv"
        if Path(labels_path).exists():
            labels_df = pd.read_csv(labels_path, header=None, usecols=[0], names=['name'])
            labels_df['index'] = range(len(labels_df))
        else:
            # Create a minimal labels DataFrame
            print(f"Warning: labels file not found at {labels_path}. Creating minimal labels DataFrame.")
            labels_df = pd.DataFrame({'index': range(2000), 'name': [f'token_{i}' for i in range(2000)]})

    # Define data files to evaluate with their prefixes
    # Format: (filename, prefix)
    if args.data_files:
        # Parse user-specified files
        data_files_list = []
        for f in args.data_files.split(','):
            f = f.strip()
            if f:
                # Generate prefix from filename
                f_lower = f.lower()
                if 'ckb' in f_lower:
                    prefix = 'extval_ckb'
                elif 'ukb' in f_lower and 'extval' in f_lower:
                    prefix = 'extval_ukb'
                elif 'jmdc' in f_lower and 'extval' in f_lower:
                    prefix = 'extval_jmdc'
                elif 'extval' in f_lower:
                    prefix = 'extval'
                elif 'val' in f_lower:
                    prefix = 'val'
                elif 'test' in f_lower:
                    prefix = 'test'
                else:
                    prefix = Path(f).stem
                data_files_list.append((f, prefix))
    else:
        # Default: evaluate all files (internal val/test + external validations)
        data_files_list = [
            ("kr_val.bin", "val"),
            ("kr_test.bin", "test"),
            ("JMDC_extval.bin", "extval_jmdc"),
            ("UKB_extval.bin", "extval_ukb"),
            ("ckb_extval.bin", "extval_ckb"),
        ]
    
    # Prepare meta info for results (base)
    base_meta_info = {
        'model_type': model_type,
        'use_moe': use_moe,
        'eval_offset_days': args.offset,
        'exclude_eot_from_auc_controls': bool(args.exclude_eot_from_auc_controls),
    }
    if eval_eot_token is not None:
        base_meta_info['eot_token'] = int(eval_eot_token)
    exclude_auc_control_tokens = []
    if args.exclude_eot_from_auc_controls:
        if eval_eot_token is None:
            print("[WARNING] --exclude_eot_from_auc_controls requested, but no EOT token was found. No control tokens will be excluded.")
        else:
            exclude_auc_control_tokens = [int(eval_eot_token)]
    if use_moe:
        base_meta_info['num_experts'] = num_experts
        base_meta_info['experts_per_token'] = experts_per_token
    
    # Add checkpoint info if available
    if 'iter_num' in checkpoint:
        base_meta_info['checkpoint_iter'] = checkpoint['iter_num']
    if 'best_val_loss' in checkpoint:
        base_meta_info['checkpoint_val_loss'] = checkpoint['best_val_loss']
    if args.aux_model_ckpt_path:
        base_meta_info['aux_model_ckpt_path'] = args.aux_model_ckpt_path

    # Define dtype for composite data
    # IMPORTANT: Must match train_model.py exactly!
    # Format: (ID, AGE, DATA, SHIFT, TOTAL) - NO DOSE, NO UNIT
    composite_dtype = np.dtype([
        ('ID', np.uint32),
        ('AGE', np.uint32),
        ('DATA', np.uint32),
        ('SHIFT', np.uint32),
        ('TOTAL', np.uint32)
    ])

    if not args.skip_next_token_prediction:
        next_token_path = Path(input_path) / args.next_token_data_file
        if not next_token_path.exists():
            print(f"\n[WARNING] Next-token data not found: {next_token_path}. Skipping next-token prediction.")
        else:
            print(f"\n{'='*60}")
            print(f"Final internal next-token prediction: {args.next_token_data_file}")
            print(f"{'='*60}\n")

            next_token_data = np.fromfile(next_token_path, dtype=composite_dtype)
            next_token_p2i = get_p2i_composite(next_token_data)
            next_token_subset_size = (
                dataset_subset_size
                if args.next_token_subset_size is None
                else args.next_token_subset_size
            )
            if next_token_subset_size == -1:
                next_token_subset_size = len(next_token_p2i)
            else:
                next_token_subset_size = min(next_token_subset_size, len(next_token_p2i))

            rng = np.random.default_rng(seed)
            next_token_patient_indices = rng.choice(
                len(next_token_p2i),
                size=next_token_subset_size,
                replace=False,
            )
            next_token_patient_indices = sorted(next_token_patient_indices.tolist())

            next_token_metrics = evaluate_next_token_prediction(
                model,
                next_token_data,
                next_token_p2i,
                next_token_patient_indices,
                block_size=args.block_size,
                batch_size=args.eval_batch_size,
                device=device,
                no_event_token_rate=no_event_token_rate,
                apply_token_shift=eval_apply_token_shift,
                separate_shift_na_from_padding=eval_separate_shift_na,
                shift_na_raw_token=eval_shift_na_raw_token,
            )
            next_token_metrics = {
                **base_meta_info,
                **next_token_metrics,
                "data_source": args.next_token_data_file,
                "data_prefix": "internal_test",
                "evaluation_role": "final_internal_test_next_token",
                "subset_size_requested": int(
                    dataset_subset_size
                    if args.next_token_subset_size is None
                    else args.next_token_subset_size
                ),
                "total_patients_available": int(len(next_token_p2i)),
            }

            print(
                "[next-token] "
                f"loss={next_token_metrics['next_token_loss']:.4f}, "
                f"data={next_token_metrics['next_token_loss_data']:.4f}, "
                f"shift={next_token_metrics['next_token_loss_shift']:.4f}, "
                f"total={next_token_metrics['next_token_loss_total']:.4f}, "
                f"time={next_token_metrics['next_token_loss_time']:.4f}, "
                f"patients={next_token_metrics['next_token_patients']}"
            )

            if output_path is not None:
                metrics_json_path = Path(output_path) / "internal_test_next_token_metrics.json"
                metrics_csv_path = Path(output_path) / "internal_test_next_token_metrics.csv"
                with open(metrics_json_path, "w") as f:
                    json.dump(next_token_metrics, f, indent=2)
                pd.DataFrame([next_token_metrics]).to_csv(metrics_csv_path, index=False)
                print(f"[next-token] metrics saved to {metrics_json_path}")
    
    # ============================================================
    # DIAGNOSTIC: Check SHIFT values in raw data (before +1 shift)
    # Only run for first data file to avoid spam
    # ============================================================
    diagnostic_run = False

    # Load train data to get valid tokens (only tokens in train should be evaluated)
    # NOTE: Train filtering only applies to data files with same prefix (e.g., kr_train → kr_val, kr_test)
    #       External validation (e.g., JMDC_extval) should NOT be filtered by kr_train
    train_data_path = f"{input_path}/{args.train_data_file}"
    train_valid_tokens = None
    train_prefix = args.train_data_file.split('_')[0] if '_' in args.train_data_file else None
    
    if Path(train_data_path).exists():
        print(f"\nLoading train data to filter valid tokens: {train_data_path}")
        print(f"  Train prefix: '{train_prefix}' (filtering will only apply to data files with same prefix)")
        train_data_raw = np.fromfile(train_data_path, dtype=composite_dtype)
        train_raw_tokens = np.unique(train_data_raw['DATA'])
        if eval_apply_token_shift:
            train_valid_tokens = set((train_raw_tokens + 1).tolist())
            token_shift_note = "after +1 shift"
        else:
            train_valid_tokens = set(train_raw_tokens.tolist())
            token_shift_note = "raw token space (no shift)"

        print(f"  Train data contains {len(train_valid_tokens)} unique tokens ({token_shift_note})")
        
        # Show which drug tokens are in train
        base_drug_token = 1279 if eval_apply_token_shift else 1278
        drug_token_names = {
            base_drug_token + 0: 'Metformin',
            base_drug_token + 1: 'Sulfonylurea',
            base_drug_token + 2: 'DPP-4',
            base_drug_token + 3: 'Insulin',
            base_drug_token + 4: 'Meglitinide',
            base_drug_token + 5: 'Thiazolidinedione',
            base_drug_token + 6: 'Alpha-glucosidase',
            base_drug_token + 7: 'GLP-1',
            base_drug_token + 8: 'SGLT-2',
            base_drug_token + 9: 'Other',
            base_drug_token + 10: 'Death',
        }
        print("  Drug tokens in train data:")
        for token, name in drug_token_names.items():
            status = "✓" if token in train_valid_tokens else "✗"
            print(f"    {status} {name} ({token})")
    else:
        print(f"\n[WARNING] Train data not found at {train_data_path}. Skipping train-based token filtering.")
        train_prefix = None

    # Process each data file
    all_results = {}
    for data_filename, prefix in data_files_list:
        data_filepath = f"{input_path}/{data_filename}"
        
        # Check if file exists
        if not Path(data_filepath).exists():
            print(f"\n{'='*60}")
            print(f"[WARNING] Skipping {data_filename}: file not found at {data_filepath}")
            print(f"{'='*60}\n")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {data_filename} (prefix: {prefix})")
        print(f"{'='*60}\n")
        
        # Load data
        data = np.fromfile(data_filepath, dtype=composite_dtype)
        data_p2i = get_p2i_composite(data)
        
        # Determine subset size
        current_subset_size = dataset_subset_size
        if current_subset_size == -1:
            current_subset_size = len(data_p2i)
        else:
            current_subset_size = min(current_subset_size, len(data_p2i))
        
        print(f"Using {current_subset_size} patients for evaluation (out of {len(data_p2i)} total)")
        
        # Sample random patients for evaluation
        np.random.seed(seed)
        patient_indices = np.random.choice(len(data_p2i), size=current_subset_size, replace=False)
        patient_indices = sorted(patient_indices)
        
        # Get a subset batch for evaluation
        d100k = get_batch_composite(
            patient_indices,
            data,
            data_p2i,
            select="left",
            block_size=args.block_size,
            device="cpu" if args.keep_batch_on_cpu else device,
            padding="random",
            no_event_token_rate=no_event_token_rate,
            apply_token_shift=eval_apply_token_shift,
            separate_shift_na_from_padding=eval_separate_shift_na,
            shift_na_raw_token=eval_shift_na_raw_token,
        )
        
        # Prepare meta info with data source
        meta_info = base_meta_info.copy()
        meta_info['data_source'] = data_filename
        meta_info['data_prefix'] = prefix

        diseases_of_interest = None
        disease_token_sets = None
        disease_label_map = None
        disease_icd_code_map = None
        exclude_diseases_of_interest = set()
        if prefix == 'extval_ckb':
            # CKB is a DATA-only external validation set with DATA values already
            # encoded as model vocabulary IDs. By default, evaluate the CKB
            # codebook disease groups: a group is positive if any of its ICD-10
            # level-3 DATA tokens appears in the target.
            token_offset = get_data_token_offset(eval_apply_token_shift)
            if args.ckb_eval_mode == "codebook_icd_overlap":
                if args.ckb_include_drug_death:
                    print("[WARNING] --ckb_include_drug_death is ignored in CKB codebook_icd_overlap mode.")
                (
                    disease_token_sets,
                    disease_label_map,
                    disease_icd_code_map,
                    ckb_missing_code_map,
                ) = build_ckb_codebook_group_maps(
                    labels_df,
                    token_offset=token_offset,
                    vocab_size=getattr(model.config, "vocab_size", None),
                )
                diseases_of_interest = sorted(disease_token_sets)
                ckb_target_mode = "codebook_icd_overlap"
                meta_info['ckb_target_mode'] = ckb_target_mode
                meta_info['ckb_codebook_groups_requested'] = ",".join(map(str, diseases_of_interest))
                meta_info['ckb_codebook_score_aggregation'] = (
                    "max_logit" if args.disease_score_mode == "logits" else "sum_scores"
                )
                if ckb_missing_code_map:
                    meta_info['ckb_codebook_groups_with_missing_codes'] = json.dumps(ckb_missing_code_map, sort_keys=True)
                print(
                    f"CKB target mode: {ckb_target_mode} "
                    f"({len(diseases_of_interest)} codebook groups before batch filtering; "
                    f"score aggregation={meta_info['ckb_codebook_score_aggregation']})"
                )
            else:
                raw_tokens = np.unique(data['DATA']).astype(np.int64)
                ckb_targets = raw_tokens[raw_tokens >= MIN_DISEASE_RAW_TOKEN_INDEX] + token_offset
                if args.ckb_include_drug_death:
                    ckb_target_mode = "exact_data_tokens_including_drug_death"
                else:
                    ckb_targets = ckb_targets[ckb_targets < _eval_drug_min]
                    ckb_target_mode = "exact_disease_tokens_only"
                diseases_of_interest = sorted({int(tok) for tok in ckb_targets.tolist()})
                meta_info['ckb_target_mode'] = ckb_target_mode
                meta_info['ckb_target_tokens_requested'] = ",".join(map(str, diseases_of_interest))
                print(
                    f"CKB target mode: {ckb_target_mode} "
                    f"({len(diseases_of_interest)} DATA targets before batch filtering)"
                )
        
        if prefix == 'extval_ckb' and disease_token_sets is not None:
            exclude_diseases_of_interest.add(CKB_MALIGNANT_NEOPLASMS_CODEBOOK_GROUP_ID)
            meta_info['excluded_ckb_codebook_groups'] = str(CKB_MALIGNANT_NEOPLASMS_CODEBOOK_GROUP_ID)
        
        # Call the internal evaluation function (don't save files yet - we'll save with prefix)
        result = evaluate_auc_pipeline(
            model,
            d100k,
            output_path=None,  # Don't save internally, we'll save with prefix
            labels_df=labels_df,
            model_type=model_type,
            # UKB/CKB external validation: only DATA AUC is needed (skip SHIFT/TOTAL).
            evaluate_composite=(prefix not in DATA_ONLY_EXTVAL_PREFIXES and not args.skip_composite_fields),
            diseases_of_interest=diseases_of_interest,
            filter_min_total=args.filter_min_total,
            disease_chunk_size=args.disease_chunk_size,
            age_groups=age_groups,
            batch_size=args.eval_batch_size,
            device=device,
            seed=seed,
            n_bootstrap=args.n_bootstrap,
            offset=args.offset,
            disease_score_mode=args.disease_score_mode,
            meta_info=meta_info,
            train_valid_tokens=None,  # No train filtering - evaluate all tokens in data
            composite_model=aux_model,
            auc_sex_slices=args.auc_sex_slices,
            use_delong=not args.skip_delong,
            disease_token_sets=disease_token_sets,
            disease_label_map=disease_label_map,
            disease_icd_code_map=disease_icd_code_map,
            exclude_diseases_of_interest=exclude_diseases_of_interest,
            exclude_auc_control_tokens=exclude_auc_control_tokens,
            composite_raw_output_path=output_path if args.save_composite_raw_predictions else None,
            composite_raw_output_prefix=prefix,
        )
        
        df_auc_unpooled_merged, df_auc_merged, composite_metrics, df_auc_age_stratified, df_auc_sex_stratified = result

        if composite_metrics is None:
            composite_metrics = {}

        # For backward-compatible AUC statistics, use sex=="all" rows only
        _df_merged_all = df_auc_merged
        if "sex" in df_auc_merged.columns:
            _df_merged_all = df_auc_merged.loc[df_auc_merged["sex"] == "all"]

        if _df_merged_all is not None and not _df_merged_all.empty and 'auc' in _df_merged_all.columns:
            auc_values = _df_merged_all['auc'].dropna()
            if not auc_values.empty:
                composite_metrics['auc_mean'] = float(auc_values.mean())
                composite_metrics['auc_median'] = float(auc_values.median())
                composite_metrics['auc_min'] = float(auc_values.min())
                composite_metrics['auc_max'] = float(auc_values.max())
                composite_metrics['auc_std'] = float(auc_values.std())
                composite_metrics['n_diseases_auc'] = int(len(auc_values))

                print(f"\n[AUC Statistics] (Next Disease Prediction)")
                print(f"  Mean:   {composite_metrics['auc_mean']:.4f}")
                print(f"  Median: {composite_metrics['auc_median']:.4f}")
                print(f"  Min/Max: {composite_metrics['auc_min']:.4f} / {composite_metrics['auc_max']:.4f}")
        
        # Save results with prefix
        if output_path is not None:
            # Backward-compatible pooled file: sex=="all" only
            if df_auc_merged is not None and not df_auc_merged.empty:
                _save_all = df_auc_merged
                if "sex" in df_auc_merged.columns:
                    _save_all = df_auc_merged.loc[df_auc_merged["sex"] == "all"]
                _save_all.to_parquet(f"{output_path}/{prefix}_df_both.parquet", index=False)
                _save_all.to_csv(f"{output_path}/{prefix}_df_both.csv", index=False)
                # Full per-sex pooled AUC
                if "sex" in df_auc_merged.columns:
                    df_auc_merged.to_parquet(f"{output_path}/{prefix}_df_both_by_sex.parquet", index=False)
                    df_auc_merged.to_csv(f"{output_path}/{prefix}_df_both_by_sex.csv", index=False)
            
            if df_auc_unpooled_merged is not None and not df_auc_unpooled_merged.empty:
                df_auc_unpooled_merged.to_parquet(f"{output_path}/{prefix}_df_auc_unpooled.parquet", index=False)
                df_auc_unpooled_merged.to_csv(f"{output_path}/{prefix}_df_auc_unpooled.csv", index=False)

            if df_auc_age_stratified is not None and not df_auc_age_stratified.empty:
                df_auc_age_stratified.to_parquet(
                    f"{output_path}/{prefix}_df_auc_age_stratified.parquet", index=False
                )
                df_auc_age_stratified.to_csv(f"{output_path}/{prefix}_df_auc_age_stratified.csv", index=False)

            if df_auc_sex_stratified is not None and not df_auc_sex_stratified.empty:
                df_auc_sex_stratified.to_parquet(
                    f"{output_path}/{prefix}_df_auc_sex_stratified.parquet", index=False
                )
                df_auc_sex_stratified.to_csv(f"{output_path}/{prefix}_df_auc_sex_stratified.csv", index=False)
            
            # Save composite metrics with prefix
            if composite_metrics:
                with open(f"{output_path}/{prefix}_composite_metrics.json", 'w') as f:
                    json_metrics = {}
                    for k, v in composite_metrics.items():
                        if isinstance(v, dict):
                            json_metrics[k] = {str(k2): float(v2) if isinstance(v2, (np.integer, np.floating, int, float)) else v2 
                                              for k2, v2 in v.items()}
                        elif isinstance(v, (np.integer, np.floating, np.ndarray)):
                            if isinstance(v, np.ndarray):
                                json_metrics[k] = v.tolist()
                            else:
                                json_metrics[k] = float(v)
                        elif isinstance(v, (int, float, bool, str)):
                            json_metrics[k] = v
                        else:
                            try:
                                json_metrics[k] = float(v)
                            except (ValueError, TypeError):
                                json_metrics[k] = str(v)
                    json.dump(json_metrics, f, indent=2)
        
        # Store results
        all_results[prefix] = {
            'df_auc_unpooled': df_auc_unpooled_merged,
            'df_auc_merged': df_auc_merged,
            'df_auc_age_stratified': df_auc_age_stratified,
            'df_auc_sex_stratified': df_auc_sex_stratified,
            'composite_metrics': composite_metrics,
            'data_filename': data_filename,
        }
        
        _n_diseases = len(_df_merged_all) if _df_merged_all is not None else 0
        print(f"\n[{prefix.upper()}] Evaluation completed!")
        print(f"  Total diseases evaluated: {_n_diseases}")
        if composite_metrics:
            metrics_label = (
                "DATA AUC summary metrics"
                if prefix in DATA_ONLY_EXTVAL_PREFIXES or args.skip_composite_fields
                else "Composite field / AUC summary metrics"
            )
            print(f"  {metrics_label} saved to {output_path}/{prefix}_composite_metrics.json")
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"ALL EVALUATIONS COMPLETED")
    print(f"{'='*60}")
    print(f"Results saved to: {output_path}")
    for prefix, result_data in all_results.items():
        print(f"\n[{prefix.upper()}] {result_data['data_filename']}:")
        print(f"  - {prefix}_df_both.parquet / .csv  (sex='all' pooled)")
        print(f"  - {prefix}_df_both_by_sex.parquet / .csv  (per-sex pooled)")
        print(f"  - {prefix}_df_auc_unpooled.parquet / .csv")
        print(f"  - {prefix}_df_auc_age_stratified.parquet / .csv")
        print(f"  - {prefix}_df_auc_sex_stratified.parquet / .csv")
        if result_data['composite_metrics']:
            print(f"  - {prefix}_composite_metrics.json")
        _df_m = result_data.get('df_auc_merged')
        _n = 0
        if _df_m is not None and not _df_m.empty:
            _n = len(_df_m.loc[_df_m["sex"] == "all"]) if "sex" in _df_m.columns else len(_df_m)
        print(f"  - Diseases evaluated: {_n}")


if __name__ == "__main__":
    main()
