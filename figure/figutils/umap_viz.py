"""
umap_viz.py - UMAP embedding visualization
============================================
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

plt.rcParams['font.family'] = 'Helvetica'

# Ensure project root is importable
_here = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_here))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

# =============================================================================
# Configuration
# =============================================================================
DEFAULT_SIZE_LEVELS = {
    'Low':  {'thresh': 1_000,        'size': 30,  'label': '< 1k'},
    'Mid':  {'thresh': 50_000,       'size': 100, 'label': '1k – 50k'},
    'High': {'thresh': float('inf'), 'size': 350, 'label': '> 50k'},
}

# =============================================================================
# Data Loading
# =============================================================================

def load_token_frequencies(data_bin_path, token_offset=0):
    """
    Compute per-token frequencies from a binary training dataset.

    Returns
    -------
    dict : token_id → count
    """
    from .common import COMPOSITE_DTYPE

    print(f"[INFO] Computing token frequencies from: {data_bin_path} (token_offset={token_offset})")
    if not os.path.exists(data_bin_path):
        print("[WARN] Dataset file not found. Returning empty counts.")
        return {}

    data_raw = np.fromfile(data_bin_path, dtype=COMPOSITE_DTYPE)
    tokens = data_raw['DATA'] + int(token_offset)
    unique, counts = np.unique(tokens, return_counts=True)
    token_counts = dict(zip(unique.tolist(), counts.tolist()))
    print(f"[OK]  {len(token_counts)} unique tokens")
    return token_counts


def load_chapter_metadata(csv_path, start_token_id=22):
    """
    Load token-to-chapter metadata CSV.

    Returns
    -------
    token_meta : dict  (token_id → {name, chapter_short, color, …})
    legend_info : pd.DataFrame  (unique chapter_short + color)
    """
    print(f"[INFO] Loading chapter metadata from: {csv_path}")
    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] == 6:
        df.columns = ['raw_idx', 'name', 'token_id', 'chapter_full', 'chapter_short', 'color']
    elif df.shape[1] == 5:
        df.columns = ['name', 'token_id', 'chapter_full', 'chapter_short', 'color']
    else:
        raise ValueError(f"Unexpected column count: {df.shape[1]}")

    df['token_id'] = pd.to_numeric(df['token_id'], errors='coerce')
    df = df.dropna(subset=['token_id'])
    df['token_id'] = df['token_id'].astype(int)

    filtered = df[df['token_id'] >= start_token_id].copy()
    token_meta = filtered.set_index('token_id').to_dict('index')
    legend_info = filtered[['chapter_short', 'color']].drop_duplicates()

    return token_meta, legend_info


# =============================================================================
# Embedding Extraction
# =============================================================================

def load_neutral_total(data_bin_path, total_vocab_size=None):
    """
    Compute a neutral TOTAL token from data (rounded mean of observed TOTAL values).
    """
    from .common import COMPOSITE_DTYPE

    print(f"[INFO] Computing neutral TOTAL from: {data_bin_path}")
    if not os.path.exists(data_bin_path):
        print("[WARN] Dataset file not found. Using neutral TOTAL token = 0.")
        return 0

    data_raw = np.fromfile(data_bin_path, dtype=COMPOSITE_DTYPE)
    total_vals = data_raw['TOTAL'].astype(np.int64)
    total_vals = total_vals[np.isfinite(total_vals)]
    mean_total = int(np.rint(total_vals.mean())) if len(total_vals) else 0
    if total_vocab_size is not None:
        mean_total = int(np.clip(mean_total, 0, int(total_vocab_size) - 1))
    print(f"[OK]  neutral TOTAL token = {mean_total}")
    return mean_total


def get_embeddings(
    ckpt_path,
    token_meta,
    repr_mode='projected_data_only',
    neutral_total=0,
    neutral_shift=0,
    model=None,
):
    """
    Extract token representations from a trained checkpoint.

    Returns
    -------
    embeddings : np.ndarray  (n_tokens, n_embd)
    valid_token_ids : list[int]
    token_offset : int
    """
    if model is None:
        from figutils.common import load_model
        print(f"[INFO] Loading embeddings from {ckpt_path}")
        model, ckpt = load_model(ckpt_path, device='cpu')
    else:
        print(f"[INFO] Using preloaded model for embeddings from {ckpt_path}")

    token_offset = 1 if bool(getattr(model.config, 'apply_token_shift', False)) else 0
    valid_ids = sorted(token_meta.keys())
    token_ids = torch.tensor(valid_ids, dtype=torch.long)

    with torch.no_grad():
        data_emb = model.composite_emb.data_emb.weight.detach()
        if repr_mode == 'raw':
            emb = data_emb[token_ids]
        elif repr_mode == 'projected_data_only':
            n_embd = data_emb.shape[1]
            proj_weight = model.composite_emb.proj.weight.detach()
            proj_data = proj_weight[:, :n_embd]
            emb = data_emb[token_ids] @ proj_data.T
        elif repr_mode == 'projected_fixed_context':
            shift_dummy = torch.full_like(token_ids, int(neutral_shift))
            total_dummy = torch.full_like(token_ids, int(neutral_total))
            emb = model.composite_emb(token_ids, shift_dummy, total_dummy)
        else:
            raise ValueError(
                f"Unknown repr_mode={repr_mode!r}. "
                "Choose from {'raw', 'projected_data_only', 'projected_fixed_context'}."
            )

    print(f"[INFO] Representation mode: {repr_mode}")
    return emb.detach().cpu().numpy(), valid_ids, token_offset


# =============================================================================
# UMAP & Plotting
# =============================================================================

def run_umap(embeddings, n_neighbors=15, min_dist=0.1, metric='cosine'):
    """Run UMAP dimensionality reduction → (N, 2) array."""
    print("[INFO] Running UMAP...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist,
        metric=metric, random_state=42,
    )
    return reducer.fit_transform(embeddings)


def draw_umap_plot(embedding_2d, valid_token_ids, token_meta,
                   token_counts, legend_info,
                   target_label_ids=None, size_levels=None,
                   figsize=(8, 6), save_path=None,
                   min_event_count=0):
    """
    Render a UMAP scatter plot.

    Parameters
    ----------
    save_path : str, optional
        If given, save figure to this path (in addition to returning it).
    min_event_count : int, default 0
        Hide tokens whose event count is below this threshold.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    if size_levels is None:
        size_levels = DEFAULT_SIZE_LEVELS

    print("[INFO] Rendering UMAP scatter plot...")

    df = pd.DataFrame(embedding_2d, columns=['x', 'y'])
    df['token_id'] = valid_token_ids
    df['color'] = [token_meta[t]['color'] for t in valid_token_ids]
    df['name'] = [token_meta[t]['name'] for t in valid_token_ids]
    df['count'] = [token_counts.get(t, 0) for t in valid_token_ids]

    if min_event_count and min_event_count > 0:
        n_before = len(df)
        df = df[df['count'] >= min_event_count].copy()
        print(f"[INFO] Kept {len(df)}/{n_before} tokens with count >= {min_event_count}")

    sizes = []
    for c in df['count']:
        if c < size_levels['Low']['thresh']:
            sizes.append(size_levels['Low']['size'])
        elif c < size_levels['Mid']['thresh']:
            sizes.append(size_levels['Mid']['size'])
        else:
            sizes.append(size_levels['High']['size'])
    df['size'] = sizes

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(df['x'], df['y'], c=df['color'], s=df['size'],
               alpha=0.7, edgecolors='white', linewidth=0.3)

    # Annotate selected tokens
    if target_label_ids:
        texts = []
        for _, row in df.iterrows():
            if int(row['token_id']) in target_label_ids:
                t = ax.text(row['x'], row['y'],
                            f"{row['name']} ({int(row['token_id'])})",
                            fontsize=10, fontweight='bold', color='black')
                texts.append(t)
        try:
            from adjustText import adjust_text
            adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5), ax=ax)
        except ImportError:
            pass

    # Legend 1: chapters
    patches = [mpatches.Patch(color=r['color'], label=r['chapter_short'])
               for _, r in legend_info.iterrows()]
    leg1 = ax.legend(handles=patches, bbox_to_anchor=(1.02, 1),
                     loc='upper left', title='Chapters', fontsize=9)
    ax.add_artist(leg1)

    # Legend 2: frequency sizes
    size_handles = [
        mlines.Line2D([], [], color='white', marker='o',
                      markerfacecolor='gray',
                      markersize=np.sqrt(size_levels[lv]['size']),
                      label=size_levels[lv]['label'])
        for lv in ('Low', 'Mid', 'High')
    ]
    ax.legend(handles=size_handles, bbox_to_anchor=(1.02, 0.4),
              loc='upper left', title='Frequency', fontsize=10, labelspacing=1.5)

    ax.set_title('Tokens UMAP Embedding', fontsize=20)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[OK]  Saved → {save_path}")

    return fig
