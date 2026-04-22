"""
Data utilities for the XAI Counterfactual Explanations project.
Handles loading images, the CSV, and finding tension cases.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# ── Paths ────────────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(__file__), "Data")
MNIST_PATH  = os.path.join(BASE, "mnist_output")
# CIFAR has a nested folder
CIFAR_PATH  = os.path.join(BASE, "cifar_resnet8_output", "cifar_resnet8_output")
CSV_PATH    = os.path.join(BASE, "evaluation_results.csv")

METHODS = ["PIECE", "Min-Edit", "C-Min-Edit", "alibi-Proto-CF", "alibi-CF"]
CIFAR_LABELS = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}


# ── CSV ───────────────────────────────────────────────────────────────────────
_df_cache = None

def load_results() -> pd.DataFrame:
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_csv(CSV_PATH)
    return _df_cache


# ── Image loaders ─────────────────────────────────────────────────────────────
def _normalize_cifar(arr: np.ndarray) -> np.ndarray:
    """Map from [-1,1] (model space) to [0,1] for display."""
    return np.clip((arr + 1) / 2, 0, 1)


def load_mnist(instance_id: int, method: str, target: int = None) -> Optional[np.ndarray]:
    """Return a (28,28) float array, or None if the file is missing/blank."""
    if method == "original":
        path = os.path.join(MNIST_PATH, "original", f"instance_{instance_id}.pt")
    else:
        path = os.path.join(MNIST_PATH, method, f"instance_{instance_id}_target_{target}.pt")

    if not os.path.exists(path):
        return None
    try:
        t = torch.load(path, map_location="cpu", weights_only=True).detach().numpy()
    except Exception:
        return None

    img = t.reshape(28, 28)
    # blank / timed-out tensor → return None
    if np.abs(img).sum() < 1e-6:
        return None
    return img


def load_cifar(instance_id: int, method: str, target: int = None) -> Optional[np.ndarray]:
    """Return a (32,32,3) float array in [0,1], or None if missing/blank."""
    if method == "original":
        path = os.path.join(CIFAR_PATH, "original", f"instance_{instance_id}.pt")
    else:
        path = os.path.join(CIFAR_PATH, method, f"instance_{instance_id}_target_{target}.pt")

    if not os.path.exists(path):
        return None
    try:
        t = torch.load(path, map_location="cpu", weights_only=True).detach().numpy()
    except Exception:
        return None

    # shape may be (1,3,32,32), (3,32,32) [CHW], or (32,32,3) [HWC already]
    if t.ndim == 4:
        t = t[0]
    # If last dim is 3 it's already HWC; otherwise transpose from CHW
    if t.shape[-1] == 3:
        img = _normalize_cifar(t)
    else:
        img = _normalize_cifar(np.transpose(t, (1, 2, 0)))

    if np.abs(img).sum() < 1e-6:
        return None
    return img


def load_image(network: str, instance_id: int, method: str, target: int = None) -> Optional[np.ndarray]:
    if "mnist" in network:
        return load_mnist(instance_id, method, target)
    return load_cifar(instance_id, method, target)


# ── Tension-case finder ────────────────────────────────────────────────────────
def find_tension_cases(
    network: str = None,
    top_n: int = 20,
    kind: str = "valid_implausible",
) -> pd.DataFrame:
    """
    Return the most interesting tension cases.

    kind='valid_implausible'  → correctness=1 but high implausibility (metric says ok, looks bad)
    kind='invalid_plausible'  → correctness=0 but low IM1 / low implausibility (looks ok, metric says bad)
    """
    df = load_results().copy()
    if network:
        df = df[df["network"] == network]

    if kind == "valid_implausible":
        sub = df[(df["correctness"] == 1) & df["implausibility"].notna()]
        return sub.nlargest(top_n, "implausibility").reset_index(drop=True)

    elif kind == "invalid_plausible":
        # IM1 close to 1 means plausible; implausibility high means looks bad
        # We want: correctness=0 AND implausibility is high (human might think it works)
        sub = df[(df["correctness"] == 0) & df["implausibility"].notna()]
        return sub.nlargest(top_n, "implausibility").reset_index(drop=True)

    raise ValueError(f"Unknown kind: {kind}")


def get_metric_row(network: str, instance_id: int, method: str, target: int) -> dict:
    """Return metric dict for a single CF, or {} if not found."""
    df = load_results()
    mask = (
        (df["network"] == network) &
        (df["image"] == instance_id) &
        (df["method"] == method) &
        (df["target"] == target)
    )
    rows = df[mask]
    if rows.empty:
        return {}
    return rows.iloc[0].to_dict()


# ── Grid visualiser ───────────────────────────────────────────────────────────
def make_grid_figure(network: str, instance_id: int, target: int) -> plt.Figure:
    """
    Create a matplotlib figure with the original image + one panel per method.
    Shows metric values below each image.
    """
    is_mnist = "mnist" in network

    ncols = 1 + len(METHODS)
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3.8))

    # Original
    orig = load_image(network, instance_id, "original")
    ax = axes[0]
    if orig is not None:
        ax.imshow(orig, cmap="gray" if is_mnist else None)
    ax.set_title("Original", fontsize=9, fontweight="bold")
    ax.axis("off")

    # One panel per method
    for i, method in enumerate(METHODS):
        ax = axes[i + 1]
        cf = load_image(network, instance_id, method, target)
        metrics = get_metric_row(network, instance_id, method, target)

        if cf is not None:
            ax.imshow(cf, cmap="gray" if is_mnist else None)
        else:
            ax.text(0.5, 0.5, "N/A\n(timeout)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_facecolor("#f0f0f0")

        correctness = metrics.get("correctness", float("nan"))
        im1 = metrics.get("IM1", float("nan"))
        implaus = metrics.get("implausibility", float("nan"))
        c_str = "✓ Valid" if correctness == 1 else ("✗ Invalid" if correctness == 0 else "?")
        color = "green" if correctness == 1 else ("red" if correctness == 0 else "gray")

        title = f"{method}\n{c_str}"
        ax.set_title(title, fontsize=8, color=color, fontweight="bold")
        subtitle = ""
        if not np.isnan(im1):
            subtitle += f"IM1={im1:.2f}  "
        if not np.isnan(implaus):
            subtitle += f"Impl={implaus:.2f}"
        ax.set_xlabel(subtitle, fontsize=7)
        ax.axis("off")

    ds_label = "MNIST" if is_mnist else "CIFAR-10"
    fig.suptitle(
        f"{ds_label} — Instance {instance_id}, Target {target}"
        + (f" ({CIFAR_LABELS.get(target, target)})" if not is_mnist else ""),
        fontsize=10, fontweight="bold"
    )
    fig.tight_layout()
    return fig


# ── Quick summary stats ────────────────────────────────────────────────────────
def summary_stats() -> dict:
    df = load_results()
    stats = {}
    stats["total_rows"] = len(df)
    stats["correctness_by_method"] = (
        df.groupby(["network", "method"])["correctness"]
        .mean()
        .round(3)
        .unstack()
        .to_dict()
    )
    stats["timeout_by_method"] = df.groupby("method")["timeout"].mean().round(3).to_dict()
    return stats
