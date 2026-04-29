"""
components.py: Shared UI components for XAI CF Explorer.
Imported by task.py, minigame.py, results.py, intro.py.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import data_utils as du

CIFAR_LABELS = du.CIFAR_LABELS

# CSS
CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }

    .step-header {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 0.25rem;
    }

    .step-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.6rem;
        font-weight: 600;
        color: #f8fafc;
        margin-bottom: 1.5rem;
        border-left: 4px solid #f59e0b;
        padding-left: 0.75rem;
    }

    .progress-bar-container {
        display: flex;
        gap: 6px;
        margin-bottom: 2rem;
        align-items: center;
    }

    .progress-step {
        height: 6px;
        flex: 1;
        border-radius: 3px;
        background: #374151;
    }

    .progress-step.done { background: #2563eb; }
    .progress-step.active { background: #93c5fd; }

    .metric-badge {
        display: inline-block;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 6px;
        padding: 4px 10px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #cbd5e1;
        margin: 2px;
    }

    .valid-badge {
        background: #14532d;
        border-color: #16a34a;
        color: #86efac;
    }

    .invalid-badge {
        background: #450a0a;
        border-color: #dc2626;
        color: #fca5a5;
    }

    .result-winner {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 1.2rem;
        font-weight: 600;
        color: #f59e0b;
        text-align: center;
        padding: 1rem;
        border: 2px solid #f59e0b;
        border-radius: 10px;
        background: rgba(245,158,11,0.1);
        margin: 1rem 0;
    }

    .metric-explainer-box {
        background: #0f172a;
        border: 1px solid #1e40af;
        border-left: 4px solid #3b82f6;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 1.5rem;
    }

    .metric-explainer-box h4 {
        color: #93c5fd;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.85rem;
        margin: 0 0 0.5rem 0;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    .metric-row {
        display: flex;
        align-items: flex-start;
        gap: 0.75rem;
        margin: 0.4rem 0;
    }

    .metric-name {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #f59e0b;
        min-width: 130px;
        font-weight: 600;
    }

    .metric-desc {
        font-size: 0.82rem;
        color: #cbd5e1;
        line-height: 1.4;
    }

    .sidebar-info {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #9ca3af;
        padding: 8px;
        border-top: 1px solid #374151;
        margin-top: 1rem;
    }

    .strike-badge {
        display: inline-block;
        font-size: 1.4rem;
        margin: 0 3px;
    }

    .stButton > button {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* Enter key fix for name input */
    input[type="text"]:focus {
        border-color: #f59e0b !important;
        box-shadow: 0 0 0 1px #f59e0b !important;
    }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


# Progress bar
def progress_bar(current_step: int, total: int = 7):
    bars = ""
    for i in range(1, total + 1):
        if i < current_step:
            cls = "done"
        elif i == current_step:
            cls = "active"
        else:
            cls = ""
        bars += f'<div class="progress-step {cls}"></div>'
    st.markdown(
        f'<div class="progress-bar-container">{bars}</div>'
        f'<div style="font-family:IBM Plex Mono;font-size:0.7rem;color:#9ca3af;'
        f'margin-top:-1.2rem;margin-bottom:1.5rem;">Step {current_step} of {total}</div>',
        unsafe_allow_html=True,
    )


def step_title(step_num: int, title: str, total: int = 7):
    st.markdown(f'<div class="step-header">Step {step_num} of {total}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-title">{title}</div>', unsafe_allow_html=True)


def page_title(title: str):
    st.markdown(f'<div class="step-title">{title}</div>', unsafe_allow_html=True)


# Metric explainer
METRIC_DEFINITIONS = [
    ("Validity (Correctness)", "Did the model actually predict the target class for this CF? 1 = Yes (valid), 0 = No (invalid). This is a binary pass/fail check, it says nothing about how the image looks, only whether the model was fooled."),
    ("Plausibility (IM1)", "Does the CF look like a realistic image? IM1 measures how far the CF is from the training data distribution. LOWER IM1 = MORE plausible (closer to real data). Higher IM1 = the image looks unnatural or distorted."),
    ("Implausibility score", "A combined score that captures how implausible the CF looks overall. HIGHER = more implausible (looks worse). This is sometimes confused with plausibility, remember: high implausibility is bad."),
    ("IM2 / L2 distance", "Measures how much the CF image differs from the original in terms of pixel values. A lower score means the CF required fewer changes to fool the model, which is better. A higher score means large parts of the image were altered."),
]


def metric_explainer(collapsed: bool = False):
    """
    Renders a metric explanation panel before any metrics are shown.
    Uses native Streamlit components to avoid HTML rendering issues.
    """
    def _render_content():
        st.markdown("**📊 What do these metrics mean?**")
        for name, desc in METRIC_DEFINITIONS:
            st.markdown(f"🔹 **{name}**")
            st.caption(desc)

    if collapsed:
        with st.expander("📊 What do these metrics mean? (click to expand)", expanded=False):
            _render_content()
    else:
        with st.container(border=True):
            _render_content()


# Image rendering
def show_image_pair(case: dict, method: str, show_method_label: bool = False):
    """Show original + one CF method side by side."""
    network = case["network"]
    instance_id = case["instance_id"]
    target = case["target"]
    is_mnist = "mnist" in network

    orig = du.load_image(network, instance_id, "original")
    cf = du.load_image(network, instance_id, method, target)

    fig, axes = plt.subplots(1, 2, figsize=(5, 3.2))
    fig.patch.set_facecolor("#111827")

    axes[0].imshow(orig, cmap="gray" if is_mnist else None)
    orig_lbl = case.get("original_label", "?")
    orig_title = f"Original: {orig_lbl}"
    if not is_mnist and orig_lbl is not None:
        orig_title += f"\n({CIFAR_LABELS.get(orig_lbl, '?')})"
    axes[0].set_title(orig_title, fontsize=10, fontfamily="monospace", color="#f8fafc")
    axes[0].axis("off")

    if cf is not None:
        axes[1].imshow(cf, cmap="gray" if is_mnist else None)
    else:
        axes[1].text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=10, color="#6b7280")
        axes[1].set_facecolor("#1f2937")

    cf_title = f"Counterfactual (target {target})"
    if not is_mnist:
        cf_title += f"\n({CIFAR_LABELS.get(target, '?')})"
    if show_method_label:
        cf_title = f"[{method}]\n" + cf_title
    axes[1].set_title(cf_title, fontsize=10, fontfamily="monospace", color="#f8fafc")
    axes[1].axis("off")

    fig.tight_layout(pad=0.8)
    st.pyplot(fig)
    plt.close(fig)

    ds_label = "MNIST" if is_mnist else "CIFAR-10"
    st.caption(f"Dataset: **{ds_label}** · Instance: **{instance_id}** · Target: **{target}**")


def show_mnist_pair(instance_id: int, method: str, target: int):
    """Render an MNIST original + CF pair for the mini game."""
    network = "mnist_output_100"
    orig = du.load_image(network, instance_id, "original")
    cf = du.load_image(network, instance_id, method, target)

    fig, axes = plt.subplots(1, 2, figsize=(5, 3.2))
    fig.patch.set_facecolor("#111827")

    if orig is not None:
        axes[0].imshow(orig, cmap="gray")
    else:
        axes[0].text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=axes[0].transAxes, fontsize=10, color="#6b7280")
        axes[0].set_facecolor("#1f2937")
    axes[0].set_title("Original digit", fontsize=10, fontfamily="monospace", color="#f8fafc")
    axes[0].axis("off")

    if cf is not None:
        axes[1].imshow(cf, cmap="gray")
    else:
        axes[1].text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=10, color="#6b7280")
        axes[1].set_facecolor("#1f2937")
    axes[1].set_title(f"Counterfactual (target: {target})", fontsize=10, fontfamily="monospace", color="#f8fafc")
    axes[1].axis("off")

    fig.tight_layout(pad=0.8)
    st.pyplot(fig)
    plt.close(fig)


def show_all_methods_grid(case: dict):
    """Show original + all 5 CF methods in a single row."""
    network = case["network"]
    instance_id = case["instance_id"]
    target = case["target"]
    is_mnist = "mnist" in network

    orig = du.load_image(network, instance_id, "original")
    ncols = 1 + len(du.METHODS)
    fig, axes = plt.subplots(1, ncols, figsize=(3.2 * ncols, 3.5))
    fig.patch.set_facecolor("#111827")

    orig_lbl = case.get("original_label", "?")
    orig_title = f"Original: {orig_lbl}"
    if not is_mnist and orig_lbl is not None:
        orig_title += f"\n({CIFAR_LABELS.get(orig_lbl, '?')})"

    axes[0].imshow(orig, cmap="gray" if is_mnist else None)
    axes[0].set_title(orig_title, fontsize=9, fontfamily="monospace",
                       fontweight="bold", color="#f8fafc")
    axes[0].axis("off")

    for i, method in enumerate(du.METHODS):
        ax = axes[i + 1]
        cf = du.load_image(network, instance_id, method, target)
        if cf is not None:
            ax.imshow(cf, cmap="gray" if is_mnist else None)
        else:
            ax.text(0.5, 0.5, "N/A\n(timeout)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="#6b7280")
            ax.set_facecolor("#1f2937")
        cf_label = f"{method}\ntarget: {target}"
        ax.set_title(cf_label, fontsize=8, fontfamily="monospace", color="#f8fafc")
        ax.axis("off")

    fig.tight_layout(pad=0.4)
    st.pyplot(fig)
    plt.close(fig)


# Metric helpers
def get_metrics_for_case(case: dict) -> dict:
    """Return metrics dict keyed by method."""
    network = case["network"]
    instance_id = case["instance_id"]
    target = case["target"]
    return {m: du.get_metric_row(network, instance_id, m, target) for m in du.METHODS}


def fmt(val, decimals: int = 3) -> str:
    """Format a float or return N/A."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def validity_badge(correctness) -> str:
    if correctness == 1:
        return '<span class="metric-badge valid-badge">✓ Valid</span>'
    elif correctness == 0:
        return '<span class="metric-badge invalid-badge">✗ Invalid</span>'
    return '<span class="metric-badge">? Unknown</span>'