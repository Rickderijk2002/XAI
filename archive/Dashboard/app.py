"""
XAI Counterfactual Explanations: Guided Task Mode
JM0110 Interactive & Explainable AI Design at JADS

Run with:
    streamlit run app.py
"""

import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import data_utils as du

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="XAI CF Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

CIFAR_LABELS = du.CIFAR_LABELS  # id→name

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
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

    .progress-step.done {
        background: #2563eb;
    }

    .progress-step.active {
        background: #93c5fd;
    }

    .metric-badge {
        display: inline-block;
        background: #f1f5f9;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 4px 10px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.8rem;
        color: #334155;
        margin: 2px;
    }

    .valid-badge {
        background: #dcfce7;
        border-color: #86efac;
        color: #166534;
    }

    .invalid-badge {
        background: #fee2e2;
        border-color: #fca5a5;
        color: #991b1b;
    }

    .method-card {
        border: 2px solid #e5e7eb;
        border-radius: 10px;
        padding: 12px 8px;
        text-align: center;
        background: #fafafa;
    }

    .selected-card {
        border-color: #2563eb;
        background: #eff6ff;
    }

    .stButton > button {
        font-family: 'IBM Plex Mono', monospace;
        font-weight: 600;
        letter-spacing: 0.05em;
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

    .sidebar-info {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: #9ca3af;
        padding: 8px;
        border-top: 1px solid #374151;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Persistent game log ───────────────────────────────────────────────────────
GAME_LOG_FILE = Path(__file__).parent / "game_log.json"

def load_game_log() -> list:
    if GAME_LOG_FILE.exists():
        with open(GAME_LOG_FILE) as f:
            return json.load(f)
    return []

def save_session(entry: dict):
    log = load_game_log()
    log.append(entry)
    with open(GAME_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


# ── Session state initialisation ──────────────────────────────────────────────
def init_state():
    defaults = {
        "page": "task",           # "task" or "results"
        "step": 0,                # 0 = name entry, 1-7 = task steps
        "player_name": "",
        "case": None,             # the sampled case dict
        "responses": {},          # all user inputs collected
        "random_method": None,    # method shown in steps 1 & 2
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Case sampling ─────────────────────────────────────────────────────────────
@st.cache_data
def get_df():
    return du.load_results()

df = get_df()

def sample_case():
    """Sample a random (network, instance_id, target) that has at least 3 valid method images."""
    sub = df[df["timeout"] != 1].copy()
    # Try up to 30 times to find a case with images for all methods
    for _ in range(30):
        row = sub.sample(1).iloc[0]
        network = row["network"]
        instance_id = int(row["image"])
        target = int(row["target"])
        # Check original image exists
        orig = du.load_image(network, instance_id, "original")
        if orig is None:
            continue
        # Check how many methods have images
        available_methods = []
        for m in du.METHODS:
            img = du.load_image(network, instance_id, m, target)
            if img is not None:
                available_methods.append(m)
        if len(available_methods) >= 3:
            return {
                "network": network,
                "instance_id": instance_id,
                "target": target,
                "original_label": int(row["original_label"]) if "original_label" in row else None,
                "available_methods": available_methods,
            }
    return None


# ── Helpers ───────────────────────────────────────────────────────────────────
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
        f'<div style="font-family:IBM Plex Mono;font-size:0.7rem;color:#9ca3af;margin-top:-1.2rem;margin-bottom:1.5rem;">Step {current_step} of {total}</div>',
        unsafe_allow_html=True
    )

def step_title(step_num: int, title: str):
    st.markdown(f'<div class="step-header">Step {step_num} of 7</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="step-title">{title}</div>', unsafe_allow_html=True)

def show_image_pair(case: dict, method: str, show_method_label: bool = False):
    """Show original + one CF method side by side."""
    network = case["network"]
    instance_id = case["instance_id"]
    target = case["target"]
    is_mnist = "mnist" in network

    orig = du.load_image(network, instance_id, "original")
    cf = du.load_image(network, instance_id, method, target)

    fig, axes = plt.subplots(1, 2, figsize=(4, 2.5))
    fig.patch.set_facecolor('#fafafa')

    axes[0].imshow(orig, cmap="gray" if is_mnist else None)
    orig_lbl = case.get("original_label", "?")
    orig_title = f"Original: {orig_lbl}"
    if not is_mnist and orig_lbl is not None:
        orig_title += f"\n({CIFAR_LABELS.get(orig_lbl, '?')})"
    axes[0].set_title(orig_title, fontsize=9, fontfamily="monospace")
    axes[0].axis("off")

    if cf is not None:
        axes[1].imshow(cf, cmap="gray" if is_mnist else None)
    else:
        axes[1].text(0.5, 0.5, "N/A", ha="center", va="center",
                     transform=axes[1].transAxes, fontsize=10, color="gray")
        axes[1].set_facecolor("#f0f0f0")

    cf_title = f"Counterfactual (target {target})"
    if not is_mnist:
        cf_title += f"\n({CIFAR_LABELS.get(target, '?')})"
    if show_method_label:
        cf_title = f"[{method}]\n" + cf_title
    axes[1].set_title(cf_title, fontsize=9, fontfamily="monospace")
    axes[1].axis("off")

    fig.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close(fig)

    ds_label = "MNIST" if is_mnist else "CIFAR-10"
    st.caption(f"Dataset: **{ds_label}** · Instance: **{instance_id}** · Target: **{target}**")


def show_all_methods_grid(case: dict):
    """Show original + all 5 method CFs in a row."""
    network = case["network"]
    instance_id = case["instance_id"]
    target = case["target"]
    is_mnist = "mnist" in network

    orig = du.load_image(network, instance_id, "original")
    ncols = 1 + len(du.METHODS)
    fig, axes = plt.subplots(1, ncols, figsize=(3 * ncols, 3.2))
    fig.patch.set_facecolor('#fafafa')

    axes[0].imshow(orig, cmap="gray" if is_mnist else None)
    axes[0].set_title("Original", fontsize=9, fontfamily="monospace", fontweight="bold")
    axes[0].axis("off")

    for i, method in enumerate(du.METHODS):
        ax = axes[i + 1]
        cf = du.load_image(network, instance_id, method, target)
        if cf is not None:
            ax.imshow(cf, cmap="gray" if is_mnist else None)
        else:
            ax.text(0.5, 0.5, "N/A\n(timeout)", ha="center", va="center",
                    transform=ax.transAxes, fontsize=8, color="gray")
            ax.set_facecolor("#f0f0f0")
        ax.set_title(method, fontsize=8, fontfamily="monospace")
        ax.axis("off")

    fig.tight_layout(pad=0.3)
    st.pyplot(fig)
    plt.close(fig)


def get_metrics_for_case(case: dict) -> dict:
    """Return metrics dict keyed by method."""
    network = case["network"]
    instance_id = case["instance_id"]
    target = case["target"]
    result = {}
    for method in du.METHODS:
        result[method] = du.get_metric_row(network, instance_id, method, target)
    return result


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🔍 XAI CF Explorer")
    st.markdown("*Counterfactual Explanation Study*")
    st.divider()

    page_choice = st.radio(
        "Navigate",
        ["Guided Task Mode", "Game Results"],
        index=0 if st.session_state.page == "task" else 1,
    )
    if page_choice == "Guided Task Mode":
        st.session_state.page = "task"
    else:
        st.session_state.page = "results"

    if st.session_state.page == "task" and st.session_state.step > 0:
        st.divider()
        st.markdown(f'<div class="sidebar-info">Player: <b>{st.session_state.player_name}</b><br>Step: {st.session_state.step}/7</div>', unsafe_allow_html=True)

    # Admin reset button
    st.divider()
    if st.button("↺ Reset session", use_container_width=True):
        for key in ["step", "player_name", "case", "responses", "random_method"]:
            st.session_state[key] = None if key in ["case", "responses", "random_method"] else (0 if key == "step" else "")
        st.session_state.responses = {}
        st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Game Results
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.page == "results":
    st.markdown('<div class="step-title">Game Results</div>', unsafe_allow_html=True)

    log = load_game_log()
    if not log:
        st.info("No sessions completed yet. Complete a Guided Task session to see results here.")
        st.stop()

    gdf = pd.DataFrame(log)
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], unit="s")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total sessions", len(gdf))
    col2.metric("Unique players", gdf["player_name"].nunique() if "player_name" in gdf else "?")
    col3.metric("Datasets seen", gdf["network"].nunique() if "network" in gdf else "?")

    st.divider()

    # ── Best method picks ──────────────────────────────────────────────────
    st.subheader("Which CF method was picked as best?")

    criteria = {
        "step3_best_method": "Step 3: Game pick (overall best)",
        "step6_best_overall": "Step 6: Best overall (valid & plausible)",
        "step6_best_plausible": "Step 6: Most plausible",
        "step6_best_valid": "Step 6: Most valid",
    }

    cols = st.columns(len(criteria))
    for col, (key, label) in zip(cols, criteria.items()):
        with col:
            st.markdown(f"**{label}**")
            if key in gdf.columns:
                counts = gdf[key].value_counts()
                fig, ax = plt.subplots(figsize=(3, 2.5))
                fig.patch.set_facecolor('#fafafa')
                bars = ax.barh(counts.index, counts.values,
                               color=["#2563eb", "#93c5fd", "#bfdbfe", "#dbeafe", "#eff6ff"][:len(counts)])
                ax.set_xlabel("Picks", fontsize=8)
                ax.tick_params(labelsize=8)
                for bar, val in zip(bars, counts.values):
                    ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                            str(val), va="center", fontsize=8)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                # Show winner
                if len(counts) > 0:
                    winner = counts.index[0]
                    st.markdown(f'<div class="result-winner">🏆 {winner}</div>', unsafe_allow_html=True)
            else:
                st.info("No data yet")

    st.divider()

    # ── Accuracy: did human game pick match best metric? ───────────────────
    st.subheader("Player accuracy in Step 3 (Game)")

    if "step3_best_method" in gdf.columns and "step3_confidence" in gdf.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Picks by method**")
            picks = gdf["step3_best_method"].value_counts().reset_index()
            picks.columns = ["Method", "Times Picked"]
            st.dataframe(picks, use_container_width=True, hide_index=True)

        with col_b:
            st.markdown("**Average confidence per pick**")
            if gdf["step3_confidence"].notna().any():
                conf_by_method = gdf.groupby("step3_best_method")["step3_confidence"].mean().round(1)
                fig, ax = plt.subplots(figsize=(4, 2.5))
                fig.patch.set_facecolor('#fafafa')
                ax.barh(conf_by_method.index, conf_by_method.values, color="#2563eb")
                ax.set_xlabel("Avg Confidence (%)", fontsize=8)
                ax.set_xlim(0, 100)
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

    st.divider()

    # ── Validity/Plausibility estimate accuracy ────────────────────────────
    st.subheader("How well did players estimate metrics? (Step 2)")

    if "step2_validity_estimate" in gdf.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Estimated vs Actual Validity**")
            # Show distribution of estimates
            fig, ax = plt.subplots(figsize=(4, 2.5))
            fig.patch.set_facecolor('#fafafa')
            ax.hist(gdf["step2_validity_estimate"].dropna(), bins=10, color="#2563eb", alpha=0.7)
            ax.set_xlabel("Estimated Validity", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            st.markdown("**Estimated vs Actual Plausibility**")
            fig, ax = plt.subplots(figsize=(4, 2.5))
            fig.patch.set_facecolor('#fafafa')
            ax.hist(gdf["step2_plausibility_estimate"].dropna(), bins=10, color="#93c5fd", alpha=0.7)
            ax.set_xlabel("Estimated Plausibility", fontsize=8)
            ax.set_ylabel("Count", fontsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    # ── Qualitative responses ──────────────────────────────────────────────
    st.subheader("Qualitative responses")

    qual_keys = [
        ("step1_why", "Step 1: Why did you think so?"),
        ("step5_why_chosen", "Step 5: What made you choose that method?"),
        ("step7_final_thoughts", "Step 7: Final thoughts"),
    ]
    for key, label in qual_keys:
        if key in gdf.columns:
            with st.expander(label):
                for _, row in gdf.iterrows():
                    val = row.get(key, "")
                    if val and str(val).strip():
                        st.markdown(f"**{row.get('player_name', '?')}**: {val}")

    st.divider()

    # ── Raw log ────────────────────────────────────────────────────────────
    with st.expander("Raw session log"):
        st.dataframe(gdf, use_container_width=True)

    if st.button("Download as CSV"):
        st.download_button(
            "Download CSV",
            gdf.to_csv(index=False),
            "game_results.csv",
            "text/csv",
        )

    st.stop()


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Guided Task Mode
# ═════════════════════════════════════════════════════════════════════════════

# ── Step 0: Name entry ────────────────────────────────────────────────────────
if st.session_state.step == 0:
    st.markdown('<div class="step-title">Welcome to the XAI Study</div>', unsafe_allow_html=True)
    st.markdown(
        """
        In this study you will evaluate **counterfactual (CF) explanations: images that show
        how an input would need to change for an AI model to predict a different class.

        You will go through **7 steps** for one randomly selected case. Your responses help us
        understand whether objective evaluation metrics align with human intuition.
        """
    )

    st.divider()
    name = st.text_input("Enter your name or participant ID to begin:", placeholder="e.g. Rick or P01")

    if st.button("Start →", type="primary", disabled=not name.strip()):
        st.session_state.player_name = name.strip()
        # Sample a case
        case = sample_case()
        if case is None:
            st.error("Could not sample a valid case. Check that the Data/ folder is set up correctly.")
            st.stop()
        st.session_state.case = case
        # Pick a random method for steps 1 & 2
        st.session_state.random_method = random.choice(case["available_methods"])
        st.session_state.step = 1
        st.session_state.responses = {
            "player_name": name.strip(),
            "timestamp": time.time(),
            "network": case["network"],
            "instance_id": case["instance_id"],
            "target": case["target"],
            "original_label": case.get("original_label"),
            "shown_method_steps_1_2": st.session_state.random_method,
        }
        st.rerun()


# ── Step 1: Visual Inspection ─────────────────────────────────────────────────
elif st.session_state.step == 1:
    progress_bar(1)
    step_title(1, "Visual Inspection")
    st.markdown("Look at the images below. **No metric information is shown yet.**")

    case = st.session_state.case
    method = st.session_state.random_method

    col_img, col_q = st.columns([1, 1])

    with col_img:
        show_image_pair(case, method)

    with col_q:
        st.markdown("**Based on visual inspection only, does the counterfactual look successful?**")
        st.caption("A successful CF should look like it belongs to the target class.")

        judgment = st.radio(
            "Your judgment:",
            ["Yes, successful", "No, not successful", "Uncertain"],
            index=2,
            key="s1_judgment",
        )

        why = st.text_area(
            "Why do you think so? *(optional but valuable)*",
            placeholder="e.g. The digit looks like a 3 but the stroke is a bit off...",
            key="s1_why",
            height=120,
        )

        if st.button("Next →", type="primary"):
            st.session_state.responses["step1_judgment"] = judgment
            st.session_state.responses["step1_why"] = why
            st.session_state.step = 2
            st.rerun()


# ── Step 2: Your Prediction ───────────────────────────────────────────────────
elif st.session_state.step == 2:
    progress_bar(2)
    step_title(2, "Your Prediction")
    st.markdown("Now **estimate** the evaluation metrics for this counterfactual.")

    case = st.session_state.case
    method = st.session_state.random_method

    col_img, col_q = st.columns([1, 1])

    with col_img:
        show_image_pair(case, method)

    with col_q:
        st.markdown("**Estimate the following metrics:**")
        st.caption("Validity: does the model actually predict the target class? Plausibility: does the CF look realistic?")

        validity_est = st.slider(
            "Validity (0 = definitely invalid, 1 = definitely valid)",
            0.0, 1.0, 0.5, 0.01,
            key="s2_validity",
        )

        plausibility_est = st.slider(
            "Plausibility (0 = very implausible, 1 = very plausible)",
            0.0, 1.0, 0.5, 0.01,
            key="s2_plausibility",
        )

        if st.button("Next →", type="primary"):
            st.session_state.responses["step2_validity_estimate"] = validity_est
            st.session_state.responses["step2_plausibility_estimate"] = plausibility_est
            st.session_state.step = 3
            st.rerun()


# ── Step 3: Game — Choose the Best ───────────────────────────────────────────
elif st.session_state.step == 3:
    progress_bar(3)
    step_title(3, "Game: Choose the Best Explanation")
    st.markdown("**Which counterfactual explanation is the most successful?** Look at all methods and select one.")

    case = st.session_state.case
    network = case["network"]
    instance_id = case["instance_id"]
    target = case["target"]
    is_mnist = "mnist" in network

    # Show all methods
    show_all_methods_grid(case)

    st.divider()

    col_pick, col_conf = st.columns([1, 1])
    with col_pick:
        best_pick = st.radio(
            "Select the best CF method:",
            du.METHODS,
            key="s3_pick",
        )

    with col_conf:
        confidence = st.slider(
            "How confident are you in your choice? (%)",
            0, 100, 60, 5,
            key="s3_confidence",
        )

    if st.button("Submit choice →", type="primary"):
        st.session_state.responses["step3_best_method"] = best_pick
        st.session_state.responses["step3_confidence"] = confidence
        st.session_state.step = 4
        st.rerun()


# ── Step 4: Actual Metrics Revealed ──────────────────────────────────────────
elif st.session_state.step == 4:
    progress_bar(4)
    step_title(4, "Actual Metrics Revealed")
    st.markdown("Here are the **actual evaluation metrics** for the CF you saw in steps 1 & 2. Compare them to your estimates.")

    case = st.session_state.case
    method = st.session_state.random_method
    network = case["network"]
    instance_id = case["instance_id"]
    target = case["target"]

    col_img, col_metrics = st.columns([1, 1])

    with col_img:
        show_image_pair(case, method, show_method_label=True)

    with col_metrics:
        metrics = du.get_metric_row(network, instance_id, method, target)

        if metrics:
            correctness = metrics.get("correctness", float("nan"))
            im1 = metrics.get("IM1", float("nan"))
            implaus = metrics.get("implausibility", float("nan"))
            im2 = metrics.get("IM2", float("nan"))

            # Validity
            your_val = st.session_state.responses.get("step2_validity_estimate", None)
            actual_val = correctness if not (isinstance(correctness, float) and np.isnan(correctness)) else None

            st.markdown("**Validity**")
            v_col1, v_col2 = st.columns(2)
            v_col1.metric("Your estimate", f"{your_val:.2f}" if your_val is not None else "N/A")
            if actual_val == 1:
                v_col2.markdown('<span class="metric-badge valid-badge">✓ Valid (1.0)</span>', unsafe_allow_html=True)
            elif actual_val == 0:
                v_col2.markdown('<span class="metric-badge invalid-badge">✗ Invalid (0.0)</span>', unsafe_allow_html=True)
            else:
                v_col2.markdown('<span class="metric-badge">? Unknown</span>', unsafe_allow_html=True)

            st.markdown("**Plausibility (IM1, lower is more plausible)**")
            your_plaus = st.session_state.responses.get("step2_plausibility_estimate", None)
            p_col1, p_col2 = st.columns(2)
            p_col1.metric("Your estimate", f"{your_plaus:.2f}" if your_plaus is not None else "N/A")
            p_col2.metric("IM1 (actual)", f"{im1:.4f}" if not (isinstance(im1, float) and np.isnan(im1)) else "N/A")

            if not (isinstance(implaus, float) and np.isnan(implaus)):
                st.metric("Implausibility score", f"{implaus:.4f}")
            if not (isinstance(im2, float) and np.isnan(im2)):
                st.metric("IM2", f"{im2:.4f}")
        else:
            st.warning("No metrics found for this case.")

    if st.button("Next →", type="primary"):
        # Store actual metrics for reference
        st.session_state.responses["step4_actual_correctness"] = metrics.get("correctness") if metrics else None
        st.session_state.responses["step4_actual_IM1"] = metrics.get("IM1") if metrics else None
        st.session_state.responses["step4_actual_implausibility"] = metrics.get("implausibility") if metrics else None
        st.session_state.step = 5
        st.rerun()


# ── Step 5: Explanation & Feedback ────────────────────────────────────────────
elif st.session_state.step == 5:
    progress_bar(5)
    step_title(5, "Explanation & Feedback")

    case = st.session_state.case
    network = case["network"]
    instance_id = case["instance_id"]
    target = case["target"]

    st.markdown(f"**You selected: `{st.session_state.responses.get('step3_best_method', '?')}`** in the game.")

    # Build actual ranking by validity then IM1
    all_metrics = get_metrics_for_case(case)
    ranked = []
    for method in du.METHODS:
        m = all_metrics.get(method, {})
        ranked.append({
            "method": method,
            "correctness": m.get("correctness", float("nan")),
            "IM1": m.get("IM1", float("nan")),
            "implausibility": m.get("implausibility", float("nan")),
        })

    ranked_df = pd.DataFrame(ranked)
    ranked_df = ranked_df.sort_values(
        ["correctness", "IM1"], ascending=[False, True]
    ).reset_index(drop=True)
    ranked_df.index = ranked_df.index + 1

    st.markdown("**Actual ranking (best → worst) by validity then plausibility:**")

    for i, row in ranked_df.iterrows():
        prefix = "⭐ " if i == 1 else f"{i}. "
        c = row["correctness"]
        c_str = "✓ Valid" if c == 1 else ("✗ Invalid" if c == 0 else "?")
        im1_str = f"IM1: {row['IM1']:.3f}" if not (isinstance(row['IM1'], float) and np.isnan(row['IM1'])) else "IM1: N/A"
        st.markdown(f"{prefix}**{row['method']}**: {c_str} | {im1_str}")

    st.divider()

    why_chosen = st.text_area(
        "What made you choose your method in Step 3?",
        placeholder="e.g. It looked most different from the original, like it could actually be a different class...",
        key="s5_why",
        height=100,
    )

    if st.button("Next →", type="primary"):
        st.session_state.responses["step5_why_chosen"] = why_chosen
        st.session_state.responses["step5_actual_best_method"] = ranked_df.iloc[0]["method"] if len(ranked_df) > 0 else None
        st.session_state.step = 6
        st.rerun()


# ── Step 6: Compare Methods on This Case ─────────────────────────────────────
elif st.session_state.step == 6:
    progress_bar(6)
    step_title(6, "Compare Methods on This Case")
    st.markdown("Now see **all methods side by side** with their metrics. Make three selections below.")

    case = st.session_state.case
    network = case["network"]
    instance_id = case["instance_id"]
    target = case["target"]
    is_mnist = "mnist" in network

    show_all_methods_grid(case)

    # Metrics table
    st.markdown("**Evaluation metrics per method:**")
    all_metrics = get_metrics_for_case(case)
    table_data = {}
    for method in du.METHODS:
        m = all_metrics.get(method, {})
        c = m.get("correctness", float("nan"))
        im1 = m.get("IM1", float("nan"))
        implaus = m.get("implausibility", float("nan"))
        l2 = m.get("l2", float("nan"))

        table_data[method] = {
            "Validity": "✓ Valid" if c == 1 else ("✗ Invalid" if c == 0 else "N/A"),
            "IM1 (plausibility)": f"{im1:.3f}" if not (isinstance(im1, float) and np.isnan(im1)) else "N/A",
            "Implausibility": f"{implaus:.3f}" if not (isinstance(implaus, float) and np.isnan(implaus)) else "N/A",
            "L2 distance": f"{l2:.3f}" if not (isinstance(l2, float) and np.isnan(l2)) else "N/A",
        }

    table_df = pd.DataFrame(table_data).T
    st.dataframe(table_df, use_container_width=True)

    st.divider()
    st.markdown("**Select the best method according to each criterion:**")

    method_options = du.METHODS
    col1, col2, col3 = st.columns(3)

    with col1:
        best_overall = st.selectbox(
            "Best overall (valid & plausible)",
            method_options,
            key="s6_overall",
        )

    with col2:
        best_plausible = st.selectbox(
            "Most plausible",
            method_options,
            key="s6_plausible",
        )

    with col3:
        best_valid = st.selectbox(
            "Most valid",
            method_options,
            key="s6_valid",
        )

    if st.button("Next →", type="primary"):
        st.session_state.responses["step6_best_overall"] = best_overall
        st.session_state.responses["step6_best_plausible"] = best_plausible
        st.session_state.responses["step6_best_valid"] = best_valid
        st.session_state.step = 7
        st.rerun()


# ── Step 7: Confidence & Final Thoughts ──────────────────────────────────────
elif st.session_state.step == 7:
    progress_bar(7)
    step_title(7, "Confidence & Final Thoughts")

    st.markdown("You have now seen all the metrics. Reflect on your initial judgments.")

    col_a, col_b = st.columns(2)

    with col_a:
        post_confidence = st.slider(
            "How confident are you in your **initial** judgment from Step 1? (%)",
            0, 100, 70, 5,
            key="s7_confidence",
        )

        change_answer = st.radio(
            "Would you change your initial answer (Step 1) after seeing the metrics?",
            ["Yes", "No"],
            key="s7_change",
        )

    with col_b:
        final_thoughts = st.text_area(
            "Any final thoughts? *(optional)*",
            placeholder="e.g. The metrics surprised me because...",
            key="s7_thoughts",
            height=150,
        )

    st.divider()

    if st.button("✓ Finish & Save", type="primary"):
        # Collect final responses
        st.session_state.responses["step7_post_confidence"] = post_confidence
        st.session_state.responses["step7_change_answer"] = change_answer
        st.session_state.responses["step7_final_thoughts"] = final_thoughts

        # Save to file
        save_session(st.session_state.responses)

        # Show thank you
        st.success("**Thank you! Your responses have been saved.**")
        st.balloons()
        st.markdown(f"**Player:** {st.session_state.player_name}")
        st.markdown(f"**Your game pick:** `{st.session_state.responses.get('step3_best_method', '?')}`")
        st.markdown(f"**Metric-best method:** `{st.session_state.responses.get('step5_actual_best_method', '?')}`")

        time.sleep(1)

        # Reset for next player
        st.session_state.step = 0
        st.session_state.player_name = ""
        st.session_state.case = None
        st.session_state.responses = {}
        st.session_state.random_method = None

        st.rerun()