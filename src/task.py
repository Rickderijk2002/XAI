"""
task.py: The 7-step Guided Task Mode flow.
Called from app.py when page == "task" and step >= 1.
"""

import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import data_utils as du
from components import (
    fmt,
    get_metrics_for_case,
    metric_explainer,
    page_title,
    progress_bar,
    show_all_methods_grid,
    show_image_pair,
    step_title,
    validity_badge,
)

# Per-step info button
STEP_INFO = {
    1: (
        "What am I judging here?",
        "You are looking at one original digit image and one counterfactual (CF) version. "
        "The CF has been modified so the AI model predicts a different digit class.\n\n"
        "Judge purely on what you see: does the CF look like it could genuinely belong to the target class?\n\n"
        "No metrics are shown yet on purpose. We want your raw visual intuition first."
    ),
    2: (
        "What do validity and plausibility mean here?",
        "**Validity** (your slider): how confident are you that the model was fooled? "
        "0 = definitely not, 1 = definitely yes.\n\n"
        "**Plausibility** (your slider): how realistic does the CF look to you as a human? "
        "0 = very distorted or unnatural, 1 = looks like a real digit.\n\n"
        "These are your personal estimates on a simple 0 to 1 human scale. "
        "The actual model metrics use different scales and will be revealed in step 4."
    ),
    3: (
        "How should I pick the best method?",
        "You now see all 5 CF methods side by side. Consider two things:\n\n"
        "1. Does the CF actually look like it belongs to the target class? (validity)\n"
        "2. Does it still look like a realistic digit image? (plausibility)\n\n"
        "The best CF does both. You will not know the metric scores yet. "
        "Trust your visual judgment and pick the one that convinces you most."
    ),
    4: (
        "How do I read these metrics?",
        "**Validity:** 1 = the model was fooled (valid), 0 = it was not. Binary, no in-between.\n\n"
        "**IM1 (plausibility):** LOWER is better. A low IM1 means the CF looks close to real training data. "
        "Values above 1.0 are normal and just mean lower plausibility. Do not treat it as a percentage.\n\n"
        "**Implausibility:** LOWER is better. High implausibility means the CF looks unnatural.\n\n"
        "**IM2 / L2:** LOWER is better. Fewer pixel changes = more minimal edit."
    ),
    5: (
        "What does this ranking mean?",
        "Methods are ranked by validity first (valid beats invalid), then by IM1 (lower IM1 wins). "
        "The top method is considered objectively best by the model metrics.\n\n"
        "Your pick may differ from the top method. That gap between human intuition "
        "and model metrics is exactly what this study is measuring. Neither answer is wrong."
    ),
    6: (
        "How do I use this comparison table?",
        "Look at each column (method) and each row (metric). "
        "For each metric, lower is better except for validity where 1 = valid is best.\n\n"
        "Use the three dropdowns to select which method you think wins on each criterion. "
        "You can pick the same method for multiple criteria or different ones."
    ),
    7: (
        "What should I reflect on here?",
        "Think back to your first impression in step 1 before you saw any metrics. "
        "Now that you have seen the full rankings and metric scores, does your initial "
        "judgment hold up? Were you surprised by any of the results?\n\n"
        "Your honest reflection here is the most valuable data point in the whole study."
    ),
}

def info_button(step: int):
    """Render a context-specific info expander for the given step."""
    if step not in STEP_INFO:
        return
    title, content = STEP_INFO[step]
    with st.expander(f"ℹ️ {title}", expanded=False):
        st.markdown(content)



GAME_LOG_FILE = Path(__file__).parent / "guided_task_results.json"


# Persistence
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


# Case sampling
@st.cache_data
def _get_df():
    return du.load_results()


def sample_case():
    """Sample a random (network, instance_id, target) with at least 3 valid CF images."""
    df = _get_df()
    sub = df[(df["timeout"] != 1) & (df["network"].str.contains("mnist", na=False))].copy()
    for _ in range(40):
        row = sub.sample(1).iloc[0]
        network = row["network"]
        instance_id = int(row["image"])
        target = int(row["target"])
        orig = du.load_image(network, instance_id, "original")
        if orig is None:
            continue
        available_methods = [
            m for m in du.METHODS
            if du.load_image(network, instance_id, m, target) is not None
        ]
        if len(available_methods) >= 3:
            return {
                "network": network,
                "instance_id": instance_id,
                "target": target,
                "original_label": int(row["original_label"]) if "original_label" in row else None,
                "available_methods": available_methods,
            }
    return None


def reset_task():
    st.session_state.task_step = 0
    st.session_state.task_player = ""
    st.session_state.task_case = None
    st.session_state.task_responses = {}
    st.session_state.task_method = None
    st.session_state.task_saved = False


def init_task_state():
    defaults = {
        "task_step": 0,
        "task_player": "",
        "task_case": None,
        "task_responses": {},
        "task_method": None,
        "task_saved": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# Main render entry point
def render_task(player_name: str, case: dict, random_method: str):
    """
    Called from app.py with already-initialised player/case.
    Renders whichever step is current in st.session_state.task_step.
    """
    step = st.session_state.task_step

    if step == 1:
        _step1(case, random_method)
    elif step == 2:
        _step2(case, random_method)
    elif step == 3:
        _step3(case)
    elif step == 4:
        _step4(case, random_method)
    elif step == 5:
        _step5(case)
    elif step == 6:
        _step6(case)
    elif step == 7:
        _step7(player_name, case)


# Step 1: Visual Inspection
def _step1(case: dict, method: str):
    progress_bar(1)
    step_title(1, "Visual Inspection")
    info_button(1)
    st.markdown("Look at the images below. **No metric information is shown yet.** Use only your eyes.")

    col_img, col_q = st.columns([1.2, 1])
    with col_img:
        show_image_pair(case, method)

    with col_q:
        st.markdown("**Based on visual inspection only, does the counterfactual look successful?**")
        st.caption("A successful CF should look like it clearly belongs to the target class.")

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
            height=130,
        )

        if st.button("Next →", type="primary"):
            st.session_state.task_responses["step1_judgment"] = judgment
            st.session_state.task_responses["step1_why"] = why
            st.session_state.task_step = 2
            st.rerun()


# Step 2: Your Prediction
def _step2(case: dict, method: str):
    progress_bar(2)
    step_title(2, "Your Prediction")
    info_button(2)
    st.markdown("Now **estimate** the evaluation metrics for this counterfactual, based on what you see.")

    col_img, col_q = st.columns([1.2, 1])
    with col_img:
        show_image_pair(case, method)

    with col_q:
        st.markdown("**Rate the following based on what you see (0 to 1):**")
        st.info(
            "These are your **human judgments** on a simple 0-1 scale. "
        "In step 4 you will see the actual model metrics, note that those use "
            "different scales and directions, which we will explain clearly at that point."
        )

        st.markdown("**Validity**: do you think the model was fooled by this CF?")
        st.caption("0 = definitely not, 1 = definitely yes")
        validity_est = st.slider(
            "Your validity estimate",
            0.0, 1.0, 0.5, 0.01,
            key="s2_validity",
        )

        st.markdown("**Plausibility**: how realistic does this CF look to you?")
        st.caption("0 = very unrealistic / distorted, 1 = very realistic / natural")
        plausibility_est = st.slider(
            "Your plausibility estimate",
            0.0, 1.0, 0.5, 0.01,
            key="s2_plausibility",
        )

        if st.button("Next →", type="primary"):
            st.session_state.task_responses["step2_validity_estimate"] = validity_est
            st.session_state.task_responses["step2_plausibility_estimate"] = plausibility_est
            st.session_state.task_step = 3
            st.rerun()


# Step 3: Game
def _step3(case: dict):
    progress_bar(3)
    step_title(3, "The Game: Choose the Best Explanation")
    info_button(3)
    st.markdown(
        "All 5 CF methods are shown below. **Which one do you think is the most successful?** "
        "Consider both whether it looks realistic and whether the model would be fooled."
    )

    show_all_methods_grid(case)
    st.divider()

    col_pick, col_conf = st.columns([1, 1])
    with col_pick:
        best_pick = st.radio(
            "Select the CF method you think is best:",
            du.METHODS,
            key="s3_pick",
        )

    with col_conf:
        confidence = st.slider(
            "How confident are you in this choice? (%)",
            0, 100, 60, 5,
            key="s3_confidence",
        )

    st.divider()
    st.markdown("**Tell us why you made this choice, this is required.**")
    why_pick = st.text_area(
        "Why did you pick this method? Were you surprised by any of the images? "
        "What stood out to you compared to the other methods?",
        placeholder="e.g. PIECE looked the most natural, the digit was still recognisable "
        "but clearly shifted towards the target. The alibi methods looked too noisy...",
        key="s3_why",
        height=120,
    )

    can_proceed = bool(why_pick.strip())
    if not can_proceed:
        st.caption("Please explain your choice above before continuing.")

    if st.button("Submit choice →", type="primary", disabled=not can_proceed):
        st.session_state.task_responses["step3_best_method"] = best_pick
        st.session_state.task_responses["step3_confidence"] = confidence
        st.session_state.task_responses["step3_why_pick"] = why_pick
        st.session_state.task_step = 4
        st.rerun()


# Step 4: Actual Metrics Revealed
def _step4(case: dict, method: str):
    progress_bar(4)
    step_title(4, "Actual Metrics Revealed")
    info_button(4)
    st.markdown(
        "Before we show you the actual metrics, here is a reminder of what each metric means. "
        "Read this carefully, the numbers may not work the way you expect."
    )

    # METRIC EXPLAINER: always shown expanded before first metric reveal
    metric_explainer(collapsed=False)

    st.divider()
    st.markdown(f"**Metrics for the CF shown in steps 1 and 2** (method hidden until step 5):")

    col_img, col_metrics = st.columns([1.2, 1])
    with col_img:
        show_image_pair(case, method, show_method_label=False)

    with col_metrics:
        metrics = du.get_metric_row(case["network"], case["instance_id"], method, case["target"])

        if metrics:
            correctness = metrics.get("correctness", float("nan"))
            im1 = metrics.get("IM1", float("nan"))
            implaus = metrics.get("implausibility", float("nan"))
            im2 = metrics.get("IM2", float("nan"))

            # Validity
            st.markdown("**Validity**")
            your_val = st.session_state.task_responses.get("step2_validity_estimate")
            v_col1, v_col2 = st.columns(2)
            v_col1.metric("Your human estimate (0-1)", fmt(your_val, 2))
            actual_val = int(correctness) if correctness in [0, 1] else None
            v_col2.metric("Actual value", actual_val if actual_val is not None else "N/A")
            v_col2.markdown(validity_badge(correctness), unsafe_allow_html=True)

            # CF quality
            if actual_val == 1:
                st.success("CF result: Valid, the model was successfully fooled by this counterfactual.")
            elif actual_val == 0:
                st.error("CF result: Invalid, the model was NOT fooled by this counterfactual.")

            # Estimate vs actual
            if your_val is not None and actual_val is not None:
                estimated_valid = your_val >= 0.5
                actually_valid = actual_val == 1
                if estimated_valid == actually_valid:
                    st.info("Your estimate matched: your intuition pointed in the right direction.")
                else:
                    st.warning(
                        f"Your estimate did not match: you estimated {fmt(your_val, 2)} "
                        f"suggesting {'valid' if estimated_valid else 'invalid'}, "
                        f"but the CF was actually {'valid' if actually_valid else 'invalid'}. "
                        "This disagreement between human intuition and model outcome is exactly what this study investigates."
                    )
            st.caption("Validity is binary: 1 = valid (model fooled), 0 = invalid (model not fooled).")

            st.divider()

            # Plausibility (IM1)
            st.markdown("**Plausibility (IM1)**")
            your_plaus = st.session_state.task_responses.get("step2_plausibility_estimate")
            st.markdown(f"Your human estimate: **{fmt(your_plaus, 2)}** (where 1.0 = very realistic)")

            if fmt(im1) != "N/A":
                import math
                im1_val = float(im1) if not (isinstance(im1, float) and math.isnan(im1)) else None
                if im1_val is not None:
                    if im1_val < 1.0:
                        plaus_label = "Good"
                        plaus_detail = f"IM1 = {fmt(im1, 4)} is below 1.0 (dataset median). This CF looks relatively close to real training data."
                        plaus_fn = st.success
                    elif im1_val < 2.0:
                        plaus_label = "Moderate"
                        plaus_detail = f"IM1 = {fmt(im1, 4)} is around the dataset mean (1.47). Somewhat plausible but not ideal."
                        plaus_fn = st.warning
                    else:
                        plaus_label = "Poor"
                        plaus_detail = f"IM1 = {fmt(im1, 4)} is well above the dataset mean (1.47). This CF looks unnatural or distorted."
                        plaus_fn = st.error
                    st.metric("Actual IM1", fmt(im1, 4))
                    plaus_fn(f"{plaus_label}: {plaus_detail}")
                    st.caption("LOWER IM1 = MORE plausible. IM1 is not a 0-1 scale, values above 1.0 are common.")

            st.divider()

            # Implausibility
            if fmt(implaus) != "N/A":
                import math
                implaus_val = float(implaus) if not (isinstance(implaus, float) and math.isnan(implaus)) else None
                if implaus_val is not None:
                    if implaus_val < 0.05:
                        impl_label = "Good"
                        impl_detail = f"{fmt(implaus, 4)} is very low (dataset median is 0.03). The CF looks realistic."
                        impl_fn = st.success
                    elif implaus_val < 0.30:
                        impl_label = "Moderate"
                        impl_detail = f"{fmt(implaus, 4)} is around the dataset mean (0.20). Somewhat plausible."
                        impl_fn = st.warning
                    else:
                        impl_label = "Poor"
                        impl_detail = f"{fmt(implaus, 4)} is high (dataset max is 0.78). The CF looks quite unnatural."
                        impl_fn = st.error
                    st.markdown("**Implausibility score**")
                    st.metric("Actual value", fmt(implaus, 4))
                    impl_fn(f"{impl_label}: {impl_detail}")
                    st.caption("LOWER is better. Dataset range: -0.085 to 0.782, mean: 0.203.")

            st.divider()

            # IM2 / L2 distance
            if fmt(im2) != "N/A":
                import math
                im2_val = float(im2) if not (isinstance(im2, float) and math.isnan(im2)) else None
                if im2_val is not None:
                    if im2_val < 0.01:
                        im2_label = "Good"
                        im2_detail = f"{fmt(im2, 4)} is very low (dataset median is 0.003). Very few pixels were changed."
                        im2_fn = st.success
                    elif im2_val < 0.15:
                        im2_label = "Moderate"
                        im2_detail = f"{fmt(im2, 4)} is around the dataset mean (0.070). A reasonable number of pixels were changed."
                        im2_fn = st.warning
                    else:
                        im2_label = "Poor"
                        im2_detail = f"{fmt(im2, 4)} is high (dataset max is 0.942). Large parts of the image were altered."
                        im2_fn = st.error
                    st.markdown("**IM2 / L2 distance**")
                    st.metric("Actual value", fmt(im2, 4))
                    im2_fn(f"{im2_label}: {im2_detail}")
                    st.caption("LOWER is better. Measures total pixel change vs the original. Dataset range: 0.0 to 0.942, mean: 0.070.")
        else:
            st.warning("No metrics found for this case in the dataset.")

    st.divider()
    st.markdown("**Your reaction to these results, this is required.**")
    surprised = st.text_area(
        "Were you surprised by the actual metrics? Did they match what you expected from looking "
        "at the image? If not, what was different and why do you think that is?",
        placeholder="e.g. I expected validity to be 1 because the image looked convincing, "
        "but it was actually 0, the model was not fooled even though it looked good to me...",
        key="s4_surprised",
        height=120,
    )

    can_proceed = bool(surprised.strip())
    if not can_proceed:
        st.caption("Please share your reaction above before continuing.")

    if st.button("Next →", type="primary", disabled=not can_proceed):
        st.session_state.task_responses["step4_actual_correctness"] = metrics.get("correctness") if metrics else None
        st.session_state.task_responses["step4_actual_IM1"] = metrics.get("IM1") if metrics else None
        st.session_state.task_responses["step4_actual_implausibility"] = metrics.get("implausibility") if metrics else None
        st.session_state.task_responses["step4_surprised_reaction"] = surprised
        st.session_state.task_step = 5
        st.rerun()


# Step 5: Explanation & Feedback
def _step5(case: dict):
    progress_bar(5)
    step_title(5, "Explanation & Feedback")
    info_button(5)

    player_pick = st.session_state.task_responses.get("step3_best_method", "?")
    method_shown = st.session_state.task_method

    # Show the image from steps 1 & 2 again for the AHA moment
    st.markdown("**Do you remember this image? This is the CF you judged in steps 1 and 2.**")
    col_img, col_rank = st.columns([1, 1.2])

    with col_img:
        show_image_pair(case, method_shown, show_method_label=True)
        st.success(f"This CF was produced by: **{method_shown}**")

    with col_rank:
        st.markdown(f"**You selected `{player_pick}` as the best method in step 3.**")
        st.markdown("Actual ranking: validity first, then plausibility (IM1):")

        all_metrics = get_metrics_for_case(case)
        ranked = []
        for method in du.METHODS:
            m = all_metrics.get(method, {})
            ranked.append({
                "method": method,
                "correctness": m.get("correctness", float("nan")),
                "IM1": m.get("IM1", float("nan")),
            })

        ranked_df = pd.DataFrame(ranked).sort_values(
            ["correctness", "IM1"], ascending=[False, True]
        ).reset_index(drop=True)
        ranked_df.index += 1

        actual_best = ranked_df.iloc[0]["method"] if len(ranked_df) > 0 else "?"
        matched = player_pick == actual_best

        for rank, (i, row) in enumerate(ranked_df.iterrows(), start=1):
            prefix = "⭐ " if rank == 1 else f"{rank}. "
            c = row["correctness"]
            c_str = "Valid" if c == 1 else ("Invalid" if c == 0 else "?")
            badge = "✓" if c == 1 else "✗"
            im1_str = f"IM1: {fmt(row['IM1'], 3)}"
            highlights = []
            if row["method"] == player_pick:
                highlights.append("**your pick**")
            if row["method"] == method_shown:
                highlights.append("**shown in steps 1 & 2**")
            suffix = " ← " + ", ".join(highlights) if highlights else ""
            st.markdown(f"{prefix}**{row['method']}** {badge} {c_str} | {im1_str}{suffix}")

        st.divider()

        if matched:
            st.success(f"✅ Your pick **{player_pick}** matched the objectively best method!")
        else:
            st.info(
                f"The objectively best method was **{actual_best}**, "
                f"but you chose **{player_pick}**. This difference is exactly what this study explores."
            )

    st.divider()
    st.markdown("**Your reflection, this is required.**")
    why_chosen = st.text_area(
        "Looking at the full ranking above: does it make sense to you? "
        "Were you surprised that the objectively best method was different from your pick "
        "(or the same)? What do you think explains the difference between human judgment and the metrics?",
        placeholder="e.g. I chose PIECE because it looked most natural, but alibi-CF had "
        "better validity. I think the model cares about pixel patterns I cannot see...",
        key="s5_why",
        height=130,
    )

    can_proceed = bool(why_chosen.strip())
    if not can_proceed:
        st.caption("Please share your reflection above before continuing.")

    if st.button("Next →", type="primary", disabled=not can_proceed):
        st.session_state.task_responses["step5_why_chosen"] = why_chosen
        st.session_state.task_responses["step5_actual_best_method"] = actual_best
        st.session_state.task_responses["step5_player_matched"] = matched
        st.session_state.task_step = 6
        st.rerun()


# Step 6: Compare Methods on This Case
def _step6(case: dict):
    progress_bar(6)
    step_title(6, "Compare All Methods on This Case")
    info_button(6)

    player_pick = st.session_state.task_responses.get("step3_best_method", "?")
    actual_best = st.session_state.task_responses.get("step5_actual_best_method", "?")

    st.markdown(
        "In step 5 you saw that the **overall best method** was determined by combining "
        "validity and plausibility. But does the best overall method also win on every "
        "individual metric? Look at the table below and find out."
    )

    # Remind the user of their step 3 pick with the image
    st.divider()
    st.markdown(f"**Reminder: in step 3 you picked `{player_pick}` as the best method.**")
    st.caption("Below is the image grid from step 3, same images, now with the full metric table.")
    show_all_methods_grid(case)

    # Metric explainer collapsed as reminder
    metric_explainer(collapsed=True)

    # Full metrics table
    st.markdown("**Full evaluation metrics per method:**")
    all_metrics = get_metrics_for_case(case)
    table_data = {}
    for method in du.METHODS:
        m = all_metrics.get(method, {})
        c = m.get("correctness", float("nan"))
        table_data[method] = {
            "Validity": "✓ Valid" if c == 1 else ("✗ Invalid" if c == 0 else "N/A"),
            "IM1 (lower=better)": fmt(m.get("IM1"), 3),
            "Implausibility (lower=better)": fmt(m.get("implausibility"), 3),
            "L2 distance (lower=better)": fmt(m.get("l2"), 3),
        }
    st.dataframe(pd.DataFrame(table_data).T, use_container_width=True)

    # Highlight per-metric winners
    st.divider()
    st.markdown("**Per-metric winners from the table above:**")

    valid_methods = [m for m in du.METHODS
                     if all_metrics.get(m, {}).get("correctness") == 1]
    best_im1 = min(
        [(m, all_metrics[m].get("IM1", float("inf"))) for m in du.METHODS
         if all_metrics.get(m, {}).get("IM1") is not None],
        key=lambda x: x[1], default=(None, None)
    )
    best_implaus = min(
        [(m, all_metrics[m].get("implausibility", float("inf"))) for m in du.METHODS
         if all_metrics.get(m, {}).get("implausibility") is not None],
        key=lambda x: x[1], default=(None, None)
    )
    best_l2 = min(
        [(m, all_metrics[m].get("l2", float("inf"))) for m in du.METHODS
         if all_metrics.get(m, {}).get("l2") is not None],
        key=lambda x: x[1], default=(None, None)
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Most valid methods", ", ".join(valid_methods) if valid_methods else "None")
    col2.metric("Best IM1 (plausibility)", f"{best_im1[0]} ({fmt(best_im1[1], 3)})" if best_im1[0] else "N/A")
    col3.metric("Best implausibility", f"{best_implaus[0]} ({fmt(best_implaus[1], 3)})" if best_implaus[0] else "N/A")
    col4.metric("Best L2 distance", f"{best_l2[0]} ({fmt(best_l2[1], 3)})" if best_l2[0] else "N/A")

    st.info(
        f"The overall best method from step 5 was **{actual_best}**. "
        "Notice whether it wins on every individual metric or only on the combined score. "
        "This is the key insight: 'best overall' does not always mean 'best at everything'."
    )

    st.divider()
    st.markdown("**One question, this is required.**")
    change_pick = st.radio(
        f"Knowing the full metric breakdown, would you change your step 3 pick of `{player_pick}`?",
        ["No, I would still pick the same method", "Yes, I would change my pick"],
        key="s6_change_pick",
    )

    new_pick = player_pick
    if change_pick == "Yes, I would change my pick":
        new_pick = st.selectbox(
            "Which method would you pick instead?",
            [m for m in du.METHODS if m != player_pick],
            key="s6_new_pick",
        )

    why_change = st.text_area(
        "Explain your reasoning, required. "
        "What does the table tell you that changed (or confirmed) your view?",
        placeholder="e.g. I would still pick alibi-Proto-CF because it wins on implausibility "
        "and L2, even though PIECE has a better IM1 score but is invalid...",
        key="s6_why_change",
        height=110,
    )

    can_proceed = bool(why_change.strip())
    if not can_proceed:
        st.caption("Please explain your reasoning above before continuing.")

    if st.button("Next →", type="primary", disabled=not can_proceed):
        st.session_state.task_responses["step6_would_change_pick"] = change_pick
        st.session_state.task_responses["step6_final_pick"] = new_pick
        st.session_state.task_responses["step6_why_change"] = why_change
        st.session_state.task_responses["step6_per_metric_best_valid"] = ", ".join(valid_methods)
        st.session_state.task_responses["step6_per_metric_best_im1"] = best_im1[0] if best_im1[0] else "N/A"
        st.session_state.task_responses["step6_per_metric_best_implaus"] = best_implaus[0] if best_implaus[0] else "N/A"
        st.session_state.task_responses["step6_per_metric_best_l2"] = best_l2[0] if best_l2[0] else "N/A"
        st.session_state.task_step = 7
        st.rerun()


# Step 7: Confidence & Final Thoughts
def _step7(player_name: str, case: dict):
    import math
    progress_bar(7)
    step_title(7, "Confidence & Final Thoughts")
    info_button(7)

    st.markdown(
        "Before you give your final rating, here is a full recap of your journey through this case. "
        "Read it carefully before filling in your confidence and final thoughts below."
    )

    # Full session recap
    r = st.session_state.task_responses
    method_shown = st.session_state.task_method

    with st.container(border=True):
        st.markdown("#### Your session recap")

        # Step 1 recap
        st.markdown("**Step 1: Your visual judgment**")
        col_img, col_s1 = st.columns([1, 1.5])
        with col_img:
            show_image_pair(case, method_shown, show_method_label=True)
        with col_s1:
            st.markdown(f"You judged this CF as: **{r.get('step1_judgment', '?')}**")
            why1 = r.get("step1_why", "")
            if why1 and why1.strip():
                st.markdown(f"Your reasoning: *{why1}*")

        st.divider()

        # Step 2 recap
        st.markdown("**Step 2: Your metric estimates**")
        col_v, col_p = st.columns(2)
        col_v.metric("Your validity estimate", fmt(r.get("step2_validity_estimate"), 2))
        col_p.metric("Your plausibility estimate", fmt(r.get("step2_plausibility_estimate"), 2))

        st.divider()

        # Step 3 recap
        st.markdown("**Step 3: Your game pick**")
        st.markdown(f"You chose **{r.get('step3_best_method', '?')}** with **{r.get('step3_confidence', '?')}%** confidence.")
        why3 = r.get("step3_why_pick", "")
        if why3 and why3.strip():
            st.markdown(f"Your reasoning: *{why3}*")

        st.divider()

        # Step 4 recap: actual metrics for the CF shown in steps 1 and 2
        st.markdown("**Step 4: Actual metrics for the CF you saw in steps 1 and 2**")
        actual_correctness = r.get("step4_actual_correctness")
        actual_im1 = r.get("step4_actual_IM1")
        actual_implaus = r.get("step4_actual_implausibility")

        mcol1, mcol2, mcol3 = st.columns(3)
        # Validity
        if actual_correctness == 1:
            mcol1.success("Validity: Valid (1)")
        elif actual_correctness == 0:
            mcol1.error("Validity: Invalid (0)")
        else:
            mcol1.metric("Validity", "N/A")

        # IM1 label
        if actual_im1 is not None and not (isinstance(actual_im1, float) and math.isnan(actual_im1)):
            if actual_im1 < 1.0:
                mcol2.success(f"IM1: {fmt(actual_im1, 4)} (Good)")
            elif actual_im1 < 2.0:
                mcol2.warning(f"IM1: {fmt(actual_im1, 4)} (Moderate)")
            else:
                mcol2.error(f"IM1: {fmt(actual_im1, 4)} (Poor)")
        else:
            mcol2.metric("IM1", "N/A")

        # Implausibility label
        if actual_implaus is not None and not (isinstance(actual_implaus, float) and math.isnan(actual_implaus)):
            if actual_implaus < 0.05:
                mcol3.success(f"Implausibility: {fmt(actual_implaus, 4)} (Good)")
            elif actual_implaus < 0.30:
                mcol3.warning(f"Implausibility: {fmt(actual_implaus, 4)} (Moderate)")
            else:
                mcol3.error(f"Implausibility: {fmt(actual_implaus, 4)} (Poor)")
        else:
            mcol3.metric("Implausibility", "N/A")

        # Did validity estimate match?
        val_est = r.get("step2_validity_estimate")
        if val_est is not None and actual_correctness is not None:
            estimated_valid = val_est >= 0.5
            actually_valid = actual_correctness == 1
            if estimated_valid == actually_valid:
                st.info("Your validity estimate pointed in the correct direction.")
            else:
                st.warning(
                    f"Your validity estimate ({fmt(val_est, 2)}) suggested "
                    f"{'valid' if estimated_valid else 'invalid'} but the CF was actually "
                    f"{'valid' if actually_valid else 'invalid'}."
                )

        surprised = r.get("step4_surprised_reaction", "")
        if surprised and surprised.strip():
            st.markdown(f"Your reaction: *{surprised}*")

        st.divider()

        # Step 5 recap
        st.markdown("**Step 5: Overall best method by metrics**")
        actual_best = r.get("step5_actual_best_method", "?")
        player_pick = r.get("step3_best_method", "?")
        matched = r.get("step5_player_matched", False)
        if matched:
            st.success(f"Your pick **{player_pick}** matched the objectively best method **{actual_best}**.")
        else:
            st.warning(f"You picked **{player_pick}** but the objectively best method was **{actual_best}**.")
        why5 = r.get("step5_why_chosen", "")
        if why5 and why5.strip():
            st.markdown(f"Your reflection: *{why5}*")

        st.divider()

        # Step 6 recap
        st.markdown("**Step 6: Would you change your pick?**")
        st.markdown(f"{r.get('step6_would_change_pick', '?')}: final pick: **{r.get('step6_final_pick', player_pick)}**")
        why6 = r.get("step6_why_change", "")
        if why6 and why6.strip():
            st.markdown(f"Your reasoning: *{why6}*")

    st.divider()

    # Final inputs
    st.markdown("**Now give your final confidence rating and thoughts:**")

    col_a, col_b = st.columns(2)

    with col_a:
        post_confidence = st.slider(
            "How confident are you in your step 1 judgment now that you have seen everything? (%)",
            0, 100, 70, 5,
            key="s7_confidence",
        )

        change_answer = st.radio(
            "Would you change your step 1 answer after seeing all the metrics?",
            ["Yes", "No"],
            key="s7_change",
        )

    with col_b:
        final_thoughts = st.text_area(
            "Final thoughts, required. Did the metrics align with your intuition overall? "
            "What surprised you most during this study?",
            placeholder="e.g. I was surprised that validity was 0 even when the image looked "
            "convincing to me. It seems the model looks at completely different features...",
            key="s7_thoughts",
            height=200,
        )

    can_finish = bool(final_thoughts.strip())
    if not can_finish:
        st.caption("Please share your final thoughts above before finishing.")

    st.divider()

    if st.button("✓ Finish & Save results", type="primary", disabled=not can_finish):
        st.session_state.task_responses["step7_post_confidence"] = post_confidence
        st.session_state.task_responses["step7_change_answer"] = change_answer
        st.session_state.task_responses["step7_final_thoughts"] = final_thoughts

        save_session(st.session_state.task_responses)

        st.success("**Thank you! Your responses have been saved.**")
        st.balloons()

        player_pick = st.session_state.task_responses.get("step3_best_method", "?")
        actual_best = st.session_state.task_responses.get("step5_actual_best_method", "?")

        st.markdown(f"**Player:** {player_name}")
        st.markdown(f"**Your game pick:** `{player_pick}`")
        st.markdown(f"**Objectively best method:** `{actual_best}`")

        if player_pick == actual_best:
            st.success("Your intuition matched the metrics!")
        else:
            st.info("Your intuition differed from the metrics, that gap is exactly what this research is studying.")

        time.sleep(1.5)
        reset_task()
        st.rerun()