"""
minigame.py: MNIST-only mini game with 2-strike system.
Players guess whether the model converged (CF is valid) for random MNIST cases.
Saves results to minigame_log.json.
"""

import json
import random
import time
from pathlib import Path

import streamlit as st

import data_utils as du
from components import metric_explainer, page_title, show_mnist_pair

MINIGAME_LOG_FILE = Path(__file__).parent / "minigame_log.json"
MAX_STRIKES = 2


# Persistence
def load_minigame_log() -> list:
    if MINIGAME_LOG_FILE.exists():
        with open(MINIGAME_LOG_FILE) as f:
            return json.load(f)
    return []


def save_minigame_session(entry: dict):
    log = load_minigame_log()
    log.append(entry)
    with open(MINIGAME_LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


# MNIST case pool
@st.cache_data
def _get_mnist_pool():
    """Return all MNIST rows with at least one valid CF."""
    df = du.load_results()
    mnist = df[df["network"].str.contains("mnist", na=False)].copy()
    mnist = mnist[mnist["timeout"] != 1]
    return mnist.reset_index(drop=True)


def sample_mnist_round(used_keys: set):
    """Sample a random MNIST (instance, method, target) not yet seen this session."""
    pool = _get_mnist_pool()
    attempts = 0
    while attempts < 60:
        attempts += 1
        row = pool.sample(1).iloc[0]
        instance_id = int(row["image"])
        target = int(row["target"])
        method = random.choice(du.METHODS)
        key = f"{instance_id}_{method}_{target}"
        if key in used_keys:
            continue
        orig = du.load_image("mnist_output_100", instance_id, "original")
        cf = du.load_image("mnist_output_100", instance_id, method, target)
        if orig is None or cf is None:
            continue
        correctness = row.get("correctness", None)
        if correctness not in [0, 1]:
            try:
                correctness = int(correctness)
            except Exception:
                continue
        return {
            "instance_id": instance_id,
            "target": target,
            "method": method,
            "correctness": int(correctness),
            "IM1": row.get("IM1", None),
            "implausibility": row.get("implausibility", None),
        }
    return None


# State
def init_minigame_state():
    defaults = {
        "mg_step": "name",       # "name" | "tutorial" | "playing" | "reveal" | "gameover"
        "mg_player": "",
        "mg_score": 0,
        "mg_strikes": 0,
        "mg_round": 0,
        "mg_current": None,       # current round dict
        "mg_used": set(),
        "mg_history": [],         # list of round dicts with guess added
        "mg_last_correct": None,
        "mg_correct": None,
        "mg_guess": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def reset_minigame():
    for k in ["mg_step", "mg_player", "mg_score", "mg_strikes", "mg_round",
              "mg_current", "mg_used", "mg_history", "mg_last_correct", "mg_guess"]:
        if k in st.session_state:
            del st.session_state[k]
    init_minigame_state()


# Render
def render_minigame():
    init_minigame_state()
    step = st.session_state.mg_step

    if step == "name":
        _mg_name()
    elif step == "tutorial":
        _mg_tutorial()
    elif step == "playing":
        _mg_playing()
    elif step == "reveal":
        _mg_reveal()
    elif step == "gameover":
        _mg_gameover()


def _mg_name():
    page_title("Mini Game: Did the Model Converge?")
    st.markdown("""
    Test your intuition! You will be shown **MNIST digit images** alongside their
    counterfactual explanations. Your job is to guess whether the model was successfully
    fooled by each CF, in other words, whether the CF is **valid**.

    You have **2 strikes**. Get 2 wrong and the game ends. How many can you get right?
    """)

    st.divider()
    name = st.text_input(
        "Enter your name or participant ID to start:",
        placeholder="e.g. Rick or P01",
        key="mg_name_input",
    )

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        if st.button("Start game →", type="primary", disabled=not name.strip()):
            st.session_state.mg_player = name.strip()
            st.session_state.mg_step = "tutorial"
            st.rerun()


def _mg_tutorial():
    page_title("Before you play: Quick Explanation")
    st.markdown("**What is validity and why does it matter?**")

    metric_explainer(collapsed=False)

    st.markdown("""
    **How the game works:**
    - You will see an original MNIST digit and its counterfactual
    - Guess: did the model predict the **target class** for this CF? (Yes = valid, No = invalid)
    - After each guess you will see the **actual answer and why**
    - **2 strikes** and the game ends, try to get as many right as possible!
    """)

    st.divider()
    if st.button("Let's play! →", type="primary"):
        # Load first round
        current = sample_mnist_round(st.session_state.mg_used)
        if current is None:
            st.error("Could not load MNIST data. Check Data/ folder setup.")
            return
        st.session_state.mg_current = current
        st.session_state.mg_used.add(
            f"{current['instance_id']}_{current['method']}_{current['target']}"
        )
        st.session_state.mg_round = 1
        st.session_state.mg_step = "playing"
        st.rerun()


def _mg_playing():
    current = st.session_state.mg_current
    score = st.session_state.mg_score
    strikes = st.session_state.mg_strikes
    rnd = st.session_state.mg_round

    # Header
    col_title, col_score, col_strikes = st.columns([3, 1, 1])
    with col_title:
        page_title(f"Round {rnd}")
    with col_score:
        st.metric("Score", score)
    with col_strikes:
        strike_display = "❌" * strikes + "⬜" * (MAX_STRIKES - strikes)
        st.markdown(f"**Strikes:** {strike_display}")

    st.markdown(
        f"Look at the original digit and the counterfactual below. "
        f"The CF is trying to make the model predict **{current['target']}**."
    )

    show_mnist_pair(current["instance_id"], current["method"], current["target"])

    st.divider()
    st.markdown("**Did the model converge? Is this CF valid?**")
    st.caption("Valid = the model actually predicts the target class for this CF image.")

    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button("✅ Yes, the model was fooled (valid)", use_container_width=True, type="primary"):
            guess = 1
            actual = current["correctness"]
            correct = guess == actual
            if correct:
                st.session_state.mg_score += 1
            else:
                st.session_state.mg_strikes += 1
            st.session_state.mg_guess = guess
            st.session_state.mg_correct = correct
            st.session_state.mg_step = "reveal"
            st.rerun()
    with col_no:
        if st.button("❌ No, the model was NOT fooled (invalid)", use_container_width=True):
            guess = 0
            actual = current["correctness"]
            correct = guess == actual
            if correct:
                st.session_state.mg_score += 1
            else:
                st.session_state.mg_strikes += 1
            st.session_state.mg_guess = guess
            st.session_state.mg_correct = correct
            st.session_state.mg_step = "reveal"
            st.rerun()


def _mg_reveal():
    current = st.session_state.mg_current
    guess = st.session_state.mg_guess
    actual = current["correctness"]
    correct = st.session_state.get("mg_correct", guess == actual)
    rnd = st.session_state.mg_round

    # Score/strikes already updated in _mg_playing, just display result
    if correct:
        st.success(f"✅ **Correct!** Round {rnd}, you got it right!")
    else:
        st.error(f"❌ **Wrong!** Round {rnd}, that is a strike ({st.session_state.mg_strikes}/{MAX_STRIKES})")

    # Always show what actually happened and why
    actual_str = "Valid (the model WAS fooled)" if actual == 1 else "Invalid (the model was NOT fooled)"
    guess_str = "Yes (valid)" if guess == 1 else "No (invalid)"

    st.markdown(f"**Your guess:** {guess_str}")
    st.markdown(f"**Actual result:** {actual_str}")

    # Explanation
    with st.expander("📊 Why is this the case? (Metrics explained)", expanded=True):
        im1 = current.get("IM1")
        implaus = current.get("implausibility")

        if actual == 1:
            st.markdown(f"""
            This CF **successfully fooled the model** (correctness = 1).
            The modified image caused the AI to predict class **{current['target']}** instead of the original class.

            - **IM1 (plausibility):** `{im1:.4f}`: {'reasonable plausibility' if im1 and im1 < 1.5 else 'somewhat implausible visually'}
            - **Implausibility:** `{implaus:.4f}`: {'low, meaning the CF looks fairly natural' if implaus and implaus < 0.5 else 'moderate, the CF may look somewhat unnatural'}

            Even if the image looks strange to you, the model found something in the pixel patterns
            that matched the target class well enough to change its prediction.
            """)
        else:
            st.markdown(f"""
            This CF **failed to fool the model** (correctness = 0).
            Despite the visual changes, the AI still predicts the original class, not class **{current['target']}**.

            - **IM1 (plausibility):** `{im1:.4f}`: {'the CF looks fairly realistic' if im1 and im1 < 1.5 else 'the CF looks quite distorted'}
            - **Implausibility:** `{implaus:.4f}`

            The changes made to the image were not enough to shift the model's internal representation
            across the decision boundary to class {current['target']}.
            This is a case where the image may look different to a human, but the model's
            features are not sensitive to those specific changes.
            """)

        st.markdown(f"**CF method used:** `{current['method']}`")

    # Save round to history
    st.session_state.mg_history.append({
        **current,
        "guess": guess,
        "correct": correct,
        "round": rnd,
    })

    # Check game over
    if st.session_state.mg_strikes >= MAX_STRIKES:
        if st.button("See your final score →", type="primary"):
            _finish_minigame()
            st.rerun()
    else:
        if st.button("Next round →", type="primary"):
            next_round = sample_mnist_round(st.session_state.mg_used)
            if next_round is None:
                st.warning("No more MNIST cases available. Game complete!")
                _finish_minigame()
                st.rerun()
            else:
                st.session_state.mg_used.add(
                    f"{next_round['instance_id']}_{next_round['method']}_{next_round['target']}"
                )
                st.session_state.mg_current = next_round
                st.session_state.mg_round += 1
                st.session_state.mg_guess = None
                st.session_state.mg_step = "playing"
                st.rerun()


def _finish_minigame():
    entry = {
        "player_name": st.session_state.mg_player,
        "timestamp": time.time(),
        "score": st.session_state.mg_score,
        "strikes": st.session_state.mg_strikes,
        "rounds_played": st.session_state.mg_round,
        "history": st.session_state.mg_history,
    }
    save_minigame_session(entry)
    st.session_state.mg_step = "gameover"


def _mg_gameover():
    page_title("Game Over!")
    score = st.session_state.mg_score
    rnd = st.session_state.mg_round

    st.markdown(f"### You scored **{score}** correct out of {rnd} rounds!")
    st.markdown(f"You got **{st.session_state.mg_strikes} strikes** and the game ended.")

    if score == 0:
        st.markdown("Better luck next time! The models are tricky to read.")
    elif score < 5:
        st.markdown("Good effort! The gap between human perception and model metrics is real.")
    elif score < 10:
        st.markdown("Solid score! Your intuition is aligning well with the model.")
    else:
        st.markdown("Excellent! You have strong intuition for model convergence.")

    st.info("Your results have been saved to the leaderboard.")
    st.balloons()

    if st.button("Play again →", type="primary"):
        reset_minigame()
        st.rerun()