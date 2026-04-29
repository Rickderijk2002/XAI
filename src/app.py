"""
XAI Counterfactual Explanations: Guided Task Mode
JM0110 Interactive & Explainable AI Design at JADS

Run with:
    streamlit run app.py
"""

import random
import time

import streamlit as st

from components import inject_css
from intro import render_intro
from leaderboard import render_leaderboard
from minigame import render_minigame
from results import render_results
from task import init_task_state, reset_task, sample_case

# Page config
st.set_page_config(
    page_title="XAI CF Explorer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()

# Global session state
if "page" not in st.session_state:
    st.session_state.page = "task"

init_task_state()

# Sidebar
with st.sidebar:
    st.markdown("### 🔍 XAI CF Explorer")
    st.markdown("*Counterfactual Explanation Study*")
    st.divider()

    page_choice = st.radio(
        "Navigate",
        ["Guided Task Mode", "Mini Game", "Leaderboard", "Guided Task Results"],
        index=["task", "minigame", "leaderboard", "results"].index(
            st.session_state.page
        ) if st.session_state.page in ["task", "minigame", "leaderboard", "results"] else 0,
        key="sidebar_nav",
    )

    page_map = {
        "Guided Task Mode": "task",
        "Mini Game": "minigame",
        "Leaderboard": "leaderboard",
        "Guided Task Results": "results",
    }
    st.session_state.page = page_map[page_choice]

    if st.session_state.page == "task" and st.session_state.task_step > 0:
        st.divider()
        st.markdown(
            f'<div style="font-family:IBM Plex Mono;font-size:0.7rem;color:#9ca3af;'
            f'padding:8px;border-top:1px solid #374151;">'
            f'Player: <b>{st.session_state.task_player}</b><br>'
            f'Step: {st.session_state.task_step}/7</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    if st.button("↺ Reset session", use_container_width=True):
        reset_task()
        st.rerun()


# Page routing
page = st.session_state.page

if page == "task":
    step = st.session_state.task_step

    if step == 0:
        name = render_intro()
        if name:
            case = sample_case()
            if case is None:
                st.error("Could not sample a valid case. Check that the Data/ folder is set up correctly.")
                st.stop()
            method = random.choice(case["available_methods"])

            st.session_state.task_player = name
            st.session_state.task_case = case
            st.session_state.task_method = method
            st.session_state.task_step = 1
            st.session_state.task_responses = {
                "player_name": name,
                "timestamp": time.time(),
                "network": case["network"],
                "instance_id": case["instance_id"],
                "target": case["target"],
                "original_label": case.get("original_label"),
                "shown_method_steps_1_2": method,
            }
            st.rerun()
    else:
        from task import render_task
        render_task(
            st.session_state.task_player,
            st.session_state.task_case,
            st.session_state.task_method,
        )

elif page == "minigame":
    render_minigame()

elif page == "leaderboard":
    render_leaderboard()

elif page == "results":
    render_results()