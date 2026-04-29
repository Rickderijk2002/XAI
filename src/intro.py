"""
intro.py Introduction screen and name entry for the Guided Task Mode.
"""

import streamlit as st
from components import page_title

METHOD_TABLE = [
    {"Method": "PIECE", "How it works": "Perturbs meaningful image segments.", "Focus": "Locally coherent changes"},
    {"Method": "Min-Edit", "How it works": "Changes as few pixels as possible to flip prediction.", "Focus": "Minimal edits"},
    {"Method": "C-Min-Edit", "How it works": "Constrained Min-Edit that stays closer to data distribution.", "Focus": "Minimal + realistic"},
    {"Method": "alibi-Proto-CF", "How it works": "Pulls CF toward prototype examples of target class.", "Focus": "Prototype-guided realism"},
    {"Method": "alibi-CF", "How it works": "Gradient-based optimisation to flip class with limited change.", "Focus": "Optimised class flip"},
]

METRIC_TABLE = [
    {"Metric": "Validity (Correctness)", "What it captures": "Did the model predict the target class?", "Good direction": "1 = valid, 0 = invalid"},
    {"Metric": "Plausibility (IM1)", "What it captures": "How realistic the CF looks vs training data.", "Good direction": "Lower is better"},
    {"Metric": "Implausibility score", "What it captures": "Overall unnaturalness of the CF.", "Good direction": "Lower is better"},
    {"Metric": "IM2 / L2 distance", "What it captures": "How many pixels changed vs original.", "Good direction": "Lower is better"},
]


def render_intro():
    page_title("Welcome to the XAI Counterfactual Study")

    st.markdown(
        "Before you begin, read this page carefully. "
        "It explains what you will do, what the images show, and what the numbers mean."
    )
    st.info(
        "ℹ️ Throughout the study you will see an **info button** on every page. "
        "Click it at any time to get a quick explanation relevant to that step. "
        "You never need to come back to this page."
    )

    with st.expander("📌 What is this study about?", expanded=True):
        st.markdown("""
        This study is part of a research project at JADS investigating how **human judgment**
        of AI explanations compares to **objective evaluation metrics**.

        You will see images of handwritten digits (MNIST) and modified versions called
        **counterfactual (CF) explanations**. These are images tweaked so that the AI model
        predicts a different digit class than the original.

        Your job is to judge these CFs using your intuition. There are no right or wrong answers.
        All responses are saved anonymously for internal research only.
        """)

    with st.expander("🔄 What is a counterfactual explanation?", expanded=True):
        st.markdown("""
        A counterfactual explanation answers:
        > *"What would need to change in this image for the AI to predict a different class?"*

        Example: the model predicts **7**. A CF is a slightly modified version the model now
        predicts as **3**.

        A good CF should:
        - **Fool the model** into predicting the target class (this is *validity*)
        - **Still look realistic** to a human (this is *plausibility*)
        """)

    with st.expander("🔬 The 5 methods and 📊 the metrics", expanded=True):
        st.markdown("**5 methods generate the CFs. Each uses a different strategy:**")
        st.table(METHOD_TABLE)

        st.divider()

        st.markdown("**4 metrics evaluate each CF. Pay attention to the direction:**")
        st.table(METRIC_TABLE)
        st.warning(
            "IM1 and implausibility go DOWN when the CF is better. "
            "Validity goes UP. These are not all on a 0 to 1 scale."
        )
        st.caption(
            "You will NOT know which method produced which image in the early steps. "
            "This is intentional so your judgment stays unbiased."
        )

    with st.expander("🗺 What will happen during the study?", expanded=False):
        st.markdown("""
        7 steps for one randomly selected case:

        1. **Visual Inspection** Judge one CF image. No metrics shown.
        2. **Your Prediction** Estimate validity and plausibility with sliders.
        3. **The Game** All 5 methods shown. Pick the best one.
        4. **Metrics Revealed** See the actual scores for the CF from steps 1 and 2.
        5. **Explanation and Feedback** See the full method ranking.
        6. **Compare Methods** Full metrics table for all methods.
        7. **Final Reflection** Did the metrics match your intuition?

        Results are saved when you complete step 7.
        """)

    st.divider()

    st.markdown("### Ready to begin?")
    st.markdown("Enter your name or participant ID below and click Start.")

    name = st.text_input(
        "Your name or participant ID:",
        placeholder="e.g. Rick or P01",
        key="intro_name_input",
    )

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        start_clicked = st.button(
            "Start the study",
            type="primary",
            disabled=not name.strip(),
            use_container_width=True,
        )
    with col_info:
        if not name.strip():
            st.caption("Please enter your name to activate the Start button.")

    return name.strip() if start_clicked and name.strip() else None