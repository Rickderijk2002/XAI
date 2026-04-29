"""
results.py: Guided Task Results page.
Reads guided_task_results.json and displays aggregate analysis of all guided task sessions.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from components import page_title

GAME_LOG_FILE = Path(__file__).parent / "guided_task_results.json"


def render_results():
    page_title("Guided Task Results")
    st.markdown("Aggregate analysis of all completed Guided Task sessions.")

    if not GAME_LOG_FILE.exists():
        st.info("No sessions completed yet. Complete a Guided Task session to see results here.")
        return

    with open(GAME_LOG_FILE) as f:
        log = json.load(f)

    if not log:
        st.info("No sessions completed yet.")
        return

    gdf = pd.DataFrame(log)
    gdf["timestamp"] = pd.to_datetime(gdf["timestamp"], unit="s")

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total sessions", len(gdf))
    col2.metric("Unique players", gdf["player_name"].nunique() if "player_name" in gdf else "?")
    col3.metric("Datasets seen", gdf["network"].nunique() if "network" in gdf else "?")
    if "step5_player_matched" in gdf.columns:
        match_rate = gdf["step5_player_matched"].mean() * 100
        col4.metric("Human matched metrics", f"{match_rate:.0f}%")

    st.divider()

    # Method picks
    st.subheader("Which CF method was picked as best?")

    criteria = {
        "step3_best_method": "Step 3: Intuitive game pick",
        "step6_final_pick": "Step 6: Final pick (after seeing full metrics)",
    }

    cols = st.columns(len(criteria))
    for col, (key, label) in zip(cols, criteria.items()):
        with col:
            st.markdown(f"**{label}**")
            if key in gdf.columns:
                counts = gdf[key].value_counts()
                fig, ax = plt.subplots(figsize=(3, 2.5))
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")
                colors = ["#f59e0b", "#2563eb", "#16a34a", "#dc2626", "#7c3aed"]
                bars = ax.barh(counts.index, counts.values,
                               color=colors[:len(counts)])
                ax.set_xlabel("Picks", fontsize=8, color="#cbd5e1")
                ax.tick_params(labelsize=8, colors="#cbd5e1")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                for spine in ["bottom", "left"]:
                    ax.spines[spine].set_color("#374151")
                for bar, val in zip(bars, counts.values):
                    ax.text(val + 0.05, bar.get_y() + bar.get_height() / 2,
                            str(val), va="center", fontsize=8, color="#f8fafc")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

                if len(counts) > 0:
                    winner = counts.index[0]
                    st.markdown(
                        f'<div style="font-family:IBM Plex Mono;font-size:1rem;font-weight:600;'
                        f'color:#f59e0b;text-align:center;padding:0.5rem;border:2px solid #f59e0b;'
                        f'border-radius:8px;background:rgba(245,158,11,0.1);margin:0.5rem 0;">'
                        f'🏆 {winner}</div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No data yet")

    st.divider()

    # Objectively best vs human pick
    if "step5_actual_best_method" in gdf.columns and "step3_best_method" in gdf.columns:
        st.subheader("Human pick vs. Objectively best method")

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**How often did the human pick match the best metric method?**")
            if "step5_player_matched" in gdf.columns:
                matched = gdf["step5_player_matched"].sum()
                total = len(gdf)
                st.metric("Matched", f"{matched}/{total} ({matched/total*100:.0f}%)")
                fig, ax = plt.subplots(figsize=(3, 3))
                fig.patch.set_facecolor("#111827")
                ax.set_facecolor("#111827")
                ax.pie(
                    [matched, total - matched],
                    labels=["Matched", "Differed"],
                    colors=["#16a34a", "#dc2626"],
                    autopct="%1.0f%%",
                    textprops={"color": "#f8fafc", "fontsize": 10},
                )
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        with col_b:
            st.markdown("**Which method was objectively best most often?**")
            best_counts = gdf["step5_actual_best_method"].value_counts()
            st.dataframe(
                best_counts.reset_index().rename(columns={"index": "Method", "step5_actual_best_method": "Times Best"}),
                use_container_width=True,
                hide_index=True,
            )

        st.divider()

    # Confidence
    st.subheader("Confidence analysis")

    if "step3_confidence" in gdf.columns and "step7_post_confidence" in gdf.columns:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Step 3 confidence (before metrics revealed)**")
            fig, ax = plt.subplots(figsize=(4, 2.5))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#111827")
            ax.hist(gdf["step3_confidence"].dropna(), bins=10, color="#2563eb", alpha=0.8)
            ax.set_xlabel("Confidence (%)", fontsize=8, color="#cbd5e1")
            ax.set_ylabel("Count", fontsize=8, color="#cbd5e1")
            ax.tick_params(colors="#cbd5e1")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        with col_b:
            st.markdown("**Step 7 confidence (after metrics revealed)**")
            fig, ax = plt.subplots(figsize=(4, 2.5))
            fig.patch.set_facecolor("#111827")
            ax.set_facecolor("#111827")
            ax.hist(gdf["step7_post_confidence"].dropna(), bins=10, color="#f59e0b", alpha=0.8)
            ax.set_xlabel("Confidence (%)", fontsize=8, color="#cbd5e1")
            ax.set_ylabel("Count", fontsize=8, color="#cbd5e1")
            ax.tick_params(colors="#cbd5e1")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

    st.divider()

    # Qualitative responses
    st.subheader("Qualitative responses")

    qual_keys = [
        ("step1_why", "Step 1: Why did you think so?"),
        ("step3_why_pick", "Step 3: Why did you pick that method?"),
        ("step4_surprised_reaction", "Step 4: Were you surprised by the metrics?"),
        ("step5_why_chosen", "Step 5: Reflection on ranking"),
        ("step6_why_change", "Step 6: Would you change your pick and why?"),
        ("step7_final_thoughts", "Step 7: Final thoughts"),
    ]
    for key, label in qual_keys:
        if key in gdf.columns:
            non_empty = gdf[gdf[key].notna() & (gdf[key].astype(str).str.strip() != "")]
            if len(non_empty) > 0:
                with st.expander(f"{label} ({len(non_empty)} responses)"):
                    for _, row in non_empty.iterrows():
                        st.markdown(f"**{row.get('player_name', '?')}:** {row[key]}")
                        st.divider()

    st.divider()

    # Raw log
    with st.expander("Raw session log"):
        st.dataframe(gdf, use_container_width=True)

    csv = gdf.to_csv(index=False)
    st.download_button("Download as CSV", csv, "guided_task_results.csv", "text/csv")