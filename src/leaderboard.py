"""
leaderboard.py: Leaderboard for the mini game.
Reads minigame_log.json and displays all-time rankings.
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from components import page_title

MINIGAME_LOG_FILE = Path(__file__).parent / "minigame_log.json"


def render_leaderboard():
    page_title("🏆 Mini Game Leaderboard")

    if not MINIGAME_LOG_FILE.exists():
        st.info("No mini game sessions yet. Play the Mini Game to appear on the leaderboard!")
        return

    with open(MINIGAME_LOG_FILE) as f:
        log = json.load(f)

    if not log:
        st.info("No mini game sessions yet. Play the Mini Game to appear on the leaderboard!")
        return

    df = pd.DataFrame(log)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d %H:%M")
    df["accuracy"] = (df["score"] / df["rounds_played"] * 100).round(1).astype(str) + "%"

    # Rank by score descending, then rounds_played ascending (fewer rounds = harder path to same score)
    df = df.sort_values(["score", "rounds_played"], ascending=[False, True]).reset_index(drop=True)
    df.index += 1
    df.index.name = "Rank"

    # Top 3 callout
    st.markdown("### Top Players")
    medals = ["🥇", "🥈", "🥉"]
    top_cols = st.columns(min(3, len(df)))
    for i, col in enumerate(top_cols):
        if i < len(df):
            row = df.iloc[i]
            with col:
                st.markdown(f"### {medals[i]}")
                st.markdown(f"**{row['player_name']}**")
                st.metric("Score", int(row["score"]))
                st.caption(f"{row['rounds_played']} rounds · {row['accuracy']} accuracy")

    st.divider()

    # Full table
    st.markdown("### Full Rankings")
    display_df = df[["player_name", "score", "rounds_played", "accuracy", "strikes", "timestamp"]].copy()
    display_df.columns = ["Player", "Score", "Rounds", "Accuracy", "Strikes", "Date"]

    st.dataframe(display_df, use_container_width=True)

    st.divider()

    # Stats
    st.markdown("### Overall Stats")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total sessions", len(df))
    col2.metric("Unique players", df["player_name"].nunique())
    col3.metric("Highest score", int(df["score"].max()))
    col4.metric("Avg score", f"{df['score'].mean():.1f}")
