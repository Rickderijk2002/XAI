# XAI Counterfactual Explanations: Guided Task Mode

An interactive Streamlit application for running a structured human evaluation study on image-based counterfactual (CF) explanations. Built for the course *JM0110 — Interactive & Explainable AI Design at JADS*.

## Research Question

> To what extent do objective evaluation metrics (validity, plausibility) align with users' intuitive judgments of counterfactual explanation quality?

---

## Project Structure

Current repository layout:

```text
XAI/
├── README.md
├── requirements.txt
├── src/
│   ├── app.py
│   ├── components.py
│   ├── data_utils.py
│   ├── intro.py
│   ├── task.py
│   ├── minigame.py
│   ├── leaderboard.py
│   ├── results.py
│   ├── guided_task_results.json
│   └── minigame_log.json
├── Dashboard/
│   ├── app.py
│   ├── data_utils.py
│   └── game_log.json
├── Game Results/
│   ├── guided_task_results.json
│   └── minigame_log.json
└── Report/
    ├── Report XAI.qmd
    ├── DliA_Final_Report.qmd
    └── My_Library.bib
```

### `src/` module overview

| File | Description |
| --- | --- |
| `src/app.py` | Main Streamlit entry point. Handles sidebar navigation and routes to Guided Task, Mini Game, Leaderboard, and Guided Task Results pages. |
| `src/components.py` | Shared UI/style utilities (CSS injection, metric explainer, image rendering, formatting helpers) used across pages. |
| `src/data_utils.py` | Data layer: CSV loading, MNIST/CIFAR tensor loading, metric row lookup, and helper functions for summary/grid generation. |
| `src/intro.py` | Intro/onboarding screen for the Guided Task flow (study explanation + participant name input). |
| `src/task.py` | Full 7-step guided study workflow, session state handling, case sampling, and persistence to `guided_task_results.json`. |
| `src/minigame.py` | Standalone MNIST mini game with score/strikes logic and persistence to `minigame_log.json`. |
| `src/leaderboard.py` | Leaderboard page for mini game sessions stored in `minigame_log.json`. |
| `src/results.py` | Aggregated analysis page for guided task sessions stored in `guided_task_results.json`. |
| `src/guided_task_results.json` | Auto-generated storage file for completed guided task sessions. |
| `src/minigame_log.json` | Auto-generated storage file for mini game sessions. |

### Notes on other folders

- `Dashboard/` contains an earlier monolithic app version kept in the repo.
- `Game Results/` contains exported/copied result logs.
- `Report/` contains Quarto report sources and bibliography.

---

## Data Setup

The data is **not included** in this repository due to confidentiality (paper under review). You need to set it up manually.

Create a `Data/` folder in the project root (not in `Dashboard`, `src`, `Game Results`, `Report`, or any other existing folder) with the following structure:

```
Data/
├── evaluation_results.csv
├── mnist_output/
│   ├── original/
│   │   └── instance_{id}.pt
│   ├── PIECE/
│   ├── Min-Edit/
│   ├── C-Min-Edit/
│   ├── alibi-Proto-CF/
│   └── alibi-CF/
└── cifar_resnet8_output/
    └── cifar_resnet8_output/   ← note: nested folder
        ├── original/
        ├── PIECE/
        ├── Min-Edit/
        ├── C-Min-Edit/
        ├── alibi-Proto-CF/
        └── alibi-CF/
```

---

## Installation

```bash
pip install streamlit torch numpy pandas matplotlib
```

---

## Running the App

```bash
streamlit run src/app.py
```

---

## Application Structure

The app has two pages accessible via the sidebar:

### 1. Guided Task Mode

The core study interface. Each participant goes through **7 linear steps** for one randomly sampled case (one instance with all 5 CF methods). Steps cannot be skipped. After completing step 7, the session is saved and the app resets for the next participant.

At the start of each session the participant enters their name or participant ID. A case is then randomly sampled from either MNIST or CIFAR-10.

The 7 steps are:

**Step 1: Visual Inspection**
The participant sees the original image and one randomly chosen CF method. No metrics are shown. They judge whether the CF looks successful and optionally explain why.

**Step 2: Your Prediction**
The same image pair is shown again. The participant estimates the validity (0 to 1) and plausibility (0 to 1) of the CF using sliders.

**Step 3: Game**
All 5 CF methods are shown side by side in a lineup (Original, PIECE, Min-Edit, C-Min-Edit, Alibi-Proto-CF, Alibi-CF). The participant picks which method produced the best CF and rates their confidence (0–100%).

**Step 4: Actual Metrics Revealed**
The real evaluation metrics for the CF shown in steps 1 and 2 are revealed: validity (correctness), IM1, and implausibility. The participant can compare these to their estimates from step 2.

**Step 5: Explanation and Feedback**
All 5 methods are ranked by validity then plausibility. The participant sees which method was objectively best and is asked to explain what drove their choice in step 3.

**Step 6: Compare Methods on This Case**
The full image grid of all methods is shown alongside a metrics table (validity, IM1, implausibility, L2 distance per method). The participant then selects the best method according to three separate criteria: best overall (valid and plausible), most plausible, and most valid.

**Step 7: Confidence and Final Thoughts**
The participant reflects on their initial judgment from step 1, rates how confident they are in it now that they have seen all the metrics, states whether they would change their answer, and optionally adds any final thoughts.

---

### 2. Game Results

Always accessible from the sidebar. Shows aggregate outcomes across all completed sessions. Refreshes live as new sessions are saved.

Displays:

- Total sessions completed and unique players
- Which CF method was picked most often as best, broken down by criterion (game pick in step 3, best overall in step 6, most plausible in step 6, most valid in step 6) with bar charts and a trophy winner per criterion
- Average confidence per method pick
- Distribution of validity and plausibility estimates from step 2
- Qualitative responses collected across sessions (step 1 reasoning, step 5 explanation, step 7 final thoughts)
- Full raw session log with CSV export

---

## Saved Data

The `Game Results/` folder currently contains two JSON logs:

- `Game Results/guided_task_results.json` (one entry per completed Guided Task session)
- `Game Results/minigame_log.json` (one entry per completed Mini Game session)

### Guided Task JSON (`guided_task_results.json`)

Each entry contains:

| Field | Source |
| ----- | ------ |
| `player_name` | Name entry screen |
| `timestamp` | Auto-generated |
| `network`, `instance_id`, `target`, `original_label` | Sampled case info |
| `shown_method_steps_1_2` | Which method was shown in steps 1 and 2 |
| `step1_judgment` | Yes / No / Uncertain |
| `step1_why` | Free text explanation |
| `step2_validity_estimate` | Slider value (0 to 1) |
| `step2_plausibility_estimate` | Slider value (0 to 1) |
| `step3_best_method` | Game pick (method name) |
| `step3_confidence` | Confidence percentage |
| `step3_why_pick` | Required explanation of game pick |
| `step4_actual_correctness` | Actual validity from CSV |
| `step4_actual_IM1` | Actual IM1 from CSV |
| `step4_actual_implausibility` | Actual implausibility from CSV |
| `step4_surprised_reaction` | Required reaction to revealed metrics |
| `step5_why_chosen` | Free text explanation |
| `step5_actual_best_method` | Objectively best method (by metrics) |
| `step5_player_matched` | Whether step-3 pick matched objective best |
| `step6_would_change_pick` | Whether participant would change their pick |
| `step6_final_pick` | Final pick after seeing full table |
| `step6_why_change` | Required explanation of step-6 choice |
| `step6_per_metric_best_valid` | Methods that were valid on this case |
| `step6_per_metric_best_im1` | Best method on IM1 |
| `step6_per_metric_best_implaus` | Best method on implausibility |
| `step6_per_metric_best_l2` | Best method on L2 distance |
| `step7_post_confidence` | Post-reveal confidence percentage |
| `step7_change_answer` | Yes or No |
| `step7_final_thoughts` | Free text |

Incomplete sessions (where the participant did not reach step 7) are not saved.

### Mini Game JSON (`minigame_log.json`)

Each mini game entry contains:

| Field | Source |
| ----- | ------ |
| `player_name` | Name entry screen |
| `timestamp` | Auto-generated |
| `score` | Number of correct answers |
| `strikes` | Number of strikes at game end |
| `rounds_played` | Total rounds in that game |
| `history` | Per-round details (`instance_id`, `target`, `method`, `correctness`, `IM1`, `implausibility`, `guess`, `correct`, `round`) |

---

## Notes for Researchers

- The sidebar shows a reset button to manually clear a session mid-way if needed.
- The randomly sampled case is consistent across all 7 steps within one session.
- The method shown in steps 1 and 2 is chosen randomly from whichever methods have a non-blank (non-timeout) CF image available for that case.
- `Game Results/guided_task_results.json` and `Game Results/minigame_log.json` are appended to after completed sessions and are safe to back up or copy at any time.
- To export all results as CSV, use the Download button on the Game Results page.
