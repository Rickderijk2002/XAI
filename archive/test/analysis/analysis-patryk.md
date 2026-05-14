## Patryk

### A1. Session and protocol

- **Participant:** Patryk
- **Date (YYYY-MM-DD):** 2026-05-11
- **Facilitator:** Stefan

### A2. Sample and stance

- **Relevant background:** Strong engagement with ML and XAI concepts during the session (counterfactual goals, realism trade-off, IM1 direction). Reads onboarding and metric explanations carefully.
- **Baseline attitude:** Curious, cooperative, frank about cognitive load.

### A3. Observations during the task (protocol Section 6)

- **Think-aloud quality:** Rich
- **Key behaviours:**
  - Read the long intro page in detail; summarised steps aloud and judged the intro clear after reading.
  - In Step 3, chose **Min-Edit** as best among five methods while stating all options were weak; justified with specific pixel detail in free text.
  - Revisited judgments in Step 7: raised confidence uncertainty (50%) and said they would change the Step 1 answer after seeing metrics, mainly due to understanding what binary validity implies.
- **Key decision:** **Min-Edit** selected in **Step 3** (game) as best explanation for the 9-to-6 case; **Step 6** explicit question on changing that pick: **No** (alibi-proto still looked like a 9 to them).
- **Confusion moments:**

  1. **Step 1:** Unclear whether “counterfactual has been modified” referred to the original counterfactual image from the dataset; resolved after facilitator clarification.
  2. **Step 2:** Uncertainty how precisely to set the **validity slider** between 0 and 1 (“how much I have to enter exactly”).
  3. **Step 3:** **Next** did not activate after typing; facilitator explained clicking outside the text box (participant agreed it was not ideal).
  4. **Step 4:** **Validity is binary** but a slider was used earlier; mapping between **plausibility human estimate** and displayed plausibility/implausibility felt “weird”; unsure whether first two metrics were their earlier inputs for “this specific” counterfactual.
  5. **Step 5–6:** Difficulty **remembering** how **Min-Edit** and **metric-best alibi-proto** images differed when interpreting rankings; wanted an easier side-by-side comparison to the objectively best method.

- **Assistance given:** Yes
- **If yes, step:** Step 1 (wording), Step 3 (Next affordance), Step 4 (which inputs map to which metrics and which image), Step 5 (IM1 lower is better; which generator), Step 6 (clarify which image was original PIECE vs user pick vs objective best)
- **If yes, what was said or done:** Short clarifications on image identity, clicking outside text field to enable Next, confirming human estimates vs model metrics for the same CF, IM1 direction, and which method was objectively best.
- **If yes, why it was necessary:** Copy and interaction affordances were ambiguous; metric screen dense; comparison across steps relied on memory.

### A4. Protocol capture and research questions (Section 9 + project RQ)

- **Initial choice / expected best:** Visually **not successful** for target 6 in Step 1; Step 2 validity ~0.3 and very low plausibility; Step 3 **Min-Edit** as best method.
- **Metric-best:** **alibi-proto-cf** described as objectively best overall; **PIECE** as original single-CF context in earlier steps.
- **Choice changed after metrics:** **No** for the **Step 3 / Step 6** “would you change your chosen best method” question; **Yes** for **Step 1** judgment in Step 7 reflection (would revise Step 1 with better understanding, especially binary validity).
- **Alignment with metrics (their words):** Mixed / surprise. Final thought: metrics did **not** align with intuition overall; the **difference** was what surprised them most (“No, this difference is what surprised me most.”).
- **RQ A (human judgment vs model):** **Yes.** They kept a **human appearance** criterion (e.g. alibi-proto “still looks like a 9”) even when metrics favoured it; articulated tension between subjective appearance and objective rankings.
- **RQ B (structured UI):** **Both.** The **sequence** (intuition first, metrics later, then comparisons) supported the study’s tension narrative, but **information density** in Step 4 and **memory load** when relating picks to “best” methods undermined clarity without facilitator support.
- **Q2 surprise:** Surprised by **objective ratings** and unclear at first glance how metrics connect to their subjective ratings; noted conflict between objective and subjective in reflection text.
- **Q3 where judgment vs metrics became clear:** **Step 4** first surfaced sustained metric confusion; **Step 6** table helped somewhat with “per-metric winner” and overall best, though human vs metric conflict remained for their preferred pick.
- **Q4 one change to the dashboard:** Reduce friction after text entry (**Ctrl+Enter / click outside** to proceed); improve ability to **compare their chosen CF to the objective best** without relying on memory (side-by-side or clearer recall aids).
- **Q4 follow-up why it helps:** Stated extra steps are annoying but doable; easier flow would make comparison and alignment checks less effortful (closing comments and Step 5 commentary).

### A5. Flow, design, evidence

- **Overall flow clarity:** Intro “kind of a lot of text” but ultimately “clear”; later metric sections felt heavy and less intuitive at first glance.
- **Emotional tone:** Neutral, analytical, mild frustration at **Next** and text-submit friction.
- **Information design:**
  - Long onboarding but participant worked through it successfully.
  - Step 4 metric reminder “quite a lot of information”.
  - Info boxes helped when read, but volume competed with task pace.
- **Bugs or interaction issues noticed:**
  - Next button inactive until clicking outside text field (Step 3; also mentioned again at end).
  - Nearly missed **final thoughts** box before finishing (facilitator pointed it out).

- **Best verified quote for this session:**

```
Okay but this... validity is binary. Why then did I get a slider?
```

- **Theme tags:** `validity_slider_binary`, `which_CF_tracking`, `metric_literacy`, `next_button_affordance`, `onboarding_load`, `positive_structure`

---
