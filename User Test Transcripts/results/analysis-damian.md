## Tim

Source transcript: `test_results/transcript_formatted.txt` (formatted excerpt; ends after two closing questions).

### A1. Session and protocol

- **Participant:** Damian
- **Date (YYYY-MM-DD):** 2026-05-11
- **Facilitator:** Tycho

### A2. Sample and stance

- **Relevant background:** Knows **MNIST** by name and quickly grasps that digits will be shown **modified** toward another class. Open to correcting his mental model when the researcher reframes **“fooled”** as **model prediction of the target**.
- **Baseline attitude:** Analytical, constructive; flags **copy logic** issues politely and suggests **concrete UI** improvements.

### A3. Observations during the task (protocol Section 6)

- **Think-aloud quality:** **Rich** (clear verbal reasoning on fooling, metric directions, and ranking trade-offs).
- **Key behaviours:**
  - **Step 2:** Initially treats “fooled” as **human appearance** (“genuinely not a 7”); after clarification, reframes to **whether the model predicts 7** and sets a **low** score (~**0.15**).
  - **Step 3:** Notes **one of five methods produced no output** (“failed completely”); narrows choice to **two** candidates that look more like a **7**; **~70%** confidence; comments on a **weird gap** on the weaker candidate.
  - **Step 4:** Questions **implausibility** vs **plausibility** wording when both seem to say **“higher is bad”**; flags **IM1** ambiguity (**realistic digit** vs **target class 7**); initially unsure how **0.2** relates to the **mean**, then accepts explanation once the **0.0–0.9** context is pointed out.
  - **Step 5:** Wants **all counterfactual images shown again next to metrics** to see **why** validity differs; links **pixel change** to “**fewest pixels wins**” versus a method that “**adjusts the entire background**.”
- **Key decision:** Chooses among the **two viable seven-like** explanations after discarding the **failed** method; **~70%** confidence. Researcher notes the **session ranking** positions the **selected best** as **method #3** with **validity then plausibility** ordering (Tim reacts with interest).
- **Confusion moments:**

  1. **Step 2:** Interpreting **“fooled”** before the researcher clarifies it is about the **model’s** target prediction, not only human resemblance.
  2. **Step 4:** **Plausibility vs implausibility** copy felt **internally inconsistent** (“for both, higher is bad?”).
  3. **Step 4:** **IM1** prompt unclear whether “realistic” means **any digit** or **the target digit** specifically; distance-from-mean not obvious until the **scale** is surfaced.
  4. **Step 5:** **Images disappear** relative to ranking; hard to connect **valid vs invalid** to concrete visuals from memory alone.

- **Assistance given:** Yes
- **If yes, step:** **Step 2** (definition of fooled / model database framing); **Step 4** (acknowledgement of unclear explanation; **IM1** comparison scope; **min–max** context for scores).
- **If yes, what was said or done:** Researcher restates task in **model-centric** terms; agrees metric **wording** was unclear; clarifies **what IM1 should measure**; points to displayed **range** for interpreting **0.2** vs **mean**.
- **If yes, why it was necessary:** **Abstract metrics** and **labels** did not fully disambiguate **target vs realism** without dialogue; **spatial memory** gap once rankings replace images.

### A4. Protocol capture and research questions (Section 9 + project RQ)

- **Initial choice / expected best:** **Not visually a successful 7** at first glance; after reframing, **low** estimated **model fooled** (~0.15); **Step 3** choice among **two** visible seven-like methods with **70%** confidence.
- **Metric-best:** Researcher describes **dashboard ranking** with **#3** as selected best and ordering **validity then plausibility**; Tim engages but transcript does not restate a single method **name** string for “metric-best.”
- **Choice changed after metrics:** **Not stated** in the excerpt (use **not stated** or treat as **no** if you only count explicit change; here: **not stated**).
- **Alignment with metrics (their words):** **Mixed.** Confusion then **partial resolution** once **scales** and **definitions** are clarified; still wants **images paired with metrics** for full alignment.
- **RQ A (human judgment vs model):** **Yes.** He separates **human “not a 7”** from **model might still be pushed toward 7** after Step 2 coaching, and continues to reason about **validity vs visuals** in Step 5.
- **RQ B (structured UI):** **Helpful when combined with visuals.** States the **difference between his judgment and metrics clicked** when metrics were **explained with images alongside**; ranking-only view **hurts** understanding without **side-by-side** CFs.
- **Q2 surprise:** Counterfactuals **did not match** his prior expectations for what a **target 7** should look like across methods; metric **wording** surprise on plausibility vs implausibility.
- **Q3 where judgment vs metrics became clear:** **When metrics were explained**, **especially with images alongside** (“it started clicking”).
- **Q4 one change to the dashboard:** Make **implausibility** (and related) copy **explicit** whether the judgment is **“looks like any digit”** vs **“resembles the target digit specifically”**; avoid **vague** phrasing.
- **Q4 follow-up why it helps:** Removes the **IM1 / implausibility** ambiguity he hit in Step 4 and supports **correct mental model** for **target-conditioned** plausibility.

### A5. Flow, design, evidence

- **Overall flow clarity:** **Guided** structure works; **metric section** needs **clearer definitions** and **tighter coupling** to **images**.
- **Emotional tone:** Curious, **constructive**, not dismissive.
- **Information design:**
  - **Positive:** Range display helped interpret **0.2** vs mean once noticed.
  - **Negative:** **Contradictory-sounding** plausibility copy; **ranking without images**.
- **Bugs or interaction issues noticed:** **Missing CF output** for one method in the five-way view; **information** issue on **plausibility vs implausibility** text (participant plus researcher treat as unclear).

- **Best verified quote for this session:**

```
Implausibility — higher is bad. But wait, it also says "high plausibility is bad" — that doesn't add up.
```

- **Theme tags:** `metric_literacy`, `implausibility_mismatch`, `info_panel_help`, `which_CF_tracking`, `positive_structure`, `step_purpose_unclear`

---
