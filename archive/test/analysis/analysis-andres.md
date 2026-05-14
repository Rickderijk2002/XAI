## Andres

Source notes: `test_results/insights_test_andres.md` (structured brief from session with facilitator Thom). Quote below is taken from that brief and should be checked against the full audio transcript if you tighten wording for the report.

### A1. Session and protocol

- **Participant:** Andres
- **Date (YYYY-MM-DD):** 2026-05-11
- **Facilitator:** Thom

### A2. Sample and stance

- **Relevant background:** Target audience profile (data science student context for the course study). Session brief does not state prior CF or XAI depth; participant inferred **CF** meaning during the task and asked how to measure **validity** early on.
- **Baseline attitude:** Cooperative; frank about confusion; think-aloud **moderate** (sparse in stretches, more facilitator-led in the second half).

### A3. Observations during the task (protocol Section 6)

- **Think-aloud quality:** Moderate (thin in places; more reliance on post-task probing in the brief).
- **Key behaviours:**
  - Early uncertainty about how to **measure validity** and what **CF** referred to until treated as counterfactual.
  - **Plausibility** rating described as hard to state precisely (“quite difficult to say exactly how precise it is”).
  - Asked **how to pick the best method** on visual criteria; noted **one method showed no output**; commented that **slider ticks** (e.g. 80, 85, 90) could make input easier.
- **Key decision:** Picked a **best** counterfactual or method during the game (exact method name **not stable** in the brief); **did not change** that choice after metrics (**“No”** in the brief’s outcome table).
- **Confusion moments:**

  1. **Early steps:** What **validity** means in practice; **CF** acronym unfamiliar at first.
  2. **Step 2 (estimates):** Difficulty giving a precise **plausibility** score on a continuous scale.
  3. **Game / comparison:** How to select the **best method** when judging mostly from images; **empty** method slot.
  4. **Metrics reveal:** “What do these metrics mean?” Confusion between **implausibility** and their own **plausibility** estimate (“in the other direction?”) and whether conclusions are **correct or not**.
  5. **Feedback copy:** Message that the pick **matches the objectively best method** while **method 5 appears invalid**; participant reported this as contradictory (“I don't know what this is for”).
  6. **Recap / compare:** Section felt **similar** to a previous screen; could not interpret how to compare earlier estimates (e.g. validity **0.81**) to what was shown or whether values were **good or bad**.

- **Assistance given:** Yes
- **If yes, step:** In-task help on **metrics meaning** (implausibility, IM2, etc., per brief); **substantial post-task debrief** (facilitator walked Q themes and design intent).
- **If yes, what was said or done:** Clarifications on metric labels; facilitator questions on decision process, metrics alignment, interface understanding, and desired changes; clarification that **mini game** was **not** part of the test.
- **If yes, why it was necessary:** Think-aloud was sparse; metrics and copy were ambiguous; core **human vs model** distinction did not land from the UI alone.

### A4. Protocol capture and research questions (Section 9 + project RQ)

- **Initial choice / expected best:** Best-method choice during the game (identifier **unclear** in brief); early judgments on validity and plausibility were **uncertain** numerically.
- **Metric-best:** Brief notes UI claimed **match to objectively best** while also showing **method 5 invalid**, which confused the participant.
- **Choice changed after metrics:** **No** (per brief outcome table).
- **Alignment with metrics (their words):** **Mixed.** Metrics “help a little bit more” when **quantified**, but **contradictions** and **direction** (implausibility vs own plausibility) stayed confusing.
- **RQ A (human judgment vs model):** **No (during task).** On whether the difference between **their judgment on the image** and **what the model predicts** was clear: **“Not to be honest. No. I didn't get that part.”** (Debrief may have added partial verbal understanding; not attributed to the prototype alone.)
- **RQ B (structured UI):** **Friction-heavy for learning.** Steps felt **clear in sequence** at a surface level, but **purpose**, **method identity**, and the **validity or plausibility tension** were not clear early; participant wanted a clearer **end goal** and **definitions** to compare methods fairly.
- **Q2 surprise:** Contradiction between **“matches best”** and **invalid** method; difficulty reading **implausibility** against own inputs.
- **Q3 where judgment vs metrics became clear:** **Not clear** during the guided flow without facilitator help (per brief).
- **Q4 one change to the dashboard:** Add **definitions** for **methods** and what differs between them; optional **short multi-step explanation** of how the model moves from input to prediction (“what is in between”); slightly more **expressive** framing of study **purpose** early (team trades off with spoiler risk).
- **Q4 follow-up why it helps:** Without method information they cannot justify **which method is better**; intermediate model steps would clarify **how** the output arises.

### A5. Flow, design, evidence

- **Overall flow clarity:** **“Very clear straightforward”** for layout and top notes, but **“a bit lost”** early on **where the flow was going** and **what the study wanted to learn**.
- **Emotional tone:** Neutral, cooperative, **confused** at metrics and copy.
- **Information design:**
  - Top instructions partly helpful.
  - Missing **method-level** explanation for comparative steps.
  - **Recap** felt redundant or hard to link to earlier numbers.
- **Bugs or interaction issues noticed:**
  - **Missing output** for one method in the lineup.
  - **Contradictory** messaging (best match vs invalid).
  - Slider **granularity** comment as a minor usability wish.

- **Best verified quote for this session:**

```
Not to be honest. No. I didn't get that part.
```

- **Theme tags:** `metric_literacy`, `contradictory_copy`, `method_identity`, `implausibility_mismatch`, `recap_redundancy`, `positive_structure`

---
