## Esmee

Source transcript: `Gilbert/transcript_clean_Esmee.md` (cleaned interview with inferred speaker labels).

### A1. Session and protocol

- **Participant:** Esmee
- **Date (YYYY-MM-DD):** 2026-05-12
- **Facilitator:** Gilbert

### A2. Sample and stance

- **Relevant background:** States she is **not familiar with the CF methods** but accepts that as fine; asks early what **metrics** mean and what **unnaturalness** means (clarified by interviewer). Engages carefully with on-screen text.
- **Baseline attitude:** Curious, careful, reflective; open to revising her own understanding after metrics.

### A3. Observations during the task (protocol Section 6)

- **Think-aloud quality:** Rich (continuous commentary through guided steps; detailed reflection and mini-game reasoning).
- **Key behaviours:**
  - **Step 1:** First thought success, then corrected: target is **3** but image still reads as **7** to her; typed rationale about edges and missing cues for a three; interviewer nudges to consider **model** recognition.
  - **Step 2:** Struggled with **0 vs 1** validity slider until interviewer explains she must **infer** model direction toward target; settles on a judgment that the model is **fooled** in the sense used in the task.
  - **Step 3:** Picks **Min-Edit** at **~70%** confidence; says another method looks **almost the same**; choice partly because Min-Edit “came first” in the list among near-ties.
  - **Step 4:** Reads metric explanations closely; finds **plausibility** hard to compare because it is **not on the same 0–1 scale** as her estimate, but **agrees** with the **worded** interpretation; accepts **implausibility** and **distance** narratives.
  - **Step 5–6:** **Surprised** by ranking detail when two methods **look identical** to her; questions **C-Min-Edit** labelling vs **Min-Edit** pick; would **still pick the same method** after full metrics.
  - **Post-guided reflection:** Says she would have set **validity lower** once she understood validity as **model predicts target** not only “looks like target”; credits **Step 4 explanations** as most helpful.
  - **Mini game (same session, after guided task):** Initial **human-intuition-first** guesses fail; shifts toward **metric-informed** reasoning; asks for **pixel or distance** feedback in the game and for **why** the model says valid/invalid.
- **Key decision:** **Min-Edit** as best method in **Step 3**; maintains that choice after **Step 6** comparison; revises **understanding of validity**, not the chosen method.
- **Confusion moments:**

  1. **Intro:** What **metrics** are; meaning of **unnaturalness** (resolved by interviewer).
  2. **Step 2:** “How do I know what the model thinks this is?” and how that maps to the **slider** (resolved by interviewer framing “assume model steps toward three”).
  3. **Step 3:** Whether a panel should look like **original** vs **target** (interviewer points to model names on top).
  4. **Step 4:** **Plausibility** score not on **same scale** as her **0–1** estimate; relies on **text explanation** more than raw number.
  5. **Step 5:** Why one ranked **third** vs **second** when images look **the same**; naming confusion **C-Min-Edit** vs **Min-Edit**.
  6. **Mini game:** Why **validity** is sometimes **yes** when human intuition says **no**; difference between **validity** and **plausibility** in the game copy (participant flags this as a good recording for the team).

- **Assistance given:** Yes
- **If yes, step:** **Intro** (unnaturalness); **Step 2** (model intent and slider); **Step 3** (layout / models on top); **Reflection** (whether validity should be front-loaded); **Mini game** (pixels and density; dataset realism; repeated coaching to try again).
- **If yes, what was said or done:** Short definitions, “assume model toward target,” hints to compare to original digit, explanations of pixel density vs human shape, suggestion to finish round and retry.
- **If yes, why it was necessary:** **Validity as model outcome** was not obvious **before** Step 2; participant repeatedly separated **human look** from **model score** and needed language to bridge that gap.

### A4. Protocol capture and research questions (Section 9 + project RQ)

- **Initial choice / expected best:** **Step 1:** CF **not successful** as a human-readable **3**; **Step 3:** **Min-Edit** as best among methods (~70% confidence).
- **Metric-best:** Participant notes **Min-Edit** aligns with her pick and list position; **actual validity** (model treats as **3**) was the **main surprise** relative to her prior belief about what the model “sees.”
- **Choice changed after metrics:** **Method choice:** **No** (still Min-Edit after Step 6). **Understanding / numeric validity estimate:** **Yes** (reflection: would lower validity estimate after understanding definition; also answers that seeing metrics **did change** her answer in the sense of **interpretation**).
- **Alignment with metrics (their words):** **Step 4** narrative metrics **matched** her reasoning when explained; **actual validity** outcome **surprised** her most (model **did** predict target though image still looks like **7** to her).
- **RQ A (human judgment vs model):** **Yes, by end of session.** She articulates that **human intuition** (looks like **7**) diverged from **model validity** (predicts **3**), and that this was the **core surprise** once metrics were understood.
- **RQ B (structured UI):** **Mostly helpful.** **Step 4** explanations were “the most helpful”; **Step 6** did less to change how she **sees images** than Step 4 did to explain **numbers**. She also states the **mini game** was the moment she started weighing **metrics** explicitly after failures, which goes **beyond** the core guided protocol but shows how she **learned** the distinction.
- **Q2 surprise:** **Actual validity** (model says **3**) vs her expectation that the model would **not** see a three; other metrics with explanations were **not surprising** once read.
- **Q3 where judgment vs metrics became clear:** **Hard during early steps**; clearest shift when **mini game** answers were **wrong** twice, then she “started thinking” from a **metric perspective**; also credits **Step 4** text for clarifying **validity** meaning.
- **Q4 one change to the dashboard:** Stronger **early definition of validity** as **model prediction of target class** even if not visually convincing; richer **metric explanations** (why this image gets this score); **mini game feedback** explaining **why** an answer is correct; optional **spatial highlights** (“hotspots”) showing what drives the model.
- **Q4 follow-up why it helps:** Would reduce **0/1 slider confusion** before metrics; connect **pixel change** and **realism** to scores; make **model logic** visible when it **conflicts** with human appearance.

### A5. Flow, design, evidence

- **Overall flow clarity:** Finds **Step 4** and its **explanations** clear; early **validity** meaning was the main gap; **long session** including mini game and leaderboard.
- **Emotional tone:** Curious, reflective, occasionally **surprised** or **stuck**, ends **positive** (“Interesting,” “nice”).
- **Information design:**
  - **Plain-language metric blurbs** highly valued in **Step 4**.
  - **Onboarding** readable but dense; **validity** should be defined **earlier** (participant agrees when interviewer suggests).
  - **Mini game** copy should separate **validity** vs **plausibility** more clearly.
- **Bugs or interaction issues noticed:**
  - **Missed a button** in Step 1 (interviewer: “You missed the button”).
  - Wishes **mini game** showed **distance or pixel-change** cues after guesses.

- **Best verified quote for this session:**

```
But does the model think this is a three? How do I know what the model thinks this is?
```

- **Theme tags:** `onboarding_load`, `step_purpose_unclear`, `metric_literacy`, `validity_slider_binary`, `info_panel_help`, `positive_structure`

---
