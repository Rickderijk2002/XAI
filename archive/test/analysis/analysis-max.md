## Max

Source transcript: `test_results/Transcript_Meeting_Max_XAI_EN.pdf` (English, ~24 min, May 9, 2026).

### A1. Session and protocol

- **Participant:** Max
- **Date (YYYY-MM-DD):** 2026-05-09
- **Facilitator:** Rick

### A2. Sample and stance

- **Relevant background:** Not spelled out in the excerpt beyond the study context; engages **pragmatically** with the dashboard and post-interview questions. Comfortable stating **visual-first** preferences.
- **Baseline attitude:** Cooperative, concise, **visually driven**; positive about **layout** early (“nice layout”).

### A3. Observations during the task (protocol Section 6)

- **Think-aloud quality:** **Moderate** (steady but often **prompted** by the facilitator; fewer long spontaneous monologues than some other sessions).
- **Key behaviours:**
  - Early **concept check** with facilitator on counterfactuals (model may “see” a digit where a human sees little).
  - Works through steps; comments on **model class labels** versus digits (“not a 5”).
  - In the **comparison step**, selects **Alibi Proto CF** because a **digit is clearly visible** to him, unlike other methods in his view.
  - On **metric display**, reads **green vs red** bands as above or below average and summarises overall as **borderline**.
  - After seeing the **dashboard’s “best”** output, distinguishes **“according to the model itself”** from his own pick.
  - In **reflection / summary UI**, asks for **shorter copy**, **metrics with explanations beside them**, or **hover** tooltips; reports **scrolling up** to read guidance then **forgetting** details when **scrolling back down**.
  - States decision-making is **purely visual** and that **metrics do not really interest** him for that purpose.
- **Key decision:** **Alibi Proto CF** as **best method** from his perspective; dashboard indicates the **counterfactual shown** was produced by **MinEdit** and reflects **model-side** “best” (his pick and model pick **diverge**).
- **Confusion moments:**

  1. **Metric blocks (IM1 / “M1” wording in transcript):** Interpreting **lower is better** and the on-screen prompt about whether the ranking “makes sense.”
  2. **Step 4 (“Action Matrix”)** felt **unclear** when trying to move quickly (participant confirms with facilitator).
  3. **Volume and layout:** Long **text** and **vertical navigation** make it hard to keep **context** between top guidance and lower content.

- **Assistance given:** Yes
- **If yes, step:** **Intro** through **final summary** (frequent facilitator explanations and checks), plus **post-interview** protocol questions.
- **If yes, what was said or done:** Explained **counterfactual purpose**, **validity and plausibility** at a high level, **guided flow** intent, **information blocks** at top, **five methods** layout, **green vs red** metric bands, which output is **“best”** on the dashboard, caution to **wait before Final** so input is not lost, **timeout** concept (model cannot always force a digit change), and **post-task Q1–Q4**.
- **If yes, why it was necessary:** Participant was willing to proceed but needed **navigation and metric literacy cues**; **think-aloud** protocol plus **dense UI** led to **ongoing facilitation**.

### A4. Protocol capture and research questions (Section 9 + project RQ)

- **Initial choice / expected best:** **Alibi Proto CF** (clear digit visible to him versus other methods).
- **Metric-best:** **MinEdit** associated with the **model-selected** best counterfactual in the flow Rick describes; participant accepts that as **“according to the model itself.”**
- **Choice changed after metrics:** **No** explicit change of preferred method; maintains **visual** rationale in **Q1**.
- **Alignment with metrics (their words):** **Low personal alignment.** A model can “produce those metrics,” but if he **still sees nothing meaningful** in the image he **does not assign value** to those metrics; he **“just see[s] a number.”**
- **RQ A (human judgment vs model):** **Partial.** In **Q3** he can articulate that metrics reflect **what the model sees** and may still be **internally consistent** even when the **front-end** looks wrong, yet he answers that metrics **did not help** him in practice (**“No”** to whether they helped). **During use**, judgment stayed **purely visual**.
- **RQ B (structured UI):** **Order is logical** (“fine” from step 1 to end). Friction from **text length**, **scrolling**, and **Step 4** clarity when moving fast; wants **inline or hover explanations** instead of distant blocks.
- **Q2 surprise:** Metrics **do not meet** his expectations as a **user** when images look empty or unconvincing, even if numbers exist.
- **Q3 where judgment vs metrics became clear:** **Little clarity during interaction** for **using** metrics; **post-interview** discussion clarifies the **conceptual** separation between **model scores** and **human appearance**, without converting him to metric-driven choices.
- **Q4 one change to the dashboard:** **Less scrolling**; **hover** (or adjacent) **short explanations** for metrics; **less text** overall.
- **Q4 follow-up why it helps:** Avoids losing thread when moving between **top guidance** and **lower panels**; makes **numbers interpretable** without reading long blocks.

### A5. Flow, design, evidence

- **Overall flow clarity:** **Layout praised**; **information architecture** criticised for **length** and **scroll distance** between explanation and task.
- **Emotional tone:** Matter-of-fact, not hostile; **direct critique** of information density.
- **Information design:**
  - Wants **shorter** summaries and **per-metric** micro-help.
  - **Top information bar** useful in principle but **scroll** breaks continuity for him.
- **Bugs or interaction issues noticed:** None as **software defects** in the transcript; **UX** issues around **scroll**, **text volume**, and **Step 4** labelling or density.

- **Best verified quote for this session:**

```
Yes, purely visual. The metrics do not really interest me.
```

- **Theme tags:** `positive_structure`, `trust_metrics`, `text_skimming`, `step_purpose_unclear`, `info_panel_help`, `recap_redundancy`

---
