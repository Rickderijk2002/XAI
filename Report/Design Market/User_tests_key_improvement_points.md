
## Key Improvement Points - User Tests

- **Validity control:** Replace or clarify the 0–1 slider for validity since validity is effectively binary; users found the slider misleading.
- **Metrics clarity:** Reduce overload in Step 4 and provide concise, clear definitions for `validity`, `plausibility`, and `implausibility` (include human vs. model distinctions); explain why metrics may disagree with intuition.
- **Method explanations:** Add brief, non-technical descriptions or info-tooltips for each generation method so users can meaningfully compare options.
- **Comparison & memory support:** Make it easier to compare selected CFs with the objectively best CF (e.g., persistent side-by-side view, highlights, or a replay), since users struggle to remember images between steps.
- **Process transparency:** Show intermediate transformation steps or a short visual pipeline so users understand how a CF was produced (a few key stages rather than a single jump).
- **Interaction/usability fixes:** Fix the `next` button activation (avoid requiring click-out or Ctrl+Enter), streamline text entry flows, and reduce excessive explanatory text on a single screen.
- **Metric input consistency:** Align how human estimates are requested across metrics (users provided estimates for validity/plausibility but not implausibility), and clarify which inputs are subjective vs. computed.
- **Visual hierarchy:** Simplify Step 4 layout to surface the most actionable information first (e.g., clear winners per metric, concise summary sentence, then detailed tables on demand).

These changes address the main sources of confusion reported in the interviews and should improve user understanding, comparison accuracy, and overall usability.

