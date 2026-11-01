# Final COSC2667 report / demonstration checklist

This guide reflects the supplied **Guide for Final Report**, not a claim that course-specific administrative requirements are already satisfied. The source Word report remains a team-review draft until approved.

## Deliverables that can now be generated directly

- Tested and documented post-MediaPipe feature engine.
- Inventory success/missing-file manifest and camera-specific coverage.
- Compact temporal feature data and clear camera/limb availability measures.
- Participant-disjoint K-means / Gaussian-mixture experiment and held-out evaluation.
- Model-selection tables, stability diagnostics, diagrams and figures.
- Offline aggregate dashboard plus `REPORT_RESULTS.md` for insertion into final report.
- Reproduction commands and GitHub-ready source structure.

## Report chapters

1. **Cover:** project, team, academic and industry supervisors, date. Verify facts with team.
2. **Introduction / scope:** initial vs current aims, partner problems and deliverables. Focus on quantitative flotation behaviours, not unvalidated safety scoring.
3. **Methodology:** input video inventory, MediaPipe 33-landmark provenance, feature definitions and QC, processing environment, train/test groups, model selection, comparison baseline, limitations.
4. **Results and evaluation:** copy figures and numerical values from `outputs/latest/REPORT_RESULTS.md` and `ml/ML_EVALUATION.md` **after running the pipeline**. Cite actual experiment results; do not recycle the earlier full-cohort silhouette as held-out accuracy.
5. **Discussion:** camera-specific measurement gaps, observed pose identity swaps, supervised-label limits, construct validity and cost of image-space assumptions.
6. **Conclusion:** working measurement/analysis prototype; identified exploratory feature groups and an independent four-person supervised expert-score pilot with LOPO evaluation; do not extrapolate the pilot MAE to population validity.
7. **Appendix:** repository access, project-management evidence, original project proposal/scope decisions, reproducibility, each member's roles/responsibilities and own reflection.

## Team-only evidence (must not be invented)

The final-report guide explicitly requires **detailed, exhaustive Roles & Responsibilities**, how collaboration worked, tools/frequency/time spent, and **individualized self-reflections**. Confirm names, author contributions, meeting records, failed experiments, project-management tools and agreed final wording with the respective team members.

Add actual references for existing scientific research and software documentation, verify partner consent for screenshots/data, follow formatting and page rules from your official rubric, and make the required GenAI disclosure accurately.

## Client demo outline

- Explain problem and two-camera approach (1 min).
- Show 33-landmark outputs / QC and why weak joints are withheld (2 min).
- Show corpus processing 383 of 394, camera-specific coverage, missing input accounting (2 min).
- Show independent-participant train/test experiment, train-selected groups and held-out scores (3 min).
- Demonstrate offline dashboard and answer what groups can/cannot mean (2 min).
- Summarize partner handover, reproducibility, validation roadmap and questions (1 min).

**Never demonstrate the pipeline as a drowning-risk or airway-safety classifier.**
