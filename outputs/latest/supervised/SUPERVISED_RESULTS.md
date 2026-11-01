# Expert-anchored supervised pilot — AI Flotation Analyser

## Expert annotations

Chris Button's scoring workbook contains four participant-level ratings made over the 30–60 s segment. The outcome scale is 0–3, and the five-category overall score ranges from 0 to 13. The labels are linked to top and side/front 30 s feature windows, not assigned to every window or video of a participant.

## Experimental design

The pilot uses **4 aligned participants**, **three preselected measurements** (overhead arm movement, side/front leg movement, side/front image-plane trunk angle), and **leave-one-participant-out evaluation**. Each fold fits missing-value imputation, standardisation and regularised Ridge regression on the other participants. The comparison is the mean of the training participants' scores. A final model is then fit to the full labelled pilot cohort, independently of its held-out evaluation.

| Target | Leave-one-participant-out MAE | Mean-predictor MAE | Independent predictions |
|---|---:|---:|---:|
| Floating outcome, 0–3 | 0.767 | 1.333 | 4 |
| Overall score, 0–13 | 2.408 | 4.333 | 4 |

Matched recording pairs: **4** of **4** expert-labelled participants. A rated participant whose specific trial is not independently documented is linked to the available paired recording by deterministic filename matching; this is a pilot alignment assumption, not confirmed session ground truth.

This experiment demonstrates functioning expert-supervised training and participant-independent evaluation on a four-person pilot. Its four held-out predictions do not establish reliable performance for unseen flotation participants or a deployable safety/quality classifier. The 0–3 scale is modelled numerically (regression), not as a validated four-class classifier.

## Reproduction

`private/expert_interval_alignment.csv` records the selected trials and usable features; `private/expert_heldout_predictions.csv` records held-out predictions, and `supervised/supervised_metrics.json` contains aggregate evaluation results. Expert input labels and predictions remain local and out of Git.
