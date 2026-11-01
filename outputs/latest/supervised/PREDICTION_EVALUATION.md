# Supervised and semi-supervised learning: measured results

Four-score expert seed; expert labels and model-generated estimates are distinct.

## Evaluation protocol
leave-one-participant-out, all preprocessing Ridge fit within each fold; transductive graph sees unlabelled features but not withheld labels

Eligible independent labelled participants: **4**
Unlabelled participants with predictions: **180**
Participants requiring additional paired measurements: **3**

## Actual held-out experiment (LOPO)

| Target | Ridge MAE | Baseline MAE | Graph MAE | Self-training MAE |
|---|---:|---:|---:|---:|
| Floating outcome 0–3 | 0.767 | 1.333 | 1.307 | 0.575 |
| Overall score 0–13 | 2.408 | 4.333 | 4.254 | 1.979 |

## Prediction and review evidence

- `private/all_participant_predictions.csv`: one row per participant, expert LOPO predictions for rated cases and provisional scores for unrated cases.
- `private/lopo_train_test_predictions.csv`: the actual withheld participant and training IDs for every fold.
- `private/participant_review_queue.csv`: 30 unlabelled cases prioritised for human assessment.
- `supervised/semisupervised_metrics.json`: model metrics, experiment settings and sample counts.
- `supervised/trained_expert_and_semi_models.joblib`: fitted numerical models.

## Interpretation

Pseudo-labels are estimates produced using four verified expert examples; they are not ground truth. 
Graph diffusion uses the unlabelled feature matrix transductively while withholding the tested outcome in each evaluation fold. 
The self-training branch uses fixed low-weight pseudo scores and excludes held-out participants from pseudo training.
The 0–3 outcome is treated as an ordered numerical value; rounded classes are proposed labels for review only.
