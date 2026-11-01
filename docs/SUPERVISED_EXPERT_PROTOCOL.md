# Supervised expert assessment protocol

## Expert score provenance

Chris Button's workbook `Four sample floating quality analyses.xlsx`, `Sheet1`, columns B–E, includes one rating for each of P2, P81, P68 and P118. The analysed interval is **30–60 seconds**. The exact score components are:

| Component | Range | P2 | P81 | P68 | P118 |
|---|---:|---:|---:|---:|---:|
| Floating outcome | 0–3 | 3 | 0 | 2 | 1 |
| Activity spectrum | 0–2 | 2 | 0 | 1 | 0 |
| Trunk position | 0–2 | 2 | 0 | 2 | 0 |
| Head position | 0–2 | 1 | 1 | 2 | 2 |
| Airway position | 0–4 | 4 | 3 | 4 | 3 |
| Overall score | 0–13 | **12** | **4** | **11** | **6** |

The overall score is the sum of the five components. `private_inputs/chris_expert_labels.csv` was transcribed programmatically from the supplied spreadsheet, checked against component sums and kept outside Git.

## Scientific task and data joining

Two target variables are modeled as *numeric regression outcomes*. We do not convert their four different values into a four-class classifier. For each labelled person, select the paired overhead and side/front recordings with corresponding trial identifiers and a window overlapping the assigned expert interval by at least 22.5 seconds (75% of the 30-second rating). The earliest matched trial key is chosen reproducibly if more than one candidate remains. This alignment rule is explicit because the workbook identifies participants but not recording session numbers; P81 has historical multiple sessions, so association with a specific video remains provisional unless confirmed by the client.

Features: median overhead arm movement speed (body-relative torso lengths/s), side/front median leg speed (same units), and side/front image-plane torso inclination (degrees). These are descriptive proxies, not measures of force or airway clearance. The per-window observations are not multiplied into separate labelled examples.

## Training and evaluation

With four individuals a random 80/20 train/test split is inappropriate: one test observation would yield no meaningful class coverage. Instead, use **leave-one-participant-out (LOPO)** cross-validation: three participants train and one is independently evaluated in each of four folds. All transformations (median missing-value imputation and standardization) fit within each training fold. The numerical regressor is Ridge (`alpha=10` fixed in advance), scored with **mean absolute error (MAE)** against each expert target. Compare to a training-fold mean-score predictor (DummyRegressor). Predictions are clipped only to the published scale ranges. After cross-validation, fit one final model per target to all aligned participants, to allow further exploratory use. The final fit is not evaluated on its own training samples.

This produces a working supervised experiment with reproducible out-of-sample predictions on the four labelled cases, **not** verified predictive performance at population level. Session alignment and independent additional labels are necessary to make general claims about flotation quality. No airway-submersion detection or classification accuracy is claimed.

## Rerun

```bash
python run.py --source-root ../AI_Flotation
```

Results: `outputs/latest/supervised/SUPERVISED_RESULTS.md`, `supervised_metrics.json`, trained pilot models, private participant folds and the dashboard's expert-supervised section. Supply updated scores through `--expert-labels` with the same column schema.
