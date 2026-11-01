# Results — AI Flotation Analyser

## Study data and processing

The inventory yielded **383 analysed recordings** from **187 participants**, with **2625 non-overlapping temporal windows**. **11** entries were missing or failed. Distinct recordings and temporal windows are not independent participant-level samples.

| Camera view | Recordings | Median valid torso frames | Reported arm metrics | Reported leg metrics |
|---|---:|---:|---:|---:|
| side_front | 191 | 99.18% | 182 | 167 |
| top | 192 | 99.93% | 190 | 147 |

Frame-level body geometry is computed from MediaPipe 33-landmark estimates. Wrist and ankle velocities are body-relative, normalised to torso length per second; image-space trunk orientation is measured against image horizontal, **not** a calibrated waterline. A provisional 10 torso-lengths/s rule flags suspect limb movements; low-coverage measurements are withheld. QC flags are not clinical or flotation assessments.

## Unsupervised machine-learning experiment

The eligible cohort contained **141 unique participants**, each represented by one quality-gated top-view video. An 80/20 participant-disjoint split allocated **112 to training** and **29 to testing**, with **0 overlapping IDs**. Log1p transformation and standardisation were fitted exclusively on the training observations. K-means k=2–5 was selected using training silhouette and minimum cluster-size constraints; Gaussian mixtures were compared using training BIC selection. The fitted models were then evaluated on previously unseen participants.

| Metric | K-means | Gaussian mixture |
|---|---:|---:|
| Selected groups (training selection) | 4 | 2 |
| Training silhouette | 0.308 | 0.236 |
| Held-out assigned silhouette | **0.212** | **0.105** |

Bootstrapped training-resample assignment stability against the fixed held-out participants (median adjusted Rand index) was **0.650**, based on 30 resamples. The selected number of groups may differ from earlier full-cohort exploratory clustering because model selection now excludes test participants. These internal and stability metrics measure feature-group structure, **not prediction accuracy, floating-quality validity or airway security**.


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

## Evaluation boundaries and further research

The P2 pilot identified visually apparent arm identity swaps and misplaced ankle landmarks in selected frames. Selected failure examples are not an unbiased error-rate estimate. The four expert-rated participants support a small supervised regression pilot with leave-one-participant-out evaluation; the session match remains an explicit assumption. The present system does not compute waterline or submersion-event duration.

The exploratory clusters may reflect pose-tracking errors, recording phase, or camera movement as well as real strategies. Independent anatomical annotation, waterline calibration, larger labelled cohorts, and expert reviews of group exemplars are needed for construct validation.

## Reproduction and handover

The repository contains code, tests, requirements and methods; the source videos, 33-landmark CSVs, participant-level exports and fitted participant assignments stay on approved research storage. Reproduce with `python run.py --source-root ../AI_Flotation` from the repository directory. Do not upload personal data, raw CSVs, video frames or participant-specific outputs to a public repository.

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
