# Supervised and semi-supervised protocol

## Input units

- The original corpus: 383 processable videos, 187 unique participant IDs, 2,625 temporal windows, 11 absent raw33 inputs.
- Outcome labels: Chris Button's four 30–60-second expert-rated samples, P2/P68/P81/P118, with distinct `floating_outcome` scores 3/2/0/1 and `overall_score` values 12/11/4/6 respectively.
- Each candidate is the **same paired top and side/front recording**, one 30–60-second window per view; never let the model mix different sessions or use unrelated whole-video aggregates.
- Inputs are the three predeclared numerical features: top-view median body-relative arm activity, side/front median body-relative leg activity, side/front median image-plane trunk angle. Measurements are not physical ground-truth waterline angles or metric displacement.

## Training/testing

1. Expert-only Ridge: 4 leave-one-participant-out folds. Preprocessing, missing-value imputation and standardisation are fitted to the remaining three labelled participants within each fold; the test participant's **label** is hidden.
2. Mean-outcome baseline: train-fold mean only. Compare held-out MAE for each of the two scores independently.
3. Graph label diffusion: uses a full feature-space affinity matrix with predeclared RBF scale and clamped expert anchors. Each held-out expert row is treated as an **unlabelled** node, so it is a transductive evaluation; graph construction uses its features but never its label.
4. Low-weight pseudo-label Ridge: predict unrated features using expert-only Ridge and graph diffusion. Rank pseudo candidates by predictor agreement and proximity to observed cases, select up to 24 in each fit, assign them a total sample weight equivalent to 0.25 of a single verified case, then refit Ridge. For leave-one-out evaluation the withheld expert participant is **excluded from pseudo-labelled retraining**.
5. All four methods report the four withheld predictions and their mean absolute errors. Candidate hyperparameters are fixed; these four folds are a small-sample evaluation, not external validation or a class-balanced four-class test.
6. After evaluation, fit final models to the available confirmed labels and generate a separate provisional prediction for each unlabelled participant with paired 30–60s measurements. Retain unmatched participants as explicit no-estimate rows.

## Outputs and interpretation

- `private/lopo_train_test_predictions.csv` is the test table and definitive evaluation provenance, including the IDs of the people fitted and withheld.
- `private/all_participant_predictions.csv` is the prediction table. Labeled rows show **held-out** predictions, not misleading in-sample fitted values. Unlabelled rows show full-model provisional estimated scores and a rounded proposed 0–3 outcome class.
- `private/pseudo_training_selection.csv` lists every pseudo-labelled participant used in final self-training; these are never counted as new expert ground truth.
- `private/participant_review_queue.csv` sorts unlabelled predictions by disagreement plus distance from verified examples. The priority score is a sorting heuristic, **not a calibrated probability** of being right or wrong.
- `private_inputs/reviewed_labels.csv` supports adding human-verified training labels. Pending/unclear records are ignored by the modelling code.

The supervised and semi-supervised results characterize a four-example pilot unless additional independently reviewed labels are supplied. A participant's ground-truth score cannot be inferred from an arbitrary model-generated pseudo-score, and only independently assessed videos can support stronger accuracy claims.
