# Machine-learning holdout evaluation — AI Flotation Analyser

> Exploratory **unsupervised** learning, not an expert-rated flotation-quality predictor. Generated from the local full-corpus feature CSV. Participant-level data remain private.

## Experimental design

- Eligible cohort: **141** unique participants (one top-camera recording each); same camera, QC gates and three movement/drift features as the original clustering experiment.
- Participant-independent 80/20 split, random seed 42: **112 training**, **29 testing** participants; overlap = **0**.
- `log1p` transform, followed by `StandardScaler` fitted on the **training cohort only**, then applied unchanged to the test cohort.
- **Training-only** K-means hyperparameter selection among k=2–5, with a minimum size threshold of five/4% of training observations and training silhouette maximization. The untouched test cohort was assigned to trained centroids.

## Held-out results

| Measure | K-means | Gaussian mixture (comparison) |
|---|---:|---:|
| Selected groups | 4 (train silhouette) | 2 (train BIC) |
| Training silhouette | 0.308 | 0.236 |
| Held-out assigned silhouette | **0.212** | 0.105 |

K-means held-out group sizes: **[10, 7, 11, 1]**. Median distance to nearest training centroid, train/test: 0.894 / 1.128 (in train-standardized feature units).

**Bootstrap sensitivity:** refitting K-means on 30 resampled sets of training observations, then assigning the same held-out participants, gave an adjusted Rand index vs the main fitted model of **0.650** median (10th percentile **0.315**). This measures consistency of assignments under training resampling, **not** accuracy.

## Interpretation and limitations

This is a **retrospective participant-level holdout**, not a completely independent prospective test: feature choices and eligibility rules were developed while exploring the full dataset. It is an unseen-participant generalisation check of **feature-group structure**, but does not create ground truth. **Silhouette is not a prediction accuracy score**. Gaussian mixture scores are descriptive because the same holdout was examined for comparison. The model cannot predict flotation quality or airway security, and participant strategies are not expert validated. In particular, the features can still be biased by body landmark swaps and by entry/recovery phases. The separately implemented expert-supervised pilot reports its own four-person leave-one-out regression evaluation, distinct from the unlabeled grouping.

### Files

- `ml_holdout_metrics.json`: reproducible parameters and scores
- `participant_split_and_predictions.csv`: participant-wise train/test allocations and exploratory assignments (PRIVATE)
- `train_only_kmeans_selection.csv`, `train_only_gmm_bic.csv`: train-only selection records
- `bootstrap_stability.csv`: training-sample sensitivity checks
- `trained_clustering_models.joblib`: trained scaler and grouping models for local use (trusted files only)
- `train_k_selection.png`, `train_test_pca.png`: non-identifying aggregate illustrations
