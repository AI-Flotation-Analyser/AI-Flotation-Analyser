#!/usr/bin/env python3
"""Participant-held-out evaluation of exploratory flotation behaviour features.

Fit all learned transformations exclusively on training participants.
This is not a supervised flotation-quality predictor.
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

FEATURES = (
    'arm_movement_speed_median_torso_s',
    'body_path_rate_median_torso_s',
    'body_max_excursion_torso',
)
GATE = {'torso_valid_fraction': .80, 'arm_speed_usable_fraction': .60,
        'arm_screened_frame_fraction': .05}


def eligible_cohort(video: pd.DataFrame) -> pd.DataFrame:
    """Reconstruct exactly the original eligible top-view one-video/person cohort."""
    needed = {'video_id', 'participant_id', 'camera_view', 'arm_measurement_basis',
              'torso_valid_fraction', 'arm_speed_usable_fraction',
              'arm_screened_frame_fraction', 'duration_sec', *FEATURES}
    absent = needed.difference(video.columns)
    if absent:
        raise ValueError(f'Missing input columns: {sorted(absent)}')
    top = video.loc[video.camera_view.eq('top')].copy()
    numeric = [*FEATURES, *GATE.keys(), 'duration_sec']
    for c in numeric:
        top[c] = pd.to_numeric(top[c], errors='coerce')
    top = top.loc[
        top.arm_measurement_basis.eq('bilateral') &
        top.torso_valid_fraction.ge(GATE['torso_valid_fraction']) &
        top.arm_speed_usable_fraction.ge(GATE['arm_speed_usable_fraction']) &
        top.arm_screened_frame_fraction.le(GATE['arm_screened_frame_fraction'])
    ].dropna(subset=[*FEATURES, 'participant_id', 'video_id'])
    top = top.loc[np.isfinite(top[list(FEATURES)].to_numpy(dtype=float)).all(axis=1)]
    top = top.loc[top[list(FEATURES)].ge(0).all(axis=1)]
    top = top.sort_values(['participant_id', 'duration_sec', 'video_id'],
                          ascending=[True, False, True]).drop_duplicates('participant_id', keep='first')
    if top.participant_id.duplicated().any():
        raise AssertionError('Duplicate participants survived filtering.')
    return top.reset_index(drop=True)


def finite_silhouette(X, labels):
    from sklearn.metrics import silhouette_score
    unique, counts = np.unique(labels, return_counts=True)
    # scikit-learn permits singleton groups (their silhouette contribution is zero).
    # A silhouette is undefined only for 1 group or one group per sample.
    if len(unique) < 2 or len(unique) >= len(labels):
        return None
    return float(silhouette_score(X, labels))


def nullable(x, digits=5):
    if x is None or not np.isfinite(x):
        return None
    return round(float(x), digits)


def train_and_evaluate(cohort: pd.DataFrame, seed: int = 42, n_bootstrap: int = 30):
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.metrics import adjusted_rand_score
    from sklearn.mixture import GaussianMixture
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.preprocessing import StandardScaler

    if len(cohort) < 30:
        raise ValueError('At least 30 eligible participants required for holdout evaluation.')
    if cohort.participant_id.duplicated().any():
        raise ValueError('Only one video per participant may enter this experiment.')

    train_i, test_i = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(
        cohort, groups=cohort.participant_id))
    train = cohort.iloc[train_i].copy().reset_index(drop=True)
    test = cohort.iloc[test_i].copy().reset_index(drop=True)
    overlap = set(train.participant_id) & set(test.participant_id)
    if overlap:
        raise AssertionError(f'Participant leakage detected: {overlap}')

    # Fit transformation ONLY on training participants.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(np.log1p(train.loc[:, FEATURES].to_numpy(dtype=float)))
    X_test = scaler.transform(np.log1p(test.loc[:, FEATURES].to_numpy(dtype=float)))

    k_rows, km_models = [], {}
    min_size = max(5, int(np.ceil(.04 * len(train))))
    for k in (2, 3, 4, 5):
        km = KMeans(n_clusters=k, n_init=20, random_state=seed)
        pred = km.fit_predict(X_train)
        sizes = np.bincount(pred, minlength=k)
        score = finite_silhouette(X_train, pred)
        eligible = score is not None and sizes.min() >= min_size
        k_rows.append({'k':k, 'train_silhouette':nullable(score),
                       'train_min_cluster_size': int(sizes.min()), 'eligible':bool(eligible)})
        if eligible:
            km_models[k] = (km, score)
    if not km_models:
        raise ValueError('No K-means candidate passed training-only size/silhouette checks.')
    best_k = sorted(km_models, key=lambda k: (-km_models[k][1], k))[0]
    km = km_models[best_k][0]
    km_train = km.labels_
    km_test = km.predict(X_test)
    km_test_s = finite_silhouette(X_test, km_test)
    km_train_s = finite_silhouette(X_train, km_train)

    # An alternative trained on the same training data, k selected using TRAIN BIC.
    bic_rows, gm_models = [], {}
    for k in (2, 3, 4, 5):
        gm = GaussianMixture(n_components=k, covariance_type='full',
                             n_init=3, reg_covar=1e-5, random_state=seed)
        gm.fit(X_train)
        bic = float(gm.bic(X_train))
        bic_rows.append({'k': k, 'train_bic': nullable(bic), 'eligible': True})
        gm_models[k] = gm
    gm_k = min(gm_models, key=lambda k: (gm_models[k].bic(X_train), k))
    gm = gm_models[gm_k]
    gm_train = gm.predict(X_train)
    gm_test = gm.predict(X_test)
    gm_test_s = finite_silhouette(X_test, gm_test)

    # Bootstrap the TRAIN participants and check assignments for the SAME untouched test people.
    # ARI measures assignment stability, not correctness or flotation strategy validity.
    rng = np.random.default_rng(seed + 1729)
    stability_rows = []
    for rep in range(n_bootstrap):
        ind = rng.integers(0, len(X_train), size=len(X_train))
        bkm = KMeans(n_clusters=best_k, n_init=10, random_state=seed + rep + 1)
        bkm.fit(X_train[ind])
        alt = bkm.predict(X_test)
        stability_rows.append({'bootstrap':rep + 1,
                               'test_assignment_ari_vs_main': round(float(adjusted_rand_score(km_test, alt)), 6),
                               'n_test_groups_predicted': int(np.unique(alt).size)})
    st = pd.DataFrame(stability_rows)

    # PCA is ONLY a visualization, fitted on training samples.
    pca = PCA(n_components=2, random_state=seed)
    train_pc = pca.fit_transform(X_train)
    test_pc = pca.transform(X_test)

    def labelled(df, split, km_pred, gm_pred, X, pc):
        out = df[['video_id', 'participant_id', 'camera_view', *FEATURES]].copy()
        out['split'] = split
        out['kmeans_cluster'] = [int(x) + 1 for x in km_pred]
        out['gmm_component'] = [int(x) + 1 for x in gm_pred]
        out['kmeans_nearest_centroid_distance'] = np.linalg.norm(X - km.cluster_centers_[km_pred], axis=1)
        out['train_pca_pc1'] = pc[:,0]
        out['train_pca_pc2'] = pc[:,1]
        return out
    assignments = pd.concat([labelled(train,'train',km_train,gm_train,X_train,train_pc),
                             labelled(test,'test',km_test,gm_test,X_test,test_pc)],
                            ignore_index=True)
    report = {
        'method':'participant-level 80/20 holdout of one eligible top-view video per participant',
        'seed': seed, 'features': list(FEATURES), 'cohort_participants':len(cohort),
        'train_participants':len(train), 'test_participants':len(test),
        'participant_overlap': len(overlap),
        'scaler_fit_on':'train_only', 'model_selection_on':'train_only',
        'holdout_not_used_for_selection':True,
        'kmeans': {
            'selected_k':int(best_k), 'train_silhouette':nullable(km_train_s),
            'test_assigned_silhouette':nullable(km_test_s),
            'train_cluster_sizes':np.bincount(km_train, minlength=best_k).astype(int).tolist(),
            'test_cluster_sizes':np.bincount(km_test, minlength=best_k).astype(int).tolist(),
            'median_train_centroid_distance':nullable(np.median(np.linalg.norm(X_train - km.cluster_centers_[km_train],axis=1))),
            'median_test_centroid_distance':nullable(np.median(np.linalg.norm(X_test - km.cluster_centers_[km_test],axis=1))),
            'bootstrap_test_ari_median':nullable(st.test_assignment_ari_vs_main.median()),
            'bootstrap_test_ari_q10':nullable(st.test_assignment_ari_vs_main.quantile(.1)),
            'bootstrap_repetitions':n_bootstrap
        },
        'gmm': {
            'selected_k_train_bic':int(gm_k),
            'train_silhouette':nullable(finite_silhouette(X_train,gm_train)),
            'test_assigned_silhouette':nullable(gm_test_s),
            'test_group_sizes':np.bincount(gm_test,minlength=gm_k).astype(int).tolist(),
        },
        'train_pca_explained_variance_ratio':[round(float(v),5) for v in pca.explained_variance_ratio_],
        'evaluation_note': 'Silhouette and ARI are internal/assignment-stability metrics, not supervised accuracy, expert-validated strategy, or flotation quality.'
    }
    return {'report':report, 'assignments':assignments, 'selection':pd.DataFrame(k_rows),
            'gmm_selection':pd.DataFrame(bic_rows), 'bootstrap':st,
            'trained_artifacts': {'scaler':scaler, 'kmeans':km, 'gmm':gm,
                                  'features':list(FEATURES), 'transform':'log1p',
                                  'seed':seed, 'kmeans_selected_k':int(best_k),
                                  'gmm_selected_k':int(gm_k),
                                  'disclaimer':'Exploratory clusters, not quality classes'}}


def make_figures(result, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sel = result['selection']; df = result['assignments']
    fig, ax = plt.subplots(figsize=(7,4.2))
    valid = sel.loc[sel.eligible]
    ax.bar(valid.k.astype(str), valid.train_silhouette.astype(float))
    ax.set_xlabel('K-means k (chosen using training data only)')
    ax.set_ylabel('Training silhouette')
    ax.set_title('Model selection on training participants')
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out/'train_k_selection.png',dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,5.4))
    for group in sorted(df.kmeans_cluster.unique()):
        for split, marker in [('train','o'), ('test','X')]:
            part = df.loc[df.kmeans_cluster.eq(group) & df.split.eq(split)]
            if len(part):
                ax.scatter(part.train_pca_pc1, part.train_pca_pc2, alpha=.7, marker=marker,
                           s=30 if split=='train' else 70, label=f'group {group}, {split} (n={len(part)})')
    ax.set_xlabel('PC1 (train-fitted visualization)')
    ax.set_ylabel('PC2 (train-fitted visualization)')
    ax.set_title('Train and held-out participant group assignments\nNOT flotation-quality classes')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'train_test_pca.png', dpi=160)
    plt.close(fig)


def make_markdown(r):
    k=r['kmeans']; gm=r['gmm']; a=r['cohort_participants']
    def f(v): return 'not estimable' if v is None else f'{v:.3f}'
    return f'''# Machine-learning holdout evaluation — AI Flotation Analyser

> Exploratory **unsupervised** learning, not an expert-rated flotation-quality predictor. Generated from the local full-corpus feature CSV. Participant-level data remain private.

## Experimental design

- Eligible cohort: **{a}** unique participants (one top-camera recording each); same camera, QC gates and three movement/drift features as the original clustering experiment.
- Participant-independent 80/20 split, random seed {r['seed']}: **{r['train_participants']} training**, **{r['test_participants']} testing** participants; overlap = **{r['participant_overlap']}**.
- `log1p` transform, followed by `StandardScaler` fitted on the **training cohort only**, then applied unchanged to the test cohort.
- **Training-only** K-means hyperparameter selection among k=2–5, with a minimum size threshold of five/4% of training observations and training silhouette maximization. The untouched test cohort was assigned to trained centroids.

## Held-out results

| Measure | K-means | Gaussian mixture (comparison) |
|---|---:|---:|
| Selected groups | {k['selected_k']} (train silhouette) | {gm['selected_k_train_bic']} (train BIC) |
| Training silhouette | {f(k['train_silhouette'])} | {f(gm['train_silhouette'])} |
| Held-out assigned silhouette | **{f(k['test_assigned_silhouette'])}** | {f(gm['test_assigned_silhouette'])} |

K-means held-out group sizes: **{k['test_cluster_sizes']}**. Median distance to nearest training centroid, train/test: {f(k['median_train_centroid_distance'])} / {f(k['median_test_centroid_distance'])} (in train-standardized feature units).

**Bootstrap sensitivity:** refitting K-means on {k['bootstrap_repetitions']} resampled sets of training observations, then assigning the same held-out participants, gave an adjusted Rand index vs the main fitted model of **{f(k['bootstrap_test_ari_median'])}** median (10th percentile **{f(k['bootstrap_test_ari_q10'])}**). This measures consistency of assignments under training resampling, **not** accuracy.

## Interpretation and limitations

This is a **retrospective participant-level holdout**, not a completely independent prospective test: feature choices and eligibility rules were developed while exploring the full dataset. It is an unseen-participant generalisation check of **feature-group structure**, but does not create ground truth. **Silhouette is not a prediction accuracy score**. Gaussian mixture scores are descriptive because the same holdout was examined for comparison. The model cannot predict flotation quality or airway security, and participant strategies are not expert validated. In particular, the features can still be biased by body landmark swaps and by entry/recovery phases. The separately implemented expert-supervised pilot reports its own four-person leave-one-out regression evaluation, distinct from the unlabeled grouping.

### Files

- `ml_holdout_metrics.json`: reproducible parameters and scores
- `participant_split_and_predictions.csv`: participant-wise train/test allocations and exploratory assignments (PRIVATE)
- `train_only_kmeans_selection.csv`, `train_only_gmm_bic.csv`: train-only selection records
- `bootstrap_stability.csv`: training-sample sensitivity checks
- `trained_clustering_models.joblib`: trained scaler and grouping models for local use (trusted files only)
- `train_k_selection.png`, `train_test_pca.png`: non-identifying aggregate illustrations
'''

