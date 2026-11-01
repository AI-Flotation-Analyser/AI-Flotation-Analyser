"""Expert-anchored pilot modelling of the 30–60 s flotation assessment interval.

Each expert rating is one independent participant, never one frame or video.
Leave-one-participant-out evaluation fits preprocessing separately in every fold.
"""
from __future__ import annotations

import re
from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = ('top_arm_speed', 'side_leg_speed', 'side_trunk_angle')
TARGETS = {'floating_outcome': (0, 3), 'overall_score': (0, 13)}
REQUIRED_LABELS = {'participant_id','interval_start_sec','interval_end_sec', *TARGETS}
SOURCE_FEATURES = {
    'top': {'top_arm_speed':'arm_speed_median_torso_s'},
    'side_front': {'side_leg_speed':'leg_speed_median_torso_s',
                   'side_trunk_angle':'trunk_angle_median_deg'},
}


def read_labels(path: Path) -> pd.DataFrame:
    """Read a local, access-controlled expert score table, one row per person."""
    labels = pd.read_csv(path)
    absent = REQUIRED_LABELS.difference(labels.columns)
    if absent:
        raise ValueError(f'Expert label table missing {sorted(absent)}')
    if labels.participant_id.isna().any() or labels.participant_id.duplicated().any():
        raise ValueError('Expert labels must have one distinct participant ID per row')
    for col in ('interval_start_sec','interval_end_sec', *TARGETS):
        labels[col] = pd.to_numeric(labels[col], errors='raise')
    if (labels.interval_start_sec >= labels.interval_end_sec).any():
        raise ValueError('Invalid expert-rated interval')
    for target,(lo,hi) in TARGETS.items():
        if not labels[target].between(lo,hi).all():
            raise ValueError(f'Expert {target} outside [{lo},{hi}]')
    return labels


def trial_key(video_id: str) -> str:
    """Reduce camera-specific identifiers to a common recording identifier."""
    key = re.sub(r'^p\d+_(?:top|side_front)_', '', str(video_id), flags=re.I)
    key = re.sub(r'_(?:2_side_front|3_top)$','',key, flags=re.I)
    return key.lower()


def _best_window(source: pd.DataFrame, start: float, end: float) -> pd.Series | None:
    """Use the existing 30 s window overlapping the expert interval most closely."""
    if source.empty:
        return None
    begin = pd.to_numeric(source.window_start_sec, errors='coerce')
    finish = pd.to_numeric(source.window_end_sec, errors='coerce')
    overlap = np.minimum(finish, end) - np.maximum(begin, start)
    overlap = np.maximum(overlap, 0)
    best_index = overlap.idxmax()
    required = min(20., 0.75 * (end-start))
    if not np.isfinite(overlap.loc[best_index]) or overlap.loc[best_index] < required:
        return None
    return source.loc[best_index]


def align_expert_intervals(video: pd.DataFrame, windows: pd.DataFrame,
                           labels: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame]:
    """Match labelled participants to one paired 30–60 s top/side trial.

    If multiple trials are present, choose an identical trial key across views.
    Selection is deterministic; metadata retains the selected trial and pairing.
    """
    needed = {'participant_id','video_id','camera_view','window_start_sec','window_end_sec'}
    if not needed.issubset(windows.columns):
        raise ValueError(f'Window table missing {sorted(needed.difference(windows.columns))}')
    samples, coverage = [], []
    for rating in labels.to_dict('records'):
        pid = str(rating['participant_id'])
        start,end = float(rating['interval_start_sec']),float(rating['interval_end_sec'])
        pool = windows.loc[windows.participant_id.astype(str).eq(pid)].copy()
        candidates = {}
        for view in ('top','side_front'):
            rows=[]
            for vid, grp in pool.loc[pool.camera_view.eq(view)].groupby('video_id'):
                picked = _best_window(grp,start,end)
                if picked is not None:
                    rows.append((trial_key(vid), str(vid), picked))
            candidates[view] = sorted(rows,key=lambda x:x[1])
        top_by_key = {key for key,_,_ in candidates['top']}
        side_by_key = {key for key,_,_ in candidates['side_front']}
        shared = sorted(top_by_key & side_by_key)
        record = {'participant_id':pid,'interval_start_sec':start,'interval_end_sec':end,
                  'n_top_candidates':len(candidates['top']),
                  'n_side_candidates':len(candidates['side_front']),
                  'trial_pair_status':'not_aligned', 'n_observed_features':0}
        if not shared:
            record['reason']='No matched top/side recording with an overlapping expert time window'
            coverage.append(record)
            continue
        # If multiple camera-matched trials are available, choose one trial only.
        selected_key=shared[0]
        found={}
        for view in ('top','side_front'):
            found[view] = next((vid,row) for key,vid,row in candidates[view] if key == selected_key)
            record[view+'_video_id']=found[view][0]
        record['trial_key'] = selected_key
        record['trial_pair_status']='matched_camera_pair'
        sample={**rating}
        sample.update({key:record[key] for key in ('top_video_id','side_front_video_id','trial_key')})
        for view in ('top','side_front'):
            row=found[view][1]
            for alias,column in SOURCE_FEATURES[view].items():
                sample[alias] = float(row[column]) if column in row.index and pd.notna(row[column]) else np.nan
        record['n_observed_features']=int(np.isfinite(np.array([sample[f] for f in FEATURES],dtype=float)).sum())
        record['reason'] = 'matched' if record['n_observed_features'] >= 2 else 'fewer_than_two_features'
        coverage.append(record)
        if record['n_observed_features']>=2:
            samples.append(sample)
    return pd.DataFrame(samples), pd.DataFrame(coverage)


def _model():
    from sklearn.dummy import DummyRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    # A predeclared regularisation strength avoids selecting hyperparameters
    # using the same four evaluation cases.
    ridge = Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                      ('scale',StandardScaler()), ('fit',Ridge(alpha=10.0))])
    baseline = DummyRegressor(strategy='mean')
    return ridge, baseline


def evaluate(samples: pd.DataFrame):
    """Fit held-out-by-participant ridge pilot and mean baselines, then final models."""
    from sklearn.metrics import mean_absolute_error
    if len(samples)<3:
        raise ValueError(f'Supervised pilot requires at least 3 aligned expert-labelled participants; got {len(samples)}')
    if samples.participant_id.duplicated().any():
        raise ValueError('Supervised pilot has duplicate participants')
    data=samples.reset_index(drop=True)
    X=data.loc[:,FEATURES].apply(pd.to_numeric,errors='coerce').to_numpy(dtype=float)
    predictions=[]
    final_models={}
    metrics={'method':'leave-one-participant-out (one fully withheld expert-rated participant per fold)',
             'supervised_model':'median imputation + training-fitted StandardScaler + Ridge(alpha=10)',
             'baseline':'training-mean DummyRegressor',
             'features':list(FEATURES), 'expert_interval_sec':[30,60],
             'participant_count':int(len(data)), 'fold_count':int(len(data)),
             'participant_leakage':0,'targets':{}}
    for target,(lo,hi) in TARGETS.items():
        y=data[target].to_numpy(dtype=float)
        y_hat=np.full(len(y),np.nan)
        y_base=np.full(len(y),np.nan)
        for held_out in range(len(y)):
            train=np.arange(len(y))!=held_out
            ridge, baseline = _model()
            ridge.fit(X[train],y[train])
            baseline.fit(X[train],y[train])
            y_hat[held_out]=np.clip(ridge.predict(X[held_out:held_out+1])[0],lo,hi)
            y_base[held_out]=np.clip(baseline.predict(X[held_out:held_out+1])[0],lo,hi)
        ridge, _ = _model()
        ridge.fit(X,y)
        final_models[target]=ridge
        metrics['targets'][target]={
            'range':[lo,hi],
            'loocv_mae':round(float(mean_absolute_error(y,y_hat)),5),
            'baseline_loocv_mae':round(float(mean_absolute_error(y,y_base)),5),
            'mean_signed_error':round(float(np.mean(y_hat-y)),5),
            'n_held_out_predictions':int(len(y)),
        }
        for idx in range(len(y)):
            predictions.append({'participant_id':str(data.iloc[idx]['participant_id']),
                 'target':target,'actual':y[idx], 'held_out_prediction':round(y_hat[idx],5),
                 'held_out_baseline':round(y_base[idx],5),
                 'absolute_error':round(abs(y_hat[idx]-y[idx]),5),
                 'fold_train_participants':int(len(data)-1)})
    return {'report':metrics, 'predictions':pd.DataFrame(predictions),
            'trained_models':final_models, 'aligned_samples':data}


def markdown(result: dict, alignment: pd.DataFrame) -> str:
    r=result['report']
    a=r['targets']['floating_outcome'];b=r['targets']['overall_score']
    return f'''# Expert-anchored supervised pilot — AI Flotation Analyser

## Expert annotations

Chris Button's scoring workbook contains four participant-level ratings made over the 30–60 s segment. The outcome scale is 0–3, and the five-category overall score ranges from 0 to 13. The labels are linked to top and side/front 30 s feature windows, not assigned to every window or video of a participant.

## Experimental design

The pilot uses **{r['participant_count']} aligned participants**, **three preselected measurements** (overhead arm movement, side/front leg movement, side/front image-plane trunk angle), and **leave-one-participant-out evaluation**. Each fold fits missing-value imputation, standardisation and regularised Ridge regression on the other participants. The comparison is the mean of the training participants' scores. A final model is then fit to the full labelled pilot cohort, independently of its held-out evaluation.

| Target | Leave-one-participant-out MAE | Mean-predictor MAE | Independent predictions |
|---|---:|---:|---:|
| Floating outcome, 0–3 | {a['loocv_mae']:.3f} | {a['baseline_loocv_mae']:.3f} | {a['n_held_out_predictions']} |
| Overall score, 0–13 | {b['loocv_mae']:.3f} | {b['baseline_loocv_mae']:.3f} | {b['n_held_out_predictions']} |

Matched recording pairs: **{int(alignment.trial_pair_status.eq('matched_camera_pair').sum())}** of **{len(alignment)}** expert-labelled participants. A rated participant whose specific trial is not independently documented is linked to the available paired recording by deterministic filename matching; this is a pilot alignment assumption, not confirmed session ground truth.

This experiment demonstrates functioning expert-supervised training and participant-independent evaluation on a four-person pilot. Its four held-out predictions do not establish reliable performance for unseen flotation participants or a deployable safety/quality classifier. The 0–3 scale is modelled numerically (regression), not as a validated four-class classifier.

## Reproduction

`private/expert_interval_alignment.csv` records the selected trials and usable features; `private/expert_heldout_predictions.csv` records held-out predictions, and `supervised/supervised_metrics.json` contains aggregate evaluation results. Expert input labels and predictions remain local and out of Git.
'''


def figure(result: dict, path: Path) -> None:
    """Write an aggregate-only pilot MAE comparison for the client dashboard."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path.parent.mkdir(parents=True, exist_ok=True)
    r=result['report']['targets']
    targets=['floating_outcome','overall_score']
    names=['Floating outcome (0–3)','Overall score (0–13)']
    x=np.arange(2)
    fig,ax=plt.subplots(figsize=(7.8,4.3))
    w=.36
    ax.bar(x-w/2,[r[t]['loocv_mae'] for t in targets],width=w,label='Ridge')
    ax.bar(x+w/2,[r[t]['baseline_loocv_mae'] for t in targets],width=w,label='Training mean')
    ax.set_xticks(x,names)
    ax.set_ylabel('Leave-one-person-out absolute error (score points)')
    ax.set_title('Pilot supervised evaluation (4 expert-rated participants)')
    ax.legend()
    ax.grid(axis='y',alpha=.2)
    fig.tight_layout()
    fig.savefig(path,dpi=150)
    plt.close(fig)