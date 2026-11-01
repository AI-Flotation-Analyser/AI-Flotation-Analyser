"""Expert-anchored semi-supervised flotation scoring and participant review.

The expert labels are the only ground-truth outcomes. Every other score is a
provisional model estimate. Held-out folds never use their test subject's label.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor

from .supervised import FEATURES, TARGETS, _best_window, trial_key, read_labels


def participant_matrix(video: pd.DataFrame, windows: pd.DataFrame,
                       interval: tuple[float,float] = (30.,60.)) -> pd.DataFrame:
    """One paired trial and one temporal-window vector per participant.

    This is the same feature definition used to train on expert-scored samples.
    An unmatched person remains in the table with a missing-feature status.
    """
    all_ids = sorted(video.participant_id.astype(str).dropna().unique())
    out = []
    for pid in all_ids:
        part = windows.loc[windows.participant_id.astype(str).eq(pid)]
        found = {}
        for view in ('top','side_front'):
            matches=[]
            for vid, g in part.loc[part.camera_view.eq(view)].groupby('video_id'):
                w = _best_window(g,*interval)
                if w is not None:
                    matches.append((trial_key(vid),str(vid),w))
            found[view] = sorted(matches,key=lambda x:(x[0],x[1]))
        keys = sorted(set(x[0] for x in found['top']) &
                      set(x[0] for x in found['side_front']))
        row={'participant_id':pid,'interval_start_sec':interval[0],
             'interval_end_sec':interval[1],
             'top_video_id':'','side_front_video_id':'',
             'trial_key':'','alignment_status':'unmatched_trial',
             'n_observed_features':0}
        for f in FEATURES:
            row[f] = np.nan
        if keys:
            key = keys[0]
            top = next((v,w) for k,v,w in found['top'] if k==key)
            side = next((v,w) for k,v,w in found['side_front'] if k==key)
            row.update(top_video_id=top[0],side_front_video_id=side[0],trial_key=key)
            fieldmap={'top_arm_speed':(top[1],'arm_speed_median_torso_s'),
                      'side_leg_speed':(side[1],'leg_speed_median_torso_s'),
                      'side_trunk_angle':(side[1],'trunk_angle_median_deg')}
            for alias,(sample,column) in fieldmap.items():
                if column in sample.index:
                    row[alias]=pd.to_numeric(sample[column],errors='coerce')
            row['n_observed_features']=int(np.isfinite(np.array([row[f] for f in FEATURES],dtype=float)).sum())
            row['alignment_status']=('eligible' if row['n_observed_features']>=2
                                      else 'insufficient_features')
        out.append(row)
    return pd.DataFrame(out)


def load_combined_labels(chris_path: Path, reviewed_path: Path|None=None) -> pd.DataFrame:
    """Optionally add human-confirmed team scores, never machine-generated scores."""
    chris=read_labels(chris_path).copy()
    chris['label_provenance']='expert_chris'
    if reviewed_path and reviewed_path.is_file() and reviewed_path.stat().st_size > 0:
        reviewed=pd.read_csv(reviewed_path)
        if len(reviewed):
            if 'review_status' not in reviewed.columns:
                raise ValueError('Reviewed labels must contain review_status=verified')
            reviewed=reviewed.loc[reviewed.review_status.eq('verified')].copy()
            if len(reviewed):
                # Match the existing label validation, without creating a temporary file.
                required={'participant_id','interval_start_sec','interval_end_sec',*TARGETS}
                missing=required.difference(reviewed.columns)
                if missing:
                    raise ValueError(f'Reviewed labels missing {sorted(missing)}')
                for col,(lo,hi) in TARGETS.items():
                    reviewed[col]=pd.to_numeric(reviewed[col],errors='raise')
                    if not reviewed[col].between(lo,hi).all():
                        raise ValueError(f'Reviewed target {col} outside [{lo},{hi}]')
                if reviewed['participant_id'].duplicated().any():
                    raise ValueError('Duplicate team-reviewed participant')
                reviewed['label_provenance']='team_verified'
                if set(reviewed.participant_id.astype(str)) & set(chris.participant_id.astype(str)):
                    raise ValueError('Team-reviewed label collides with a Chris expert label')
                chris=pd.concat([chris,reviewed],ignore_index=True)
    return chris


def _numeric_matrix(frame: pd.DataFrame):
    return frame.loc[:,FEATURES].apply(pd.to_numeric,errors='coerce').to_numpy(float)


def _graph_regress(X: np.ndarray, labelled: np.ndarray, y: np.ndarray) -> np.ndarray:
    """RBF graph label diffusion with clamped expert nodes, numeric targets.

    All X may be known (transductive learning); only labelled rows provide y.
    The fixed mixing term stabilises disconnected or weakly linked cases.
    """
    if not len(labelled):
        raise ValueError('At least one labelled node required')
    D = euclidean_distances(X,squared=True)
    positive = D[D>1e-12]
    scale = max(float(np.median(positive)),1e-8) if len(positive) else 1.
    W = np.exp(-D / scale)
    np.fill_diagonal(W,0.)
    P = W / np.maximum(W.sum(axis=1,keepdims=True),1e-12)
    mean=float(np.mean(y))
    values=np.full(len(X),mean,dtype=float)
    values[labelled]=y
    for _ in range(180):
        new=.88*(P @ values)+.12*mean
        new[labelled]=y
        if np.max(np.abs(new-values))<1e-8:
            values=new
            break
        values=new
    return values


def _ridge():
    return Pipeline([
      ('imputer',SimpleImputer(strategy='median',keep_empty_features=True)),
      ('scaler',StandardScaler()),
      ('ridge',Ridge(alpha=10.))])


def _fitted_space(X: np.ndarray):
    """Unlabelled covariates are allowed in transductive graph preprocessing."""
    imp=SimpleImputer(strategy='median',keep_empty_features=True)
    Z=imp.fit_transform(X)
    scaler=StandardScaler()
    return scaler.fit_transform(Z),imp,scaler


def _train_pseudo_ridge(X: np.ndarray, Z: np.ndarray, train_idx: np.ndarray,
                        pseudo_candidates: np.ndarray, target: np.ndarray,
                        gpred: np.ndarray, bounds: tuple[int,int],
                        max_pseudo:int=24) -> tuple[Pipeline,np.ndarray]:
    """Fixed low-weight self-training on agreement-ranked provisional scores."""
    model=_ridge()
    model.fit(X[train_idx],target)
    if not len(pseudo_candidates):
        return model,np.array([],dtype=int)
    base=np.clip(model.predict(X[pseudo_candidates]),*bounds)
    graph=np.clip(gpred[pseudo_candidates],*bounds)
    disagreements=np.abs(base-graph)
    distances=euclidean_distances(Z[pseudo_candidates],Z[train_idx]).min(axis=1)
    # Fixed priorities: agreement first, then distance from known cases.
    ranks=np.lexsort((distances,disagreements))
    keep=pseudo_candidates[ranks[:min(max_pseudo,len(pseudo_candidates))]]
    pseudo=(np.clip(model.predict(X[keep]),*bounds)+np.clip(gpred[keep],*bounds))/2
    train_X=np.vstack([X[train_idx],X[keep]])
    train_y=np.concatenate([target,pseudo])
    # All machine-generated pseudo labels together carry at most 25% of one
    # confirmed label's total weight. They are not new ground-truth labels.
    sample_weight=np.r_[np.ones(len(train_idx)),
                        np.full(len(keep),0.25/max(1,len(keep)))]
    model.fit(train_X,train_y,ridge__sample_weight=sample_weight)
    return model,keep


def run(matrix: pd.DataFrame, labels: pd.DataFrame, seed: int=42) -> dict:
    """Leave-person-out test + full cohort provisional predictions and review queue."""
    from sklearn.metrics import r2_score
    if matrix.participant_id.duplicated().any():
        raise ValueError('Matrix repeats a participant')
    matrix=matrix.reset_index(drop=True).copy()
    eligible=matrix.alignment_status.eq('eligible')
    working=matrix.loc[eligible].copy().reset_index(drop=True)
    labels=labels.copy()
    labels['participant_id']=labels.participant_id.astype(str)
    present = labels.set_index('participant_id')
    known=working.participant_id.isin(present.index)
    labelled_indices=np.where(known.to_numpy())[0]
    if len(labelled_indices)<3:
        raise ValueError(f'Need >=3 aligned independently rated participants; found {len(labelled_indices)}')
    X=_numeric_matrix(working)
    Z,imp,scaler=_fitted_space(X)
    ids=working.participant_id.astype(str).to_numpy()
    labelled_ids=ids[labelled_indices]
    if len(set(labelled_ids))!=len(labelled_indices):
        raise AssertionError('One sample per rated person required')
    unlabelled_indices=np.where(~known.to_numpy())[0]
    pred=working[['participant_id','top_video_id','side_front_video_id','trial_key',
                  'interval_start_sec','interval_end_sec','alignment_status',
                  'n_observed_features',*FEATURES]].copy()
    pred['label_provenance']='model_estimate'
    pred.loc[known,'label_provenance']=[present.loc[x,'label_provenance'] for x in ids[known.to_numpy()]]
    pred['evaluation_kind']=np.where(known,'leave_one_person_out_test','unlabelled_inference')
    folds=[]
    trained={}
    scores={}
    pseudo_ids={}
    for target,bounds in TARGETS.items():
        y=np.array([float(present.loc[ids[i],target]) for i in labelled_indices],dtype=float)
        targets_for_rows=dict(zip(labelled_ids,y))
        pred[f'actual_{target}']=pred.participant_id.map(targets_for_rows)
        lopo={'ridge':[], 'baseline':[], 'graph':[], 'selftrain':[]}
        fold_pred={}
        for held_pos,held_idx in enumerate(labelled_indices):
            tr=np.delete(labelled_indices,held_pos)
            values=np.array([targets_for_rows[ids[i]] for i in tr])
            held_label=targets_for_rows[ids[held_idx]]
            ridge=_ridge(); ridge.fit(X[tr],values)
            base=DummyRegressor(strategy='mean').fit(X[tr],values)
            graph=_graph_regress(Z,tr,values)
            # Test row is NOT used as a pseudo-labelled training sample.
            possible=np.setdiff1d(np.arange(len(ids)),np.append(tr,held_idx))
            st,_=_train_pseudo_ridge(X,Z,tr,possible,values,graph,bounds)
            testX=X[held_idx:held_idx+1]
            evaluated={'ridge':float(np.clip(ridge.predict(testX)[0],*bounds)),
                       'baseline':float(np.clip(base.predict(testX)[0],*bounds)),
                       'graph':float(np.clip(graph[held_idx],*bounds)),
                       'selftrain':float(np.clip(st.predict(testX)[0],*bounds))}
            for method,value in evaluated.items():
                lopo[method].append(abs(held_label-value))
            folds.append({'target':target,'fold':held_pos+1,'train_ids':';'.join(ids[tr]),
                'test_id':ids[held_idx],'n_train_verified':len(tr),
                'n_test_verified':1,'n_test_label_in_training':0,
                'actual':held_label,**{f'{k}_prediction':round(v,5) for k,v in evaluated.items()}})
            fold_pred[ids[held_idx]]=evaluated
        full_model=_ridge(); full_model.fit(X[labelled_indices],y)
        graph_all=_graph_regress(Z,labelled_indices,y)
        final_model, selected=_train_pseudo_ridge(
            X,Z,labelled_indices,unlabelled_indices,y,graph_all,bounds)
        trained[target]={'expert_ridge':full_model,'semi_selftrain_ridge':final_model}
        pseudo_ids[target]=list(ids[selected])
        full_ridge=np.clip(full_model.predict(X),*bounds)
        full_st=np.clip(final_model.predict(X),*bounds)
        graph_all=np.clip(graph_all,*bounds)
        for index,pid in enumerate(ids):
            if pid in fold_pred:
                # These are predictions from the model that did not see this label.
                rr=fold_pred[pid]
                pred.loc[index,f'ridge_prediction_{target}']=round(rr['ridge'],5)
                pred.loc[index,f'graph_prediction_{target}']=round(rr['graph'],5)
                pred.loc[index,f'semisupervised_prediction_{target}']=round(rr['selftrain'],5)
                pred.loc[index,f'baseline_prediction_{target}']=round(rr['baseline'],5)
            else:
                pred.loc[index,f'ridge_prediction_{target}']=round(float(full_ridge[index]),5)
                pred.loc[index,f'graph_prediction_{target}']=round(float(graph_all[index]),5)
                pred.loc[index,f'semisupervised_prediction_{target}']=round(float(full_st[index]),5)
                pred.loc[index,f'baseline_prediction_{target}']=np.nan
        scores[target]={'lopo_mae':{method:round(float(np.mean(errors)),5)
                     for method,errors in lopo.items()},
            'n_test_folds':int(len(labelled_indices)),
            'selected_pseudo_train_count':int(len(selected)),
            'target_range':list(bounds)}
    expert_distances=euclidean_distances(Z,Z[labelled_indices]).min(axis=1)
    pred['distance_to_nearest_verified_feature_vector']=np.round(expert_distances,5)
    outcome_delta=(pred['ridge_prediction_floating_outcome'] - pred['graph_prediction_floating_outcome']).abs()
    score_delta=(pred['ridge_prediction_overall_score'] - pred['graph_prediction_overall_score']).abs()/13*3
    pred['model_disagreement']=np.round(outcome_delta+score_delta,5)
    pred['review_priority_score']=np.round(pred.model_disagreement+0.3*expert_distances,5)
    pred['provisional_outcome_class']=np.where(pred.evaluation_kind.eq('unlabelled_inference'),
                  np.clip(np.rint(pred.semisupervised_prediction_floating_outcome),0,3),np.nan)
    pred['is_pseudo_ground_truth']=False
    # Write an all-person table, even if one video view or one feature is absent.
    missing = matrix.loc[~eligible].copy()
    if len(missing):
        missing['label_provenance']=missing.participant_id.apply(
            lambda pid: present.loc[str(pid),'label_provenance'] if str(pid) in present.index else 'not_estimable')
        missing['evaluation_kind']='no_paired_features'
        for col in pred.columns:
            if col not in missing:
                missing[col]=np.nan
        pred=pd.concat([pred,missing[pred.columns]],ignore_index=True)
    pred=pred.sort_values('participant_id').reset_index(drop=True)
    queue=pred.loc[pred.evaluation_kind.eq('unlabelled_inference')].sort_values(
        'review_priority_score',ascending=False).head(30).copy()
    queue['review_status']='pending'
    queue['reviewer_id']=''
    queue['verified_floating_outcome']=''
    queue['verified_overall_score']=''
    queue['review_notes']=''
    methods={'ridge':'Expert-only Ridge','baseline':'Expert mean',
             'graph':'Graph propagation','selftrain':'Low-weight pseudo-label Ridge'}
    report={'method':'four-expert-seed semi-supervised numerical score pilot',
        'seed':seed,'features':list(FEATURES),'rated_participants_available':len(labelled_indices),
        'unlabelled_predicted':int(len(unlabelled_indices)),
        'participant_rows_total':int(len(pred)),
        'not_estimable':int(len(missing)),
        'evaluation':'leave-one-participant-out, all preprocessing Ridge fit within each fold; transductive graph sees unlabelled features but not withheld labels',
        'n_lopo_folds':int(len(labelled_indices)),'fold_overlap':0,
        'methods':methods,'targets':scores,
        'pseudo_label_definition':'model-derived estimates, never expert ground truth',
        'review_queue_size':int(len(queue)),
        'notes':'Four independent expert cases; LOPO scores reflect four test predictions per target.'}
    return {'report':report,'predictions':pred,'folds':pd.DataFrame(folds),
            'review_queue':queue,'trained_models':trained,
            'pseudo_training_ids':pseudo_ids,'graph_preprocessing':{'imputer':imp,'scaler':scaler}}


def describe(result:dict)->str:
    r=result['report']
    lines=['# Supervised and semi-supervised learning: measured results','',
        'Four-score expert seed; expert labels and model-generated estimates are distinct.',
        '', '## Evaluation protocol', r['evaluation'],'',
        f"Eligible independent labelled participants: **{r['rated_participants_available']}**",
        f"Unlabelled participants with predictions: **{r['unlabelled_predicted']}**",
        f"Participants requiring additional paired measurements: **{r['not_estimable']}**",'',
        '## Actual held-out experiment (LOPO)','',
        '| Target | Ridge MAE | Baseline MAE | Graph MAE | Self-training MAE |',
        '|---|---:|---:|---:|---:|']
    for k,desc in [('floating_outcome','Floating outcome 0–3'),('overall_score','Overall score 0–13')]:
        m=r['targets'][k]['lopo_mae']
        lines.append(f"| {desc} | {m['ridge']:.3f} | {m['baseline']:.3f} | {m['graph']:.3f} | {m['selftrain']:.3f} |")
    lines+=['','## Prediction and review evidence','',
      '- `private/all_participant_predictions.csv`: one row per participant, expert LOPO predictions for rated cases and provisional scores for unrated cases.',
      '- `private/lopo_train_test_predictions.csv`: the actual withheld participant and training IDs for every fold.',
      '- `private/participant_review_queue.csv`: 30 unlabelled cases prioritised for human assessment.',
      '- `supervised/semisupervised_metrics.json`: model metrics, experiment settings and sample counts.',
      '- `supervised/trained_expert_and_semi_models.joblib`: fitted numerical models.',
      '', '## Interpretation','',
      'Pseudo-labels are estimates produced using four verified expert examples; they are not ground truth. ',
      'Graph diffusion uses the unlabelled feature matrix transductively while withholding the tested outcome in each evaluation fold. ',
      'The self-training branch uses fixed low-weight pseudo scores and excludes held-out participants from pseudo training.',
      'The 0–3 outcome is treated as an ordered numerical value; rounded classes are proposed labels for review only.',
      '']
    return '\n'.join(lines)


def comparison_figure(result:dict,path:Path)->None:
    """Held-out absolute error comparison, generated from real test predictions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    scores=result['report']['targets']
    methods=[('ridge','Expert-only Ridge'),('baseline','Training mean'),
             ('graph','Graph diffusion'),('selftrain','Pseudo-label Ridge')]
    fig,ax=plt.subplots(figsize=(9.2,4.8))
    pos=np.arange(2)
    width=.19
    for j,(key,label) in enumerate(methods):
        v=[scores[t]['lopo_mae'][key] for t in ('floating_outcome','overall_score')]
        ax.bar(pos+(j-1.5)*width,v,width,label=label)
    ax.set_xticks(pos,['Floating outcome (0–3)','Overall score (0–13)'])
    ax.set_ylabel('Leave-one-participant-out MAE (score points)')
    ax.set_title('Supervised vs semi-supervised test results (expert-rated cases)')
    ax.legend(ncol=2,fontsize=9)
    ax.grid(axis='y',alpha=.18)
    fig.tight_layout()
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path,dpi=160)
    plt.close(fig)