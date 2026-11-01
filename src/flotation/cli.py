"""End-to-end orchestration, with fixed outputs and no destructive steps."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

import pandas as pd

from . import __version__
from . import ml, supervised, semisupervised, extract
from .pipeline import (find_existing_batch, load_completed_batch, rebuild_from_raw33,
                       save_csv, save_json, snapshot_provenance, summarize_views)
from .presentation import figures, dashboard, report_text


def complete(args) -> Path:
    repository = Path(__file__).resolve().parents[2]
    source = Path(args.source_root).expanduser().resolve()
    output = Path(args.output).expanduser().resolve() if args.output else repository / 'outputs' / 'latest'
    # A new research dataset may be created in place by the extraction stage.
    if getattr(args, 'video_manifest', None):
        extract.extract_dataset(
            source, args.video_manifest, video_root=args.video_root,
            frame_step=args.frame_step, model_complexity=args.model_complexity,
            max_frames=args.max_frames, overwrite=args.overwrite_landmarks)
    if not source.exists():
        raise FileNotFoundError(f'Dataset directory not found: {source}')
    if shutil.disk_usage(output.parent if output.parent.exists() else repository).free < 150 * (1024**2):
        raise RuntimeError('Less than 150 MiB free. Free disk space before running; no source files were changed.')
    output.mkdir(parents=True, exist_ok=True)
    print('=== FLOTATION ANALYSER: COMPLETE REPRODUCIBLE RUN ===', flush=True)
    print('Source (read-only):', source, flush=True)
    print('Derived outputs:', output, flush=True)

    # Existing final feature batches are authoritative for the current 383-video
    # corpus. Rebuilding is explicit, so the default operation is fast and safe.
    # Import a completed run when available; otherwise analyse the raw33 CSVs.
    batch = find_existing_batch(source)
    if args.rebuild or getattr(args, 'video_manifest', None) or batch is None:
        video, windows, failures, origin = rebuild_from_raw33(source, output)
    else:
        video, windows, failures, origin = load_completed_batch(source, batch)
    quality = summarize_views(video)
    evidence = snapshot_provenance(source, video, windows, failures, origin)
    save_csv(video, output / 'private' / 'video_features.csv')
    save_csv(windows, output / 'private' / 'window_features.csv')
    save_csv(failures, output / 'private' / 'input_failures.csv')
    save_csv(quality, output / 'aggregate' / 'view_quality.csv')
    print(f'Imported {len(video)} video feature rows, {video.participant_id.nunique()} participants, {len(windows)} windows, {len(failures)} missing/failed.', flush=True)

    model = None
    status = {'run': 'not_run', 'reason': None}
    cohort = ml.eligible_cohort(video)
    print('Model eligibility: one top-view video per participant, QC gated =>', len(cohort), flush=True)
    if len(cohort) >= 30:
        model = ml.train_and_evaluate(cohort, seed=args.seed, n_bootstrap=args.bootstrap)
        mldir = output / 'ml'
        mldir.mkdir(parents=True, exist_ok=True)
        save_csv(model['assignments'], output / 'private' / 'participant_model_assignments.csv')
        save_csv(model['selection'], mldir / 'train_k_selection.csv')
        save_csv(model['gmm_selection'], mldir / 'train_gmm_bic_selection.csv')
        save_csv(model['bootstrap'], mldir / 'bootstrap_stability.csv')
        r = model['report']
        r['created_utc'] = datetime.now(timezone.utc).isoformat()
        r['scikit_learn_version'] = __import__('sklearn').__version__
        save_json(r, mldir / 'heldout_metrics.json')
        (mldir / 'ML_EVALUATION.md').write_text(ml.make_markdown(r), encoding='utf-8')
        from joblib import dump
        dump(model['trained_artifacts'], mldir / 'trained_models.joblib')
        # Group profiles are descriptive summaries, not quality labels.
        groups = model['assignments'].groupby(['split','kmeans_cluster'], as_index=False).agg(
            n_participants=('participant_id','nunique'),
            median_arm_activity=('arm_movement_speed_median_torso_s','median'),
            median_body_path_rate=('body_path_rate_median_torso_s','median'),
            median_excursion=('body_max_excursion_torso','median'))
        save_csv(groups, output / 'aggregate' / 'exploratory_group_profiles.csv')
        status = {'run': 'completed', 'cohort': len(cohort), 'holdout': r}
        print(f'ML holdout: {r["train_participants"]} train, {r["test_participants"]} test, overlap {r["participant_overlap"]}; K-means k={r["kmeans"]["selected_k"]}; test silhouette={r["kmeans"]["test_assigned_silhouette"]}', flush=True)
    else:
        status = {'run': 'skipped', 'reason': f'{len(cohort)} eligible participants < 30'}
        print('ML skipped:', status['reason'], flush=True)
    evidence['ml'] = status
    # Expert ratings form a separate, genuinely supervised experiment. This
    # uses participant-level leave-one-out folds, not the unsupervised holdout.
    expert_path = (Path(args.expert_labels).expanduser().resolve()
                   if getattr(args, 'expert_labels', None)
                   else repository / 'private_inputs' / 'chris_expert_labels.csv')
    supervised_result = None
    semi_result = None
    alignment = None
    if expert_path.is_file():
        reviewed_path = (Path(args.reviewed_labels).expanduser().resolve()
                         if getattr(args,'reviewed_labels',None)
                         else repository/'private_inputs'/'reviewed_labels.csv')
        expert = semisupervised.load_combined_labels(expert_path, reviewed_path)
        samples, alignment = supervised.align_expert_intervals(video, windows, expert)
        save_csv(alignment, output / 'private' / 'expert_interval_alignment.csv')
        if len(samples) >= 3:
            supervised_result = supervised.evaluate(samples)
            report = supervised_result['report']
            report['source_name'] = expert_path.name
            report['expert_label_count'] = len(expert)
            report['aligned_count'] = len(samples)
            save_csv(supervised_result['predictions'], output / 'private' / 'expert_heldout_predictions.csv')
            save_csv(supervised_result['aligned_samples'], output / 'private' / 'expert_aligned_features.csv')
            save_json(report, output / 'supervised' / 'supervised_metrics.json')
            (output / 'supervised' / 'SUPERVISED_RESULTS.md').write_text(
                supervised.markdown(supervised_result, alignment), encoding='utf-8')
            from joblib import dump
            dump(supervised_result['trained_models'], output / 'supervised' / 'trained_expert_pilot_models.joblib')
            evidence['supervised'] = {'run':'completed','expert_labels':len(expert),
                   'aligned_participants':len(samples), 'loocv_folds':len(samples),
                   'pilot_metrics':report['targets']}
            print('Expert-supervised LOPO: ' + str(len(samples)) + ' rated participants; '
                  + 'floating outcome MAE=' + str(report['targets']['floating_outcome']['loocv_mae'])
                  + ', overall score MAE=' + str(report['targets']['overall_score']['loocv_mae']),flush=True)
        else:
            evidence['supervised'] = {'run':'not_estimable','expert_labels':len(expert),
                  'aligned_participants':len(samples)}
            print('Expert-supervised: '+str(len(samples))+' aligned expert cases; model not fitted.',flush=True)
    else:
        evidence['supervised'] = {'run':'no_label_input'}
    # A prediction is generated for every participant with an eligible paired
    # 30–60 s measurement. Expert LOPO predictions remain distinct from
    # provisional estimates for the previously unrated participants.
    if expert_path.is_file():
        candidates = semisupervised.participant_matrix(video,windows)
        save_csv(candidates,output/'private'/'all_participant_feature_matrix.csv')
        aligned_ids=set(candidates.loc[candidates.alignment_status.eq('eligible'),'participant_id'])
        n_aligned_labels=sum(str(pid) in aligned_ids for pid in expert.participant_id)
        if n_aligned_labels>=3:
            semi_result=semisupervised.run(candidates,expert,seed=args.seed)
        else:
            evidence['semisupervised']={'run':'not_estimable',
                                       'aligned_labels':int(n_aligned_labels)}
            print('Semi-supervised experiment skipped:',n_aligned_labels,
                  'aligned labels (<3 required minimum).',flush=True)
    if semi_result:
        save_csv(semi_result['predictions'],output/'private'/'all_participant_predictions.csv')
        save_csv(semi_result['folds'],output/'private'/'lopo_train_test_predictions.csv')
        save_csv(semi_result['review_queue'],output/'private'/'participant_review_queue.csv')
        pseudo_rows=[]
        for target,pids in semi_result['pseudo_training_ids'].items():
            for pid in pids:
                pseudo_rows.append({'participant_id':pid,'target':target,
                    'label_source':'model_pseudo_label',
                    'sample_weight':0.25/max(1,len(pids))})
        save_csv(pd.DataFrame(pseudo_rows,columns=['participant_id','target',
                         'label_source','sample_weight']),
                 output/'private'/'pseudo_training_selection.csv')
        save_json(semi_result['report'],output/'supervised'/'semisupervised_metrics.json')
        (output/'supervised'/'PREDICTION_EVALUATION.md').write_text(
            semisupervised.describe(semi_result),encoding='utf-8')
        from joblib import dump
        dump({'models':semi_result['trained_models'],
              'graph_preprocessing':semi_result['graph_preprocessing'],
              'features':list(semisupervised.FEATURES)},
             output/'supervised'/'trained_expert_and_semi_models.joblib')
        evidence['semisupervised']=semi_result['report']
        print('Prediction table: '
              +str(semi_result['report']['participant_rows_total'])+' participants; '
              +str(semi_result['report']['unlabelled_predicted'])+' provisional; '
              +str(semi_result['report']['not_estimable'])+' without eligible paired features.',flush=True)
        save_json(evidence, output / 'RUN_EVIDENCE.json')
    figures(quality, model, output / 'figures')
    if semi_result:
        semisupervised.comparison_figure(semi_result, output/'figures'/'semi_supervised_model_comparison.png')
    if supervised_result:
        supervised.figure(supervised_result, output/'figures'/'supervised_model_vs_baseline.png')
    dashboard(evidence, quality, model, output / 'figures', output / 'CLIENT_DASHBOARD.html', supervised_result, semi_result)
    (output / 'REPORT_RESULTS.md').write_text(report_text(evidence, quality, model, supervised_result)
         + ('\n' + semisupervised.describe(semi_result) if semi_result else ''), encoding='utf-8')
    (output / 'START_HERE.txt').write_text(
        'CLIENT_DASHBOARD.html - open in a web browser (offline)\n'
        'REPORT_RESULTS.md - report-ready text derived from actual run\n'
        'RUN_EVIDENCE.json - record of source and result provenance\n'
        'aggregate/ - non-identifying summary tables\n'
        'figures/ - non-identifying figures\n'
        'ml/ - Unsupervised models, holdout scores and training selection\n'
        'supervised/ - Expert-rated pilot, semi-supervised metrics and trained models\n'
        'private/all_participant_predictions.csv - full prediction table, one row per person\n'
        'private/lopo_train_test_predictions.csv - test folds, test scores and train IDs\n'
        'private/participant_review_queue.csv - prioritised videos for human label review\n'
        'private/pseudo_training_selection.csv - model-generated pseudo samples used in training\n'
        'private/ - PARTICIPANT LEVEL CSVs: do not upload to public GitHub\n',
        encoding='utf-8')
    print('=== COMPLETE ===', flush=True)
    print('Client dashboard:', output / 'CLIENT_DASHBOARD.html', flush=True)
    print('Report results:', output / 'REPORT_RESULTS.md', flush=True)
    print('Unsupervised evidence:', output / 'ml/heldout_metrics.json', flush=True)
    print('Supervised evidence:', output / 'supervised/supervised_metrics.json', flush=True)
    if semi_result:
        print('ALL PREDICTIONS:', output / 'private/all_participant_predictions.csv', flush=True)
        print('TRAIN/TEST TABLE:', output / 'private/lopo_train_test_predictions.csv', flush=True)
        print('REVIEW QUEUE:', output / 'private/participant_review_queue.csv', flush=True)
    print('Analysis complete.', flush=True)
    return output


def main(argv=None) -> int:
    """Run extraction separately or execute the complete workflow."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'extract':
        return extract.main(argv[1:])
    parser = argparse.ArgumentParser(description='End-to-end flotation measurement and modelling')
    parser.add_argument('--source-root', type=Path, default=None,
                        help='Input dataset root containing 33-landmark CSVs, or a completed analysis batch')
    parser.add_argument('--output', type=Path, help='Result directory (default: outputs/latest)')
    parser.add_argument('--rebuild', action='store_true', help='Rebuild features from raw33 CSVs')
    parser.add_argument('--video-manifest', type=Path,
                        help='Extract videos listed in a CSV, then run all analysis stages')
    parser.add_argument('--video-root', type=Path,
                        help='Base folder for relative video paths in the manifest')
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--model-complexity', type=int, choices=(0,1,2), default=1)
    parser.add_argument('--max-frames', type=int, default=None)
    parser.add_argument('--overwrite-landmarks', action='store_true')
    parser.add_argument('--expert-labels', type=Path, default=None,
                        help='Expert-scored outcomes CSV (optional; uses private_inputs by default)')
    parser.add_argument('--reviewed-labels', type=Path, default=None,
                        help='Additional human-reviewed labels CSV')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--bootstrap', type=int, default=30)
    args = parser.parse_args(argv)
    if args.bootstrap < 1 or args.frame_step < 1:
        parser.error('--bootstrap and --frame-step must be >=1')
    repository = Path(__file__).resolve().parents[2]
    if args.source_root is None:
        sibling = repository.parent / 'AI_Flotation'
        args.source_root = sibling if sibling.is_dir() else repository / 'data'
    try:
        complete(args)
        return 0
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}', file=sys.stderr)
        return 1