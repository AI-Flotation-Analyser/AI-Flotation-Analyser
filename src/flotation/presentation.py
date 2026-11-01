"""Static, offline client report and non-identifying aggregate figures."""
from __future__ import annotations

import base64
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def figures(view_summary: pd.DataFrame, ml: dict, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    vs = view_summary.sort_values('camera_view')
    xs = np.arange(len(vs))
    fig, ax = plt.subplots(figsize=(8.3, 4.7))
    ax.bar(xs - .23, vs.arm_feature_available / vs.video_count * 100, width=.44, label='Arm')
    ax.bar(xs + .23, vs.leg_feature_available / vs.video_count * 100, width=.44, label='Leg')
    ax.set_xticks(xs, vs.camera_view.str.replace('_', '/', regex=False))
    ax.set_ylim(0, 105)
    ax.set_ylabel('Recordings with reportable features (%)')
    ax.set_title('Measurement availability by camera')
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / 'camera_coverage.png', dpi=150)
    plt.close(fig)

    if not ml:
        return
    assign = ml['assignments']
    fig, ax = plt.subplots(figsize=(8.3, 5.5))
    for c in sorted(assign.kmeans_cluster.unique()):
        for split, mark in (('train', 'o'), ('test', 'X')):
            d = assign[(assign.kmeans_cluster == c) & (assign.split == split)]
            if len(d):
                ax.scatter(d.train_pca_pc1, d.train_pca_pc2, marker=mark,
                           s=25 if split == 'train' else 70,
                           alpha=.8, label=f'Group {c} / {split}: {len(d)}')
    ax.set_title('Participant-held-out exploratory group assignments')
    ax.set_xlabel('PC1 (training-fitted display)')
    ax.set_ylabel('PC2 (training-fitted display)')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / 'heldout_groups.png', dpi=150)
    plt.close(fig)

    sel = ml['selection']
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    ax.bar(sel.k.astype(str), sel.train_silhouette, label='Training silhouette')
    ax.set_ylabel('Silhouette')
    ax.set_xlabel('Candidate number of groups (k)')
    ax.set_title('K-means selection (training participants only)')
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(directory / 'train_k_scan.png', dpi=150)
    plt.close(fig)


def _img(path: Path, label: str) -> str:
    if not path.exists():
        return ''
    contents = base64.b64encode(path.read_bytes()).decode('ascii')
    return f'<img loading="lazy" alt="{escape(label)}" src="data:image/png;base64,{contents}">'


def dashboard(provenance: dict, quality: pd.DataFrame, ml: dict | None, figure_dir: Path, output: Path, expert: dict | None = None, semi: dict | None = None):
    """Single self-contained HTML, suitable for offline partner demonstration."""
    n = provenance['processed_videos']
    inventory_n = provenance.get('source', {}).get('existing_evidence', {}).get('source_pipeline_manifest', {}).get('selected_inventory_videos')
    if inventory_n is None:
        inventory_n = n + provenance['missing_or_failed_entries']
    cells = []
    for _, r in quality.iterrows():
        cells.append('<tr>'+''.join(f'<td>{escape(str(v))}</td>' for v in (
            r.camera_view, int(r.video_count), f'{r.median_torso_valid_fraction * 100:.1f}%',
            f'{int(r.arm_feature_available)}/{int(r.video_count)}',
            f'{int(r.leg_feature_available)}/{int(r.video_count)}',
            int(r.unilateral_arm_records), int(r.review_required)))+'</tr>')
    if ml:
        r = ml['report']
        km = r['kmeans']; gm = r['gmm']
        def metric(x):
            return 'Not estimable' if x is None else f'{x:.3f}'
        model_html = f'''<section id="model"><h2>Machine-learning evaluation</h2>
        <p>Unsupervised clustering of eligible top-camera, one-video-per-person measurements. All model selection and scaling used training participants only. A held-out participant never contributed to model fitting.</p>
        <div class="tiles"><div class="tile"><small>Eligible participants</small><strong>{r['cohort_participants']}</strong></div>
        <div class="tile"><small>Training / held-out</small><strong>{r['train_participants']} / {r['test_participants']}</strong></div>
        <div class="tile"><small>Selected K-means groups</small><strong>{km['selected_k']}</strong></div></div>
        <table><thead><tr><th>Model</th><th>Training grouping score</th><th>Unseen participant silhouette</th></tr></thead><tbody>
        <tr><td>K-means (training silhouette)</td><td>{metric(km['train_silhouette'])}</td><td>{metric(km['test_assigned_silhouette'])}</td></tr>
        <tr><td>Gaussian mixture (training BIC selected)</td><td>{metric(gm['train_silhouette'])}</td><td>{metric(gm['test_assigned_silhouette'])}</td></tr>
        </tbody></table><p>Bootstrap held-out assignment ARI (median): <strong>{metric(km['bootstrap_test_ari_median'])}</strong>. This is stability of feature grouping, not accuracy against flotation quality.</p>
        {_img(figure_dir/'heldout_groups.png', 'Held-out exploratory assignments')}
        {_img(figure_dir/'train_k_scan.png', 'Training-only k scan')}
        </section>'''
    else:
        model_html = '<section><h2>Machine learning</h2><p>The eligible independent-participant cohort was too small for the configured experiment; details are in the run log.</p></section>'
    # Expert-supervised results are displayed separately from clustering.
    if expert:
        e=expert['report']
        fo=e['targets']['floating_outcome']; overall=e['targets']['overall_score']
        supervised_html=f'''<section id="supervised"><h2>Expert-supervised pilot (Chris's scores)</h2>
        <p>Leave-one-participant-out Ridge regression trained against the 30–60 s expert-assessed interval. A different participant is excluded from training in each fold. Model inputs are three pose-derived activity and body-position measurements.</p>
        <div class="tiles"><div class="tile"><small>Expert-rated participants</small><strong>{e['participant_count']}</strong></div>
        <div class="tile"><small>Independent test folds</small><strong>{e['fold_count']}</strong></div>
        <div class="tile"><small>Training/test overlap</small><strong>{e['participant_leakage']}</strong></div></div>
        <table><thead><tr><th>Expert target</th><th>Model LOPO MAE</th><th>Training-mean baseline MAE</th></tr></thead><tbody>
        <tr><td>Floating outcome (0–3)</td><td>{fo['loocv_mae']:.3f}</td><td>{fo['baseline_loocv_mae']:.3f}</td></tr>
        <tr><td>Overall score (0–13)</td><td>{overall['loocv_mae']:.3f}</td><td>{overall['baseline_loocv_mae']:.3f}</td></tr>
        </tbody></table><p>Four expert-rated participants constitute a model-development pilot. This evaluates predictions against Chris's ratings, distinct from the 141-participant unsupervised movement grouping.</p>
        {_img(figure_dir/'supervised_model_vs_baseline.png', 'Supervised model versus baseline mean absolute error')}</section>'''
    else:
        supervised_html='<section id="supervised"><h2>Expert-supervised pilot</h2><p>Expert case alignment or labels are not available for training in this run.</p></section>'
    semi_html=''
    if semi:
        r=semi['report']
        m1=r['targets']['floating_outcome']['lopo_mae']
        m2=r['targets']['overall_score']['lopo_mae']
        rows=[]
        for _,row in semi['predictions'].iterrows():
            columns=['participant_id','evaluation_kind','actual_floating_outcome',
                'semisupervised_prediction_floating_outcome','graph_prediction_floating_outcome',
                'actual_overall_score','semisupervised_prediction_overall_score',
                'review_priority_score']
            def v(k):
                val=row.get(k,'')
                if pd.isna(val): return '—'
                if isinstance(val,(float,np.floating)):return f'{val:.2f}'
                return escape(str(val))
            rows.append('<tr data-kind="'+escape(str(row['evaluation_kind']))+'">'
                       + ''.join('<td>'+v(k)+'</td>' for k in columns)+'</tr>')
        summary=''.join(
            f'<tr><td>{escape(name)}</td><td>{m["ridge"]:.3f}</td><td>{m["baseline"]:.3f}</td>'
            f'<td>{m["graph"]:.3f}</td><td>{m["selftrain"]:.3f}</td></tr>'
            for name,m in [('Outcome score 0–3',m1),('Overall score 0–13',m2)])
        semi_html=f'''<section id="predictions"><h2>Supervised, semi-supervised and prediction table</h2>
        <p>Four expert-scored participants anchor the models. Each model is evaluated with
        a genuine leave-one-participant-out test. All unrated participant results are
        provisional estimates for assessment, not additional expert ratings.</p>
        <div class="tiles"><div class="tile"><small>Rated test participants</small><strong>{r['rated_participants_available']}</strong></div>
        <div class="tile"><small>Provisional predictions</small><strong>{r['unlabelled_predicted']}</strong></div>
        <div class="tile"><small>Unavailable paired features</small><strong>{r['not_estimable']}</strong></div>
        <div class="tile"><small>Human-review queue</small><strong>{r['review_queue_size']}</strong></div></div>
        <h3>Held-out supervised evaluation — mean absolute error</h3>
        <table><thead><tr><th>Target</th><th>Expert-only Ridge</th><th>Train mean</th><th>Graph diffusion</th><th>Pseudo-label self-training</th></tr></thead><tbody>{summary}</tbody></table>
        <p>Each target has {r['n_lopo_folds']} withheld test predictions. Training IDs and actual predictions are saved to
        <code>private/lopo_train_test_predictions.csv</code>.</p>
        {_img(figure_dir/'semi_supervised_model_comparison.png','Expert-only vs pseudo-label test scores')}
        <h3>Participant-level prediction table ({r['participant_rows_total']} rows)</h3>
        <div class="tools"><input id="search-pid" type="search" placeholder="Search participant ID or status" oninput="filterPredictions()" aria-label="Search predictions">
        <select id="filter-kind" onchange="filterPredictions()" aria-label="Prediction category"><option value="">All participants</option>
        <option value="leave_one_person_out_test">Expert-labelled test cases</option><option value="unlabelled_inference">Provisional predictions</option>
        <option value="no_paired_features">Missing paired measurements</option></select>
        <button onclick="exportPredictionTable()">Export visible rows (CSV)</button></div>
        <div class="table-scroll"><table id="all-predictions"><thead><tr><th>Participant</th><th>Prediction source</th>
        <th>Expert outcome</th><th>Predicted outcome</th><th>Graph outcome</th><th>Expert total</th><th>Predicted total</th><th>Review priority</th></tr></thead>
        <tbody>{''.join(rows)}</tbody></table></div><p id="shown-count" class="minor"></p>
        <p>Full-precision outputs: <code>private/all_participant_predictions.csv</code>. The separate
        <code>private/participant_review_queue.csv</code> prioritises 30 cases for human labelling.
        For rated people the table shows <em>held-out</em> predictions; for unrated people it
        shows models fitted to all available verified labels.</p></section>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Flotation | Research Dashboard</title><style>
    :root{{--ink:#162944;--blue:#214e80;--bg:#f4f7fb;--muted:#566479;--line:#d9e1e9}}
    body{{font:15px system-ui,Segoe UI,sans-serif;margin:0;background:var(--bg);color:var(--ink);line-height:1.65}}
    header{{padding:32px max(calc((100vw - 1020px)/2),24px);background:var(--blue);color:white}}
    header h1{{margin:0;font-size:30px}}nav{{display:flex;gap:25px;flex-wrap:wrap;margin-top:15px}}nav a{{color:white;text-decoration:underline}}
    main{{max-width:1020px;margin:24px auto;padding:0 18px 48px}}section{{background:white;border:1px solid var(--line);padding:24px;margin:18px 0;border-radius:12px}}
    h2{{margin:0 0 15px;font-size:21px}}.tiles{{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px;margin:15px 0}}
    .tile{{background:#f0f5fa;padding:15px;border-radius:8px}}.tile strong{{font-size:25px;display:block}}.tile small{{color:var(--muted)}}
    table{{border-collapse:collapse;width:100%;overflow:auto;display:block}}td,th{{text-align:left;border-bottom:1px solid var(--line);padding:10px}}th{{background:#f3f6fa}}
    img{{display:block;max-width:100%;height:auto;margin:12px auto 24px}}p{{max-width:85ch}}.minor{{color:var(--muted)}}
    footer{{color:var(--muted);font-size:13px;text-align:center;padding:20px}}@media print{{header{{background:white;color:var(--ink)}}nav{{display:none}}section{{break-inside:avoid}}}}
    .tools{{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}}
    .tools input,.tools select,.tools button{{padding:9px 12px;border:1px solid #cad5e0;border-radius:7px;background:white;color:#162944}}
    .tools button{{cursor:pointer;background:#214e80;color:white}}
    .table-scroll{{overflow-x:auto;max-height:500px;overflow-y:auto}}
    #all-predictions{{font-size:13px;min-width:900px}}
    #all-predictions th{{position:sticky;top:0}}
    </style></head><body><header><h1>AI Flotation Analyser</h1><p>Multi-view movement analysis — research prototype</p>
    <nav><a href="#coverage">Dataset and coverage</a><a href="#model">Unsupervised</a><a href="#supervised">Expert-supervised</a><a href="#predictions">Prediction table</a><a href="#caveats">Scientific limitations</a></nav></header><main>
    <section id="coverage"><h2>Dataset and quality assurance</h2>
    <div class="tiles"><div class="tile"><small>Video entries analysed</small><strong>{n} / {inventory_n}</strong></div>
    <div class="tile"><small>Unique participants</small><strong>{provenance['participants']}</strong></div>
    <div class="tile"><small>Temporal measurement windows</small><strong>{provenance['temporal_windows']}</strong></div>
    <div class="tile"><small>Missing or failed entries</small><strong>{provenance['missing_or_failed_entries']}</strong></div></div>
    <p>These statistics describe technical feature availability, not human flotation performance. A pose detection can be confident yet anatomically incorrect.</p>
    <table><thead><tr><th>Camera</th><th>Videos</th><th>Median torso tracking</th><th>Arms available</th><th>Legs available</th><th>Unilateral arms</th><th>QC review</th></tr></thead><tbody>{''.join(cells)}</tbody></table>
    {_img(figure_dir/'camera_coverage.png','Arm and leg measurement availability')}</section>
    {model_html}
    {supervised_html}
    {semi_html}
    <section id="caveats"><h2>Interpretation and limits</h2><p>This dashboard characterises image-space body posture and movement. It does <strong>not</strong> determine whether a participant floats safely, has their airway above water, or has a high flotation-quality score.</p>
    <p>Observed errors include left/right landmark identity swaps, low ankle coverage, camera projection effects and entry/recovery movement. The speed screening threshold is provisional; the selected groups have not been independently assessed as distinct flotation strategies by the domain experts.</p>
    <p>One participant can contribute multiple video entries and windows. Model training and evaluation use disjoint participant IDs and one eligible overhead video per participant.</p></section>
    </main><footer>Locally generated research output. Participant-level files remain private and are excluded from the public repository.</footer>
    <script>
    function filterPredictions() {{
      const phrase=document.getElementById('search-pid').value.toLowerCase();
      const kind=document.getElementById('filter-kind').value;
      const rows=[...document.querySelectorAll('#all-predictions tbody tr')];
      let shown=0;
      for(const row of rows){{
        const match=(!kind || row.dataset.kind===kind) && row.innerText.toLowerCase().includes(phrase);
        row.style.display=match?'':'none'; if(match)shown++;
      }}
      document.getElementById('shown-count').textContent=shown+' participants shown';
    }}
    function exportPredictionTable() {{
      const rows=[...document.querySelectorAll('#all-predictions tr')].filter(r=>r.style.display!=='none');
      const value=t=>'"'+t.replaceAll('"','""')+'"';
      const csv=rows.map(r=>[...r.querySelectorAll('th,td')].map(c=>value(c.innerText.trim())).join(',')).join('\\n');
      const url=URL.createObjectURL(new Blob([csv],{{type:'text/csv;charset=utf-8'}}));
      const link=document.createElement('a');link.href=url;link.download='visible_flotation_predictions.csv';link.click();
      URL.revokeObjectURL(url);
    }}
    if(document.getElementById('search-pid'))filterPredictions();
    </script></body></html>'''
    output.write_text(html, encoding='utf-8')


def report_text(provenance: dict, quality: pd.DataFrame, ml: dict | None, expert: dict | None = None) -> str:
    viewrows = '\n'.join('| {camera_view} | {video_count} | {median_torso_valid_fraction:.2%} | {arm_feature_available} | {leg_feature_available} |'.format(**r)
                     for r in quality.to_dict('records'))
    text = f'''# Results — AI Flotation Analyser\n\n## Study data and processing\n\nThe inventory yielded **{provenance['processed_videos']} analysed recordings** from **{provenance['participants']} participants**, with **{provenance['temporal_windows']} non-overlapping temporal windows**. **{provenance['missing_or_failed_entries']}** entries were missing or failed. Distinct recordings and temporal windows are not independent participant-level samples.\n\n| Camera view | Recordings | Median valid torso frames | Reported arm metrics | Reported leg metrics |\n|---|---:|---:|---:|---:|\n{viewrows}\n\nFrame-level body geometry is computed from MediaPipe 33-landmark estimates. Wrist and ankle velocities are body-relative, normalised to torso length per second; image-space trunk orientation is measured against image horizontal, **not** a calibrated waterline. A provisional 10 torso-lengths/s rule flags suspect limb movements; low-coverage measurements are withheld. QC flags are not clinical or flotation assessments.\n\n'''
    if ml:
        r = ml['report']; km=r['kmeans']; gm=r['gmm']
        fmt=lambda a: 'not estimable' if a is None else f'{a:.3f}'
        text += f'''## Unsupervised machine-learning experiment\n\nThe eligible cohort contained **{r['cohort_participants']} unique participants**, each represented by one quality-gated top-view video. An 80/20 participant-disjoint split allocated **{r['train_participants']} to training** and **{r['test_participants']} to testing**, with **{r['participant_overlap']} overlapping IDs**. Log1p transformation and standardisation were fitted exclusively on the training observations. K-means k=2–5 was selected using training silhouette and minimum cluster-size constraints; Gaussian mixtures were compared using training BIC selection. The fitted models were then evaluated on previously unseen participants.\n\n| Metric | K-means | Gaussian mixture |\n|---|---:|---:|\n| Selected groups (training selection) | {km['selected_k']} | {gm['selected_k_train_bic']} |\n| Training silhouette | {fmt(km['train_silhouette'])} | {fmt(gm['train_silhouette'])} |\n| Held-out assigned silhouette | **{fmt(km['test_assigned_silhouette'])}** | **{fmt(gm['test_assigned_silhouette'])}** |\n\nBootstrapped training-resample assignment stability against the fixed held-out participants (median adjusted Rand index) was **{fmt(km['bootstrap_test_ari_median'])}**, based on {km['bootstrap_repetitions']} resamples. The selected number of groups may differ from earlier full-cohort exploratory clustering because model selection now excludes test participants. These internal and stability metrics measure feature-group structure, **not prediction accuracy, floating-quality validity or airway security**.\n\n'''
    else:
        text += '## Unsupervised machine learning\n\nA participant-held-out evaluation was not performed because the eligible cohort was insufficient or the model stage was disabled.\n\n'
    if expert:
        from .supervised import markdown
        text += '\n' + markdown(expert, pd.DataFrame([{'trial_pair_status':'matched_camera_pair'}]*expert['report']['participant_count'])) + '\n'
    text += '''## Evaluation boundaries and further research\n\nThe P2 pilot identified visually apparent arm identity swaps and misplaced ankle landmarks in selected frames. Selected failure examples are not an unbiased error-rate estimate. The four expert-rated participants support a small supervised regression pilot with leave-one-participant-out evaluation; the session match remains an explicit assumption. The present system does not compute waterline or submersion-event duration.\n\nThe exploratory clusters may reflect pose-tracking errors, recording phase, or camera movement as well as real strategies. Independent anatomical annotation, waterline calibration, larger labelled cohorts, and expert reviews of group exemplars are needed for construct validation.\n\n## Reproduction and handover\n\nThe repository contains code, tests, requirements and methods; the source videos, 33-landmark CSVs, participant-level exports and fitted participant assignments stay on approved research storage. Reproduce with `python run.py --source-root ../AI_Flotation` from the repository directory. Do not upload personal data, raw CSVs, video frames or participant-specific outputs to a public repository.\n'''
    return text