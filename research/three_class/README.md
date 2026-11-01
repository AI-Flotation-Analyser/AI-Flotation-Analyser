# Side-front three-class research — Chetan workstream

This folder retains the relevant **historical research scripts**, copied from [vchetanyadav/swim_analysis](https://github.com/vchetanyadav/swim_analysis) at commit `4345025af894030513dd3e61035e6ffa9d30610e`.

## Pipeline

1. Early exploratory and no-expert-label work: `swim_cluster*.py`, `swim_classify.py`.
2. Video/pose processing: `inspect_video.py`, `build_manifest.py`, `swim_features.py`, `mp_features.py`.
3. Building labels and participant splits: `label_ui_v3.html`, `label_ui_expert.html`, `select_*.py`, `build_training_labels.py`, `make_splits.py`.
4. Three-class passive flotation model: `swim_train.py` and `swim_test.py`.
5. Expert-aligned comparison: `swim_multi.py`, `predict_multi.py`, `expert_agreement.py`, `irr_kappa.py`.

Two original scripts (`swim_predict.py` and `select_test.py`) were not copied successfully during the connector upload. They remain available in the original repository; this directory is therefore **not a standalone reproduction** of every upstream command.

The historical results are reported in the team's documentation and have not been reproduced using public participant-level data. Child video, annotation tables, fitted models and individual predictions are intentionally excluded. There is no assumption that the original code is openly licensed for redistribution.

**Important:** the official 1 November three-member team report focuses on a **different portable 33-landmark application** (unsupervised holdout plus a four-person expert-scored Ridge LOPO pilot). Do not present the three-class 467-clip study as the source of its pilot MAE results.
