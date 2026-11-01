# Methods, decisions and limitations

## Scope and source

The system begins **after** MediaPipe Pose has exported a CSV containing all 33 landmarks for each frame. The raw33 extraction stage is deliberately not rerun. Source data remain under the existing private research project. Completed per-video v1.1 features may be imported directly; the raw33 rebuild path provides reproducibility and caching.

Research questions: *Can camera-view-specific features characterize different flotation behaviours?* And *can an expert-rated pilot train a supervised model of the defined floating-outcome and overall-score scales?* The clustering and supervised pilot are analytically separate.

## Quantitative feature definitions

- **Torso geometry:** average valid left/right shoulders and hips; unilateral torso fallback is possible. A torso length less than 1.5% of the shorter frame dimension is rejected.
- **Landmark eligibility:** `pose_detected` true, visibility ≥0.5, presence ≥0.5, finite coordinates. Missing observations are not interpolated.
- **Smoothing:** five-frame centred median, preserving missing positions.
- **Side/front trunk angle:** angle of shoulder–hip vector to the **image horizontal**, folded into 0–90°. It is **not** an angle relative to a detected waterline.
- **Head proxy:** nose minus shoulder image-y coordinate in torso lengths; it is not head rotation, airway status, or submersion duration.
- **Relative limb activity:** frame-adjacent displacement of wrist/ankle *relative to estimated torso centre* divided by frame-to-frame seconds and by median torso length. Changes due to body rotation and pose swaps are not completely removed.
- **Screening:** candidate speeds above 10 torso lengths/s are marked and excluded from summary estimates. The threshold is a provisional engineering screen, **not** a validated physiological upper bound.
- **Coverage:** limb speed medians are reported only when at least 60% of frames support the screened speed and at least one limb side has ≥60% coverage. Left-only/right-only/bilateral status is retained for comparability.
- **Top-view drift:** camera-relative body centre displacement from an early reference, and image-plane body-centre path rate, normalized by torso length. No metric is claimed in metres.
- **Temporal resolution:** non-overlapping 30-second windows; incomplete intervals shorter than 10 seconds are excluded. Windows are not assumed to align with expert ratings.

## Model design and evaluation

The **overhead view** is selected to avoid mixing camera-specific feature definitions. Only one recording per participant enters the clustering cohort. Eligibility: bilateral arms, torso validity ≥80%, usable arm speed ≥60%, arm screen fraction ≤5%; all three model features finite and nonnegative. For participants with multiple valid overhead recordings, select the longest, then lexicographically smallest `video_id` for deterministic tie-breaking.

An 80/20 grouped split by participant ID ensures no participant appears in both datasets. Preprocessing is fitted only on training data. Select K-means group count by training silhouette, subject to a minimum group-size rule; Gaussian mixture components are selected on training BIC. Test individuals are assigned to already fitted models, and test silhouette is measured **without refitting**. Compare assignments under 30 training bootstrap resamples with adjusted Rand index on the *same untouched* test participants.

The bootstrap ARI assesses stability of assigned feature groups, not correctness, and the Gaussian mixture is a comparison model rather than a separately tuned clinical predictor. Results depend on the seed, selection gates, features and video representation. This is a **retrospective** holdout: the three feature choices and gates were developed earlier using the full project corpus, so it is not a fully independent prospective evaluation.

Silhouette scores are **internal** clustering diagnostics, not supervised classification accuracy. Cluster IDs are nominal, not an ordering of ability. A moderate silhouette does not demonstrate domain-valid flotation strategies.

## Scientific validity and unaddressed requirements

- MediaPipe landmarks can swap left/right identity or locate feet near the upper body; high detection coverage does not imply anatomical correctness.
- Observed swimming/flotation movements may include water entry and recovery, not only sustained flotation.
- Side-front and overhead videos differ by projection; directly comparing normalized speed across views is not valid without cross-view calibration.
- The system cannot determine mouth/nose versus waterline, sustained submersion or airway security. The 33 landmarks alone are insufficient.
- Arm/leg **frequency**, coordinated cycling, underwater head dipping, hand-specific sculling, and rotations are *not* validated outcomes of the current feature engine. Do not present them as completed without new scientific validation.
- Chris provided four expert-rated participants (P2, P68, P81, P118), assessed over 30–60 s. A separate supervised Ridge regression pipeline now aligns and tests the two numerical targets using leave-one-participant-out cross-validation. Its case-level errors do not constitute generalised flotation-quality validation. The workbook does not identify the precise recorded session (particularly relevant for P81); the trial pairing is based on filename matching. See `docs/SUPERVISED_EXPERT_PROTOCOL.md`.
- The selected P2 overlay pilot provides evidence of some tracking failure modes but is not a representative population-level benchmark.

## Practical follow-up if additional research time becomes available

Independently annotate a stratified sample of frames, check landmark-to-video alignment, calibrate the waterline, align expert ratings to sessions and windows, audit clusters with domain experts, and assess sensitivity across viewpoint, recording phase and track-confidence thresholds. Keep this separate from the already completed engineering and held-out feature-grouping deliverable.


## Semi-supervised addition (v3)

The system also generates provisional floating-outcome and overall-score estimates for previously unlabelled people using Chris's four anchors. Fixed-parameter graph label diffusion and lightly weighted self-training are evaluated by withholding each expert participant's label in turn. Those provisional scores are review candidates, not expanded expert ground truth. For independent fold-level training IDs, withheld predictions, and sample selection, see `docs/SEMI_SUPERVISED_PROTOCOL.md`.
