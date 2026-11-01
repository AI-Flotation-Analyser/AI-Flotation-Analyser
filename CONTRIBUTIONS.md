# Contributions and code provenance

**Official final team (three members, per report dated 1 November 2026):**
- **Sovannreaksa Chea** — pose extraction experiments, measurement and landmark quality control, batch processing/recovery, feature analyses and integration of the portable processing pipeline.
- **Aakash Kolte** — computer vision and video/pose workstream, checking camera-dependent landmarks and visual outputs, MediaPipe feature-extraction trials, research review and technical progress updates; contributed to discussions and tests after the expert-defined targets were available.
- **Chetan Yadav Venkatesh** — side-front three-class passive-flotation research: target definitions, feature selection, labelling, grouped model comparison, frozen testing and expert-alignment model scripts. Original source: [vchetanyadav/swim_analysis](https://github.com/vchetanyadav/swim_analysis).

These are **team-reported workstreams**, not independently audited line-by-line authorship. Commit dates for this repository show consolidation and publication, not original authorship. Refer to the individual weekly logs and report reflections for credit; each member should confirm named files before final assessment.

## Distinct implementations

- **Main report deliverable:** portable 33-landmark processing pipeline, held-out K-means/GMM, expert-scored four-person Ridge LOPO study, semi-supervised workflow, dashboard. Local team release is required to reproduce these results.
- **Separate Chetan research:** `research/three_class/` includes historical three-class side-front classification and expert-rubric experiments. Its 467 clip labels and 0.66 sealed-test macro-F1 must **not** be conflated with the main report's four-person pilot.

## External source credit

The three-class sources were copied from `vchetanyadav/swim_analysis` at commit `4345025af894030513dd3e61035e6ffa9d30610e`. Contact original authors regarding permission for reuse or relicensing. No open-source licence is asserted for that source.

## Privacy

No participant videos, raw pose rows, identity-linked labels, private model weights or personally identifiable expert records belong in this public repository.
