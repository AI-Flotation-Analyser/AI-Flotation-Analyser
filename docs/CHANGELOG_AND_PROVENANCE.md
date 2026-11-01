# Release notes and method lineage

## Portable full-pipeline release

- Added independent video ingestion with MediaPipe Pose 33-landmark CSV exports, manifest validation, standard metadata inventory and resumable outputs.
- Retained the v1.1 confidence-gated feature and non-overlapping temporal window definitions for compatibility with the existing complete corpus.
- Added automatic raw33 feature reconstruction when no completed feature batch exists.
- Retained participant-held-out K-means/GMM evaluation, expert-rated leave-one-person-out regression, graph propagation, weighted pseudo-label self-training and human review queue.
- Exported fully auditable predictions, model choices, train/test memberships and review records in one result directory.
- Added independent extraction schema tests and cross-platform installation commands.

`RUN_EVIDENCE.json` stores per-run processing and modelling provenance. Research inputs remain outside the tracked source repository.
