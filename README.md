# AI Flotation Analyser

**RMIT COSC2667 | Data Science Postgraduate Project | P000308DS**

Research prototype for multi-view pose estimation, movement analysis and participant-level modelling of flotation behaviour.

**Project team:** Sovannreaksa Chea, Aakash Kolte, Chetan Yadav Venkatesh.

## What the final report describes

- MediaPipe 33-landmark video processing and camera-specific quality checks.
- Unsupervised overhead movement grouping: participant-disjoint K-means and Gaussian mixture evaluation.
- Four-person expert-scored Ridge regression pilot using leave-one-participant-out evaluation.
- Semi-supervised graph/pseudo-label experiments and review-queue design.
- Reproducible command-line application and offline dashboard.

The report describes **383 processed videos from 187 participants**, **2,625 analysis windows**, a **112/29 overhead clustering split**, and a separate **four-person expert pilot**. These counts refer to different data units.

## Code availability

The original three-class classifier and validation work is maintained in [vchetanyadav/swim_analysis](https://github.com/vchetanyadav/swim_analysis). It is a **separate experiment** and its 0.66 sealed-test macro-F1 must not be confused with the four-person Ridge pilot metrics in the final team report.

This repository is being assembled from the shared release ZIP, separate modelling notebooks and the upstream classifier. **Check the commit history before treating a listed component as uploaded.**

## Running the portable team release

The team release's canonical entry point is `python run.py --source-root /path/to/study`. It uses `src/flotation/`, `tests/`, `requirements.txt` and `requirements-video.txt`. If those files are not yet visible in this repository, use the bundled release archive supplied by the team.

## Ethics and limits

No identifiable participant videos, raw landmark CSVs, expert label tables, participant-level predictions or trained model binaries are to be published here. Underwater pose cannot establish airway clearance. Model predictions are research outputs, not safety assessments.

## Verification and contributions

See `CONTRIBUTIONS.md` for carefully distinguished authorship. Code upload under Aakash's account does **not** imply that all code was written by Aakash.
