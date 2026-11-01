# AI Flotation Analyser

**RMIT University | COSC2667 — Data Science Postgraduate Project | P000308DS**

The **AI Flotation Analyser** is a research project investigating how swimming flotation behaviour can be measured from video using computer vision, pose estimation and machine learning. The system converts body-landmark observations into movement and posture features, explores patterns in those measurements, and compares model predictions with available expert assessments.

**Project team:** Sovannreaksa Chea · Aakash Kolte · Chetan Yadav Venkatesh

## Project overview

The work developed across several stages, from early trials with unlabelled video data to feature refinement, expert-labelled modelling and evaluation. The aim is to provide a consistent way to examine flotation behaviour while recognising the limitations of underwater camera observations.

The project includes:

- **Pose estimation and preprocessing:** MediaPipe-based extraction of 33 body landmarks, camera-view handling and quality checks.
- **Feature engineering:** measurements of torso orientation, body displacement, arm and leg movement, and landmark visibility.
- **Exploratory analysis:** feature selection, PCA and clustering trials to investigate movement patterns before expert labels were available.
- **Machine learning:** K-means and Gaussian mixture comparisons, expert-scored Ridge regression, and semi-supervised modelling experiments.
- **Evaluation and reporting:** participant-level data separation, cross-validation, baseline comparisons, visual summaries and supporting analysis.

## Project results

The final report documents the processing of **383 out of 394 video entries** from **187 participants**, producing **2,625 temporal analysis windows**.

| Analysis | Reported outcome |
| --- | --- |
| Overhead movement grouping | 141 eligible participants; 112 training and 29 held-out |
| K-means clustering | Held-out silhouette: **0.212** |
| Gaussian mixture model | Held-out silhouette: **0.105** |
| K-means stability | Median bootstrap assignment ARI: **0.650** |
| Expert-scored Ridge pilot | Four independently rated participants, evaluated with leave-one-participant-out cross-validation |
| Floating-outcome prediction | MAE **0.767**, compared with a training-mean baseline of **1.333** |
| Overall-score prediction | MAE **2.408**, compared with a training-mean baseline of **4.333** |

These results refer to different experimental stages and data subsets. The clustering results describe movement patterns, **not** independent grades of flotation quality. The four-participant supervised evaluation is an initial pilot, not a population-level performance claim.

## Repository structure

```text
AI-Flotation-Analyser/
├── src/flotation/          Main processing and modelling modules
├── research/three_class/   Additional flotation classification experiments
├── notebooks/exploratory/ Feature selection and modelling notebooks
├── tests/                  Automated software tests
├── docs/                   Methods, source schemas and report mapping
├── run.py                  Main application entry point
├── examples/               Sample input schemas (synthetic data)
├── outputs/latest/         Public aggregate metrics and result summaries
├── requirements.txt        Core Python dependencies
├── requirements-video.txt  Video-processing dependencies
└── CONTRIBUTIONS.md        Team contributions and source attribution
```

The repository brings together the team's processing pipeline, experimental scripts, model-development work and documentation. The `research/three_class/` folder retains additional side-front classification and expert-alignment experiments alongside the principal analysis described in the final report.

## Working with the code

The project is written in Python. To set up a local environment:

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

The main application starts from `run.py`; its data-processing workflow requires an appropriately structured research dataset supplied separately through authorised access. Additional video-processing dependencies are listed in `requirements-video.txt`. The notebooks and research scripts document earlier experiments and model comparisons.

## Results and reproducibility evidence

The repository includes aggregate results and documentation from the team release. Useful starting points are:

- [Full results summary](outputs/latest/REPORT_RESULTS.md)
- [Unsupervised evaluation](outputs/latest/ml/ML_EVALUATION.md) and [held-out metrics](outputs/latest/ml/heldout_metrics.json)
- [Expert-scored pilot](outputs/latest/supervised/SUPERVISED_RESULTS.md) and [supervised metrics](outputs/latest/supervised/supervised_metrics.json)
- [Semi-supervised evaluation metrics](outputs/latest/supervised/semisupervised_metrics.json)
- [Input and output schema](docs/INPUT_AND_OUTPUT_SCHEMA.md)
- [Methods and limitations](docs/METHODS_AND_LIMITATIONS.md)
- [Example input files](examples/)
- [Release evidence and privacy note](docs/RELEASE_EVIDENCE_NOTE.md)

The application was tested on the supplied release fixtures: **31 automated tests passed**. These tests verify software behaviour; reported research metrics require authorised access to the underlying study data to reproduce. The original interactive dashboard, private participant-level tables and fitted model artefacts are not publicly distributed.

## Scope and limitations

The project is intended for **research and analysis**, not for making swimming-safety decisions. Camera occlusion, underwater distortion and missing landmarks can affect measurements. In particular, body-pose information alone cannot establish whether a swimmer's airway is clear.

Original participant videos, identifiable landmark records, expert label files and participant-level model outputs are not included in this public repository to protect research data privacy.

## Team and project documentation

This project was completed by **Sovannreaksa Chea, Aakash Kolte and Chetan Yadav Venkatesh**. For the project workstreams and code attribution, see [CONTRIBUTIONS.md](CONTRIBUTIONS.md). For connections between the report and the source files, see [docs/REPORT_MAPPING.md](docs/REPORT_MAPPING.md).

**Institution:** RMIT University  
**Project:** P000308DS — AI Flotation Analyser  
**Course:** COSC2667 — Data Science Postgraduate Project
