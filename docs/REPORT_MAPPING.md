# Report-to-code map

This map follows **AI_Flotation_Final_Report_3_Member_Team.docx (1 November 2026)**.

| Report section | Reproducibility component |
| --- | --- |
| §3.2 Video and raw33 ingestion | Portable team release: `run.py`, `src/flotation/extract.py`, `pipeline.py` |
| §3.3 QC and feature engineering | `src/flotation/features.py`; documented 33-keypoint schema |
| §4.1 Overhead clustering comparison | `src/flotation/ml.py`: participant grouped K-means/GMM |
| §4.2–4.3 Expert alignment and Ridge LOPO | `src/flotation/supervised.py` |
| §4.4 Graph and pseudo-labelled learning | `src/flotation/semisupervised.py` |
| §4.5 Testing, dashboard | `tests/`, `src/flotation/presentation.py` |
| Appendix C — Chetan's alternate three-class trial | `research/three_class/swim_train.py`, `swim_test.py`, `swim_multi.py` |
| Aakash's two exploratory notebooks | `notebooks/` in team release, pending direct upload |

**Evidence distinction:** The report's primary final-corpus analysis covers 383/394 video entries, 187 participants and 2,625 30-second windows. The original three-class research uses separate 3-second overlapping windows and a different label set; the numbers must not be added or treated as identical experiments.

**Availability:** This text index does not imply that the portable team release is already present on GitHub. Check individual paths. The main report application must be committed independently from the supplied clean ZIP.
