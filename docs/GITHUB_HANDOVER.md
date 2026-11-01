# GitHub publishing and reproducibility

From this product folder, create a private GitHub repository and publish its source:

```bash
git init
git branch -M main
git add .github .gitignore README.md run.py pyproject.toml requirements.txt requirements-video.txt RUN_WINDOWS.bat RUN_GIT_BASH.sh EXTRACT_AND_RUN_WINDOWS.bat src tests docs examples START_HERE.txt
git status --short
git commit -m "Release reproducible flotation analysis and modelling software"
git remote add origin https://github.com/OWNER/REPOSITORY.git
git push -u origin main
```

Use the exact repository URL returned by GitHub. Keep the repository private until collaborator and data-sharing permissions are confirmed.

The code repository intentionally excludes `data/`, `outputs/`, `private_inputs/`, `*.mp4`, raw landmark CSVs, model artifacts and participant identifiers. A collaborator reproduces the analysis by cloning the repository, installing dependencies and providing either their own video manifest/videos, a raw33 inventory, or an existing feature batch via `--source-root`.

For a scored model experiment, they also provide an expert label CSV via `--expert-labels` (column specification in `docs/INPUT_AND_OUTPUT_SCHEMA.md`). GitHub Actions runs the automated code tests independently of private datasets.
