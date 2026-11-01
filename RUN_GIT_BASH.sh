#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
if [[ $# -gt 0 ]]; then
  python run.py --source-root "$1"
else
  python run.py
fi
printf 'Results available at outputs/latest/CLIENT_DASHBOARD.html\n'
