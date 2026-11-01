"""Entrypoint for the portable AI Flotation Analyser research application."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from flotation.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
