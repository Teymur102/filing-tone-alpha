"""Reproduce everything: python run.py  (downloads are cached in data/, so re-runs are fast)."""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

warnings.filterwarnings("ignore")
from tone.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
