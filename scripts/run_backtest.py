"""Backtest entrypoint. Loads the trained bundle, evaluates on the
held-out window, and writes data/outputs/backtest_results.csv."""
from __future__ import annotations

import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from src.trading.backtest import run

if __name__ == "__main__":
    run()
