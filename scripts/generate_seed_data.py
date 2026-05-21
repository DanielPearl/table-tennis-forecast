"""DEPRECATED — synthetic seed generator, no longer in use.

This script previously generated ``data/raw/seed_matches.csv`` with
~6000 fictional table-tennis matches (skill draws + simulated game
scores). It has been removed because we now only use real data.

There's no free zero-risk public archive of per-match TT results (the
romanzdk/ittf-data-scrape repo only has rankings). Configure a paid
provider to re-enable training; until then the bot is disabled.
"""
from __future__ import annotations

import sys


def main() -> int:
    raise SystemExit(
        "generate_seed_data.py is deprecated. No real free source of "
        "per-match TT data exists. Configure a paid provider (BetsAPI, "
        "Sportradar, etc.) to re-enable training."
    )


if __name__ == "__main__":
    sys.exit(main())
