"""Match-history loader for the table-tennis bot.

Real-data source: ``ahartness/table-tennis-bet-analyzer`` on GitHub
(see ``ttelite_archive.py``). The upstream archive publishes JSON
snapshots of TT Elite Series betting analysis whose embedded
head-to-head history we parse into a per-match panel.

The previous synthetic ``seed_matches.csv`` shim has been removed —
the bot trains exclusively on real TT Elite matches.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging
from .ttelite_archive import fetch_archive_matches

log = setup_logging("data.fetch_matches")


_EXPECTED_COLS = [
    "match_date", "tournament", "tournament_level", "round", "best_of",
    "winner_name", "loser_name", "winner_hand", "loser_hand",
    "winner_rank", "loser_rank",
    "w_games_won", "l_games_won", "w_points_won", "l_points_won",
    "w_deuce_games_won", "w_deuce_games_played",
    "l_deuce_games_won", "l_deuce_games_played",
    "w_first_game_won", "winner_won_from_down",
]


def fetch_all() -> pd.DataFrame:
    """Return the per-match DataFrame the trainer consumes.

    Pulled from the ahartness/table-tennis-bet-analyzer GitHub
    archive. Schema matches ``_EXPECTED_COLS``; any optional columns
    missing in the upstream data are zero-filled.
    """
    df = fetch_archive_matches()
    for col in _EXPECTED_COLS:
        if col not in df.columns:
            df[col] = 0
    df = df[_EXPECTED_COLS].copy()
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date", "winner_name", "loser_name"])
    df = df.sort_values("match_date").reset_index(drop=True)
    log.info("loaded %d real TT Elite matches from ahartness archive",
              len(df))
    return df


def save_clean(matches: pd.DataFrame) -> Path:
    cfg = load_config()
    out = resolve_path(cfg["paths"]["processed_dir"]) / "matches_clean.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    matches.to_csv(out, index=False)
    log.info("wrote %s (%d rows)", out, len(matches))
    return out


if __name__ == "__main__":
    df = fetch_all()
    save_clean(df)
