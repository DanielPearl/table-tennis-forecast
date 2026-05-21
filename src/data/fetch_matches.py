"""Match-history loader for the table-tennis bot — DISABLED.

There is no free, zero-risk public archive of per-match table tennis
results comparable to ``JeffSackmann/tennis_atp`` (regular tennis) or
``wonderkiduk/darts_data`` (PDC darts).

The closest free GitHub source — ``romanzdk/ittf-data-scrape`` — only
carries weekly ITTF rankings, not match outcomes. Without match
outcomes there is nothing to train a "who wins?" classifier on.

The previous synthetic seed (~6000 simulated matches with fictional
players like "Felix FRA39") has been removed. Calling ``fetch_all``
raises immediately. Re-enable the bot by wiring up a real per-match
source (e.g. a paid BetsAPI subscription that covers TT Elite Series
and ITTF events).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..utils.logging_setup import setup_logging

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


class TableTennisDataUnavailable(RuntimeError):
    """Raised when the bot is asked for training data but no real
    source is configured."""


def fetch_all() -> pd.DataFrame:
    raise TableTennisDataUnavailable(
        "No real table-tennis match data is available from free public "
        "sources. The synthetic seed has been removed. Configure a paid "
        "provider (BetsAPI, Sportradar, etc.) to re-enable training."
    )


def save_clean(matches: pd.DataFrame) -> Path:
    raise TableTennisDataUnavailable(
        "save_clean called but no real data source is configured — "
        "see fetch_matches.fetch_all docstring."
    )


if __name__ == "__main__":
    print(fetch_all())
