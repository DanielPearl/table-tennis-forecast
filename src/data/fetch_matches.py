"""Match-history loader for the table-tennis bot.

Unlike tennis (where the Sackmann GitHub mirrors give us free, current
data), table tennis has no equivalent free historical dataset that's
trivially scrapable. We support two sources:

  1. A bundled seed CSV at ``data/raw/seed_matches.csv`` (committed to
     the repo). Synthetic but realistic — generated from a fixed
     skill distribution + hand-mix + tournament tier mix, with stats
     consistent with elite-level WTT play. Lets the model bootstrap on
     a fresh checkout without an external feed.
  2. An optional external CSV pull (``config.data.external_csv_url``).
     When set, this overrides / supplements the seed data. Schema must
     match the seed CSV (see below).

Seed CSV schema:

  match_date,tournament,tournament_level,round,best_of,
  winner_name,loser_name,winner_hand,loser_hand,
  winner_rank,loser_rank,
  w_games_won,l_games_won,w_points_won,l_points_won,
  w_deuce_games_won,w_deuce_games_played,
  l_deuce_games_won,l_deuce_games_played,
  w_first_game_won,winner_won_from_down
"""
from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

from ..utils.config import load_config, resolve_path
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


def _load_seed() -> pd.DataFrame:
    cfg = load_config()
    seed_path = resolve_path(cfg["paths"]["seed_matches_csv"])
    if not seed_path.exists():
        raise RuntimeError(
            f"seed_matches.csv not found at {seed_path}. "
            "Run scripts/generate_seed_data.py to regenerate, or commit "
            "the bundled seed CSV into data/raw/."
        )
    df = pd.read_csv(seed_path, parse_dates=["match_date"])
    missing = [c for c in _EXPECTED_COLS if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"seed_matches.csv is missing required columns: {missing}"
        )
    return df


def _load_external() -> pd.DataFrame | None:
    cfg = load_config()
    url = (cfg.get("data") or {}).get("external_csv_url") or ""
    if not url:
        return None
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text), parse_dates=["match_date"])
    except Exception as exc:  # noqa: BLE001
        log.warning("external CSV fetch failed (%s) — falling back to seed", exc)
        return None
    missing = [c for c in _EXPECTED_COLS if c not in df.columns]
    if missing:
        log.warning("external CSV missing cols %s — falling back to seed",
                     missing)
        return None
    return df


def fetch_all() -> pd.DataFrame:
    """Return the combined match-history DataFrame.

    Priority: external (if URL set + healthy) → seed CSV. Both share the
    schema in _EXPECTED_COLS.
    """
    df_ext = _load_external()
    df_seed = _load_seed()
    if df_ext is not None and len(df_ext) > 0:
        # Concatenate — the external data is typically more recent so we
        # let it win on duplicate (match_date, winner_name, loser_name).
        merged = pd.concat([df_seed, df_ext], ignore_index=True)
        merged = merged.drop_duplicates(
            subset=["match_date", "winner_name", "loser_name"], keep="last"
        )
        df = merged
    else:
        df = df_seed
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date", "winner_name", "loser_name"])
    df = df.sort_values("match_date").reset_index(drop=True)
    log.info("loaded %d table-tennis matches", len(df))
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
