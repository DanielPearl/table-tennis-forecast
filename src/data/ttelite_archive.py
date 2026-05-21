"""Real-data table-tennis loader sourced from ``ahartness/table-tennis-
bet-analyzer`` on GitHub.

This replaces the previous synthetic seed CSV. The upstream repo
publishes JSON snapshots of TT Elite Series betting analysis, each of
which carries the most recent 30 head-to-head matches between the
home/away pair — including dates, per-set scores, and total points
per player. We download those JSON files, dedupe by (date, ordered
pair), and produce a per-match panel in the schema the bot's trainer
consumes.

Source repo (MIT-licensed):
    https://github.com/ahartness/table-tennis-bet-analyzer
Files consumed:
    data/table-tennis-all-plays.json
    data/all_h2h_plays.json

What's real / what's missing
----------------------------
Real, parsed from the upstream JSON:
  • match_date          (from ``dates``)
  • winner_name / loser_name (from ``games_dataset``: whoever has 3
                              games is the winner)
  • w_games_won / l_games_won (game scores like 3-0, 3-2, etc.)
  • w_points_won / l_points_won (sum of per-game points from
                                 ``score_history``)
  • w_first_game_won    (set 0 winner)
  • w_deuce_games_won / played (count of games where both reached >=10)
  • winner_won_from_down (1 if loser took ≥1 of the first two sets)

Defaulted (no signal in the upstream archive):
  • tournament         "TT Elite Series" — all of this data is
                        from that league
  • tournament_level   "TT-Elite"
  • round              ""
  • best_of            5
  • winner_hand / loser_hand  "R" (unknown; right-handed is the prior)
  • winner_rank / loser_rank  0 (no ranking snapshot per match)
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

from ..utils.config import load_config, resolve_path

log = logging.getLogger("data.ttelite_archive")


_REPO_OWNER = "ahartness"
_REPO_NAME = "table-tennis-bet-analyzer"
_REPO_BRANCH = "main"
_REPO_DATA_DIR = "data"

# Two JSON snapshots that overlap heavily on player pairs but the
# h2h variant carries the cleanest per-game scores. We use both and
# dedupe on (date, ordered_pair).
_DATA_FILES: List[str] = [
    "all_h2h_plays.json",
    "table-tennis-all-plays.json",
]


def _cache_dir() -> Path:
    cfg = load_config()
    raw_dir = resolve_path(cfg["paths"]["raw_dir"])
    out = Path(raw_dir) / "ttelite_archive"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _raw_url(filename: str) -> str:
    return (f"https://raw.githubusercontent.com/{_REPO_OWNER}/{_REPO_NAME}/"
            f"{_REPO_BRANCH}/{_REPO_DATA_DIR}/{filename}")


def _download(filename: str, refresh: bool) -> Path:
    dest = _cache_dir() / filename
    if dest.exists() and not refresh:
        return dest
    log.info("downloading %s", _raw_url(filename))
    r = requests.get(_raw_url(filename), timeout=60)
    r.raise_for_status()
    # Some upstream files contain merge-conflict markers (`<<<<<<<`,
    # `=======`, `>>>>>>>`); strip them so the JSON parses.
    text = r.text
    cleaned = re.sub(r"^(<<<<<<< |=======|>>>>>>> ).*$\n?", "",
                      text, flags=re.MULTILINE)
    dest.write_text(cleaned)
    return dest


def _safe_load(filename: str, refresh: bool) -> List[Dict[str, Any]]:
    p = _download(filename, refresh)
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        log.warning("could not parse %s (%s) — skipping", filename, exc)
        return []
    return data if isinstance(data, list) else []


# --------------------------------------------------------------------- #
# Per-record match extraction                                            #
# --------------------------------------------------------------------- #

def _parse_date(s: str) -> Optional[datetime]:
    """Parse "2024-12-31 11:10PM" → datetime."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip()
    for fmt in ("%Y-%m-%d %I:%M%p", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _label_to_data(entries: List[Dict[str, Any]],
                    name: str) -> List[Any]:
    """``[{label, data: [...]}, ...]`` -> the entry whose label matches
    ``name``'s data list (or [] if not found)."""
    for e in entries or []:
        if (e or {}).get("label") == name:
            return list(e.get("data") or [])
    return []


def _extract_from_h2h_record(rec: Dict[str, Any]
                              ) -> List[Dict[str, Any]]:
    """Pull the embedded historical matches out of one
    ``all_h2h_plays.json`` record."""
    home = rec.get("home_player") or ""
    away = rec.get("away_player") or ""
    h2h = rec.get("h2h_datasets") or {}
    dates = h2h.get("dates") or []
    games_ds = h2h.get("games_dataset") or []
    score_hist = h2h.get("score_history") or {}
    home_games = _label_to_data(games_ds, home)
    away_games = _label_to_data(games_ds, away)
    home_scores = list(score_hist.get("home") or [])
    away_scores = list(score_hist.get("away") or [])
    rows: List[Dict[str, Any]] = []
    n = min(len(dates), len(home_games), len(away_games),
            len(home_scores), len(away_scores))
    for i in range(n):
        row = _build_row(
            date=dates[i],
            home=home, away=away,
            home_g=home_games[i], away_g=away_games[i],
            home_pts_per_game=home_scores[i],
            away_pts_per_game=away_scores[i],
        )
        if row is not None:
            rows.append(row)
    return rows


def _extract_from_all_plays_record(rec: Dict[str, Any]
                                     ) -> List[Dict[str, Any]]:
    """Pull embedded historical matches from a
    ``table-tennis-all-plays.json`` record."""
    home = rec.get("home_player") or ""
    away = rec.get("away_player") or ""
    labels = rec.get("labels") or []
    games_ds = rec.get("game_dataset_array") or []
    points_ds = rec.get("dataset_array") or []
    home_games = _label_to_data(games_ds, home)
    away_games = _label_to_data(games_ds, away)
    home_pts = _label_to_data(points_ds, home)
    away_pts = _label_to_data(points_ds, away)
    rows: List[Dict[str, Any]] = []
    n = min(len(labels), len(home_games), len(away_games))
    for i in range(n):
        # all-plays carries total points per match (not per-game), so
        # we treat the points-per-game list as a single-element list.
        hp = [int(home_pts[i])] if i < len(home_pts) else []
        ap = [int(away_pts[i])] if i < len(away_pts) else []
        row = _build_row(
            date=labels[i],
            home=home, away=away,
            home_g=home_games[i], away_g=away_games[i],
            home_pts_per_game=hp,
            away_pts_per_game=ap,
        )
        if row is not None:
            rows.append(row)
    return rows


def _build_row(*, date: str, home: str, away: str,
                 home_g: Any, away_g: Any,
                 home_pts_per_game: List[int],
                 away_pts_per_game: List[int]) -> Optional[Dict[str, Any]]:
    """Materialise one per-match row in the schema the bot consumes."""
    dt = _parse_date(date)
    if dt is None or not home or not away or home == away:
        return None
    try:
        hg = int(home_g)
        ag = int(away_g)
    except (TypeError, ValueError):
        return None
    if hg == ag:  # No tie in TT — skip if data is ambiguous.
        return None
    if hg > ag:
        winner, loser = home, away
        w_games, l_games = hg, ag
        w_pts_pg, l_pts_pg = list(home_pts_per_game or []), list(away_pts_per_game or [])
        first_game_winner_is_w = (
            len(home_pts_per_game) > 0 and len(away_pts_per_game) > 0
            and home_pts_per_game[0] > away_pts_per_game[0]
        )
    else:
        winner, loser = away, home
        w_games, l_games = ag, hg
        w_pts_pg, l_pts_pg = list(away_pts_per_game or []), list(home_pts_per_game or [])
        first_game_winner_is_w = (
            len(home_pts_per_game) > 0 and len(away_pts_per_game) > 0
            and away_pts_per_game[0] > home_pts_per_game[0]
        )

    # Aggregate points across games when per-game scores are available.
    w_points = sum(w_pts_pg) if w_pts_pg else 0
    l_points = sum(l_pts_pg) if l_pts_pg else 0
    # Deuce game stats — count games where both sides reached >= 10.
    w_deuce_won = w_deuce_played = 0
    l_deuce_won = l_deuce_played = 0
    n_games = min(len(w_pts_pg), len(l_pts_pg))
    for i in range(n_games):
        wp, lp = w_pts_pg[i], l_pts_pg[i]
        if max(wp, lp) >= 11 and min(wp, lp) >= 10:
            w_deuce_played += 1
            l_deuce_played += 1
            if wp > lp:
                w_deuce_won += 1
            else:
                l_deuce_won += 1
    # Won-from-down: loser took at least one of the first two sets.
    won_from_down = 0
    if n_games >= 2:
        # Reconstruct per-set winner sequence to detect 0-1 / 0-2 starts.
        set_winners_w_perspective: List[int] = []
        for i in range(n_games):
            set_winners_w_perspective.append(
                1 if w_pts_pg[i] > l_pts_pg[i] else 0
            )
        if 0 in set_winners_w_perspective[:2]:
            won_from_down = 1

    return {
        "match_date": dt.strftime("%Y-%m-%d"),
        "tournament": "TT Elite Series",
        "tournament_level": "TT-Elite",
        "round": "",
        "best_of": 5,
        "winner_name": winner,
        "loser_name": loser,
        "winner_hand": "R",
        "loser_hand": "R",
        "winner_rank": 0,
        "loser_rank": 0,
        "w_games_won": w_games,
        "l_games_won": l_games,
        "w_points_won": w_points,
        "l_points_won": l_points,
        "w_deuce_games_won": w_deuce_won,
        "w_deuce_games_played": w_deuce_played,
        "l_deuce_games_won": l_deuce_won,
        "l_deuce_games_played": l_deuce_played,
        "w_first_game_won": 1 if first_game_winner_is_w else 0,
        "winner_won_from_down": won_from_down,
        # Dedupe key (date + ordered pair). Stripped before output.
        "_dedup": (dt.strftime("%Y-%m-%d %H:%M"),
                    tuple(sorted([winner, loser]))),
    }


# --------------------------------------------------------------------- #
# Public API                                                             #
# --------------------------------------------------------------------- #

def fetch_archive_matches(refresh: bool = False) -> pd.DataFrame:
    """Build a per-match DataFrame from the ahartness archive."""
    all_rows: List[Dict[str, Any]] = []

    h2h_records = _safe_load("all_h2h_plays.json", refresh)
    log.info("loaded %d h2h records from all_h2h_plays.json", len(h2h_records))
    for r in h2h_records:
        all_rows.extend(_extract_from_h2h_record(r))

    ap_records = _safe_load("table-tennis-all-plays.json", refresh)
    log.info("loaded %d all-plays records from table-tennis-all-plays.json",
              len(ap_records))
    for r in ap_records:
        all_rows.extend(_extract_from_all_plays_record(r))

    if not all_rows:
        raise RuntimeError(
            "ahartness archive produced 0 matches — upstream JSON files "
            "may have moved or changed format"
        )

    # Dedupe by (date, ordered_pair). When both files cover the same
    # match, the h2h record (which carries per-game scores) is parsed
    # first and wins by virtue of being added first.
    seen: set = set()
    deduped: List[Dict[str, Any]] = []
    for r in all_rows:
        key = r.pop("_dedup")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    df = pd.DataFrame(deduped)
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date", "winner_name", "loser_name"])
    df = df.sort_values("match_date").reset_index(drop=True)
    log.info("built %d unique TT Elite matches (date range %s..%s)",
              len(df),
              df["match_date"].min().date(),
              df["match_date"].max().date())
    return df
