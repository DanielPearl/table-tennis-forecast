"""Build per-snapshot training rows from TT Elite archive matches.

The TTelite-bet-analyzer JSON archive carries per-set point totals for
every historical H2H matchup (e.g. ``home_per_set=[11, 9, 11, 11]``
vs ``away_per_set=[9, 11, 6, 7]`` for a 3-1 home win). That's coarser
than tennis PBP — we only have set boundaries, not within-set point
state — but it's enough to train an in-match model on the features
that *do* update at set boundaries: set score, cumulative point
share, recent-games window, closing-set flag.

The rules layer (live_adjustment_model.py) keeps doing within-set
adjustments (point streaks, deuce/game/set-point volatility) from
the live feed; the trained model replaces only the **directional**
nudge that the rules layer applies from set-state and point share.

Real-data only: every row is parsed from the upstream JSON snapshot;
no fixtures, no synthetic state.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Iterable

import pandas as pd

from ..data.ttelite_archive import _safe_load, _parse_date, _label_to_data
from ..utils.logging_setup import setup_logging

log = setup_logging("features.build_pbp_snapshots")


# Each row from ``_extract_from_h2h_record`` already has the final
# game count + per-game points (winner_pts_per_game / loser_pts_per_game).
# We replay those games to emit set-boundary snapshots.

FEATURE_COLUMNS: tuple[str, ...] = (
    "set_score_a", "set_score_b", "sets_diff",
    "best_of",
    "current_set",
    "is_closing_game",
    "is_decider",
    "progress",
    "point_win_pct_a_live", "point_win_pct_b_live",
    "point_share_a",
    "points_diff",
    "games_won_last_3_a", "games_won_last_3_b",
    "deuce_games_a", "deuce_games_b",
    "deuce_share_a",
    "first_set_winner_a",
    "ran_up_a", "ran_up_b",  # margin of victory in last set
)


def _iter_match_snapshots(row: dict) -> Iterable[dict]:
    """Yield one snapshot row per set boundary inside one historical match.

    ``row`` is what ``ttelite_archive._build_row`` returned: per-game
    point lists for the winner and loser, plus the final game count.
    We replay the per-set sequence and emit a snapshot after each set
    finishes — that's the moment the live monitor's set_score / point
    share would change in production.
    """
    w_pts_pg = row.get("w_pts_per_game") or []
    l_pts_pg = row.get("l_pts_per_game") or []
    w_games = int(row.get("w_games_won") or 0)
    l_games = int(row.get("l_games_won") or 0)
    n_games = min(len(w_pts_pg), len(l_pts_pg))
    if n_games < 1:
        return

    # The archive is winner/loser-keyed; the live model is player_a /
    # player_b keyed. Randomize so the trained model sees both labels
    # equally — otherwise the label would be trivially predictable from
    # "we always call the winner player_a". We deterministically map
    # by alphabetical order of the player names to keep training
    # reproducible without RNG.
    winner_name = str(row.get("winner_name") or "")
    loser_name = str(row.get("loser_name") or "")
    if not winner_name or not loser_name:
        return
    a_is_winner = winner_name.lower() < loser_name.lower()
    a_name = winner_name if a_is_winner else loser_name
    b_name = loser_name if a_is_winner else winner_name
    a_pts_pg = w_pts_pg if a_is_winner else l_pts_pg
    b_pts_pg = l_pts_pg if a_is_winner else w_pts_pg
    won_a = int(a_is_winner)
    # Best-of: derive from final game count. TT Elite is bo5, but a
    # 3-2 match means 5 games played; 3-0 means 3 games. Use 5 as
    # the format (matches the league convention) so ``is_decider``
    # behaves correctly.
    best_of = 5

    sets_a = 0
    sets_b = 0
    cum_pts_a = 0
    cum_pts_b = 0
    deuce_a = 0
    deuce_b = 0
    first_set_winner_a = 0
    games_history: list[int] = []  # 1 if A won that set, 2 if B
    last_margin_a = 0
    last_margin_b = 0

    match_id = f"{row.get('match_date','')}|{a_name}|{b_name}"

    for s in range(n_games):
        ap = int(a_pts_pg[s] or 0)
        bp = int(b_pts_pg[s] or 0)
        if ap == bp:
            continue
        cum_pts_a += ap
        cum_pts_b += bp
        a_won_set = ap > bp
        if a_won_set:
            sets_a += 1
            games_history.append(1)
            last_margin_a = ap - bp
            last_margin_b = 0
        else:
            sets_b += 1
            games_history.append(2)
            last_margin_a = 0
            last_margin_b = bp - ap
        if max(ap, bp) >= 11 and min(ap, bp) >= 10:
            if a_won_set:
                deuce_a += 1
            else:
                deuce_b += 1
        if s == 0:
            first_set_winner_a = 1 if a_won_set else 0

        is_decider = (sets_a == best_of // 2 and sets_b == best_of // 2)
        is_closing_game = (sets_a == best_of // 2) or (sets_b == best_of // 2)
        progress = (s + 1) / float(best_of)
        total_pts = cum_pts_a + cum_pts_b
        deuce_total = deuce_a + deuce_b
        snap = {
            "match_id": match_id,
            "date": row.get("match_date"),
            "set_score_a": float(sets_a),
            "set_score_b": float(sets_b),
            "sets_diff": float(sets_a - sets_b),
            "best_of": float(best_of),
            "current_set": float(s + 1),
            "is_closing_game": float(is_closing_game),
            "is_decider": float(is_decider),
            "progress": progress,
            "point_win_pct_a_live": cum_pts_a / total_pts if total_pts else 0.5,
            "point_win_pct_b_live": cum_pts_b / total_pts if total_pts else 0.5,
            "point_share_a": cum_pts_a / total_pts if total_pts else 0.5,
            "points_diff": float(cum_pts_a - cum_pts_b),
            "games_won_last_3_a": float(sum(1 for g in games_history[-3:] if g == 1)),
            "games_won_last_3_b": float(sum(1 for g in games_history[-3:] if g == 2)),
            "deuce_games_a": float(deuce_a),
            "deuce_games_b": float(deuce_b),
            "deuce_share_a": deuce_a / deuce_total if deuce_total else 0.5,
            "first_set_winner_a": float(first_set_winner_a),
            "ran_up_a": float(last_margin_a),
            "ran_up_b": float(last_margin_b),
            "won_a": won_a,
        }
        yield snap
        # A snapshot after a match-clinching set is still useful as a
        # check that the model learns "3 sets won → 100% win prob",
        # but we don't continue past the match end.
        if sets_a >= best_of // 2 + 1 or sets_b >= best_of // 2 + 1:
            break


def _parse_h2h_records(records: list[dict]) -> list[dict]:
    """Parse ``all_h2h_plays.json`` into per-match rows that retain
    the per-set point arrays needed for snapshot replay.

    The upstream record is one entry per upcoming matchup with its
    H2H history embedded under ``h2h_datasets.score_history.home``
    / ``away`` (list-of-lists of per-set points) and ``dates``.
    """
    rows: list[dict] = []
    for rec in records:
        home = rec.get("home_player") or ""
        away = rec.get("away_player") or ""
        h2h = rec.get("h2h_datasets") or {}
        dates = h2h.get("dates") or []
        sh = h2h.get("score_history") or {}
        home_sets = sh.get("home") or []
        away_sets = sh.get("away") or []
        games_ds = h2h.get("games_dataset") or []
        home_games = _label_to_data(games_ds, home)
        away_games = _label_to_data(games_ds, away)
        n = min(len(dates), len(home_sets), len(away_sets),
                len(home_games), len(away_games))
        for i in range(n):
            dt = _parse_date(dates[i])
            if dt is None:
                continue
            h_pts = list(home_sets[i] or [])
            a_pts = list(away_sets[i] or [])
            try:
                hg = int(home_games[i])
                ag = int(away_games[i])
            except (TypeError, ValueError):
                continue
            if hg == ag or not h_pts or not a_pts:
                continue
            if hg > ag:
                rows.append({
                    "match_date": dt.strftime("%Y-%m-%d %H:%M"),
                    "winner_name": home, "loser_name": away,
                    "w_games_won": hg, "l_games_won": ag,
                    "w_pts_per_game": h_pts, "l_pts_per_game": a_pts,
                })
            else:
                rows.append({
                    "match_date": dt.strftime("%Y-%m-%d %H:%M"),
                    "winner_name": away, "loser_name": home,
                    "w_games_won": ag, "l_games_won": hg,
                    "w_pts_per_game": a_pts, "l_pts_per_game": h_pts,
                })
    return rows


def load_archive_matches(refresh: bool = False) -> pd.DataFrame:
    """Load every historical match row from the TTelite archive JSON
    files. Mirrors what ``ttelite_archive`` would produce but keeps
    the per-set point lists on the row so we can replay sets."""
    rows: list[dict] = []
    records = _safe_load("all_h2h_plays.json", refresh)
    if records:
        rows.extend(_parse_h2h_records(records))
    # Note: table-tennis-all-plays.json has only aggregate totals, not
    # per-set arrays, so it can't contribute to snapshot replay. The
    # h2h file already carries 4.5K+ unique matches.
    if not rows:
        raise RuntimeError(
            "no TTelite archive rows parsed — check data/raw/ttelite_archive"
        )
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["match_date", "winner_name", "loser_name"])
    df["match_date"] = pd.to_datetime(df["match_date"], errors="coerce")
    df = df.dropna(subset=["match_date"]).reset_index(drop=True)
    return df


def build_snapshots(matches: pd.DataFrame | None = None,
                    refresh: bool = False) -> pd.DataFrame:
    """Return all per-set snapshots ready for training."""
    if matches is None:
        matches = load_archive_matches(refresh=refresh)
    rows: list[dict] = []
    for _, row in matches.iterrows():
        rows.extend(_iter_match_snapshots(row.to_dict()))
    if not rows:
        raise RuntimeError("no snapshots built — check archive parser")
    df = pd.DataFrame(rows)
    log.info(
        "built %d snapshots across %d matches",
        len(df), df["match_id"].nunique(),
    )
    return df


if __name__ == "__main__":
    from ..utils.config import load_config, resolve_path

    snaps = build_snapshots(refresh=False)
    cfg = load_config()
    out_dir = resolve_path(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "pbp_snapshots.csv"
    snaps.to_csv(out, index=False)
    log.info("wrote %s (%d rows, %d features)", out, len(snaps), len(FEATURE_COLUMNS))
