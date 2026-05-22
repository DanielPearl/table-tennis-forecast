"""Build the pre-match feature panel.

Per user spec: **start broad**, then let the trainer prune. We try to
expose every potentially-useful pre-match feature here; the walk-
forward permutation importance step in ``train_prematch_model.py``
identifies noisy / non-contributing features and the second-pass refit
keeps the strong ones.

Each row is one historical match. Columns include:

  * Elo features (overall + style — added by ``elo.build_elo_features``)
  * Recent form windows (last 5 / 10 / 20 matches, point-win %)
  * Rolling per-match stats (games per match, sets won %, deuce rate)
  * Comeback / closing tendencies (won-from-down, last-game win %)
  * Head-to-head (total + last 5)
  * Days of rest / matches-last-7
  * Tournament level + round
  * Best-of (5 vs 7)
  * Upset rate against ELO-favored player in this tournament tier
  * Volatility (std of recent point-win %, std of recent margin)
  * Hand matchup features (left/right indicator + diff)

We orient everything from "player_a" perspective and ``y = 1`` if
player_a wins, then mirror each row so the model sees both orientations
equally — kills the orientation bias the raw winner/loser layout
injects.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Iterable

import numpy as np
import pandas as pd

from .elo import build_elo_features, EloState


# Tournament-level encoding. Single-letter codes mirror the tennis side
# so the dashboard's level filter logic transfers verbatim.
#   GS = WTT Grand Smash / Olympics / Worlds
#   CH = WTT Champions / WTT Cup Finals
#   ST = WTT Star Contender / WTT Champions Asia
#   FD = WTT Feeder / Contender
#   OT = other
_LEVEL_RANK = {
    "GS": 5,
    "CH": 4,
    "ST": 3,
    "FD": 2,
    "OT": 1,
}


def _round_rank(r: str) -> int:
    """Numeric ordering for round strings."""
    if not isinstance(r, str):
        return 0
    table = {
        "R128": 1, "R64": 2, "R32": 3, "R16": 4,
        "QF": 5, "SF": 6, "F": 8, "RR": 4,
        # Table-tennis-specific group-stage labels
        "GRP": 2, "QUAL": 1,
    }
    return table.get(r.upper(), 0)


def _safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(b > 0, a / np.maximum(b, 1e-9), 0.0)


def _per_match_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row derived stats. Columns are oriented winner/loser; the
    caller re-orients to player_a/player_b later.

    Required input columns (any may be NaN; we coerce gracefully):
      - w_games_won, l_games_won
      - w_points_won, l_points_won
      - w_deuce_games_won, w_deuce_games_played  (and l_*)
      - w_first_game_won (1/0)
      - best_of
      - winner_won_from_down (1 if winner lost game 1 then won)
    """
    out = df.copy()

    # Game win % (out of total games played in the match)
    total_games = out["w_games_won"].astype(float) + out["l_games_won"].astype(float)
    out["w_game_win_pct"] = _safe_div(out["w_games_won"], total_games)
    out["l_game_win_pct"] = _safe_div(out["l_games_won"], total_games)

    # Point win % (out of total points played)
    total_points = out["w_points_won"].astype(float) + out["l_points_won"].astype(float)
    out["w_point_win_pct"] = _safe_div(out["w_points_won"], total_points)
    out["l_point_win_pct"] = _safe_div(out["l_points_won"], total_points)

    # Deuce conversion %
    out["w_deuce_pct"] = _safe_div(out.get("w_deuce_games_won", 0),
                                     out.get("w_deuce_games_played", 0))
    out["l_deuce_pct"] = _safe_div(out.get("l_deuce_games_won", 0),
                                     out.get("l_deuce_games_played", 0))

    # Closing game indicator: 1 if winner won the last game of the match.
    # In a fully-played bo7 the winner won game N where N = w_games + l_games;
    # since we don't have game-by-game outcomes in the panel, this is a
    # weak heuristic: scoring 4 games means the winner closed.
    out["w_closing_game_won"] = (out["w_games_won"].astype(float)
                                  >= np.where(out["best_of"].astype(float) >= 7, 4, 3)).astype(int)
    out["l_closing_game_won"] = 0

    # Margin of victory (in games)
    out["w_game_margin"] = out["w_games_won"].astype(float) - out["l_games_won"].astype(float)
    out["l_game_margin"] = -out["w_game_margin"]
    return out


def _rolling_features(df: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Per-player rolling form / point-rate / volatility / comeback /
    h2h, computed in chronological order with no in-row leakage.

    The output adds columns for every player on every row using only
    state that was known BEFORE this match. Buffers are updated after
    the row's features are recorded.

    Returns ``(df, h2h_table, last_match_date, player_history)`` where
    ``player_history`` is the per-player buffer state at the end of the
    panel — predict.py reads it so inference doesn't have to default
    every rolling feature to zero.
    """
    df = df.sort_values("match_date").reset_index(drop=True)

    # Per-player rolling buffers
    win5: dict[str, deque] = defaultdict(lambda: deque(maxlen=5))
    win10: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    win20: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    pointwin_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    margin_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    closing_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    deuce_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    comeback_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
    last_match_date: dict[str, pd.Timestamp] = {}
    # Matches per player in the last 7 days (rolling buffer of dates)
    recent_dates: dict[str, deque] = defaultdict(lambda: deque(maxlen=14))
    h2h: dict[tuple[str, str], int] = defaultdict(int)
    h2h_last5: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=5))

    # New buffers: strength-of-schedule, momentum, format/tier splits.
    opp_elo_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    elo_hist: dict[str, deque] = defaultdict(lambda: deque(maxlen=11))  # 10 deltas
    win_streak: dict[str, int] = defaultdict(int)
    matches_total: dict[str, int] = defaultdict(int)
    wins_total: dict[str, int] = defaultdict(int)
    bo7_form_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    bo5_form_buf: dict[str, deque] = defaultdict(lambda: deque(maxlen=10))
    tier_form_buf: dict[tuple[str, str], deque] = defaultdict(lambda: deque(maxlen=10))

    cols = {
        "w_form_last5": [], "l_form_last5": [],
        "w_form_last10": [], "l_form_last10": [],
        "w_form_last20": [], "l_form_last20": [],
        "w_form_ewm10": [], "l_form_ewm10": [],
        "w_avg_point_win_pct_10": [], "l_avg_point_win_pct_10": [],
        "w_std_point_win_pct_10": [], "l_std_point_win_pct_10": [],
        "w_avg_game_margin_10": [], "l_avg_game_margin_10": [],
        "w_std_game_margin_10": [], "l_std_game_margin_10": [],
        "w_closing_win_pct_10": [], "l_closing_win_pct_10": [],
        "w_deuce_win_pct_10": [], "l_deuce_win_pct_10": [],
        "w_comeback_rate_20": [], "l_comeback_rate_20": [],
        "w_days_rest": [], "l_days_rest": [],
        "w_matches_last_7d": [], "l_matches_last_7d": [],
        # New columns
        "w_opp_avg_elo_10": [], "l_opp_avg_elo_10": [],
        "w_elo_momentum_10": [], "l_elo_momentum_10": [],
        "w_win_streak": [], "l_win_streak": [],
        "w_career_win_pct": [], "l_career_win_pct": [],
        "w_career_matches": [], "l_career_matches": [],
        "w_tier_form_10": [], "l_tier_form_10": [],
        "w_bo_form_10": [], "l_bo_form_10": [],
        "h2h_w_wins_minus_l_wins": [],
        "h2h_w_wins_last5": [],
        "h2h_meetings_last5": [],
    }

    def _avg(buf: deque, default: float = 0.5) -> float:
        return float(np.mean(buf)) if len(buf) else default

    def _std(buf: deque, default: float = 0.05) -> float:
        return float(np.std(buf)) if len(buf) >= 2 else default

    def _ewm(buf: deque, alpha: float = 0.4, default: float = 0.5) -> float:
        """Exponentially-weighted mean (most recent gets the highest weight)."""
        if not buf:
            return default
        vals = list(buf)
        w = [(1 - alpha) ** (len(vals) - 1 - i) for i in range(len(vals))]
        s = float(sum(w))
        return float(sum(v * wi for v, wi in zip(vals, w)) / s) if s > 0 else default

    def _streak_capped(n: int, cap: int = 10) -> float:
        return float(max(-cap, min(cap, n)))

    for _, row in df.iterrows():
        w, l = row["winner_name"], row["loser_name"]
        date = row["match_date"]
        bo = int(row.get("best_of") or 7)
        tier = str(row.get("tournament_level") or "OT").upper()

        cols["w_form_last5"].append(_avg(win5[w], 0.5))
        cols["l_form_last5"].append(_avg(win5[l], 0.5))
        cols["w_form_last10"].append(_avg(win10[w], 0.5))
        cols["l_form_last10"].append(_avg(win10[l], 0.5))
        cols["w_form_last20"].append(_avg(win20[w], 0.5))
        cols["l_form_last20"].append(_avg(win20[l], 0.5))
        cols["w_form_ewm10"].append(_ewm(win10[w]))
        cols["l_form_ewm10"].append(_ewm(win10[l]))

        cols["w_avg_point_win_pct_10"].append(_avg(pointwin_buf[w], 0.50))
        cols["l_avg_point_win_pct_10"].append(_avg(pointwin_buf[l], 0.50))
        cols["w_std_point_win_pct_10"].append(_std(pointwin_buf[w], 0.05))
        cols["l_std_point_win_pct_10"].append(_std(pointwin_buf[l], 0.05))

        cols["w_avg_game_margin_10"].append(_avg(margin_buf[w], 0.0))
        cols["l_avg_game_margin_10"].append(_avg(margin_buf[l], 0.0))
        cols["w_std_game_margin_10"].append(_std(margin_buf[w], 1.0))
        cols["l_std_game_margin_10"].append(_std(margin_buf[l], 1.0))

        cols["w_closing_win_pct_10"].append(_avg(closing_buf[w], 0.5))
        cols["l_closing_win_pct_10"].append(_avg(closing_buf[l], 0.5))

        cols["w_deuce_win_pct_10"].append(_avg(deuce_buf[w], 0.5))
        cols["l_deuce_win_pct_10"].append(_avg(deuce_buf[l], 0.5))

        cols["w_comeback_rate_20"].append(_avg(comeback_buf[w], 0.0))
        cols["l_comeback_rate_20"].append(_avg(comeback_buf[l], 0.0))

        # Days rest
        wd = (date - last_match_date[w]).days if w in last_match_date else 7
        ld = (date - last_match_date[l]).days if l in last_match_date else 7
        cols["w_days_rest"].append(min(60, max(0, wd)))
        cols["l_days_rest"].append(min(60, max(0, ld)))

        # Matches in last 7 days
        cols["w_matches_last_7d"].append(_count_within_days(recent_dates[w], date, 7))
        cols["l_matches_last_7d"].append(_count_within_days(recent_dates[l], date, 7))

        # Strength-of-schedule — avg Elo of recent opponents (defaults to
        # the base rating so an unknown player reads as "average opp").
        default_elo = 1500.0
        cols["w_opp_avg_elo_10"].append(_avg(opp_elo_buf[w], default_elo))
        cols["l_opp_avg_elo_10"].append(_avg(opp_elo_buf[l], default_elo))

        # Elo momentum — current Elo minus Elo from up to 10 matches ago.
        # Captures "trending up" vs "trending down" without changing the
        # underlying Elo system. Uses the pre-match Elo (added by
        # build_elo_features upstream) for the *current* read.
        cur_w_elo = float(row.get("winner_elo_pre") or default_elo)
        cur_l_elo = float(row.get("loser_elo_pre") or default_elo)
        w_hist = elo_hist[w]
        l_hist = elo_hist[l]
        cols["w_elo_momentum_10"].append(cur_w_elo - (w_hist[0] if w_hist else cur_w_elo))
        cols["l_elo_momentum_10"].append(cur_l_elo - (l_hist[0] if l_hist else cur_l_elo))

        # Win streak (signed: + for consecutive wins, − for consecutive losses)
        cols["w_win_streak"].append(_streak_capped(win_streak[w]))
        cols["l_win_streak"].append(_streak_capped(win_streak[l]))

        # Career win rate (laplace-smoothed so a 1-1 player ≠ 1.0)
        def _wr(player: str) -> float:
            m = matches_total[player]
            if m == 0:
                return 0.5
            return float((wins_total[player] + 2.0) / (m + 4.0))
        cols["w_career_win_pct"].append(_wr(w))
        cols["l_career_win_pct"].append(_wr(l))
        cols["w_career_matches"].append(float(matches_total[w]))
        cols["l_career_matches"].append(float(matches_total[l]))

        # Tier-specific form
        cols["w_tier_form_10"].append(_avg(tier_form_buf[(w, tier)], 0.5))
        cols["l_tier_form_10"].append(_avg(tier_form_buf[(l, tier)], 0.5))

        # Best-of-specific form (bo7 vs bo5 — same player can swing 5-7pp)
        if bo >= 7:
            cols["w_bo_form_10"].append(_avg(bo7_form_buf[w], 0.5))
            cols["l_bo_form_10"].append(_avg(bo7_form_buf[l], 0.5))
        else:
            cols["w_bo_form_10"].append(_avg(bo5_form_buf[w], 0.5))
            cols["l_bo_form_10"].append(_avg(bo5_form_buf[l], 0.5))

        # H2H (winner-side perspective). We also carry the bounded
        # count of recent meetings so the loser-orientation panel build
        # can compute the loser's wins as (total_meetings − winner_wins)
        # rather than the buggy ``5 − winner_wins`` (which over-credits
        # the loser when the H2H buffer is shorter than 5).
        key = tuple(sorted([w, l]))
        sign = 1 if key[0] == w else -1
        cols["h2h_w_wins_minus_l_wins"].append(h2h[key] * sign)
        last5 = h2h_last5[key]
        recent_wins_w = sum(1 for v in last5 if v == w) if last5 else 0
        cols["h2h_w_wins_last5"].append(recent_wins_w)
        cols["h2h_meetings_last5"].append(len(last5))

        # Post-row buffer updates.
        win5[w].append(1.0); win5[l].append(0.0)
        win10[w].append(1.0); win10[l].append(0.0)
        win20[w].append(1.0); win20[l].append(0.0)

        if not pd.isna(row.get("w_point_win_pct")):
            pointwin_buf[w].append(float(row["w_point_win_pct"]))
        if not pd.isna(row.get("l_point_win_pct")):
            pointwin_buf[l].append(float(row["l_point_win_pct"]))
        if not pd.isna(row.get("w_game_margin")):
            margin_buf[w].append(float(row["w_game_margin"]))
        if not pd.isna(row.get("l_game_margin")):
            margin_buf[l].append(float(row["l_game_margin"]))
        if not pd.isna(row.get("w_closing_game_won")):
            closing_buf[w].append(float(row["w_closing_game_won"]))
        if not pd.isna(row.get("l_closing_game_won")):
            closing_buf[l].append(float(row["l_closing_game_won"]))
        if not pd.isna(row.get("w_deuce_pct")):
            deuce_buf[w].append(float(row["w_deuce_pct"]))
        if not pd.isna(row.get("l_deuce_pct")):
            deuce_buf[l].append(float(row["l_deuce_pct"]))
        # Comeback: 1 if w lost the first game then won the match
        won_from_down = int(row.get("winner_won_from_down") or 0)
        comeback_buf[w].append(float(won_from_down))
        # Loser doesn't get comeback credit
        comeback_buf[l].append(0.0)

        # Update new buffers.
        opp_elo_buf[w].append(cur_l_elo)
        opp_elo_buf[l].append(cur_w_elo)
        elo_hist[w].append(cur_w_elo)
        elo_hist[l].append(cur_l_elo)
        win_streak[w] = win_streak[w] + 1 if win_streak[w] >= 0 else 1
        win_streak[l] = win_streak[l] - 1 if win_streak[l] <= 0 else -1
        matches_total[w] += 1; matches_total[l] += 1
        wins_total[w] += 1
        tier_form_buf[(w, tier)].append(1.0)
        tier_form_buf[(l, tier)].append(0.0)
        if bo >= 7:
            bo7_form_buf[w].append(1.0); bo7_form_buf[l].append(0.0)
        else:
            bo5_form_buf[w].append(1.0); bo5_form_buf[l].append(0.0)

        last_match_date[w] = date; last_match_date[l] = date
        recent_dates[w].append(date); recent_dates[l].append(date)
        h2h[key] += 1 if key[0] == w else -1
        h2h_last5[key].append(w)

    for k, v in cols.items():
        df[k] = v

    player_history = {
        "win5": {k: list(v) for k, v in win5.items()},
        "win10": {k: list(v) for k, v in win10.items()},
        "win20": {k: list(v) for k, v in win20.items()},
        "pointwin_10": {k: list(v) for k, v in pointwin_buf.items()},
        "margin_10": {k: list(v) for k, v in margin_buf.items()},
        "closing_10": {k: list(v) for k, v in closing_buf.items()},
        "deuce_10": {k: list(v) for k, v in deuce_buf.items()},
        "comeback_20": {k: list(v) for k, v in comeback_buf.items()},
        "opp_elo_10": {k: list(v) for k, v in opp_elo_buf.items()},
        "elo_hist": {k: list(v) for k, v in elo_hist.items()},
        "win_streak": dict(win_streak),
        "matches_total": dict(matches_total),
        "wins_total": dict(wins_total),
        "tier_form_10": {f"{k[0]}|{k[1]}": list(v)
                          for k, v in tier_form_buf.items()},
        "bo7_form_10": {k: list(v) for k, v in bo7_form_buf.items()},
        "bo5_form_10": {k: list(v) for k, v in bo5_form_buf.items()},
        "recent_dates": {k: [pd.Timestamp(d).isoformat() for d in v]
                          for k, v in recent_dates.items()},
    }
    return df, h2h, last_match_date, player_history


def _count_within_days(date_buf: deque, ref: pd.Timestamp, days: int) -> int:
    cutoff = ref - pd.Timedelta(days=days)
    return sum(1 for d in date_buf if d > cutoff)


def build_full_panel(matches: pd.DataFrame, elo_cfg: dict | None = None
                     ) -> tuple[pd.DataFrame, EloState, dict, dict, dict]:
    """End-to-end enrichment. Returns the wide panel + trained Elo state
    + accumulated H2H / last-match-date dicts / per-player rolling
    history (all used at inference time)."""
    df = _per_match_stats(matches)
    df, elo_state = build_elo_features(df, elo_cfg)
    df, h2h_table, last_match_date, player_history = _rolling_features(df)

    # Tournament + round encodings (with defaults for missing values).
    df["level_rank"] = df["tournament_level"].fillna("OT").map(_LEVEL_RANK).fillna(1).astype(int)
    df["round_rank"] = df["round"].fillna("R32").map(_round_rank).astype(int)
    df["is_bo7"] = (df["best_of"].fillna(7).astype(int) >= 7).astype(int)

    # Rank-based features (when present in source data).
    df["winner_rank"] = pd.to_numeric(df.get("winner_rank"), errors="coerce")
    df["loser_rank"] = pd.to_numeric(df.get("loser_rank"), errors="coerce")
    df["rank_diff"] = (df["loser_rank"] - df["winner_rank"]).fillna(0.0)

    # Hand matchup features
    df["winner_hand_left"] = (df["winner_hand"].fillna("R").str.upper() == "L").astype(int)
    df["loser_hand_left"] = (df["loser_hand"].fillna("R").str.upper() == "L").astype(int)
    df["hand_matchup_lr"] = (df["winner_hand_left"] != df["loser_hand_left"]).astype(int)

    return df, elo_state, h2h_table, last_match_date, player_history


def build_player_a_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Re-orient the winner/loser-encoded panel into player_a/player_b
    with a balanced ``y`` label."""
    keep = ["match_date", "tournament", "tournament_level", "round", "best_of"]
    base = panel[keep].copy()

    a_pos = base.copy()
    a_pos["player_a"] = panel["winner_name"].values
    a_pos["player_b"] = panel["loser_name"].values
    a_pos["y"] = 1
    _attach_oriented(a_pos, side="w_to_a", panel=panel)

    a_neg = base.copy()
    a_neg["player_a"] = panel["loser_name"].values
    a_neg["player_b"] = panel["winner_name"].values
    a_neg["y"] = 0
    _attach_oriented(a_neg, side="l_to_a", panel=panel)

    return pd.concat([a_pos, a_neg], ignore_index=True)


def _attach_oriented(out: pd.DataFrame, side: str, panel: pd.DataFrame) -> None:
    """Copy winner/loser-prefixed columns into a_/b_ columns based on
    which orientation this row represents."""
    pairs = [
        ("elo_pre", "winner_elo_pre", "loser_elo_pre"),
        ("style_elo_pre", "winner_style_elo_pre", "loser_style_elo_pre"),
        ("form_last5", "w_form_last5", "l_form_last5"),
        ("form_last10", "w_form_last10", "l_form_last10"),
        ("form_last20", "w_form_last20", "l_form_last20"),
        ("form_ewm10", "w_form_ewm10", "l_form_ewm10"),
        ("avg_point_win_pct_10", "w_avg_point_win_pct_10", "l_avg_point_win_pct_10"),
        ("std_point_win_pct_10", "w_std_point_win_pct_10", "l_std_point_win_pct_10"),
        ("avg_game_margin_10", "w_avg_game_margin_10", "l_avg_game_margin_10"),
        ("std_game_margin_10", "w_std_game_margin_10", "l_std_game_margin_10"),
        ("closing_win_pct_10", "w_closing_win_pct_10", "l_closing_win_pct_10"),
        ("deuce_win_pct_10", "w_deuce_win_pct_10", "l_deuce_win_pct_10"),
        ("comeback_rate_20", "w_comeback_rate_20", "l_comeback_rate_20"),
        ("days_rest", "w_days_rest", "l_days_rest"),
        ("matches_last_7d", "w_matches_last_7d", "l_matches_last_7d"),
        ("hand_left", "winner_hand_left", "loser_hand_left"),
        # New rolling features
        ("opp_avg_elo_10", "w_opp_avg_elo_10", "l_opp_avg_elo_10"),
        ("elo_momentum_10", "w_elo_momentum_10", "l_elo_momentum_10"),
        ("win_streak", "w_win_streak", "l_win_streak"),
        ("career_win_pct", "w_career_win_pct", "l_career_win_pct"),
        ("career_matches", "w_career_matches", "l_career_matches"),
        ("tier_form_10", "w_tier_form_10", "l_tier_form_10"),
        ("bo_form_10", "w_bo_form_10", "l_bo_form_10"),
    ]
    for short, w_col, l_col in pairs:
        if side == "w_to_a":
            out[f"a_{short}"] = panel[w_col].values
            out[f"b_{short}"] = panel[l_col].values
        else:
            out[f"a_{short}"] = panel[l_col].values
            out[f"b_{short}"] = panel[w_col].values

    # Diffs (a − b) for everything that's numeric.
    for short, _, _ in pairs:
        out[f"diff_{short}"] = out[f"a_{short}"] - out[f"b_{short}"]

    # Shared (orientation-invariant) features
    out["level_rank"] = panel["level_rank"].values
    out["round_rank"] = panel["round_rank"].values
    out["is_bo7"] = panel["is_bo7"].values
    out["hand_matchup_lr"] = panel["hand_matchup_lr"].values
    if side == "w_to_a":
        out["a_rank"] = pd.to_numeric(panel.get("winner_rank"), errors="coerce").values
        out["b_rank"] = pd.to_numeric(panel.get("loser_rank"), errors="coerce").values
        out["h2h_a_wins_minus_b_wins"] = panel["h2h_w_wins_minus_l_wins"].values
        out["h2h_a_wins_last5"] = panel["h2h_w_wins_last5"].values
    else:
        out["a_rank"] = pd.to_numeric(panel.get("loser_rank"), errors="coerce").values
        out["b_rank"] = pd.to_numeric(panel.get("winner_rank"), errors="coerce").values
        out["h2h_a_wins_minus_b_wins"] = -panel["h2h_w_wins_minus_l_wins"].values
        # Loser-side h2h count = total recent meetings − winner's wins.
        # Carrying the meetings count from the panel build keeps this
        # exact even when the H2H buffer hasn't yet filled to 5.
        winner_wins = panel["h2h_w_wins_last5"].values
        total_meetings = panel["h2h_meetings_last5"].values
        out["h2h_a_wins_last5"] = np.maximum(0, total_meetings - winner_wins)
    out["rank_diff"] = (out["b_rank"].fillna(500) - out["a_rank"].fillna(500))


# Broad initial feature list. The trainer prunes anything whose
# permutation importance is consistently noisy (see
# ``train_prematch_model.py``).
PREMATCH_FEATURES_BROAD = [
    "diff_elo_pre",
    "diff_style_elo_pre",
    "diff_form_last5",
    "diff_form_last10",
    "diff_form_last20",
    "diff_form_ewm10",
    "diff_avg_point_win_pct_10",
    "diff_std_point_win_pct_10",
    "diff_avg_game_margin_10",
    "diff_std_game_margin_10",
    "diff_closing_win_pct_10",
    "diff_deuce_win_pct_10",
    "diff_comeback_rate_20",
    "diff_days_rest",
    "diff_matches_last_7d",
    "h2h_a_wins_minus_b_wins",
    "h2h_a_wins_last5",
    "rank_diff",
    "level_rank",
    "round_rank",
    "is_bo7",
    "diff_hand_left",
    "hand_matchup_lr",
    # New strength-of-schedule / momentum / specialization features
    "diff_opp_avg_elo_10",
    "diff_elo_momentum_10",
    "diff_win_streak",
    "diff_career_win_pct",
    "diff_career_matches",
    "diff_tier_form_10",
    "diff_bo_form_10",
]


def select_features(df: pd.DataFrame,
                     features: list[str] | None = None) -> pd.DataFrame:
    feats = features or PREMATCH_FEATURES_BROAD
    return df[feats].fillna(0.0)
