"""Table-tennis Elo (overall + style-matchup).

Why we use a custom Elo:

  * In table tennis, Elo is the dominant pre-match feature — official
    ITTF / TTW ratings update on a fixed cadence and lag real form by
    weeks. A self-maintained Elo captures form continuously.
  * Style matchup matters: left-handed players have a small but
    consistent edge against right-handers (~3-5pp at the elite level)
    because of the angles their forehand/backhand opens up. A separate
    style-Elo dimension tracks ``(player, opponent_handedness)`` so the
    overall rating doesn't get polluted by this asymmetry.

K-factor: starts at ``k_base`` and decays toward ``k_floor`` as a player
accumulates matches. Best-of-7 matches carry slightly more K because
the result has more signal than a bo5 win.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

import pandas as pd


_HANDS = ("R", "L")  # right / left


@dataclass
class EloState:
    default_rating: float = 1500.0
    k_base: float = 40.0
    k_floor: float = 16.0
    k_decay_matches: int = 50
    bo7_k_multiplier: float = 1.10
    style_k_multiplier: float = 1.0
    style_blend: float = 0.25

    overall: dict = field(default_factory=dict)
    # Style Elo keyed by (player, opponent_hand).
    style: dict = field(default_factory=dict)
    matches_played: dict = field(default_factory=lambda: defaultdict(int))
    style_matches: dict = field(default_factory=lambda: defaultdict(int))

    def get_overall(self, player: str) -> float:
        return self.overall.get(player, self.default_rating)

    def get_style(self, player: str, opp_hand: str) -> float:
        opp_hand = opp_hand if opp_hand in _HANDS else "R"
        return self.style.get((player, opp_hand), self.default_rating)

    def k_for(self, player: str) -> float:
        n = self.matches_played[player]
        if n >= self.k_decay_matches:
            return self.k_floor
        frac = n / max(1, self.k_decay_matches)
        return self.k_base - (self.k_base - self.k_floor) * frac


def _expected(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def _update_pair(state: EloState, winner: str, loser: str,
                  winner_hand: str, loser_hand: str,
                  best_of: int) -> tuple[float, float, float, float]:
    w_pre = state.get_overall(winner)
    l_pre = state.get_overall(loser)
    ws_pre = state.get_style(winner, loser_hand)
    ls_pre = state.get_style(loser, winner_hand)

    bo_mult = state.bo7_k_multiplier if int(best_of) >= 7 else 1.0

    e_w = _expected(w_pre, l_pre)
    k_w = state.k_for(winner) * bo_mult
    k_l = state.k_for(loser) * bo_mult
    state.overall[winner] = w_pre + k_w * (1.0 - e_w)
    state.overall[loser] = l_pre + k_l * (0.0 - (1.0 - e_w))

    e_ws = _expected(ws_pre, ls_pre)
    k_ws = state.k_for(winner) * state.style_k_multiplier * bo_mult
    k_ls = state.k_for(loser) * state.style_k_multiplier * bo_mult
    state.style[(winner, loser_hand)] = ws_pre + k_ws * (1.0 - e_ws)
    state.style[(loser, winner_hand)] = ls_pre + k_ls * (0.0 - (1.0 - e_ws))

    state.matches_played[winner] += 1
    state.matches_played[loser] += 1
    state.style_matches[(winner, loser_hand)] += 1
    state.style_matches[(loser, winner_hand)] += 1

    return w_pre, l_pre, ws_pre, ls_pre


def build_elo_features(matches: pd.DataFrame, state_cfg: dict | None = None
                       ) -> tuple[pd.DataFrame, EloState]:
    """Add pre-match Elo columns to a matches dataframe.

    Required columns:
      - match_date, winner_name, loser_name
      - winner_hand, loser_hand (one of "R" / "L"; missing → "R")
      - best_of (5 or 7; missing → 7)
    """
    state = EloState(**(state_cfg or {}))
    df = matches.sort_values("match_date").reset_index(drop=True).copy()
    cols = {
        "winner_elo_pre": [], "loser_elo_pre": [],
        "winner_style_elo_pre": [], "loser_style_elo_pre": [],
    }
    for _, row in df.iterrows():
        wh = str(row.get("winner_hand") or "R").upper()[:1]
        lh = str(row.get("loser_hand") or "R").upper()[:1]
        bo = int(row.get("best_of") or 7)
        w_pre, l_pre, ws_pre, ls_pre = _update_pair(
            state, row["winner_name"], row["loser_name"], wh, lh, bo
        )
        cols["winner_elo_pre"].append(w_pre)
        cols["loser_elo_pre"].append(l_pre)
        cols["winner_style_elo_pre"].append(ws_pre)
        cols["loser_style_elo_pre"].append(ls_pre)
    for k, v in cols.items():
        df[k] = v
    df["elo_diff"] = df["winner_elo_pre"] - df["loser_elo_pre"]
    df["style_elo_diff"] = df["winner_style_elo_pre"] - df["loser_style_elo_pre"]
    df["blended_elo_diff"] = (
        state.style_blend * df["style_elo_diff"]
        + (1.0 - state.style_blend) * df["elo_diff"]
    )
    df["elo_winprob"] = 1.0 / (1.0 + 10.0 ** (-df["elo_diff"] / 400.0))
    return df, state


def lookup_pair_features(state: EloState, player_a: str, player_b: str,
                          hand_a: str = "R", hand_b: str = "R") -> dict:
    """Build Elo features for a hypothetical (a vs b) matchup."""
    a = state.get_overall(player_a)
    b = state.get_overall(player_b)
    a_s = state.get_style(player_a, hand_b)
    b_s = state.get_style(player_b, hand_a)
    elo_diff = a - b
    style_elo_diff = a_s - b_s
    blended = (state.style_blend * style_elo_diff
               + (1.0 - state.style_blend) * elo_diff)
    elo_winprob = 1.0 / (1.0 + 10.0 ** (-elo_diff / 400.0))
    return {
        "player_a_elo": a, "player_b_elo": b,
        "player_a_style_elo": a_s, "player_b_style_elo": b_s,
        "elo_diff": elo_diff, "style_elo_diff": style_elo_diff,
        "blended_elo_diff": blended, "elo_winprob_a": elo_winprob,
    }
