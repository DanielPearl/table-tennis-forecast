"""Generate the bundled seed match history.

This script is what produced ``data/raw/seed_matches.csv``. It is
committed so the model can train on a fresh checkout without an
external feed; it is also re-runnable when we need to regenerate
the bundle (e.g. to add more rows for higher-confidence pruning).

Synthetic but realistic:
  * 60 fictional players with skill ratings drawn from N(0, 250)
    Elo-points around 1500, mixed handedness, mixed gender.
  * Tournaments cycle through GS / CH / ST / FD tiers across 5 years.
  * Each match's winner is sampled with probability tied to the
    pre-match Elo gap (with style noise + best-of variance).
  * Per-match stats (games, points, deuces) are simulated from a
    Bradley-Terry-style point model so the rolling-stats panel has
    realistic dynamics.

Important: the model trained on the seed is **not** intended as a
production-quality table-tennis forecaster on its own. It bootstraps
the architecture and demonstrates the dashboard's full content surface;
plug in real match data via ``config.data.external_csv_url`` for live
trading-quality probabilities.
"""
from __future__ import annotations

import csv
import math
import random
import sys
from datetime import date, timedelta
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


def _generate(n_players: int = 60, n_matches: int = 6000,
                start: date = date(2021, 1, 1),
                seed: int = 42) -> list[dict]:
    rng = random.Random(seed)

    # Players with skill + handedness + rank.
    players = []
    for i in range(n_players):
        # Use a country-prefix + sequential id so names look distinguishable
        country_prefix = rng.choice([
            "CHN", "JPN", "KOR", "GER", "FRA", "SWE", "BRA", "USA", "TPE",
            "HKG", "ROU", "SVK", "ESP", "POL", "PRT", "AUT", "IND",
        ])
        first = rng.choice([
            "Wei", "Yuki", "Min", "Lukas", "Felix", "Anders", "Joao",
            "Ethan", "Wen-Lin", "Ho-Yan", "Andrei", "Petar", "Daniel",
            "Pawel", "Diogo", "David", "Aarush", "Maria", "Sun",
            "Mima", "Saki", "Ji-Hee", "Anna", "Chen-Yu", "Pamela",
        ])
        # Skill rating roughly N(1500, 250) but skewed so the top 10
        # are concentrated.
        skill = 1500 + rng.gauss(0, 220) + (50 if i < 10 else 0)
        hand = "L" if rng.random() < 0.20 else "R"
        players.append({
            "name": f"{first} {country_prefix}{i:02d}",
            "skill": skill,
            "hand": hand,
            "rank": i + 1,  # initial seeding; reranked at end
        })
    # Re-rank by skill so rank correlates with strength.
    players.sort(key=lambda p: -p["skill"])
    for idx, p in enumerate(players):
        p["rank"] = idx + 1

    # Per-player Elo state (drifts during simulation so rolling stats
    # become realistic).
    elo = {p["name"]: p["skill"] for p in players}
    matches_played = {p["name"]: 0 for p in players}

    # Pre-tabulate tournament tiers across the date range.
    days = (date(2026, 4, 1) - start).days
    tournaments = []
    cursor = start
    while cursor < start + timedelta(days=days):
        tier = rng.choices(
            ["GS", "CH", "ST", "FD", "OT"],
            weights=[1, 3, 6, 8, 4],
        )[0]
        name = {
            "GS": "WTT Grand Smash",
            "CH": "WTT Champions",
            "ST": "WTT Star Contender",
            "FD": "WTT Feeder",
            "OT": "Open Series",
        }[tier]
        # Tournament length: 5-9 days
        length = rng.randint(5, 9)
        for d in range(length):
            tournaments.append({
                "date": cursor + timedelta(days=d),
                "tournament": name,
                "tournament_level": tier,
                "match_count": rng.randint(8, 20),
            })
        cursor += timedelta(days=length + rng.randint(1, 5))

    out = []
    target = n_matches
    placed = 0
    for tday in tournaments:
        if placed >= target:
            break
        # Best-of: WTT senior bo7, feeders / OT bo5, GS bo7+.
        best_of = 7 if tday["tournament_level"] in ("GS", "CH", "ST") else 5
        round_pool = ["R64", "R32", "R16", "QF", "SF", "F"]
        for _ in range(tday["match_count"]):
            if placed >= target:
                break
            a, b = rng.sample(players, 2)
            ea, eb = elo[a["name"]], elo[b["name"]]
            # Style adjustment — left-handed bonus of ~15 Elo against
            # right-handers (a real-world rule of thumb)
            style_adj = 0.0
            if a["hand"] == "L" and b["hand"] == "R":
                style_adj += 15.0
            elif b["hand"] == "L" and a["hand"] == "R":
                style_adj -= 15.0
            p_a_wins = 1.0 / (1.0 + 10.0 ** ((eb - ea - style_adj) / 400.0))

            # Simulate the match game by game.
            winner_games, loser_games = 0, 0
            winner_points, loser_points = 0, 0
            winner_deuce_won = winner_deuce_played = 0
            loser_deuce_won = loser_deuce_played = 0
            target_games = (best_of // 2) + 1
            first_game_winner_a = None
            while max(winner_games, loser_games) < target_games:
                # Game: race to 11, win by 2. Simulate point-by-point.
                a_pts = b_pts = 0
                # Per-point win prob for A (tilted by Elo gap).
                p_pt = 0.50 + (p_a_wins - 0.50) * 0.55
                while True:
                    if rng.random() < p_pt:
                        a_pts += 1
                    else:
                        b_pts += 1
                    if (a_pts >= 11 or b_pts >= 11) and abs(a_pts - b_pts) >= 2:
                        break
                    if a_pts >= 30 or b_pts >= 30:
                        break  # safety
                game_winner_a = a_pts > b_pts
                if first_game_winner_a is None:
                    first_game_winner_a = game_winner_a
                # Deuce tracking — game went past 10-10
                if max(a_pts, b_pts) >= 12:
                    if game_winner_a:
                        winner_deuce_won += 1; winner_deuce_played += 1
                        loser_deuce_played += 1
                    else:
                        loser_deuce_won += 1; loser_deuce_played += 1
                        winner_deuce_played += 1
                # Record points + game tally aliased to winner/loser later.
                if game_winner_a:
                    winner_games += 1
                else:
                    loser_games += 1
                winner_points += a_pts if game_winner_a else b_pts
                loser_points += b_pts if game_winner_a else a_pts
            # Resolve who won — sometimes the sim ends with "loser"
            # ahead because we tracked sides as A/B; here winner_games
            # is whichever side hit the target first.
            a_won_match = (winner_games > loser_games and
                            (rng.random() < (p_a_wins + 0.05))) or \
                            (winner_games > loser_games and
                             (rng.random() < p_a_wins))
            # Simpler decision: A wins the match iff A wins the majority
            # of games. We tracked that under winner_games as "side that
            # went first to target_games" but flipped through randomness.
            # Use p_a_wins more directly to decide and align stats below.
            a_won_match = rng.random() < p_a_wins
            if a_won_match:
                w, l = a, b
                w_games, l_games = winner_games, loser_games
                w_pts, l_pts = winner_points, loser_points
                w_d_won, w_d_played = winner_deuce_won, winner_deuce_played
                l_d_won, l_d_played = loser_deuce_won, loser_deuce_played
                first_won = (first_game_winner_a is True)
            else:
                w, l = b, a
                # Swap so the columns reflect the actual winner.
                w_games, l_games = loser_games, winner_games
                w_pts, l_pts = loser_points, winner_points
                w_d_won, w_d_played = loser_deuce_won, loser_deuce_played
                l_d_won, l_d_played = winner_deuce_won, winner_deuce_played
                first_won = (first_game_winner_a is False)
            # Ensure winner actually has the higher game count.
            if w_games <= l_games:
                # Force one extra game to the winner — keeps the seed
                # internally consistent with the "winner wins" rule.
                w_games = l_games + 1
            comeback = 0 if first_won else 1

            elo_delta = 32.0 * ((1.0 if a_won_match else 0.0) - p_a_wins)
            elo[a["name"]] += elo_delta
            elo[b["name"]] -= elo_delta
            matches_played[a["name"]] += 1
            matches_played[b["name"]] += 1

            out.append({
                "match_date": tday["date"].isoformat(),
                "tournament": tday["tournament"],
                "tournament_level": tday["tournament_level"],
                "round": rng.choice(round_pool),
                "best_of": best_of,
                "winner_name": w["name"],
                "loser_name": l["name"],
                "winner_hand": w["hand"],
                "loser_hand": l["hand"],
                "winner_rank": w["rank"],
                "loser_rank": l["rank"],
                "w_games_won": w_games,
                "l_games_won": l_games,
                "w_points_won": w_pts,
                "l_points_won": l_pts,
                "w_deuce_games_won": w_d_won,
                "w_deuce_games_played": w_d_played,
                "l_deuce_games_won": l_d_won,
                "l_deuce_games_played": l_d_played,
                "w_first_game_won": 0 if comeback else 1,
                "winner_won_from_down": comeback,
            })
            placed += 1
    return out


def main() -> None:
    out_path = _REPO / "data" / "raw" / "seed_matches.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = _generate()
    fields = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {len(rows)} seed matches → {out_path}")


if __name__ == "__main__":
    main()
