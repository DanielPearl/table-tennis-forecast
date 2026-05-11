"""Real Kalshi table-tennis market fetcher.

Mirrors the tennis sibling: iterates each configured series via the
kalshi_sdk, collapses two-sided markets into one record per match,
and writes the canonical live-state file the watchlist exporter
reads.

Title parsing
  Kalshi table-tennis titles vary by series but generally read
  "Will {Player Name} win the {LastA} vs {LastB}: {Round} match?"
  — same pattern as tennis. We use a flexible regex and fall back to
  picking the player from the ``yes_sub_title`` field when present.

Pricing
  ``yes_ask`` → implied probability (cents → dollars). Two-sided NO
  fallback when only one side is quoted, identical to tennis.
"""
from __future__ import annotations

import os
import re
import time
from typing import Iterable

from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("data.kalshi_markets")


_TITLE_RE = re.compile(
    r"^Will (?P<player>.+?) win the (?P<lastA>[^\s]+) vs (?P<lastB>[^\s]+):"
    r"\s*(?P<round>[^?]+) match\?\s*$"
)


def _client():
    try:
        from kalshi_sdk import KalshiClient
    except ImportError as exc:
        raise RuntimeError(
            "kalshi_sdk not installed in this venv — pip install -e the "
            "shared sdk under /root/kalshi_sdk (or set up the editable "
            "dep in your local checkout)"
        ) from exc
    api_key = os.environ.get("KALSHI_API_KEY_ID", "").strip()
    pkey = os.environ.get("KALSHI_PRIVATE_KEY_PATH", "").strip()
    if not api_key or not pkey:
        raise RuntimeError(
            "KALSHI_API_KEY_ID and KALSHI_PRIVATE_KEY_PATH must be set "
            "in the env (see the bot's systemd EnvironmentFile)"
        )
    return KalshiClient(api_key_id=api_key, private_key_path=pkey)


def _parse_title(title: str) -> dict[str, str]:
    if not title:
        return {}
    m = _TITLE_RE.match(title.strip())
    if not m:
        return {}
    return {
        "player": m.group("player").strip(),
        "round": m.group("round").strip(),
        "lastA": m.group("lastA").strip(),
        "lastB": m.group("lastB").strip(),
    }


def _to_float(v):
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def _market_price_prob(market: dict, side: str) -> float | None:
    """Side ask price as an implied probability. Tries Kalshi's new
    ``*_ask_dollars`` field first (0.60 → 60%), then the legacy cents
    field, so the same code works across both API generations."""
    if side == "yes":
        ya = (_to_float(market.get("yes_ask_dollars"))
               or (_to_float(market.get("yes_ask")) or 0.0) / 100.0)
        if ya:
            return max(0.01, min(0.99, ya))
    else:
        na = (_to_float(market.get("no_ask_dollars"))
               or (_to_float(market.get("no_ask")) or 0.0) / 100.0)
        if na:
            return max(0.01, min(0.99, na))
    return None


def _yes_price_dollars(market: dict) -> float | None:
    p = _market_price_prob(market, "yes")
    if p is not None:
        return p
    no_p = _market_price_prob(market, "no")
    if no_p is not None:
        return max(0.01, min(0.99, 1.0 - no_p))
    yb = (_to_float(market.get("yes_bid_dollars"))
           or (_to_float(market.get("yes_bid")) or 0.0) / 100.0)
    if yb:
        return max(0.01, min(0.99, yb))
    return None


def _ask_cents(market: dict, side: str) -> int | None:
    if side == "yes":
        d = _to_float(market.get("yes_ask_dollars"))
        if d is not None:
            return int(round(d * 100))
        c = _to_float(market.get("yes_ask"))
        return int(c) if c is not None else None
    d = _to_float(market.get("no_ask_dollars"))
    if d is not None:
        return int(round(d * 100))
    c = _to_float(market.get("no_ask"))
    return int(c) if c is not None else None


def _spread_cents(market: dict) -> float | None:
    ya_d = _to_float(market.get("yes_ask_dollars"))
    yb_d = _to_float(market.get("yes_bid_dollars"))
    if ya_d is not None and yb_d is not None:
        return (ya_d - yb_d) * 100.0
    ya = _to_float(market.get("yes_ask"))
    yb = _to_float(market.get("yes_bid"))
    if ya is not None and yb is not None:
        return float(ya) - float(yb)
    return None


def _volume(market: dict) -> float | None:
    return _to_float(market.get("volume_fp")) or _to_float(market.get("volume"))


def _open_interest(market: dict) -> float | None:
    return (_to_float(market.get("open_interest_fp"))
            or _to_float(market.get("open_interest")))


def _tournament_from_rules(rules: str) -> str:
    if not rules:
        return "WTT"
    m = re.search(
        r"\d{4}\s+([A-Z][A-Za-z .'-]+?"
        r"(?:Open|Cup|Championships|Smash|Star Contender|Contender|"
        r"Champions|Finals|Olympic|World))",
        rules,
    )
    if m:
        return m.group(1).strip()
    return "WTT"


def _round_to_code(round_str: str) -> str:
    if not round_str:
        return "R32"
    s = round_str.lower()
    if "round of 128" in s: return "R128"
    if "round of 64" in s: return "R64"
    if "round of 32" in s: return "R32"
    if "round of 16" in s: return "R16"
    if "quarter" in s: return "QF"
    if "semi" in s: return "SF"
    if "final" in s: return "F"
    if "group" in s: return "GRP"
    return "R32"


def _best_of_from_rules(rules: str) -> int:
    """Kalshi table-tennis rules sometimes name the match format.
    Default to best-of-7 (WTT senior tour standard); world-tour group
    stages occasionally use bo5."""
    if not rules:
        return 7
    s = rules.lower()
    if "best of 5" in s or "best-of-5" in s:
        return 5
    if "best of 7" in s or "best-of-7" in s:
        return 7
    if "best of 3" in s:
        return 3
    return 7


def fetch_table_tennis_markets(
    series: Iterable[str] | None = None,
    inter_series_pause_s: float = 1.0,
) -> list[dict]:
    """Pull every active Kalshi table-tennis market.

    Series come from ``config.kalshi.series`` by default. We iterate
    each in turn with a short pause to stay clear of the rate limit;
    any series that 404s or rate-limits is logged and skipped.
    """
    if series is None:
        cfg = load_config()
        series = list(cfg.get("kalshi", {}).get("series") or [])
    c = _client()
    out: list[dict] = []
    for s in series:
        try:
            for m in c.iter_open_markets(series_ticker=s):
                out.append(m)
        except Exception as exc:  # noqa: BLE001
            log.warning("fetch %s failed: %s", s, exc)
        time.sleep(inter_series_pause_s)
    log.info("fetched %d table-tennis markets across %d series",
             len(out), len(list(series)))
    return out


_TOURNAMENT_TITLE_RE = re.compile(
    r"^Will (?P<player>.+?) win the (?P<tourney>.+?)\??\s*$",
    re.IGNORECASE,
)


def _parse_tournament_title(title: str) -> dict[str, str]:
    """Parse a tournament-winner title like 'Will Ma Long win the 2026 ITTF
    World Championships?'. Used for >2-side events where each market is
    a separate player-as-YES contract for the tournament outcome."""
    if not title:
        return {}
    m = _TOURNAMENT_TITLE_RE.match(title.strip())
    if not m:
        return {}
    return {
        "player": m.group("player").strip(),
        "tournament": m.group("tourney").strip(),
    }


def collapse_to_matches(markets: list[dict],
                         prev_markets_by_ticker: dict[str, dict] | None = None
                         ) -> list[dict]:
    """Group Kalshi markets into watchlist records.

    Two shapes are handled:

      * **Head-to-head match events** (2 markets per event_ticker, one
        YES per player) — collapsed into a single match record where
        player_a is the alphabetically-first ticker's player and
        player_b is the other side.

      * **Tournament-winner events** (3+ markets per event_ticker, one
        YES per contender) — emitted as ONE ROW PER PLAYER, with
        player_b set to "Field". These don't fit the head-to-head
        model, so the exporter marks them ``_market_type=tournament``
        and skips edge-based signals; they're surfaced for visibility
        only.
    """
    by_event: dict[str, list[dict]] = {}
    for m in markets:
        ev = m.get("event_ticker") or ""
        if not ev:
            continue
        by_event.setdefault(ev, []).append(m)

    out: list[dict] = []
    for event_ticker, sides in by_event.items():
        if len(sides) < 1:
            continue
        sides.sort(key=lambda x: x.get("ticker") or "")

        # Tournament-winner shape — one row per side.
        if len(sides) >= 3:
            for m in sides:
                t = _parse_tournament_title(m.get("title", ""))
                player = t.get("player") or (m.get("ticker") or "").split("-")[-1]
                tournament_name = t.get("tournament") or _tournament_from_rules(
                    m.get("rules_primary") or "")
                mkt = _yes_price_dollars(m)
                prev = (prev_markets_by_ticker or {}).get(m.get("ticker") or "")
                prev_mkt = _yes_price_dollars(prev) if prev else None
                status_closed = (m.get("status") or "").lower() in (
                    "closed", "settled", "finalized"
                )
                out.append({
                    "match_id": m.get("ticker"),  # unique per player
                    "ticker_a": m.get("ticker"),
                    "ticker_b": None,
                    "tournament": tournament_name,
                    "surface": "Indoor",
                    "level": "GS",  # tournament markets are big events
                    "round": "F",
                    "best_of": 7,
                    "player_a": player,
                    "player_b": "Field",
                    "_market_type": "tournament",
                    "set_score_a": 0, "set_score_b": 0,
                    "current_game_score_a": 0, "current_game_score_b": 0,
                    "point_streak_a": 0, "point_streak_b": 0,
                    "is_deuce": False, "is_game_point_a": False,
                    "is_game_point_b": False, "is_set_point_a": False,
                    "is_set_point_b": False, "is_match_point_a": False,
                    "is_match_point_b": False, "is_closing_game": False,
                    "medical_timeout": False,
                    "injury_news_flag": False,
                    "retirement_risk_flag": False,
                    "market_prob_a": mkt,
                    "market_prob_a_prev": prev_mkt,
                    "completed": status_closed,
                    "winner_side": None,
                    "expected_expiration_time": m.get("expected_expiration_time"),
                    "rules_primary": m.get("rules_primary") or "",
                    "yes_ask_cents_a": _ask_cents(m, "yes"),
                    "yes_ask_cents_b": None,
                    "volume_a": _volume(m),
                    "open_interest_a": _open_interest(m),
                    "spread_cents": _spread_cents(m),
                    "title_a": m.get("title"),
                    "title_b": None,
                })
            continue

        # Head-to-head match shape — 2 markets, one row.
        a_market = sides[0]
        b_market = sides[1] if len(sides) >= 2 else None
        a_title = _parse_title(a_market.get("title", ""))
        b_title = (_parse_title(b_market.get("title", ""))
                    if b_market else {})
        player_a = (a_title.get("player")
                     or (a_market.get("ticker") or "").split("-")[-1])
        player_b = b_title.get("player") or ""
        if not player_b and b_market is None:
            lastA, lastB = a_title.get("lastA", ""), a_title.get("lastB", "")
            player_b = (lastB if a_title.get("player", "").endswith(lastA)
                         else lastA)
        rules = a_market.get("rules_primary") or ""
        tournament = _tournament_from_rules(rules)
        best_of = _best_of_from_rules(rules)
        round_str = a_title.get("round") or "R32"
        round_code = _round_to_code(round_str)

        market_yes_a = _yes_price_dollars(a_market)
        prev = (prev_markets_by_ticker or {}).get(a_market.get("ticker") or "")
        market_yes_a_prev = (_yes_price_dollars(prev) if prev else None)
        is_closed = (a_market.get("status") or "").lower() in (
            "closed", "settled", "finalized"
        )
        winner_side = None
        if is_closed and b_market is not None:
            ya = _ask_cents(a_market, "yes") or 0
            yb = _ask_cents(b_market, "yes") or 0
            if ya >= 99:
                winner_side = "PLAYER_A"
            elif yb >= 99:
                winner_side = "PLAYER_B"

        out.append({
            "match_id": event_ticker,
            "ticker_a": a_market.get("ticker"),
            "ticker_b": (b_market.get("ticker") if b_market else None),
            "tournament": tournament,
            "surface": "Indoor",
            "level": "ST",  # default to Star Contender; refined later
            "round": round_code,
            "best_of": best_of,
            "player_a": player_a,
            "player_b": player_b,
            "set_score_a": 0, "set_score_b": 0,
            "current_game_score_a": 0, "current_game_score_b": 0,
            "point_streak_a": 0, "point_streak_b": 0,
            "is_deuce": False,
            "is_game_point_a": False, "is_game_point_b": False,
            "is_set_point_a": False, "is_set_point_b": False,
            "is_match_point_a": False, "is_match_point_b": False,
            "is_closing_game": False,
            "medical_timeout": False,
            "injury_news_flag": False,
            "retirement_risk_flag": False,
            "market_prob_a": market_yes_a,
            "market_prob_a_prev": market_yes_a_prev,
            "completed": is_closed,
            "winner_side": winner_side,
            "expected_expiration_time": a_market.get("expected_expiration_time"),
            "rules_primary": rules,
            "yes_ask_cents_a": _ask_cents(a_market, "yes"),
            "yes_ask_cents_b": (_ask_cents(b_market, "yes") if b_market else None),
            "volume_a": _volume(a_market),
            "open_interest_a": _open_interest(a_market),
            "spread_cents": _spread_cents(a_market),
            "title_a": a_market.get("title"),
            "title_b": (b_market.get("title") if b_market else None),
        })
    return out


def write_live_state(records: list[dict]) -> str:
    import json
    cfg = load_config()
    fp = resolve_path(cfg["paths"]["raw_dir"]) / "live_state.json"
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, default=str)
    return str(fp)
