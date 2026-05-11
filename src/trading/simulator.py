"""Paper-trade simulator for the table-tennis bot.

Same shape + same state-file schema as the Tennis sibling so the
trading-dashboard's tennis adapter (which reads sim_state.json) renders
table-tennis positions out of the box. Differences from tennis:

  * No surface column carried through the position record (always
    "Indoor"); the dashboard renderer treats this as a stylistic field.
  * Cooldowns are slightly shorter because table-tennis matches are
    much faster than tennis — multiple settles can occur in a single
    Kalshi-fetch tick window.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging
from .buy_gate import evaluate as evaluate_buy
from .ev import ev as ev_calc

log = setup_logging("trading.simulator")


_TRADEABLE_LABELS = {"STRONG_EDGE", "SMALL_EDGE", "MARKET_OVERREACTION"}
# Shorter cooldown than tennis (60s) — table-tennis matches settle fast
# and we don't want to miss a fresh edge on the next match.
_SAME_MATCH_COOLDOWN_SECONDS = 45


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _state_path() -> Path:
    cfg = load_config()
    return resolve_path(cfg["paths"]["outputs_dir"]) / "sim_state.json"


def _empty_state() -> dict[str, Any]:
    return {
        "started_at": _now_iso(),
        "last_tick_at": _now_iso(),
        "open_positions": [],
        "closed_positions": [],
        "stats": _zero_stats(),
        "last_settled_at_by_match_id": {},
    }


def _zero_stats() -> dict[str, Any]:
    return {
        "total_opened": 0, "total_closed": 0, "open_count": 0,
        "wins": 0, "losses": 0, "win_rate": None,
        "total_realized_pnl": 0.0, "total_unrealized_pnl": 0.0,
        "total_staked": 0.0, "roi": None,
    }


def _load_state() -> dict[str, Any]:
    fp = _state_path()
    if not fp.exists():
        return _empty_state()
    try:
        with open(fp, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        log.warning("sim state at %s unreadable — starting fresh", fp)
        return _empty_state()
    for k, v in _empty_state().items():
        data.setdefault(k, v)
    return data


def _save_state(state: dict[str, Any]) -> None:
    fp = _state_path()
    fp.parent.mkdir(parents=True, exist_ok=True)
    with open(fp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def _aggregate_stats(state: dict[str, Any]) -> dict[str, Any]:
    open_positions = state.get("open_positions") or []
    closed = state.get("closed_positions") or []
    wins = sum(1 for c in closed if c.get("won"))
    losses = len(closed) - wins
    total_realized = sum(float(c.get("realized_pnl", 0)) for c in closed)
    total_unrealized = sum(float(p.get("unrealized_pnl", 0)) for p in open_positions)
    total_staked = sum(float(c.get("stake", 0)) for c in closed)
    win_rate = (wins / len(closed)) if closed else None
    roi = (total_realized / total_staked) if total_staked > 0 else None
    return {
        "total_opened": int(state["stats"].get("total_opened", 0)),
        "total_closed": len(closed),
        "open_count": len(open_positions),
        "wins": wins, "losses": losses,
        "win_rate": win_rate,
        "total_realized_pnl": round(total_realized, 4),
        "total_unrealized_pnl": round(total_unrealized, 4),
        "total_staked": round(total_staked, 4),
        "roi": roi,
    }


def _pick_side(model_prob_a: float, market_prob_a: float | None
                ) -> tuple[str, float, float]:
    if market_prob_a is None:
        return "PLAYER_A", 0.5, model_prob_a
    edge_a = model_prob_a - market_prob_a
    if edge_a >= 0:
        return "PLAYER_A", float(market_prob_a), float(model_prob_a)
    return "PLAYER_B", float(1.0 - market_prob_a), float(1.0 - model_prob_a)


def _within_cooldown(state: dict[str, Any], match_id: str) -> bool:
    last = state.get("last_settled_at_by_match_id", {}).get(match_id)
    if not last:
        return False
    try:
        ts = datetime.fromisoformat(last.replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    return (datetime.now(timezone.utc) - ts).total_seconds() < _SAME_MATCH_COOLDOWN_SECONDS


def _settle_position(p: dict[str, Any], live_record: dict[str, Any],
                      slippage: float) -> dict[str, Any]:
    won = (p["side"] == live_record.get("winner_side"))
    stake = float(p.get("stake", 1.0))
    entry = float(p["entry_market_prob"])
    if won:
        realized = stake * (1.0 - entry - slippage)
    else:
        realized = -stake * (entry + slippage)
    return {
        **p,
        "closed_at": _now_iso(),
        "winner_side": live_record.get("winner_side"),
        "won": bool(won),
        "result": "WIN" if won else "LOSS",
        "settle_market_prob": float(
            live_record.get("market_prob_a") if p["side"] == "PLAYER_A"
            else 1.0 - (live_record.get("market_prob_a") or 0.5)
        ),
        "realized_pnl": round(realized, 4),
    }


def tick(watchlist_rows: list[dict[str, Any]],
          live_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Run one simulator tick. Writes ``data/outputs/sim_state.json``."""
    cfg = load_config()
    t = cfg["trading"]
    slippage = float(t["slippage_pct"])
    stake = float(t["bet_size"])

    state = _load_state()
    live_by_id = {str(r.get("match_id") or ""): r for r in live_records}

    # 1) Settle completed matches
    still_open: list[dict[str, Any]] = []
    newly_closed: list[dict[str, Any]] = []
    for p in (state.get("open_positions") or []):
        live = live_by_id.get(p.get("match_id", ""))
        if live and live.get("completed") and live.get("winner_side"):
            closed = _settle_position(p, live, slippage)
            newly_closed.append(closed)
            state.setdefault("last_settled_at_by_match_id", {})[p["match_id"]] = closed["closed_at"]
            log.info("settled %s on %s vs %s — won=%s, P&L=%+.3f",
                      p["side"], p["player_a"], p["player_b"],
                      closed["won"], closed["realized_pnl"])
        else:
            still_open.append(p)

    state["closed_positions"] = (state.get("closed_positions") or []) + newly_closed
    state["open_positions"] = still_open

    # 2) Mark-to-market
    wl_by_id = {str(r.get("match_id") or ""): r for r in watchlist_rows}
    for p in state["open_positions"]:
        wl = wl_by_id.get(p.get("match_id", ""))
        if not wl:
            continue
        market_a = wl.get("market_prob_a")
        live_a = wl.get("live_prob_a")
        if market_a is not None:
            mark_for_side = float(market_a if p["side"] == "PLAYER_A"
                                    else 1.0 - market_a)
            p["current_market_prob"] = round(mark_for_side, 4)
            p["unrealized_pnl"] = round(
                stake * (mark_for_side - p["entry_market_prob"]), 4
            )
        if live_a is not None:
            p["current_model_prob"] = round(
                live_a if p["side"] == "PLAYER_A" else 1.0 - live_a, 4
            )

    # 3) Open new positions on the TOP-N buy-eligible rows ranked by
    #    edge × EV. Sort-and-cap here enforces "best N candidates" —
    #    we never open on a marginal eligible row while a stronger
    #    eligible row exists.
    open_match_ids = {p["match_id"] for p in state["open_positions"]}
    max_open = int(t.get("max_open_positions", 10))

    ranked = sorted(
        (r for r in watchlist_rows
            if r.get("buy_eligible") and r.get("match_id")),
        key=lambda r: -float(r.get("buy_score") or 0),
    )
    for r in ranked:
        if len(state["open_positions"]) >= max_open:
            break
        match_id = str(r.get("match_id") or "")
        if match_id in open_match_ids:
            continue
        if _within_cooldown(state, match_id):
            continue
        decision = evaluate_buy(r, t)
        if not decision.eligible:
            continue
        side = "PLAYER_A" if decision.side == "A" else "PLAYER_B"
        mkt_for_side = decision.side_market
        model_for_side = (float(r["live_prob_a"]) if side == "PLAYER_A"
                            else 1.0 - float(r["live_prob_a"]))
        # ── all gates passed — open the position ─────────────────────────
        side_player = r["player_a"] if side == "PLAYER_A" else r["player_b"]
        position_id = f"{match_id}-{side}-{int(datetime.now(timezone.utc).timestamp())}"
        title = (r.get("title_a") if side == "PLAYER_A"
                  else r.get("title_b")) or r.get("title") or ""
        new_p = {
            "position_id": position_id,
            "match_id": match_id,
            "tournament": r.get("tournament", ""),
            "surface": r.get("surface", "Indoor"),
            "player_a": r.get("player_a", ""),
            "player_b": r.get("player_b", ""),
            "side": side,
            "side_player": side_player,
            "title": title,
            "entry_market_prob": round(mkt_for_side, 4),
            "entry_model_prob": round(model_for_side, 4),
            "label_at_open": (r.get("recommended_action") or ""),
            "stake": stake,
            "slippage": slippage,
            "opened_at": _now_iso(),
            "current_market_prob": round(mkt_for_side, 4),
            "current_model_prob": round(model_for_side, 4),
            "unrealized_pnl": 0.0,
            "reason_at_open": r.get("reason_for_signal", ""),
        }
        state["open_positions"].append(new_p)
        open_match_ids.add(match_id)
        state["stats"]["total_opened"] = int(state["stats"].get("total_opened", 0)) + 1
        log.info("opened %s on %s (%s vs %s) — entry %.2f, model %.2f, label %s",
                  side, side_player, r["player_a"], r["player_b"],
                  mkt_for_side, model_for_side, new_p["label_at_open"])

    state["last_tick_at"] = _now_iso()
    state["stats"] = {**_aggregate_stats(state),
                      "total_opened": int(state["stats"].get("total_opened", 0))}
    _save_state(state)
    return state


def load_state() -> dict[str, Any]:
    return _load_state()
