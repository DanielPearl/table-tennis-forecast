"""End-to-end watchlist exporter for table tennis.

Same shape as the tennis sibling — runs each live-state record through:

  pre-match model → live adjustment → EV/edge → signal label

…and writes the canonical watchlist JSON + CSV the dashboard reads.
Output schema is identical to tennis so the trading-dashboard tennis
adapter can render table-tennis rows verbatim.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from kalshi_sdk.pinnacle import pick_pair_entry

from ..data.fetch_live_scores import load_live_state
from ..features.build_live_features import standardize
from ..models.live_adjustment_model import adjust as live_adjust
from ..models.predict import players_known, safe_predict
from ..data.fetch_odds import pinnacle_probs_by_pair
from ..trading.buy_gate import evaluate as evaluate_buy
from ..trading.ev import ev as ev_calc
from ..trading.signals import label_match
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("dashboard.export")


def _format_score(rec: dict[str, Any]) -> str:
    a = int(rec.get("set_score_a") or 0)
    b = int(rec.get("set_score_b") or 0)
    return f"{a}-{b}"


def _round_label(level: str, round_: str) -> str:
    return f"{level} / {round_}" if round_ else level


_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12}


def _ticker_anchor(ticker: str | None) -> tuple[str | None, float]:
    """(ISO anchor, matching window hours) from a Kalshi TT ticker.

    ``KXTTELITEMATCH-26SEP102130GMIMMA-GMI`` embeds date + HHMM (ET,
    Kalshi's ticker convention) → tight 6h window, because Liga Pro /
    TT Elite put the SAME pair on the table twice in one day and a
    date-level match would share one line across both. Date-only
    tickers fall back to noon UTC + 24h.
    """
    if not ticker:
        return None, 24.0
    try:
        body = ticker.split("-")[1]
        yy = int(body[:2])
        mon = _MONTHS.get(body[2:5])
        dd = int(body[5:7])
        if mon is None:
            return None, 24.0
        hhmm = body[7:11]
        if len(hhmm) == 4 and hhmm.isdigit():
            from zoneinfo import ZoneInfo
            et = datetime(2000 + yy, mon, dd, int(hhmm[:2]), int(hhmm[2:]),
                          tzinfo=ZoneInfo("America/New_York"))
            return et.astimezone(timezone.utc).isoformat(), 6.0
        return f"20{yy:02d}-{mon:02d}-{dd:02d}T12:00:00Z", 24.0
    except (IndexError, ValueError):
        return None, 24.0


def _pinnacle_prob_for(lookup: dict, player_a: str, player_b: str,
                        anchor_iso: str | None = None,
                        max_delta_hours: float = 24.0,
                        ) -> tuple[float | None, str | None]:
    """(player-A probability, scheduled start ISO) from the Pinnacle
    pair lookup, or (None, None).

    Same matching as tennis-forecast: exact frozenset first, then a
    loose last-name scan so a diacritic / initial difference between
    Kalshi's and Pinnacle's spellings doesn't drop the line. Keys
    starting with "_" ("_source") are SDK metadata, not names.
    ``anchor_iso`` (the contract's own day) picks the right entry when
    the pair has lines on more than one day (Liga Pro rematches are
    routine); the returned start feeds the prematch-only trade rule.
    """
    if not lookup or not player_a or not player_b:
        return None, None
    pinn_map = lookup.get(frozenset({player_a, player_b}))
    if pinn_map is None:
        la = player_a.split()[-1].lower()
        lb = player_b.split()[-1].lower()
        for key_set, probs in lookup.items():
            names = [str(n) for n in key_set
                     if not str(n).startswith("_")]
            if len(names) != 2:
                continue
            lnames = [n.split()[-1].lower() for n in names]
            if la in lnames and lb in lnames:
                pinn_map = probs
                break
    if pinn_map is not None and anchor_iso is not None:
        pinn_map = pick_pair_entry(pinn_map, anchor_iso,
                                   max_delta_hours=max_delta_hours)
    if pinn_map is None:
        return None, None
    start = pinn_map.get("_start")
    for name, prob in pinn_map.items():
        if str(name).startswith("_"):
            continue
        if name == player_a:
            return float(prob), start
    la = player_a.split()[-1].lower()
    for name, prob in pinn_map.items():
        if str(name).startswith("_"):
            continue
        if la in str(name).lower():
            return float(prob), start
    return None, start


def _load_comeback_rates() -> dict[str, float]:
    """Optional per-player comeback rates that the live-adjustment model
    uses to dampen score-state nudges. Stored in
    ``data/processed/artifacts/comeback_rates.joblib`` when the trainer
    has emitted them; absent on first boot."""
    cfg = load_config()
    fp = resolve_path(cfg["paths"]["artifacts_dir"]) / "comeback_rates.joblib"
    if not fp.exists():
        return {}
    try:
        return dict(joblib.load(fp))
    except Exception:  # noqa: BLE001
        return {}


def build_watchlist_records(live_records: list[dict[str, Any]] | None = None
                             ) -> list[dict[str, Any]]:
    cfg = load_config()
    slip = float(cfg["trading"]["slippage_pct"])

    if live_records is None:
        live_records = load_live_state()

    comeback_rates = _load_comeback_rates()

    # Professional benchmark for the whole batch — Pinnacle's guest
    # feed quotes the TT Elite / Liga Pro circuits heavily. Cached
    # inside the SDK helper; empty dict when the feed is down.
    pinnacle_lookup = pinnacle_probs_by_pair()

    out: list[dict[str, Any]] = []
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")

    for raw in live_records:
        rec = standardize(raw)
        market_type = raw.get("_market_type") or "match"
        market_prob_a = rec.get("market_prob_a")

        a_known, b_known = players_known(rec["player_a"], rec["player_b"])
        no_model_opinion = (market_type == "tournament"
                              or not (a_known or b_known))
        if no_model_opinion:
            # Either tournament-winner market (no head-to-head shape) or
            # a head-to-head where neither side is in the Elo state
            # (e.g. ITTF World Team Championships — the model was
            # trained on individual-player Elo, has no country
            # ratings). Surface the row with the market price + no
            # model opinion — edge = 0 by construction, signals emit
            # WATCH only, BUY gate never fires.
            # No model probability AT ALL (user 2026-09-08: a market
            # echo displayed as Model % "is not valid — don't show
            # it"). Rows with None model prob drop out of
            # Model-vs-market via the dashboard's no-model-%-no-row
            # rule; held rows stay visible regardless.
            pre_prob_a = None
            live_prob_a = None
            edge_a = 0.0 if market_prob_a is not None else None
            edge_b = 0.0 if market_prob_a is not None else None
            ev_a = 0.0 if market_prob_a is not None else None
            ev_b = 0.0 if market_prob_a is not None else None
            from types import SimpleNamespace
            adj = SimpleNamespace(
                volatility_score=0.05, injury_news_flag=False,
                market_overreaction=False, rules_fired=[],
            )
            reason = ("tournament-winner market — informational only"
                       if market_type == "tournament"
                       else f"no Elo data for {rec['player_a']} or "
                            f"{rec['player_b']} — model has no opinion")
            sig = SimpleNamespace(
                label="WATCH", reason=reason, confidence_score=0.5,
            )
        else:
            pre = safe_predict(
                rec["player_a"], rec["player_b"],
                level=raw.get("level", "ST"),
                round_=raw.get("round", "R32"),
                best_of=int(raw.get("best_of") or 7),
                rank_a=raw.get("rank_a"), rank_b=raw.get("rank_b"),
                hand_a=raw.get("hand_a", "R"), hand_b=raw.get("hand_b", "R"),
            )
            pre_prob_a = pre["prob_a"]
            adj = live_adjust(
                pre_prob_a, rec,
                player_a_comeback_rate=float(comeback_rates.get(rec["player_a"], 0.0)),
                player_b_comeback_rate=float(comeback_rates.get(rec["player_b"], 0.0)),
            )
            live_prob_a = adj.live_prob_a
            edge_a = (live_prob_a - market_prob_a) if market_prob_a is not None else None
            edge_b = -edge_a if edge_a is not None else None
            ev_a = (ev_calc(live_prob_a, market_prob_a, slip).ev_per_contract
                     if market_prob_a is not None else None)
            ev_b = (ev_calc(1 - live_prob_a, 1 - market_prob_a, slip).ev_per_contract
                     if market_prob_a is not None else None)
            sig = label_match(
                live_prob_a, market_prob_a,
                volatility=adj.volatility_score,
                injury_flag=adj.injury_news_flag,
                market_overreaction=adj.market_overreaction,
                rules_fired=adj.rules_fired,
            )

        # Sharp-reference override, mirroring tennis (2026-09-07):
        # when Pinnacle quotes the match, its devigged probability
        # drives edge / EV / the signal label (and, below, the buy
        # gate) — the Elo model becomes the fallback reference plus a
        # disagreement veto. This also gives rows the Elo state has
        # no opinion on a real, tradeable benchmark.
        _anchor, _win = _ticker_anchor(raw.get("ticker_a")
                                       or rec.get("match_id"))
        pinnacle_prob_a, bench_start = _pinnacle_prob_for(
            pinnacle_lookup, rec["player_a"], rec["player_b"],
            anchor_iso=_anchor, max_delta_hours=_win)
        pinnacle_prob_b = (1.0 - pinnacle_prob_a
                           if pinnacle_prob_a is not None else None)
        if pinnacle_prob_a is not None:
            edge_a = ((pinnacle_prob_a - market_prob_a)
                      if market_prob_a is not None else None)
            edge_b = -edge_a if edge_a is not None else None
            ev_a = (ev_calc(pinnacle_prob_a, market_prob_a,
                            slip).ev_per_contract
                    if market_prob_a is not None else None)
            ev_b = (ev_calc(1 - pinnacle_prob_a, 1 - market_prob_a,
                            slip).ev_per_contract
                    if market_prob_a is not None else None)
            sig = label_match(
                pinnacle_prob_a, market_prob_a,
                volatility=adj.volatility_score,
                injury_flag=bool(adj.injury_news_flag),
                market_overreaction=bool(adj.market_overreaction),
                rules_fired=list(getattr(adj, "rules_fired", []) or []),
            )

        market_prob_b = (1.0 - market_prob_a) if market_prob_a is not None else None

        row = {
            "match_id": rec["match_id"] or f"{rec['player_a']}-{rec['player_b']}",
            "tournament": rec["tournament"],
            "surface": rec["surface"],
            "player_a": rec["player_a"],
            "player_b": rec["player_b"],
            "current_score": _format_score(rec),
            # Prematch-only rule (2026-09-10): benchmark's scheduled
            # start (falls back to the ticker's own ET timestamp) plus
            # a live-score started flag; the shared gate and the live
            # executor both refuse entries once either marks the match
            # under way.
            "kickoff": bench_start or _anchor,
            "match_started": bool(
                (rec.get("set_score_a") or 0) > 0
                or (rec.get("set_score_b") or 0) > 0
                or rec.get("completed")),
            "round_label": _round_label(raw.get("level", "ST"),
                                          raw.get("round", "")),
            "pre_match_prob_a": (round(pre_prob_a, 4)
                                  if pre_prob_a is not None else None),
            "pre_match_prob_b": (round(1 - pre_prob_a, 4)
                                  if pre_prob_a is not None else None),
            "live_prob_a": (round(live_prob_a, 4)
                             if live_prob_a is not None else None),
            "live_prob_b": (round(1 - live_prob_a, 4)
                             if live_prob_a is not None else None),
            "pinnacle_prob_a": (round(pinnacle_prob_a, 4)
                                 if pinnacle_prob_a is not None else None),
            "pinnacle_prob_b": (round(pinnacle_prob_b, 4)
                                 if pinnacle_prob_b is not None else None),
            "market_prob_a": round(market_prob_a, 4) if market_prob_a is not None else None,
            "market_prob_b": round(market_prob_b, 4) if market_prob_b is not None else None,
            "edge_a": round(edge_a, 4) if edge_a is not None else None,
            "edge_b": round(edge_b, 4) if edge_b is not None else None,
            "ev_a": round(ev_a, 4) if ev_a is not None else None,
            "ev_b": round(ev_b, 4) if ev_b is not None else None,
            "confidence_score": round(sig.confidence_score, 4),
            "volatility_score": round(adj.volatility_score, 4),
            "injury_news_flag": bool(adj.injury_news_flag),
            "recommended_action": sig.label,
            "reason_for_signal": sig.reason,
            "last_updated": now,
            "open_interest": raw.get("open_interest_a"),
            "volume": ((raw.get("volume_a") or 0)
                       + (raw.get("volume_b") or 0)),
            "spread_cents": raw.get("spread_cents"),
            "yes_ask_cents_a": raw.get("yes_ask_cents_a"),
            "yes_ask_cents_b": raw.get("yes_ask_cents_b"),
            "title_a": raw.get("title_a"),
            "title_b": raw.get("title_b"),
            "title": (raw.get("title_a") if (edge_a or 0) >= 0
                       else raw.get("title_b")),
            # Kalshi event-page heading — passed through so the
            # dashboard's Title column matches the click target.
            "event_title": raw.get("event_title"),
            # Kalshi's rules paragraph — the dashboard's per-row
            # Rules "i" popover reads it; without this the TT pane's
            # Model-vs-market rows rendered blank Rules cells (user
            # 2026-09-11: "include the rules ... in every table").
            "rules_primary": raw.get("rules_primary"),
        }
        # Buy-gate reference cascade — Pinnacle when quoted, else the
        # Elo model. Same shape as tennis, including the relaxed
        # model-disagreement veto: the internal model only blocks a
        # sharp-book signal when it ACTIVELY disagrees by >10pp on the
        # same side; a silent model is "no vote", not "no".
        if live_prob_a is None and pinnacle_prob_a is None:
            # No model, no benchmark — nothing to evaluate.
            row["buy_eligible"] = False
            row["buy_score"] = 0.0
            row["buy_side"] = None
            row["buy_side_edge"] = 0.0
            row["buy_side_ev"] = None
            row["buy_gates"] = {}
            row["buy_blockers"] = ["no_model_or_benchmark"]
            out.append(row)
            continue
        if pinnacle_prob_a is not None:
            gate_row = dict(row)
            gate_row["live_prob_a"] = pinnacle_prob_a
            decision = evaluate_buy(gate_row, cfg.get("trading") or {})
            _disagree_floor = 0.10
            if (decision.eligible and decision.side in ("A", "B")
                    and live_prob_a is not None and not no_model_opinion):
                _ask_c = row.get("yes_ask_cents_a" if decision.side == "A"
                                  else "yes_ask_cents_b")
                _model_side = (float(live_prob_a) if decision.side == "A"
                                else 1.0 - float(live_prob_a))
                _model_edge = ((_model_side - float(_ask_c) / 100.0)
                                if _ask_c is not None else None)
                if _model_edge is not None and _model_edge < -_disagree_floor:
                    decision.eligible = False
                    decision.blockers = list(decision.blockers) + [
                        f"internal_model_disagrees_{_model_edge*100:+.1f}pp"
                        f"<-{_disagree_floor*100:.0f}pp"
                    ]
                    decision.gates = dict(decision.gates,
                                           model_confirms=False)
        else:
            decision = evaluate_buy(row, cfg.get("trading") or {})
        row["buy_eligible"] = bool(decision.eligible)
        row["buy_score"] = round(float(decision.score), 6)
        row["buy_side"] = decision.side
        row["buy_side_edge"] = round(float(decision.side_edge), 4)
        row["buy_side_ev"] = (round(float(decision.side_ev), 4)
                                if decision.side_ev is not None else None)
        row["buy_gates"] = decision.gates
        row["buy_blockers"] = decision.blockers
        out.append(row)

    return out


def export(records: list[dict[str, Any]] | None = None) -> tuple[Path, Path]:
    cfg = load_config()
    csv_path = resolve_path(cfg["paths"]["watchlist_csv"])
    json_path = resolve_path(cfg["paths"]["watchlist_json"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    rows = records if records is not None else build_watchlist_records()
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"generated_at": datetime.now(timezone.utc).isoformat(),
             "rows": rows},
            f, indent=2, default=str,
        )
    log.info("wrote %s + %s (%d rows)", csv_path, json_path, len(rows))
    return csv_path, json_path


if __name__ == "__main__":
    export()
