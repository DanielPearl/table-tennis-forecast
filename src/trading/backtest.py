"""Backtest harness. Same scaffolding as the tennis sibling — runs the
trained model against the held-out match window, simulates a unit-stake
trading policy, and reports ROI / drawdown / per-cohort splits.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..features.build_prematch_features import (
    build_full_panel, build_player_a_panel, select_features,
)
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging
from .ev import ev as ev_calc

log = setup_logging("trading.backtest")


@dataclass
class BacktestResult:
    rows_tested: int
    accuracy: float
    avg_edge_taken: float
    bets_placed: int
    win_rate: float
    avg_ev: float
    simulated_roi: float
    max_drawdown: float


def _simulated_market_prob(model_prob: float, rng: np.random.Generator,
                             noise_sigma: float = 0.04) -> float:
    """For a no-real-prices backtest, sample a noisy version of the
    Elo-implied probability. NOT real ROI — useful as a sanity check.
    Drop your captured closing market prices into a ``closing_market_prob``
    column on the test panel to override this with the real number."""
    p = model_prob + rng.normal(0, noise_sigma)
    return float(np.clip(p, 0.05, 0.95))


def run() -> dict:
    from ..data.fetch_matches import fetch_all
    cfg = load_config()
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])

    bundle = joblib.load(artifacts_dir / "prematch_model.joblib")
    feature_list = bundle["feature_list"]

    matches = fetch_all()
    panel, _, _, _ = build_full_panel(matches, elo_cfg=cfg["elo"])
    oriented = build_player_a_panel(panel)
    oriented = oriented.sort_values("match_date").reset_index(drop=True)

    cutoff_months = int(cfg["model"]["test_window_months"])
    cutoff = oriented["match_date"].max() - pd.DateOffset(months=cutoff_months)
    test = oriented[oriented["match_date"] >= cutoff].copy()
    if test.empty:
        log.warning("no test rows in last %d months — using last 20%%",
                    cutoff_months)
        n = max(1, int(len(oriented) * 0.2))
        test = oriented.iloc[-n:].copy()

    X_test = select_features(test, feature_list)
    y_test = test["y"].values

    p_ens = bundle["ensemble"].predict_proba(X_test)[:, 1]
    p_log = bundle["logistic"].predict_proba(
        X_test[bundle["elo_only_features"]])[:, 1]
    p = (bundle["blend_weight_ensemble"] * p_ens
         + bundle["blend_weight_logistic"] * p_log)

    rng = np.random.default_rng(int(cfg["model"]["random_state"]))
    market_p = np.array(
        [_simulated_market_prob(float(pi), rng) for pi in p]
    )
    slippage = float(cfg["trading"]["slippage_pct"])
    edges = p - market_p

    bets = []
    pnl = 0.0
    pnl_curve = []
    for prob, mp, edge, y in zip(p, market_p, edges, y_test):
        # Buy the side with the bigger edge (mirrors the live simulator).
        if abs(edge) < float(cfg["trading"]["small_edge_min"]):
            pnl_curve.append(pnl); continue
        if edge >= 0:
            side_p, side_mkt, side_y = prob, mp, y
        else:
            side_p, side_mkt, side_y = 1 - prob, 1 - mp, 1 - y
        if not (float(cfg["trading"]["min_market_prob"]) <= side_mkt
                <= float(cfg["trading"]["max_market_prob"])):
            pnl_curve.append(pnl); continue
        bets.append(1)
        ev_obj = ev_calc(side_p, side_mkt, slippage)
        if int(side_y) == 1:
            pnl += 1.0 - side_mkt - slippage
        else:
            pnl -= side_mkt + slippage
        pnl_curve.append(pnl)

    accuracy = float(((p >= 0.5) == y_test).mean()) if len(p) else 0.0
    win_rate = (float(sum(1 for c in pnl_curve if c > 0)) / max(1, len(pnl_curve))
                 if pnl_curve else 0.0)
    avg_ev = float(np.mean([
        ev_calc(float(pi), float(mi), slippage).ev_per_contract
        for pi, mi in zip(p, market_p)
    ])) if len(p) else 0.0
    bets_placed = int(sum(bets))
    avg_edge_taken = float(np.mean([abs(e) for e, b in zip(edges, bets) if b])
                            if bets_placed else 0.0)
    # ROI = realized P&L / total stake. With $1 unit stakes:
    sim_roi = (float(pnl) / max(1, bets_placed)) if bets_placed else 0.0
    # Max drawdown: peak-to-trough on the running P&L curve.
    peaks = np.maximum.accumulate(np.array(pnl_curve)) if pnl_curve else np.array([0.0])
    drawdowns = peaks - np.array(pnl_curve) if pnl_curve else np.array([0.0])
    max_dd = float(np.max(drawdowns)) if len(drawdowns) else 0.0

    res = BacktestResult(
        rows_tested=len(p),
        accuracy=accuracy,
        avg_edge_taken=avg_edge_taken,
        bets_placed=bets_placed,
        win_rate=win_rate,
        avg_ev=avg_ev,
        simulated_roi=sim_roi,
        max_drawdown=max_dd,
    )
    log.info("backtest: %s", json.dumps(asdict(res), indent=2))

    csv_path = resolve_path(cfg["paths"]["backtest_csv"])
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    out_df = test.copy()
    out_df["model_prob_a"] = p
    out_df["sim_market_prob_a"] = market_p
    out_df["edge"] = edges
    out_df.to_csv(csv_path, index=False)
    return asdict(res)


if __name__ == "__main__":
    run()
