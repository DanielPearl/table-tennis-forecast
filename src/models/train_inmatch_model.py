"""Train the table-tennis in-match adjustment model.

Same shape as the tennis sibling: replays per-set snapshots from the
TTelite archive (real, parsed point totals per game) and trains a
calibrated classifier that maps current set state → P(player_a wins
match). The rules layer in ``live_adjustment_model.py`` keeps doing
the within-set adjustments (streaks, deuce/game-point volatility)
and remains the source of the dashboard's ``rules_fired`` audit
string; only the numerical probability is taken from the model.

Validation: temporal holdout — most-recent ``test_window_days``
calendar window goes to the test set, everything earlier is train.
Test metrics versus the rules baseline are written to
``artifacts/inmatch_metrics.json``.
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import brier_score_loss, log_loss

from ..features.build_live_features import standardize
from ..features.build_pbp_snapshots import FEATURE_COLUMNS
from ..models.live_adjustment_model import adjust as rules_adjust
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("models.train_inmatch_model")


# Hold out the most-recent ~20% of matches by date as the test set.
TEST_FRAC = 0.20


def _split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    cutoff = int(n * (1 - TEST_FRAC))
    return df.iloc[:cutoff].reset_index(drop=True), df.iloc[cutoff:].reset_index(drop=True)


def _snapshot_to_live_record(row: pd.Series) -> dict:
    """Map a snapshot to the canonical live_record so we can score the
    rules baseline on the same rows."""
    rec = {
        "match_id": row.get("match_id", ""),
        "tournament": "TT Elite Series",
        "surface": "Indoor",
        "player_a": "P1",
        "player_b": "P2",
        "set_score_a": float(row.get("set_score_a", 0)),
        "set_score_b": float(row.get("set_score_b", 0)),
        "best_of": float(row.get("best_of", 5)),
        "current_game_score_a": 0.0,
        "current_game_score_b": 0.0,
        "point_streak_a": 0.0,
        "point_streak_b": 0.0,
        "point_win_pct_a_live": float(row.get("point_win_pct_a_live", 0.5)),
        "point_win_pct_b_live": float(row.get("point_win_pct_b_live", 0.5)),
        "games_won_last_3_a": float(row.get("games_won_last_3_a", 0)),
        "games_won_last_3_b": float(row.get("games_won_last_3_b", 0)),
        "is_closing_game": bool(row.get("is_closing_game", 0)),
        "serving_a": False,
    }
    return standardize(rec)


def _rules_baseline_probs(test: pd.DataFrame) -> np.ndarray:
    out = np.empty(len(test), dtype=float)
    for i, row in enumerate(test.itertuples(index=False)):
        s = pd.Series(row._asdict())
        live = _snapshot_to_live_record(s)
        adj = rules_adjust(0.5, live)
        out[i] = float(adj.live_prob_a)
    return out


def _bucket_metrics(p_pred: np.ndarray, y: np.ndarray,
                    key: np.ndarray,
                    buckets: list[tuple[float, float, str]]) -> list[dict]:
    rows = []
    for lo, hi, name in buckets:
        mask = (key >= lo) & (key < hi)
        n = int(mask.sum())
        if n == 0:
            continue
        rows.append({
            "bucket": name, "n": n,
            "brier": float(brier_score_loss(y[mask], p_pred[mask])),
            "logloss": float(log_loss(y[mask],
                                       np.clip(p_pred[mask], 1e-6, 1 - 1e-6),
                                       labels=[0, 1])),
            "acc": float(((p_pred[mask] >= 0.5) == y[mask].astype(bool)).mean()),
        })
    return rows


def train_and_eval() -> dict:
    cfg = load_config()
    proc_dir = resolve_path(cfg["paths"]["processed_dir"])
    snaps_path = proc_dir / "pbp_snapshots.csv"
    if not snaps_path.exists():
        raise FileNotFoundError(
            f"missing {snaps_path} — run features.build_pbp_snapshots first"
        )
    df = pd.read_csv(snaps_path)
    log.info("loaded %d snapshots (%d matches)", len(df), df["match_id"].nunique())

    train, test = _split(df)
    log.info("split: train=%d (oldest), test=%d (newest)", len(train), len(test))

    X_train = train[list(FEATURE_COLUMNS)].astype(float).values
    y_train = train["won_a"].astype(int).values
    X_test = test[list(FEATURE_COLUMNS)].astype(float).values
    y_test = test["won_a"].astype(int).values

    base = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.5,
        random_state=int(cfg["model"]["random_state"]),
    )
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=5)
    clf.fit(X_train, y_train)
    p_test = clf.predict_proba(X_test)[:, 1]

    p_rules = _rules_baseline_probs(test)

    metrics = {
        "n_train": len(train),
        "n_test": len(test),
        "model": {
            "brier": float(brier_score_loss(y_test, p_test)),
            "logloss": float(log_loss(y_test, np.clip(p_test, 1e-6, 1 - 1e-6),
                                       labels=[0, 1])),
            "acc": float(((p_test >= 0.5) == y_test.astype(bool)).mean()),
        },
        "rules_baseline": {
            "brier": float(brier_score_loss(y_test, p_rules)),
            "logloss": float(log_loss(y_test, np.clip(p_rules, 1e-6, 1 - 1e-6),
                                       labels=[0, 1])),
            "acc": float(((p_rules >= 0.5) == y_test.astype(bool)).mean()),
        },
        "by_progress_model": _bucket_metrics(
            p_test, y_test, test["progress"].values,
            [(0.0, 0.4, "early"), (0.4, 0.7, "mid"),
             (0.7, 0.95, "late"), (0.95, 5.0, "endgame")]),
        "by_progress_rules": _bucket_metrics(
            p_rules, y_test, test["progress"].values,
            [(0.0, 0.4, "early"), (0.4, 0.7, "mid"),
             (0.7, 0.95, "late"), (0.95, 5.0, "endgame")]),
        "by_setsdiff_model": _bucket_metrics(
            p_test, y_test, test["sets_diff"].values,
            [(-5, -1.5, "down 2+"), (-1.5, -0.5, "down 1"),
             (-0.5, 0.5, "even"), (0.5, 1.5, "up 1"),
             (1.5, 5, "up 2+")]),
        "feature_columns": list(FEATURE_COLUMNS),
    }

    art_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    art_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, art_dir / "inmatch_model.joblib")
    with open(art_dir / "inmatch_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    m = metrics["model"]
    r = metrics["rules_baseline"]
    log.info("OVERALL  model:  brier=%.4f  logloss=%.4f  acc=%.3f",
             m["brier"], m["logloss"], m["acc"])
    log.info("OVERALL  rules:  brier=%.4f  logloss=%.4f  acc=%.3f",
             r["brier"], r["logloss"], r["acc"])
    log.info("Brier improvement: %.4f (%.1f%% relative)",
             r["brier"] - m["brier"],
             100 * (r["brier"] - m["brier"]) / r["brier"])
    return metrics


if __name__ == "__main__":
    train_and_eval()
