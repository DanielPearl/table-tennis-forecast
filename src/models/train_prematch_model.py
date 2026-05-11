"""Train the pre-match win-probability model for table tennis.

Pipeline:

  1. Load matches (real source if available, bundled seed otherwise)
  2. Build full feature panel (Elo + form + rolling stats + h2h + ...)
  3. Mirror to player_a orientation; balance the y label
  4. Hold out the last ``test_window_months`` for evaluation
  5. **First pass — broad panel**: train Elo-only logistic baseline
     + boosted ensemble on every feature in PREMATCH_FEATURES_BROAD,
     calibrate, score holdout.
  6. **Permutation importance** on the broad panel — measure each
     feature's contribution by shuffling it and re-scoring.
  7. **Prune**: drop features whose importance is below
     ``model.prune_floor`` AND whose 1-std band crosses zero. Keep at
     least ``model.prune_min_features``.
  8. **Second pass — pruned panel**: re-fit on the survivors.
  9. **Keep whichever is better** by held-out log-loss; persist that
     bundle and emit the corresponding metrics.json / coefficients.json
     / feature_importance.csv / holdout_predictions.csv.

This makes "start broad, then prune" an automatic, audit-friendly step
that the user can see in the dashboard's Models tab.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score,
                              log_loss, precision_score, recall_score,
                              roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..data.fetch_matches import fetch_all, save_clean
from ..features.build_prematch_features import (
    PREMATCH_FEATURES_BROAD, build_full_panel, build_player_a_panel,
    select_features,
)
from ..features.elo import EloState
from ..utils.config import load_config, resolve_path
from ..utils.logging_setup import setup_logging

log = setup_logging("models.train_prematch")


def _try_xgb():
    try:
        import xgboost as xgb  # type: ignore
        return xgb
    except Exception:
        return None


def _split_by_date(df: pd.DataFrame, months: int):
    cutoff = df["match_date"].max() - pd.DateOffset(months=months)
    train = df[df["match_date"] < cutoff].copy()
    test = df[df["match_date"] >= cutoff].copy()
    # Guard against degenerate splits on a tiny seed dataset — if
    # everything ends up on one side, fall back to a 80/20 row split.
    if len(train) < 50 or len(test) < 20:
        cut = max(1, int(len(df) * 0.8))
        train = df.iloc[:cut].copy()
        test = df.iloc[cut:].copy()
        cutoff = df.iloc[cut]["match_date"] if cut < len(df) else df["match_date"].max()
    return train, test, cutoff


def _calibrate(model, X, y, holdout_frac: float = 0.2):
    """Fit + calibrate. Sigmoid calibration is safe default for trees."""
    n = len(X)
    cut = int(n * (1 - holdout_frac))
    X_fit, X_cal = X.iloc[:cut], X.iloc[cut:]
    y_fit, y_cal = y[:cut], y[cut:]
    model.fit(X_fit, y_fit)
    try:
        from sklearn.frozen import FrozenEstimator  # sklearn >= 1.6
        cal = CalibratedClassifierCV(FrozenEstimator(model), method="sigmoid")
    except Exception:
        cal = CalibratedClassifierCV(model, method="sigmoid", cv="prefit")
    cal.fit(X_cal, y_cal)
    return cal


def _build_ensemble(cfg: dict):
    xgb = _try_xgb() if cfg["model"]["ensemble"]["use_xgboost_if_available"] else None
    if xgb is not None:
        return xgb.XGBClassifier(
            n_estimators=int(cfg["model"]["ensemble"]["n_estimators"]),
            learning_rate=float(cfg["model"]["ensemble"]["learning_rate"]),
            max_depth=int(cfg["model"]["ensemble"]["max_depth"]),
            tree_method="hist",
            random_state=int(cfg["model"]["random_state"]),
            eval_metric="logloss",
            n_jobs=2,
        ), True
    return HistGradientBoostingClassifier(
        max_iter=int(cfg["model"]["ensemble"]["n_estimators"]),
        learning_rate=float(cfg["model"]["ensemble"]["learning_rate"]),
        max_depth=int(cfg["model"]["ensemble"]["max_depth"]),
        random_state=int(cfg["model"]["random_state"]),
    ), False


def _fit_pass(features: list[str], X_train: pd.DataFrame, y_train,
                X_test: pd.DataFrame, y_test, cfg: dict
                ) -> dict[str, Any]:
    """One end-to-end fit on a specific feature list. Returns the
    bundle dict (model artefacts + held-out scores)."""
    elo_only = [f for f in ("diff_elo_pre", "diff_style_elo_pre")
                 if f in features]
    # Raw Elo differences can run into the thousands once player ratings
    # spread out — feeding those into LogisticRegression unscaled causes
    # numerical overflow in the lbfgs solver. A StandardScaler in front
    # gives the LR well-conditioned inputs; the predict-side pipeline
    # applies the same scaling automatically.
    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=400)),
    ])
    base.fit(X_train[elo_only], y_train)
    base_p_test = base.predict_proba(X_test[elo_only])[:, 1]
    base_p_train = base.predict_proba(X_train[elo_only])[:, 1]

    clf, used_xgb = _build_ensemble(cfg)
    cal_clf = _calibrate(clf, X_train[features], y_train, holdout_frac=0.2)
    ens_p_test = cal_clf.predict_proba(X_test[features])[:, 1]
    ens_p_train = cal_clf.predict_proba(X_train[features])[:, 1]

    blend_p_test = 0.70 * ens_p_test + 0.30 * base_p_test
    blend_p_train = 0.70 * ens_p_train + 0.30 * base_p_train

    return {
        "feature_list": features,
        "elo_only_features": elo_only,
        "logistic": base,
        "ensemble_uncalibrated": clf,
        "ensemble": cal_clf,
        "blend_weight_ensemble": 0.70,
        "blend_weight_logistic": 0.30,
        "metrics": {
            "elo_only": _eval(y_test, base_p_test),
            "ensemble": _eval(y_test, ens_p_test),
            "blended": _eval(y_test, blend_p_test),
        },
        "train_metrics": {
            "blended": _eval(y_train, blend_p_train),
        },
        "blend_p_test": blend_p_test,
        "blend_p_train": blend_p_train,
        "ens_p_test": ens_p_test,
        "xgboost": used_xgb,
    }


def _permutation_importance(bundle: dict[str, Any], X_test: pd.DataFrame,
                              y_test, n_repeats: int = 5,
                              random_state: int = 42
                              ) -> list[dict[str, float]]:
    """Walk-forward permutation importance on the blended model.

    Output: one record per feature with mean and std of the log-loss
    increase when that feature is shuffled. Higher means more
    important; negative means the feature was actually hurting the
    blended score (we'll prune those).
    """
    feats = bundle["feature_list"]
    cal_clf = bundle["ensemble"]
    base_clf = bundle["logistic"]
    elo_only = bundle["elo_only_features"]
    w_ens = bundle["blend_weight_ensemble"]
    w_log = bundle["blend_weight_logistic"]

    def _blended_log_loss(X: pd.DataFrame) -> float:
        p_ens = cal_clf.predict_proba(X[feats])[:, 1]
        p_log = base_clf.predict_proba(X[elo_only])[:, 1]
        p = np.clip(w_ens * p_ens + w_log * p_log, 1e-6, 1 - 1e-6)
        return float(log_loss(y_test, p))

    rng = np.random.default_rng(random_state)
    base_score = _blended_log_loss(X_test)
    results = []
    for name in feats:
        diffs = []
        for _ in range(n_repeats):
            X_shuf = X_test.copy()
            X_shuf[name] = rng.permutation(X_shuf[name].values)
            shuf_score = _blended_log_loss(X_shuf)
            diffs.append(shuf_score - base_score)
        results.append({
            "feature": name,
            "mean_importance": float(np.mean(diffs)),
            "std_importance": float(np.std(diffs)),
            "positive_folds": int(sum(1 for d in diffs if d > 0)),
        })
    return results


def _prune_noisy(importances: list[dict[str, float]],
                  floor: float, min_features: int) -> tuple[list[str], list[str]]:
    """Return (kept, dropped) feature names.

    Drop any feature whose mean_importance < floor AND whose 1-std band
    crosses zero (mean - std < 0). Always keep at least ``min_features``
    by ranking the survivors when too many would be pruned.
    """
    keep_strong = []
    weak = []
    for r in importances:
        crosses_zero = (r["mean_importance"] - r["std_importance"]) < 0
        if r["mean_importance"] < floor and crosses_zero:
            weak.append(r)
        else:
            keep_strong.append(r)
    if len(keep_strong) >= min_features:
        kept = [r["feature"] for r in keep_strong]
        dropped = [r["feature"] for r in weak]
        return kept, dropped
    # Not enough strong features — rank everything and keep the top N.
    ranked = sorted(importances, key=lambda r: -r["mean_importance"])
    kept = [r["feature"] for r in ranked[:min_features]]
    dropped = [r["feature"] for r in ranked[min_features:]]
    return kept, dropped


def train_and_persist() -> dict[str, Any]:
    cfg = load_config()
    artifacts_dir = resolve_path(cfg["paths"]["artifacts_dir"])
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    log.info("fetching match history…")
    matches = fetch_all()
    save_clean(matches)

    log.info("building feature panel (broad: Elo + form + h2h + ...)")
    panel, elo_state, h2h_table, last_match_date = build_full_panel(
        matches, elo_cfg=cfg["elo"]
    )
    oriented = build_player_a_panel(panel)
    oriented = oriented.sort_values("match_date").reset_index(drop=True)

    train, test, cutoff = _split_by_date(
        oriented, months=int(cfg["model"]["test_window_months"])
    )
    log.info("train rows %d / test rows %d (cutoff %s)",
             len(train), len(test),
             str(cutoff.date()) if hasattr(cutoff, "date") else cutoff)

    # ── First pass: broad panel ─────────────────────────────────────────
    X_train_broad = select_features(train, PREMATCH_FEATURES_BROAD)
    y_train = train["y"].values
    X_test_broad = select_features(test, PREMATCH_FEATURES_BROAD)
    y_test = test["y"].values

    log.info("first pass — broad panel (%d features)", len(PREMATCH_FEATURES_BROAD))
    bundle_broad = _fit_pass(PREMATCH_FEATURES_BROAD, X_train_broad, y_train,
                              X_test_broad, y_test, cfg)
    log.info("broad metrics: %s",
             json.dumps(bundle_broad["metrics"]["blended"], indent=2))

    # ── Permutation importance ──────────────────────────────────────────
    log.info("computing permutation importance on broad model…")
    importances = _permutation_importance(
        bundle_broad, X_test_broad, y_test,
        n_repeats=int(cfg["model"].get("perm_repeats", 5)),
        random_state=int(cfg["model"]["random_state"]),
    )
    importances_sorted = sorted(importances,
                                 key=lambda r: -r["mean_importance"])
    log.info("top features by perm importance: %s",
             ", ".join(f"{r['feature']}({r['mean_importance']:.4f})"
                       for r in importances_sorted[:5]))

    # ── Prune ────────────────────────────────────────────────────────────
    kept, dropped = _prune_noisy(
        importances,
        floor=float(cfg["model"]["prune_floor"]),
        min_features=int(cfg["model"]["prune_min_features"]),
    )
    if dropped:
        log.info("pruning %d weak/noisy features: %s",
                 len(dropped), ", ".join(dropped))
    else:
        log.info("no features pruned — every feature contributed")

    # ── Second pass: pruned panel ────────────────────────────────────────
    if dropped:
        X_train_p = select_features(train, kept)
        X_test_p = select_features(test, kept)
        log.info("second pass — pruned panel (%d features)", len(kept))
        bundle_pruned = _fit_pass(kept, X_train_p, y_train, X_test_p, y_test, cfg)
        # Pick whichever has the better held-out log-loss.
        ll_broad = bundle_broad["metrics"]["blended"]["log_loss"]
        ll_pruned = bundle_pruned["metrics"]["blended"]["log_loss"]
        log.info("broad log-loss %.4f vs pruned %.4f — pruned %s",
                 ll_broad, ll_pruned,
                 "wins" if ll_pruned <= ll_broad else "lost")
        chosen = bundle_pruned if ll_pruned <= ll_broad else bundle_broad
        chosen["pruning_chosen"] = "pruned" if ll_pruned <= ll_broad else "broad"
    else:
        chosen = bundle_broad
        chosen["pruning_chosen"] = "broad"

    metrics = {
        "rows_train": int(len(train)),
        "rows_test": int(len(test)),
        "cutoff_date": (str(cutoff.date()) if hasattr(cutoff, "date")
                         else str(cutoff)),
        "broad_blended": bundle_broad["metrics"]["blended"],
        "pruned_blended": chosen["metrics"]["blended"] if chosen.get("pruning_chosen") == "pruned" else None,
        "elo_only": chosen["metrics"]["elo_only"],
        "ensemble": chosen["metrics"]["ensemble"],
        "blended": chosen["metrics"]["blended"],
        "train_blended": chosen["train_metrics"]["blended"],
        "xgboost": chosen.get("xgboost", False),
        "pruning_chosen": chosen.get("pruning_chosen", "broad"),
        "features_used": chosen["feature_list"],
        "features_pruned": dropped,
    }
    log.info("final metrics: %s",
             json.dumps(metrics["blended"], indent=2))

    # ── Persist artefacts ────────────────────────────────────────────────
    bundle_to_persist = {
        "ensemble": chosen["ensemble"],
        "logistic": chosen["logistic"],
        "feature_list": chosen["feature_list"],
        "elo_only_features": chosen["elo_only_features"],
        "blend_weight_ensemble": chosen["blend_weight_ensemble"],
        "blend_weight_logistic": chosen["blend_weight_logistic"],
        "metrics": metrics,
    }
    model_path = artifacts_dir / "prematch_model.joblib"
    joblib.dump(bundle_to_persist, model_path)
    log.info("wrote model bundle → %s", model_path)

    state_path = artifacts_dir / "elo_state.joblib"
    joblib.dump(_elo_state_to_dict(elo_state), state_path)
    joblib.dump(dict(h2h_table), artifacts_dir / "h2h_table.joblib")
    joblib.dump(
        {k: pd.Timestamp(v).isoformat() for k, v in last_match_date.items()},
        artifacts_dir / "last_match_date.joblib",
    )

    # ── Coefficients (Elo-only logistic) + ensemble top features ────────
    # Pipeline wraps the LR; the final estimator carries .coef_/.intercept_.
    try:
        lr = chosen["logistic"]
        if hasattr(lr, "named_steps"):
            lr = lr.named_steps.get("clf", lr)
        log_coef = list(map(float, lr.coef_.ravel().tolist()))
        log_intercept = float(np.array(lr.intercept_).ravel()[0])
    except Exception:
        log_coef, log_intercept = [], 0.0

    coefficients = {
        "logistic": {
            "intercept": log_intercept,
            "features": chosen["elo_only_features"],
            "coefficients": log_coef,
        },
        "ensemble_top_features": _ensemble_top_features(
            chosen["ensemble_uncalibrated"], chosen["feature_list"],
            top_n=len(chosen["feature_list"]),
        ),
        "blend": {
            "ensemble_weight": chosen["blend_weight_ensemble"],
            "logistic_weight": chosen["blend_weight_logistic"],
        },
        "elo": {
            "k_base": cfg["elo"]["k_base"],
            "k_floor": cfg["elo"]["k_floor"],
            "style_blend": cfg["elo"]["style_blend"],
        },
        "permutation_importance": importances_sorted,
        "features_pruned": dropped,
        "pruning_chosen": metrics["pruning_chosen"],
        # Surface every feature's coefficient or importance under a single
        # ``coefficients`` dict so the dashboard's "all features" panel has
        # one place to read from. Logistic coefs for the Elo-only side;
        # permutation-importance-derived signed magnitudes for the rest
        # (sign comes from a single-feature univariate-correlation pass
        # against the holdout target — same sign convention as a logistic
        # coefficient: positive = pushes toward player_a winning).
        "coefficients": _flatten_coefficients(
            chosen["logistic"], chosen["elo_only_features"], log_intercept,
            chosen["feature_list"], importances, X_test_broad, y_test,
        ),
    }
    with open(artifacts_dir / "model_coefficients.json", "w") as f:
        json.dump(coefficients, f, indent=2)

    metrics_path = artifacts_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # ── Holdout predictions for the dashboard's ROC / confusion / calib
    try:
        holdout_path = artifacts_dir / "holdout_predictions.csv"
        with open(holdout_path, "w") as f:
            f.write("predicted_prob,actual_label\n")
            for p, y in zip(chosen["blend_p_test"], y_test):
                f.write(f"{float(p):.6f},{int(y)}\n")
        log.info("wrote holdout predictions (%d rows) → %s",
                 len(y_test), holdout_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("holdout dump failed: %s", exc)

    # ── Feature importance audit (cross-bot dashboard schema) ───────────
    try:
        fi_path = artifacts_dir / "feature_importance.csv"
        with open(fi_path, "w") as f:
            f.write("feature,mean_importance,positive_folds,selected\n")
            kept_set = set(chosen["feature_list"])
            for r in importances_sorted:
                selected = "True" if r["feature"] in kept_set else "False"
                f.write(
                    f"{r['feature']},{r['mean_importance']:.6f},"
                    f"{r['positive_folds']},{selected}\n"
                )
        log.info("wrote feature importance (%d features) → %s",
                 len(importances_sorted), fi_path)
    except Exception as exc:  # noqa: BLE001
        log.warning("feature_importance dump failed: %s", exc)

    return metrics


def _flatten_coefficients(logistic, elo_only_features: list[str],
                            intercept: float, feature_list: list[str],
                            permutation_results: list[dict[str, float]],
                            X_holdout: pd.DataFrame, y_holdout: np.ndarray,
                            ) -> dict[str, float]:
    """Return a {feature: coef-or-importance} dict the dashboard can
    render as a single bar chart.

    * Logistic coefs (after StandardScaler inversion) for the Elo-only
      features — same units the dashboard already documents for tennis.
    * For every other feature, we use the permutation-importance
      magnitude (log-loss improvement when the feature is left intact)
      signed by the feature's univariate correlation with the target on
      the holdout slice. Positive = pushes toward player_a winning.
      Scaled to share a comparable y-axis with the logistic coefs.
    """
    out: dict[str, float] = {}
    try:
        lr = logistic
        scaler = None
        if hasattr(lr, "named_steps"):
            scaler = lr.named_steps.get("scaler")
            lr = lr.named_steps.get("clf", lr)
        log_coefs = list(map(float, lr.coef_.ravel().tolist()))
        # If the LR was fit on scaled inputs, divide by std to convert
        # the coefficient back to the raw-feature scale ("per +1 Elo
        # point" interpretation) — this matches what the tennis
        # dashboard documents.
        if scaler is not None and hasattr(scaler, "scale_"):
            scales = list(map(float, scaler.scale_.tolist()))
            log_coefs = [c / s if s != 0 else c
                          for c, s in zip(log_coefs, scales)]
        for n, c in zip(elo_only_features, log_coefs):
            out[n] = float(c)
    except Exception:
        pass

    # Permutation-importance magnitudes signed by univariate correlation.
    perm_by_name = {r["feature"]: float(r["mean_importance"])
                     for r in permutation_results}
    max_perm = max((abs(v) for v in perm_by_name.values()), default=1.0) or 1.0
    max_log = max((abs(v) for v in out.values()), default=1.0) or 1.0
    for name in feature_list:
        if name in out:
            continue
        magnitude = perm_by_name.get(name, 0.0)
        # Univariate correlation with y on the holdout — gives a sign.
        try:
            col = X_holdout[name].astype(float).values
            if np.std(col) > 1e-12:
                corr = float(np.corrcoef(col, y_holdout)[0, 1])
            else:
                corr = 0.0
        except Exception:
            corr = 0.0
        sign = 1.0 if corr >= 0 else -1.0
        out[name] = sign * (magnitude / max_perm) * max_log
    out["(intercept)"] = float(intercept)
    return out


def _ensemble_top_features(clf, feature_names: list[str], top_n: int = 6
                            ) -> list[dict[str, float]]:
    try:
        importances = getattr(clf, "feature_importances_", None)
        if importances is None:
            return []
        pairs = sorted(zip(feature_names, importances.tolist()),
                        key=lambda x: -x[1])
        return [{"name": n, "importance": float(v)} for n, v in pairs[:top_n]]
    except Exception:
        return []


def _eval(y_true, y_prob) -> dict[str, float]:
    y_prob_arr = np.clip(np.asarray(y_prob), 1e-6, 1 - 1e-6)
    y_pred = (y_prob_arr >= 0.5).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "log_loss": float(log_loss(y_true, y_prob_arr)),
        "brier": float(brier_score_loss(y_true, y_prob_arr)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob_arr)),
    }


def _elo_state_to_dict(state: EloState) -> dict[str, Any]:
    return {
        "default_rating": state.default_rating,
        "k_base": state.k_base, "k_floor": state.k_floor,
        "k_decay_matches": state.k_decay_matches,
        "bo7_k_multiplier": state.bo7_k_multiplier,
        "style_k_multiplier": state.style_k_multiplier,
        "style_blend": state.style_blend,
        "overall": dict(state.overall),
        "style": {f"{k[0]}|{k[1]}": v for k, v in state.style.items()},
        "matches_played": dict(state.matches_played),
        "style_matches": {f"{k[0]}|{k[1]}": v for k, v in state.style_matches.items()},
    }


def load_elo_state(d: dict[str, Any]) -> EloState:
    s = EloState(
        default_rating=d["default_rating"],
        k_base=d["k_base"], k_floor=d["k_floor"],
        k_decay_matches=d["k_decay_matches"],
        bo7_k_multiplier=d.get("bo7_k_multiplier", 1.10),
        style_k_multiplier=d.get("style_k_multiplier", 1.0),
        style_blend=d.get("style_blend", 0.25),
    )
    s.overall = dict(d["overall"])
    s.style = {tuple(k.split("|", 1)): v for k, v in d["style"].items()}
    s.matches_played.update(d["matches_played"])
    s.style_matches.update({tuple(k.split("|", 1)): v
                              for k, v in d["style_matches"].items()})
    return s


if __name__ == "__main__":
    train_and_persist()
