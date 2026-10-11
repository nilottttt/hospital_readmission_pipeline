"""Model monitoring (run daily by the readmission_model_monitoring DAG).

1. score_recent_batch  - send a batch of the most recent encounters to the API
                         (simulates daily production traffic; labels arrive later)
2. run_monitoring      - over the last MONITOR_WINDOW_DAYS of mlops.prediction_log:
     * operations : volume, error/rejection rate, p50/p95 latency, input-quality flags
     * input drift: PSI of every feature vs. the champion's training profile
     * performance: ROC-AUC, precision, recall, F1 on predictions whose outcome is known
     * calibration: Brier score and expected calibration error (ECE)
     * fairness   : recall / precision / AUC per gender, race and age group
   and decides whether retraining is recommended (see docs/model_lifecycle.md).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

from etl import db
from ml import config, evaluate
from ml.drift import feature_psi
from ml.features import FEATURES, LABEL, SUBGROUP_COLUMNS, coerce_types

log = logging.getLogger(__name__)


def api_health() -> dict:
    r = requests.get(f"{config.API_URL}/health", timeout=10)
    r.raise_for_status()
    return r.json()


def score_recent_batch(n: int | None = None, seed: int | None = None) -> dict:
    """Score a random sample of the newest 15 % of encounters (the period after training)."""
    n = n or config.MONITOR_BATCH_SIZE
    seed = seed if seed is not None else int(datetime.now(timezone.utc).strftime("%Y%m%d"))
    total = db.fetch_one("SELECT COUNT(*) FROM mart.v_ml_features")[0]
    start = int(total * (config.TRAIN_FRACTION + config.VALID_FRACTION))
    df = db.fetch_df("SELECT * FROM mart.v_ml_features WHERE encounter_order > %s", (start,))
    df = coerce_types(df).sample(min(n, len(df)), random_state=seed % (2**32 - 1))
    sent, latencies = 0, []
    for chunk_start in range(0, len(df), 1000):
        chunk = df.iloc[chunk_start:chunk_start + 1000]
        payload = {"source": "batch", "explain": False, "encounters": [
            {**{k: _py(v) for k, v in row[FEATURES].items()}, "encounter_sk": row["encounter_sk"]}
            for _, row in chunk.iterrows()]}
        r = requests.post(f"{config.API_URL}/predict/batch", json=payload, timeout=300)
        r.raise_for_status()
        sent += r.json()["count"]
        latencies.append(r.json()["latency_ms"])
    log.info("Scored %s recent encounters through the API", sent)
    return {"scored": sent, "batch_latency_ms": latencies}


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _champion_reference_auc(version: str) -> float | None:
    row = db.fetch_one("SELECT test_roc_auc FROM mlops.training_runs WHERE registered_version=%s "
                       "AND selected ORDER BY trained_at DESC LIMIT 1", (version,))
    return float(row[0]) if row and row[0] is not None else None


def run_monitoring(model_version: str | None = None) -> dict:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=config.MONITOR_WINDOW_DAYS)
    mon_id = f"mon_{now:%Y%m%dT%H%M%S}"
    if model_version is None:
        model_version = str(api_health().get("model_version"))

    logs = db.fetch_df(
        "SELECT request_id, predicted_at, source, model_version, encounter_sk, features, probability, "
        "flagged, threshold, latency_ms, status, dq_flags FROM mlops.prediction_log "
        "WHERE predicted_at >= %s", (start,))
    alerts, retrain = [], False
    result = {"monitoring_id": mon_id, "model_version": model_version,
              "window_start": start, "window_end": now, "n_predictions": 0}

    # ---------------- operations ----------------
    if len(logs):
        result["n_predictions"] = int((logs["status"] == "ok").sum())
        result["error_rate"] = float((logs["status"] != "ok").mean())
        interactive = logs[(logs["status"] == "ok") & (logs["source"] != "batch")]
        lat = (interactive if len(interactive) else logs[logs["status"] == "ok"])["latency_ms"].dropna()
        result["p50_latency_ms"] = float(lat.quantile(0.5)) if len(lat) else None
        result["p95_latency_ms"] = float(lat.quantile(0.95)) if len(lat) else None
        ok = logs[logs["status"] == "ok"]
        result["dq_flag_rate"] = float(ok["dq_flags"].map(lambda f: bool(f)).mean()) if len(ok) else 0.0
        if result["error_rate"] > config.ERROR_RATE_ALERT:
            alerts.append(f"error rate {result['error_rate']:.1%} > {config.ERROR_RATE_ALERT:.0%}")
        if result["p95_latency_ms"] and result["p95_latency_ms"] > config.P95_LATENCY_ALERT_MS:
            alerts.append(f"p95 latency {result['p95_latency_ms']:.0f} ms > {config.P95_LATENCY_ALERT_MS} ms")

    ok = logs[(logs["status"] == "ok") & (logs["model_version"].astype(str) == model_version)] if len(logs) else logs

    # ---------------- input drift ----------------
    drift = pd.DataFrame()
    prof = db.fetch_df("SELECT feature, profile FROM mlops.reference_profile WHERE model_version=%s",
                       (model_version,))
    if len(ok) and len(prof):
        feats = pd.DataFrame([f if isinstance(f, dict) else json.loads(f) for f in ok["features"]])
        profile = {r.feature: (r.profile if isinstance(r.profile, dict) else json.loads(r.profile))
                   for r in prof.itertuples()}
        drift = feature_psi(profile, coerce_types(feats))
        result["max_psi"] = float(drift["psi"].max())
        result["drifted_features"] = drift.loc[drift["status"] == "drift", "feature"].tolist()
        if result["drifted_features"]:
            retrain = True
            alerts.append(f"input drift (PSI >= {config.PSI_ALERT}) in: {', '.join(result['drifted_features'])}")

    # ---------------- performance, calibration, fairness ----------------
    sub = pd.DataFrame()
    labeled = ok.dropna(subset=["encounter_sk"]) if len(ok) else ok
    if len(labeled):
        labeled = labeled.sort_values("predicted_at").drop_duplicates("encounter_sk", keep="last")
        truth = db.fetch_df("SELECT encounter_sk, race, age_group, gender, label_readmitted_30d "
                            "FROM mart.v_ml_features WHERE encounter_sk = ANY(%s)",
                            (labeled["encounter_sk"].tolist(),))
        joined = labeled.merge(truth, on="encounter_sk", how="inner")
        result["n_labeled"] = len(joined)
        if len(joined) >= config.MIN_LABELED_FOR_PERFORMANCE and joined[LABEL].nunique() == 2:
            thr = float(joined["threshold"].iloc[-1])
            m = evaluate.classification_metrics(joined[LABEL], joined["probability"], thr)
            result.update({k: m[k] for k in ["roc_auc", "precision", "recall", "f1", "brier", "ece"]})
            ref_auc = _champion_reference_auc(model_version)
            result["reference_roc_auc"] = ref_auc
            if ref_auc and m["roc_auc"] < ref_auc - config.AUC_DROP_ALERT:
                retrain = True
                alerts.append(f"ROC-AUC {m['roc_auc']:.3f} dropped > {config.AUC_DROP_ALERT} below {ref_auc:.3f}")
            if m["ece"] > config.ECE_ALERT:
                retrain = True
                alerts.append(f"calibration ECE {m['ece']:.3f} > {config.ECE_ALERT}")
            sub = evaluate.subgroup_metrics(joined, LABEL, "probability", thr, SUBGROUP_COLUMNS,
                                            min_n=config.SUBGROUP_MIN_N)
            comparable = sub[sub["positives"] >= config.SUBGROUP_MIN_POSITIVES]
            if len(comparable):
                gaps = comparable.groupby("group_type")["recall"].agg(lambda s: s.max() - s.min())
                result["max_subgroup_recall_gap"] = float(gaps.max())
                if gaps.max() > config.SUBGROUP_RECALL_GAP_ALERT:
                    alerts.append(f"subgroup recall gap {gaps.max():.2f} ({gaps.idxmax()}) - fairness review")
    result["retrain_recommended"] = retrain
    result["alerts"] = alerts

    _save(result, drift, sub)
    log.info("Monitoring %s: %s", mon_id, {k: v for k, v in result.items() if k not in ("window_start",)})
    return {**result, "window_start": str(start), "window_end": str(now)}


def _save(result: dict, drift: pd.DataFrame, sub: pd.DataFrame) -> None:
    cols = ["monitoring_id", "model_version", "window_start", "window_end", "n_predictions", "n_labeled",
            "error_rate", "p50_latency_ms", "p95_latency_ms", "dq_flag_rate", "max_psi", "drifted_features",
            "roc_auc", "reference_roc_auc", "precision", "recall", "f1", "brier", "ece",
            "max_subgroup_recall_gap", "retrain_recommended", "alerts"]
    db.insert_rows("mlops.monitoring_runs", cols, [tuple(result.get(c) for c in cols)])
    if len(drift):
        db.insert_rows("mlops.drift_metrics", ["monitoring_id", "feature", "kind", "psi", "status"],
                       [(result["monitoring_id"], r.feature, r.kind, float(r.psi), r.status)
                        for r in drift.itertuples()])
    if len(sub):
        from ml.train import to_db_rows
        db.insert_rows("mlops.subgroup_metrics",
                       ["monitoring_id", "group_type", "group_value", "n", "positives", "roc_auc", "recall",
                        "precision", "flag_rate", "mean_predicted", "observed_rate"],
                       [(result["monitoring_id"], *r) for r in to_db_rows(
                           sub[["group_type", "group_value", "n", "positives", "roc_auc", "recall",
                                "precision", "flag_rate", "mean_predicted", "observed_rate"]])])


def retrain_allowed() -> tuple[bool, str]:
    """Cool-down rule: no automatic retraining if a model was trained recently."""
    row = db.fetch_one("SELECT MAX(trained_at) FROM mlops.training_runs")
    if not row or row[0] is None:
        return True, "no previous training run"
    age = datetime.now(timezone.utc) - row[0]
    if age < timedelta(days=config.RETRAIN_COOLDOWN_DAYS):
        return False, f"last training {age.days} day(s) ago (< {config.RETRAIN_COOLDOWN_DAYS}-day cool-down)"
    return True, f"last training {age.days} days ago"


def mark_retrain_triggered(monitoring_id: str) -> None:
    db.execute("UPDATE mlops.monitoring_runs SET retrain_triggered=TRUE WHERE monitoring_id=%s",
               (monitoring_id,))
