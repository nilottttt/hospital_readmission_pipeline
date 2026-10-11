"""
Readmission-risk inference API (FastAPI)

Serves the MLflow model registered as  models:/readmission_risk_model@champion

  GET  /health          liveness + whether a model is loaded
  GET  /model-info      version, metrics, threshold, features of the served model
  POST /predict         score one encounter (+ top contributing factors)
  POST /predict/batch   score up to 5,000 encounters
  POST /reload          reload the current champion (called after training)
  GET  /metrics         Prometheus metrics (requests, failures, latency)
  GET  /docs            interactive Swagger UI

Every prediction and every failure is logged to mlops.prediction_log.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import asynccontextmanager
import time
import uuid

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from api import store
from api.schemas import DISCLAIMER, BatchRequest, BatchResponse, Encounter, Prediction, PredictRequest
from ml import config
from ml.features import FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")



@asynccontextmanager
async def lifespan(_app):
    if os.getenv("API_PRELOAD", "1") != "0":   # tests switch this off
        store.init_schema()
        holder.load(force=True)
    yield


app = FastAPI(title="Hospital Readmission Risk API",
              description="30-day readmission risk - academic decision-support prototype. " + DISCLAIMER,
              version="2.0", lifespan=lifespan)

REQUESTS = Counter("predictions_total", "Encounters scored", ["endpoint", "status"])
LATENCY = Histogram("prediction_latency_seconds", "Request latency", ["endpoint"],
                    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10))
MODEL_VERSION = Gauge("model_version", "Version number of the served model")
RISK = Counter("predictions_by_band_total", "Predictions per risk band", ["band"])


class ModelHolder:
    """Loads the champion from the MLflow registry; retries lazily if unavailable."""

    def __init__(self):
        self.model = None          # ReadmissionRiskModel
        self.version = None
        self.info: dict = {}
        self.last_attempt = 0.0
        self.lock = threading.Lock()

    def load(self, force: bool = False) -> bool:
        with self.lock:
            if self.model is not None and not force:
                return True
            if not force and time.time() - self.last_attempt < 30:
                return False
            self.last_attempt = time.time()
            try:
                import mlflow
                from mlflow import MlflowClient
                mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
                client = MlflowClient()
                mv = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
                pyfunc = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}/{mv.version}")
                self.model = pyfunc.unwrap_python_model()
                clf = self.model.pipeline.named_steps["model"]
                if hasattr(clf, "n_jobs"):
                    clf.n_jobs = 1        # single-row requests: thread start-up costs more than it saves
                self.version = str(mv.version)
                run = client.get_run(mv.run_id)
                self.info = {
                    "model_name": config.MODEL_NAME, "version": self.version,
                    "alias": config.CHAMPION_ALIAS, "model_type": self.model.model_type,
                    "threshold": self.model.threshold, "base_rate": self.model.base_rate,
                    "mlflow_run_id": mv.run_id, "registered_at": mv.creation_timestamp,
                    "description": mv.description,
                    "test_metrics": {k.replace("test_", ""): round(v, 4) for k, v in run.data.metrics.items()
                                     if k.startswith("test_") and k[5:] in
                                     ("roc_auc", "pr_auc", "precision", "recall", "f1", "brier", "ece")},
                    "features": FEATURES,
                    "categories": self.model.feature_ranges.get("categorical", {}),
                    "numeric_ranges": self.model.feature_ranges.get("numeric", {}),
                }
                self._warm_up()
                MODEL_VERSION.set(float(self.version))
                log.info("Loaded %s v%s (%s)", config.MODEL_NAME, self.version, self.model.model_type)
                return True
            except Exception as exc:
                log.warning("Model not available yet: %s", exc)
                return False


    def _warm_up(self):
        """Score one typical row so the first real request is not slow
        (builds the SHAP explainer, loads libraries)."""
        try:
            rng = self.model.feature_ranges
            row = {c: (lo + hi) / 2 for c, (lo, hi) in rng["numeric"].items()}
            row.update({c: cats[0] for c, cats in rng["categorical"].items() if cats})
            df = pd.DataFrame([row])
            self.model.predict_proba(df)
            self.model.explain(df)
        except Exception as exc:
            log.warning("Warm-up failed (not fatal): %s", exc)


holder = ModelHolder()


def require_model():
    if holder.model is None and not holder.load():
        raise HTTPException(503, "No champion model is registered yet. Run the "
                                 "readmission_model_training DAG, then POST /reload.")
    return holder.model


def score(encounters: list[Encounter], explain: bool, source: str, endpoint: str) -> list[Prediction]:
    model = require_model()
    t0 = time.perf_counter()
    df = pd.DataFrame([e.model_dump(exclude={"encounter_sk"}) for e in encounters])
    try:
        probs = model.predict_proba(df)
        factors = model.explain(df) if explain else [[] for _ in encounters]
        flags = model.input_quality_flags(df)
    except Exception as exc:
        elapsed = (time.perf_counter() - t0) * 1000
        REQUESTS.labels(endpoint, "error").inc(len(encounters))
        store.write([{"request_id": str(uuid.uuid4()), "source": source, "model_name": config.MODEL_NAME,
                      "model_version": holder.version, "encounter_sk": e.encounter_sk,
                      "features": e.model_dump(), "latency_ms": elapsed, "status": "error",
                      "error": f"{exc.__class__.__name__}: {exc}"[:1000]} for e in encounters])
        log.exception("Scoring failed")
        raise HTTPException(500, f"Scoring failed: {exc.__class__.__name__}")
    elapsed = (time.perf_counter() - t0) * 1000
    per_row = elapsed / len(encounters)

    preds, rows = [], []
    for e, p, f, dq in zip(encounters, probs, factors, flags):
        rid = str(uuid.uuid4())
        band = model.band(float(p))
        pred = Prediction(request_id=rid, encounter_sk=e.encounter_sk, probability=round(float(p), 4),
                          risk_band=band, flagged=bool(p >= model.threshold),
                          threshold=round(model.threshold, 4), model_name=config.MODEL_NAME,
                          model_version=holder.version, top_factors=f, dq_flags=dq,
                          latency_ms=round(per_row, 2))
        preds.append(pred)
        RISK.labels(band).inc()
        rows.append({"request_id": rid, "source": source, "model_name": config.MODEL_NAME,
                     "model_version": holder.version, "encounter_sk": e.encounter_sk,
                     "features": e.model_dump(exclude={"encounter_sk"}), "probability": float(p),
                     "risk_band": band, "flagged": pred.flagged, "threshold": model.threshold,
                     "latency_ms": per_row, "status": "ok", "dq_flags": dq})
    REQUESTS.labels(endpoint, "ok").inc(len(encounters))
    LATENCY.labels(endpoint).observe(elapsed / 1000)
    store.write(rows)
    return preds


# ------------------------------------------------------------------ endpoints
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": holder.model is not None, "model_version": holder.version}


@app.get("/model-info")
def model_info():
    require_model()
    return holder.info


@app.post("/predict", response_model=Prediction)
def predict(req: PredictRequest):
    return score([req.encounter], req.explain, req.source, "predict")[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    t0 = time.perf_counter()
    preds = score(req.encounters, req.explain, req.source, "batch")
    return BatchResponse(model_version=holder.version, count=len(preds),
                         latency_ms=round((time.perf_counter() - t0) * 1000, 2), predictions=preds)


@app.post("/reload")
def reload_model():
    ok = holder.load(force=True)
    if not ok:
        raise HTTPException(503, "Champion model could not be loaded")
    return {"reloaded": True, "model_version": holder.version}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    """Invalid input is a monitored failure: count it and log it."""
    REQUESTS.labels(request.url.path.strip("/").replace("/", "_") or "root", "rejected").inc()
    store.write([{"request_id": str(uuid.uuid4()), "source": "api", "model_name": config.MODEL_NAME,
                  "model_version": holder.version, "status": "rejected",
                  "error": str(exc.errors())[:1000]}])
    return JSONResponse(status_code=422, content=jsonable_encoder({"detail": exc.errors()}))

