"""Writes every prediction (or failure) to mlops.prediction_log.
Logging problems never break a prediction - they are only written to stdout."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values

log = logging.getLogger("api.store")
SQL_FILE = Path(os.getenv("MLOPS_SQL", Path(__file__).resolve().parent.parent / "sql" / "init_mlops.sql"))
COLUMNS = ["request_id", "source", "model_name", "model_version", "encounter_sk", "features",
           "probability", "risk_band", "flagged", "threshold", "latency_ms", "status", "error", "dq_flags"]


def _connect():
    return psycopg2.connect(
        host=os.getenv("HC_DB_HOST", "localhost"), port=int(os.getenv("HC_DB_PORT", "5432")),
        dbname=os.getenv("HC_DB_NAME", "healthcare"), user=os.getenv("HC_DB_USER", "hc_admin"),
        password=os.getenv("HC_DB_PASSWORD", ""), connect_timeout=5)


def init_schema() -> None:
    try:
        with _connect() as conn, conn.cursor() as cur:
            cur.execute(SQL_FILE.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Could not initialise mlops schema: %s", exc)


def write(rows: list[dict]) -> None:
    if not rows or os.getenv("DISABLE_PREDICTION_LOG") == "1":
        return
    values = [tuple(json.dumps(r.get(c), default=str) if c == "features" else r.get(c) for c in COLUMNS)
              for r in rows]
    try:
        with _connect() as conn, conn.cursor() as cur:
            execute_values(cur, f"INSERT INTO mlops.prediction_log ({', '.join(COLUMNS)}) VALUES %s", values)
    except Exception as exc:
        log.warning("Prediction log write failed: %s", exc)
