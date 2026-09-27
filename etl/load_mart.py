"""STEP 5 - BUILD THE MART (clean -> mart)

Runs sql/build_mart.sql, which rebuilds the star-schema tables and the
reporting views, then checks that the mart reconciles with the clean layer.
"""
import logging

from etl import config, db
from etl.validate import record_dq

log = logging.getLogger(__name__)


def run(run_id: str) -> dict:
    db.run_sql_file(config.SQL_DIR / "build_mart.sql")

    counts = {}
    for t in ["dim_patient", "dim_admission_type", "dim_discharge_disposition",
              "dim_admission_source", "fact_admission", "fact_diagnosis", "fact_readmission"]:
        counts[t] = db.fetch_one(f"SELECT COUNT(*) FROM mart.{t}")[0]

    clean_rows = db.fetch_one("SELECT COUNT(*) FROM clean.encounters")[0]
    clean_patients = db.fetch_one("SELECT COUNT(DISTINCT patient_sk) FROM clean.encounters")[0]
    checks = [
        ("fact_admission_matches_clean", abs(counts["fact_admission"] - clean_rows)),
        ("fact_readmission_matches_clean", abs(counts["fact_readmission"] - clean_rows)),
        ("dim_patient_matches_distinct_patients", abs(counts["dim_patient"] - clean_patients)),
    ]
    for name, failed in checks:
        record_dq(run_id, "mart", name, "critical", failed == 0, failed,
                  f"difference = {failed}")
        if failed:
            raise RuntimeError(f"Mart reconciliation failed: {name}")

    log.info("Mart built: %s", counts)
    return counts
