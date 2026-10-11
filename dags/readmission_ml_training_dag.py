"""
Airflow DAG: readmission_model_training  (Part 2)

    init_mlops -> check_training_data -> train_compare_register -> reload_api

* Trains Logistic Regression, Random Forest and XGBoost on mart.v_ml_features
  (chronological split), tracks everything in MLflow, registers the best model
  and promotes it to the 'champion' alias if it beats the current champion.
* Runs monthly, and whenever the monitoring DAG detects drift / degradation.
"""
import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task

log = logging.getLogger(__name__)


@dag(
    dag_id="readmission_model_training",
    description="Train, compare (LR / RF / XGBoost), track in MLflow, register & promote",
    schedule="@monthly",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "student", "retries": 1, "retry_delay": timedelta(minutes=2)},
    tags=["healthcare", "mlops", "part2"],
)
def readmission_model_training():

    @task
    def init_mlops():
        from etl import config as etl_config, db
        db.run_sql_file(etl_config.SQL_DIR / "init_mlops.sql")

    @task
    def check_training_data() -> dict:
        """Fail early (with a clear message) if the ETL output is not ready."""
        from etl import db
        from ml import config
        if db.fetch_one("SELECT to_regclass('mart.v_ml_features')")[0] is None:
            raise RuntimeError("mart.v_ml_features is missing - run the hospital_readmission_etl DAG first")
        n, pos, missing_order = db.fetch_one(
            "SELECT COUNT(*), AVG(label_readmitted_30d), COUNT(*) - COUNT(encounter_order) "
            "FROM mart.v_ml_features")
        if missing_order:
            raise RuntimeError("encounter_order is empty - re-run the hospital_readmission_etl DAG once "
                               "so the Part 2 columns are filled")
        if n < config.MIN_TRAINING_ROWS:
            raise RuntimeError(f"only {n} training rows (< {config.MIN_TRAINING_ROWS})")
        if not 0.02 <= float(pos) <= 0.40:
            raise RuntimeError(f"implausible readmission rate {float(pos):.3f}")
        last_etl = db.fetch_one("SELECT MAX(finished_at) FROM audit.pipeline_runs WHERE status='success'")[0]
        info = {"rows": int(n), "positive_rate": round(float(pos), 4), "last_etl_run": str(last_etl)}
        log.info("Training data OK: %s", info)
        return info

    @task(execution_timeout=timedelta(minutes=45))
    def train_compare_register(data_info: dict, dag_run=None) -> dict:
        from ml import train
        trigger = "monitoring" if dag_run and dag_run.external_trigger and dag_run.conf.get("reason") \
            else "airflow"
        result = train.run(triggered_by=trigger)
        log.info("Training result: %s", {k: v for k, v in result.items() if k != "comparison"})
        log.info("Model comparison: %s", result["comparison"])
        return {k: v for k, v in result.items() if k != "comparison"}

    @task
    def reload_api(result: dict):
        """Tell the API to load the new champion (not fatal if the API is down)."""
        import requests
        from ml import config
        if not result.get("promoted"):
            log.info("Champion unchanged - no reload needed (%s)", result.get("decision"))
            return
        try:
            r = requests.post(f"{config.API_URL}/reload", timeout=120)
            log.info("API reload: %s %s", r.status_code, r.text)
        except Exception as exc:
            log.warning("API not reachable, it will load the champion on its next start: %s", exc)

    init = init_mlops()
    checked = check_training_data()
    init >> checked
    reload_api(train_compare_register(checked))


readmission_model_training()
