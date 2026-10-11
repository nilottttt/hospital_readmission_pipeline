"""
Airflow DAG: readmission_model_monitoring  (Part 2)

    check_api -> score_recent_batch -> compute_monitoring -> decide
                                                              |-> trigger_retraining -> mark_triggered
                                                              |-> no_retraining

Daily: scores recent encounters through the API, then measures input drift (PSI),
performance, calibration, subgroup performance, latency and failures, and
triggers the training DAG when the retraining criteria are met.
"""
import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

log = logging.getLogger(__name__)


@dag(
    dag_id="readmission_model_monitoring",
    description="Drift, performance, calibration, fairness and API health monitoring",
    schedule="@daily",
    start_date=datetime(2026, 10, 1),
    catchup=False,
    max_active_runs=1,
    default_args={"owner": "student", "retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["healthcare", "mlops", "monitoring", "part2"],
)
def readmission_model_monitoring():

    @task.short_circuit
    def check_api() -> bool:
        """Skip the run (not fail) if the API has no champion model yet."""
        from ml import monitoring
        try:
            health = monitoring.api_health()
        except Exception as exc:
            log.warning("API not reachable: %s", exc)
            return False
        log.info("API health: %s", health)
        return bool(health.get("model_loaded"))

    @task
    def score_recent_batch() -> dict:
        from ml import monitoring
        return monitoring.score_recent_batch()

    @task
    def compute_monitoring(_scored: dict) -> dict:
        from ml import monitoring
        result = monitoring.run_monitoring()
        for a in result["alerts"]:
            log.warning("ALERT: %s", a)
        return {k: result.get(k) for k in ["monitoring_id", "model_version", "retrain_recommended",
                                           "alerts", "max_psi", "roc_auc", "ece", "n_labeled"]}

    @task.branch
    def decide(result: dict) -> str:
        from ml import monitoring
        if not result["retrain_recommended"]:
            return "no_retraining"
        allowed, why = monitoring.retrain_allowed()
        log.info("Retraining recommended; allowed=%s (%s)", allowed, why)
        return "trigger_retraining" if allowed else "no_retraining"

    trigger = TriggerDagRunOperator(
        task_id="trigger_retraining",
        trigger_dag_id="readmission_model_training",
        conf={"reason": "monitoring alert"},
        wait_for_completion=False,
    )

    @task
    def mark_triggered(result: dict):
        from ml import monitoring
        monitoring.mark_retrain_triggered(result["monitoring_id"])

    no_retrain = EmptyOperator(task_id="no_retraining")

    scored = score_recent_batch()
    check_api() >> scored
    result = compute_monitoring(scored)
    branch = decide(result)
    branch >> [trigger, no_retrain]
    trigger >> mark_triggered(result)


readmission_model_monitoring()
