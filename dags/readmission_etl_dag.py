"""
Airflow DAG: hospital_readmission_etl

    init_database -> start_run -> extract -> stage -> validate -> transform -> build_mart -> finish_run

Each task calls a function from the etl package, so the same logic can also
be run without Airflow (python -m etl.run_pipeline).
"""
import logging
from datetime import datetime, timedelta

from airflow.decorators import dag, task

log = logging.getLogger(__name__)


def mark_run_failed(context):
    """Called by Airflow when any task fails (after its retries)."""
    from etl import run_log
    run_log.finish_run(context["run_id"], "failed",
                       f"{context['task_instance'].task_id}: {context.get('exception')}"[:2000])


default_args = {
    "owner": "student",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "on_failure_callback": mark_run_failed,
}


@dag(
    dag_id="hospital_readmission_etl",
    description="UCI Diabetes 130-US hospitals: raw -> staging -> clean -> mart",
    schedule="@weekly",
    start_date=datetime(2026, 9, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["healthcare", "etl", "part1"],
)
def hospital_readmission_etl():

    @task
    def init_database():
        from etl import run_log
        run_log.init_database()

    @task
    def start_run(run_id=None):
        from etl import run_log
        run_log.start_run(run_id, "airflow")

    @task
    def extract(run_id=None) -> dict:
        from etl import extract as step
        return step.run(run_id)

    @task
    def stage(extract_info: dict, run_id=None) -> int:
        from etl import stage as step
        return step.run(run_id, extract_info["encounters_path"])

    @task
    def validate(_staged_rows: int, run_id=None) -> dict:
        from etl import validate as step
        return step.run(run_id)

    @task
    def transform(_validation: dict, run_id=None) -> int:
        from etl import transform as step
        return step.run(run_id)

    @task
    def build_mart(_clean_rows: int, run_id=None) -> dict:
        from etl import load_mart as step
        return step.run(run_id)

    @task
    def finish_run(mart_counts: dict, run_id=None):
        from etl import run_log
        run_log.finish_run(run_id, "success")
        log.info("Run %s finished. Mart row counts: %s", run_id, mart_counts)

    init = init_database()
    started = start_run()
    init >> started
    extracted = extract()
    started >> extracted
    finish_run(build_mart(transform(validate(stage(extracted)))))


hospital_readmission_etl()
