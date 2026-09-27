"""Run the whole pipeline from the command line, without Airflow.

Useful for development and debugging:
    python -m etl.run_pipeline

The Airflow DAG (dags/readmission_etl_dag.py) calls exactly the same functions.
"""
import logging
from datetime import datetime, timezone

from etl import extract, load_mart, run_log, stage, transform, validate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline")


def main() -> None:
    run_id = "cli__" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    run_log.init_database()
    run_log.start_run(run_id, "cli")
    try:
        info = extract.run(run_id)
        stage.run(run_id, info["encounters_path"])
        validate.run(run_id)
        transform.run(run_id)
        counts = load_mart.run(run_id)
        run_log.finish_run(run_id, "success")
        log.info("Pipeline %s finished successfully: %s", run_id, counts)
    except Exception as exc:
        run_log.finish_run(run_id, "failed", str(exc)[:2000])
        log.exception("Pipeline %s failed", run_id)
        raise


if __name__ == "__main__":
    main()
