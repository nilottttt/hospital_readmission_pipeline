"""Start / finish bookkeeping for audit.pipeline_runs."""
from etl import config, db


def init_database() -> None:
    """Create schemas and tables if they don't exist yet (idempotent)."""
    db.run_sql_file(config.SQL_DIR / "init_schema.sql")


def start_run(run_id: str, triggered_by: str) -> None:
    db.execute(
        "INSERT INTO audit.pipeline_runs (run_id, triggered_by, status) VALUES (%s, %s, 'running') "
        "ON CONFLICT (run_id) DO UPDATE SET status='running', started_at=now(), "
        "finished_at=NULL, error_message=NULL",
        (run_id, triggered_by),
    )


def finish_run(run_id: str, status: str, error: str | None = None) -> None:
    db.execute(
        "UPDATE audit.pipeline_runs SET status=%s, finished_at=now(), error_message=%s "
        "WHERE run_id=%s",
        (status, error, run_id),
    )
