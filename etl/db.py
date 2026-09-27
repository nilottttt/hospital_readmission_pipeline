"""Small PostgreSQL helper built on psycopg2.

We use psycopg2 directly (instead of pandas.to_sql) because it is fast
(COPY) and avoids SQLAlchemy version conflicts inside the Airflow image.
"""
import io
import logging
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values

from etl import config

log = logging.getLogger(__name__)


def get_connection():
    return psycopg2.connect(
        host=config.DB_HOST,
        port=config.DB_PORT,
        dbname=config.DB_NAME,
        user=config.DB_USER,
        password=config.DB_PASSWORD,
    )


@contextmanager
def connect():
    """Yield a connection; commit on success, roll back on error."""
    conn = get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def run_sql_file(path: Path) -> None:
    sql = Path(path).read_text(encoding="utf-8")
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql)
    log.info("Executed SQL file %s", Path(path).name)


def execute(sql: str, params=None) -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)


def fetch_df(sql: str, params=None) -> pd.DataFrame:
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        cols = [c.name for c in cur.description]
        return pd.DataFrame(cur.fetchall(), columns=cols)


def fetch_one(sql: str, params=None):
    with connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()


def copy_df(df: pd.DataFrame, table: str, conn=None, truncate: bool = False) -> int:
    """Bulk-load a DataFrame into an existing table with COPY.
    Column names in df must match the table's column names."""
    buf = io.StringIO()
    df.to_csv(buf, index=False, header=False, na_rep="\\N")
    buf.seek(0)
    cols = ", ".join(f'"{c}"' for c in df.columns)
    sql = f"COPY {table} ({cols}) FROM STDIN WITH (FORMAT csv, NULL '\\N')"

    own_conn = conn is None
    conn = conn or get_connection()
    try:
        with conn.cursor() as cur:
            if truncate:
                cur.execute(f"TRUNCATE {table}")
            cur.copy_expert(sql, buf)
        if own_conn:
            conn.commit()
    except Exception:
        if own_conn:
            conn.rollback()
        raise
    finally:
        if own_conn:
            conn.close()
    log.info("Loaded %s rows into %s", len(df), table)
    return len(df)


def insert_rows(table: str, columns: list, rows: list) -> None:
    if not rows:
        return
    cols = ", ".join(columns)
    with connect() as conn, conn.cursor() as cur:
        execute_values(cur, f"INSERT INTO {table} ({cols}) VALUES %s", rows)
