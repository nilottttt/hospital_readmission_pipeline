"""STEP 2 - STAGE

Loads the raw CSV into staging.encounters exactly as it is (every value as
TEXT, nothing cleaned). Each row keeps its source line number so any problem
can be traced back to the original file.
"""
import csv
import logging
from pathlib import Path

import pandas as pd

from etl import db
from etl.columns import SOURCE_COLUMNS, to_snake

log = logging.getLogger(__name__)


def read_raw_encounters(csv_path: Path) -> pd.DataFrame:
    rows, line_numbers = [], []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for line_no, row in enumerate(reader, start=2):
            if len(row) == len(header):      # malformed lines were logged in extract
                rows.append(row)
                line_numbers.append(line_no)
    df = pd.DataFrame(rows, columns=header, dtype=str)
    df = df[SOURCE_COLUMNS]                   # fixed column order, ignore extras
    df.columns = [to_snake(c) for c in df.columns]
    df.insert(0, "source_row_num", line_numbers)
    return df


def run(run_id: str, encounters_path: str) -> int:
    df = read_raw_encounters(Path(encounters_path))
    df.insert(1, "run_id", run_id)
    n = db.copy_df(df, "staging.encounters", truncate=True)
    db.execute("UPDATE audit.pipeline_runs SET rows_staged=%s WHERE run_id=%s", (n, run_id))
    log.info("Staged %s rows", n)
    return n
