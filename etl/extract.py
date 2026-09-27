"""STEP 1 - EXTRACT

* Downloads the UCI dataset zip (with retries).
* Falls back to a manually downloaded copy in data/manual/ if the download fails.
* Saves an untouched raw copy in data/raw/<run_id>/ (the raw layer).
* Checks each CSV line; malformed lines are logged as rejected records.
* Writes extraction date, source, file status and row count to audit.ingestion_log.
* Loads IDs_mapping.csv into raw.lookup_ids.
"""
import csv
import hashlib
import io
import json
import logging
import re
import shutil
import stat
import time
import zipfile
from pathlib import Path

import requests

from etl import config, db
from etl.columns import SOURCE_COLUMNS

log = logging.getLogger(__name__)


class ExtractionError(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def safe_run_folder(run_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_read_only(path: Path) -> None:
    """Protect raw files from accidental manual edits (best effort on Windows mounts)."""
    try:
        path.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass


def log_ingestion(run_id, status, file_path=None, http_status=None, row_count=None,
                  bad_line_count=0, message=None):
    p = Path(file_path) if file_path else None
    db.insert_rows(
        "audit.ingestion_log",
        ["run_id", "source_name", "source_url", "file_name", "file_path",
         "file_size_bytes", "sha256", "status", "http_status", "row_count",
         "bad_line_count", "message"],
        [(run_id, config.SOURCE_NAME, config.DATA_SOURCE_URL,
          p.name if p else None, str(p) if p else None,
          p.stat().st_size if p and p.exists() else None,
          sha256_of(p) if p and p.exists() else None,
          status, http_status, row_count, bad_line_count, message)],
    )


def download_zip(dest: Path) -> int:
    """Download the source zip with retries. Returns the HTTP status code."""
    last_error = None
    for attempt in range(1, config.DOWNLOAD_RETRIES + 1):
        try:
            log.info("Downloading %s (attempt %s)", config.DATA_SOURCE_URL, attempt)
            with requests.get(config.DATA_SOURCE_URL, stream=True,
                              timeout=config.DOWNLOAD_TIMEOUT_SEC) as r:
                r.raise_for_status()
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
            if not zipfile.is_zipfile(dest):
                raise ExtractionError("Downloaded file is not a valid zip archive")
            return r.status_code
        except Exception as exc:  # network error, HTTP error, corrupt file
            last_error = exc
            log.warning("Download attempt %s failed: %s", attempt, exc)
            dest.unlink(missing_ok=True)
            time.sleep(5 * attempt)
    raise ExtractionError(f"Download failed after {config.DOWNLOAD_RETRIES} attempts: {last_error}")


def unzip_members(zip_path: Path, target_dir: Path) -> None:
    """Extract the two CSVs we need, wherever they sit inside the zip.
    Names are matched case-insensitively (the UCI zip ships 'IDS_mapping.csv')
    and saved under our standard names."""
    wanted = {config.ENCOUNTERS_FILE.lower(): config.ENCOUNTERS_FILE,
              config.MAPPING_FILE.lower(): config.MAPPING_FILE}
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            key = Path(member).name.lower()
            if key in wanted:
                with zf.open(member) as src, open(target_dir / wanted[key], "wb") as dst:
                    shutil.copyfileobj(src, dst)
                del wanted[key]
    if wanted:
        raise ExtractionError(f"Zip archive is missing: {sorted(wanted.values())}")


def use_manual_copy(target_dir: Path) -> str:
    """Fallback: a zip or the two CSV files placed by hand in data/manual/."""
    zips = sorted(config.MANUAL_DIR.glob("*.zip"))
    if zips:
        shutil.copy2(zips[0], target_dir / zips[0].name)
        unzip_members(target_dir / zips[0].name, target_dir)
        return f"used manual zip {zips[0].name}"
    files = {p.name.lower(): p for p in config.MANUAL_DIR.glob("*.csv")}
    enc = files.get(config.ENCOUNTERS_FILE.lower())
    mapping = files.get(config.MAPPING_FILE.lower())
    if enc and mapping:
        shutil.copy2(enc, target_dir / config.ENCOUNTERS_FILE)
        shutil.copy2(mapping, target_dir / config.MAPPING_FILE)
        return "used manual CSV files"
    raise ExtractionError(
        "Download failed and no manual copy found in data/manual/. "
        "Download the zip from the UCI website and place it there."
    )


def check_encounter_lines(csv_path: Path, run_id: str) -> tuple[int, int]:
    """Verify the header and every line's field count.
    Malformed lines go to audit.rejected_records (stage='extract').
    Returns (good_row_count, bad_line_count)."""
    bad_rows = []
    good = 0
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        missing = [c for c in SOURCE_COLUMNS if c not in header]
        if missing:
            raise ExtractionError(f"Source file is missing expected columns: {missing}")
        n = len(header)
        for line_no, row in enumerate(reader, start=2):
            if len(row) == n:
                good += 1
            else:
                bad_rows.append((
                    run_id, "extract", line_no, row[0] if row else None, "E01",
                    f"Malformed line: expected {n} fields, found {len(row)}",
                    str(len(row)), json.dumps({"raw_line": ",".join(row)[:2000]}),
                ))
    db.insert_rows(
        "audit.rejected_records",
        ["run_id", "stage", "source_row_num", "encounter_id", "rule_id",
         "rule_description", "offending_value", "record"],
        bad_rows,
    )
    return good, len(bad_rows)


def parse_mapping_file(path: Path) -> list[tuple]:
    """IDs_mapping.csv holds three small tables stacked on top of each other,
    each starting with a header like 'admission_type_id,description'."""
    rows, current = [], None
    text = path.read_text(encoding="utf-8-sig")
    for rec in csv.reader(io.StringIO(text)):
        if not rec or all(not c.strip() for c in rec):
            continue
        first = rec[0].strip()
        if first.endswith("_id"):
            current = first
            continue
        if current and first.isdigit():
            desc = rec[1].strip() if len(rec) > 1 else ""
            rows.append((current, int(first), desc))
    if not rows:
        raise ExtractionError("IDs_mapping.csv could not be parsed")
    return rows


# ----------------------------------------------------------------------------
# main entry point
# ----------------------------------------------------------------------------
def run(run_id: str) -> dict:
    target_dir = config.RAW_DIR / safe_run_folder(run_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    # If Airflow retries this task for the same run, clear the partial files first
    for old in target_dir.iterdir():
        old.chmod(0o644)
        old.unlink()
    zip_path = target_dir / "source.zip"

    # 1) acquire the files
    try:
        http_status = download_zip(zip_path)
        unzip_members(zip_path, target_dir)
        status, message = "downloaded", "downloaded from source URL"
        log_ingestion(run_id, status, zip_path, http_status=http_status, message=message)
    except Exception as exc:
        log.warning("Falling back to manual copy: %s", exc)
        log_ingestion(run_id, "failed", None, message=f"download failed: {exc}")
        try:
            message = use_manual_copy(target_dir)
            status = "manual_fallback"
        except ExtractionError as exc2:
            log_ingestion(run_id, "failed", None, message=str(exc2))
            raise

    enc_path = target_dir / config.ENCOUNTERS_FILE
    map_path = target_dir / config.MAPPING_FILE

    # 2) validate file structure, count rows, log malformed lines
    row_count, bad_lines = check_encounter_lines(enc_path, run_id)
    lookup_rows = parse_mapping_file(map_path)

    log_ingestion(run_id, status, enc_path, row_count=row_count,
                  bad_line_count=bad_lines, message=message)
    log_ingestion(run_id, status, map_path, row_count=len(lookup_rows), message=message)

    # 3) raw copies are never edited again
    for p in target_dir.iterdir():
        make_read_only(p)

    # 4) load lookup table as-is into the raw schema
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("TRUNCATE raw.lookup_ids")
        cur.executemany(
            "INSERT INTO raw.lookup_ids (lookup_type, code, description) VALUES (%s,%s,%s) "
            "ON CONFLICT DO NOTHING",
            lookup_rows,
        )

    db.execute("UPDATE audit.pipeline_runs SET rows_extracted=%s WHERE run_id=%s",
               (row_count, run_id))
    log.info("Extracted %s rows (%s malformed lines) into %s", row_count, bad_lines, target_dir)
    return {
        "raw_dir": str(target_dir),
        "encounters_path": str(enc_path),
        "mapping_path": str(map_path),
        "row_count": row_count,
        "bad_lines": bad_lines,
        "status": status,
    }
