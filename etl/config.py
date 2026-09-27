"""Central configuration. Every setting comes from an environment variable
(set in .env / docker-compose.yml) so no credentials live in the code."""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Database -----------------------------------------------------------
DB_HOST = os.getenv("HC_DB_HOST", "localhost")
DB_PORT = int(os.getenv("HC_DB_PORT", os.getenv("POSTGRES_HOST_PORT", "5432")))
DB_NAME = os.getenv("HC_DB_NAME", "healthcare")
DB_USER = os.getenv("HC_DB_USER", os.getenv("POSTGRES_USER", "hc_admin"))
DB_PASSWORD = os.getenv("HC_DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", ""))

# --- Folders --------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", PROJECT_ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"          # untouched downloaded files (raw layer)
MANUAL_DIR = DATA_DIR / "manual"    # optional: manually downloaded zip/CSVs (fallback)
REJECTED_DIR = DATA_DIR / "rejected"  # CSV copy of the rejected-record log
SQL_DIR = Path(os.getenv("SQL_DIR", PROJECT_ROOT / "sql"))

# --- Source ----------------------------------------------------------------
SOURCE_NAME = "UCI Diabetes 130-US Hospitals for Years 1999-2008 (dataset id 296)"
DATA_SOURCE_URL = os.getenv(
    "DATA_SOURCE_URL",
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip",
)
ENCOUNTERS_FILE = "diabetic_data.csv"
MAPPING_FILE = "IDs_mapping.csv"
DOWNLOAD_TIMEOUT_SEC = 120
DOWNLOAD_RETRIES = int(os.getenv("DOWNLOAD_RETRIES", "3"))

# --- Privacy ---------------------------------------------------------------
HASH_SALT = os.getenv("HASH_SALT", "")


def require_salt() -> str:
    if not HASH_SALT:
        raise RuntimeError("HASH_SALT is not set. Add it to your .env file.")
    return HASH_SALT
