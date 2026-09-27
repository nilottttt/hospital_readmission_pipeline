"""Patient-safe surrogate keys.

The source IDs (patient_nbr, encounter_id) are replaced by a salted SHA-256
hash. The salt is secret (HASH_SALT in .env), so the original IDs cannot be
recovered from the keys, but the same patient always gets the same key, which
keeps joins and patient-level summaries possible.
"""
import hashlib

import pandas as pd

from etl import config


def surrogate_key(value, prefix: str) -> str:
    salt = config.require_salt()
    digest = hashlib.sha256(f"{salt}|{prefix}|{value}".encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:20]}"


def surrogate_keys(values: pd.Series, prefix: str) -> pd.Series:
    return values.astype(str).map(lambda v: surrogate_key(v, prefix))
