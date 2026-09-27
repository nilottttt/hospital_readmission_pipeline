"""STEP 3 - VALIDATE

Applies the data-quality rules to staging.encounters.
* A row that breaks any REJECT rule is written to audit.rejected_records
  (one line per broken rule) and to data/rejected/<run_id>_rejected.csv,
  and is marked dq_passed = false in staging.
* WARNING checks do not reject rows; their counts go to audit.dq_check_results.
* CRITICAL checks stop the pipeline if they fail (e.g. too many rejects).
"""
import json
import logging

import pandas as pd

from etl import config, db
from etl.columns import (
    EXPIRED_OR_HOSPICE_DISCHARGE_IDS, MEDICATION_COLUMNS, VALID_AGE_BRACKETS,
    VALID_GENDERS, VALID_MED_VALUES, VALID_READMITTED,
)
from etl.extract import safe_run_folder

log = logging.getLogger(__name__)

MAX_REJECT_RATE = 0.10   # more than 10 % rejected => source problem, stop the run

COUNT_COLUMNS = [
    "num_lab_procedures", "num_procedures", "num_medications", "number_outpatient",
    "number_emergency", "number_inpatient", "number_diagnoses",
]
ID_COLUMNS = ["admission_type_id", "discharge_disposition_id", "admission_source_id"]


def _is_int(s: pd.Series) -> pd.Series:
    return s.fillna("").str.fullmatch(r"\d+")


# Each rule returns (failed_mask, offending_value_series)
def rule_encounter_id(df):
    return ~_is_int(df["encounter_id"]), df["encounter_id"]


def rule_duplicate_encounter(df):
    return df["encounter_id"].duplicated(keep="first"), df["encounter_id"]


def rule_patient_nbr(df):
    return ~_is_int(df["patient_nbr"]), df["patient_nbr"]


def rule_gender(df):
    return ~df["gender"].isin(VALID_GENDERS), df["gender"]


def rule_age(df):
    return ~df["age"].isin(VALID_AGE_BRACKETS), df["age"]


def rule_length_of_stay(df):
    los = pd.to_numeric(df["time_in_hospital"], errors="coerce")
    ok = _is_int(df["time_in_hospital"]) & los.between(1, 14)
    return ~ok, df["time_in_hospital"]


def rule_counts(df):
    bad = pd.Series(False, index=df.index)
    value = pd.Series("", index=df.index)
    for c in COUNT_COLUMNS:
        col_bad = ~_is_int(df[c])
        value = value.mask(col_bad & ~bad, c + "=" + df[c].astype(str))
        bad |= col_bad
    return bad, value


def rule_id_codes(df):
    bad = pd.Series(False, index=df.index)
    value = pd.Series("", index=df.index)
    for c in ID_COLUMNS:
        col_bad = ~_is_int(df[c])
        value = value.mask(col_bad & ~bad, c + "=" + df[c].astype(str))
        bad |= col_bad
    return bad, value


def rule_readmitted(df):
    return ~df["readmitted"].isin(VALID_READMITTED), df["readmitted"]


def rule_expired_or_hospice(df):
    ids = pd.to_numeric(df["discharge_disposition_id"], errors="coerce")
    return ids.isin(EXPIRED_OR_HOSPICE_DISCHARGE_IDS), df["discharge_disposition_id"]


def rule_medication_values(df):
    bad = pd.Series(False, index=df.index)
    value = pd.Series("", index=df.index)
    for c in MEDICATION_COLUMNS:
        col_bad = ~df[c].isin(VALID_MED_VALUES)
        value = value.mask(col_bad & ~bad, c + "=" + df[c].astype(str))
        bad |= col_bad
    return bad, value


REJECT_RULES = [
    ("V01", "encounter_id is missing or not a whole number", rule_encounter_id),
    ("V02", "Duplicate encounter_id (first occurrence kept)", rule_duplicate_encounter),
    ("V03", "patient_nbr is missing or not a whole number", rule_patient_nbr),
    ("V04", "gender is not Male/Female (Unknown/Invalid)", rule_gender),
    ("V05", "age is not a valid 10-year bracket", rule_age),
    ("V06", "time_in_hospital is not a whole number between 1 and 14", rule_length_of_stay),
    ("V07", "A count field is missing, negative or not a whole number", rule_counts),
    ("V08", "An admission/discharge/source ID is not a whole number", rule_id_codes),
    ("V09", "readmitted is not one of <30, >30, NO", rule_readmitted),
    ("V10", "Patient expired or discharged to hospice - cannot be readmitted", rule_expired_or_hospice),
    ("V11", "A medication field is not one of No/Steady/Up/Down", rule_medication_values),
]

# Non-blocking completeness checks: (name, column, missing values)
WARNING_CHECKS = [
    ("race_missing", "race", {"?", ""}),
    ("weight_missing", "weight", {"?", ""}),
    ("payer_code_missing", "payer_code", {"?", ""}),
    ("medical_specialty_missing", "medical_specialty", {"?", ""}),
    ("diag_1_missing", "diag_1", {"?", ""}),
]


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per (record, broken rule)."""
    findings = []
    for rule_id, desc, fn in REJECT_RULES:
        mask, value = fn(df)
        if mask.any():
            hit = df.loc[mask, ["source_row_num", "encounter_id"]].copy()
            hit["rule_id"] = rule_id
            hit["rule_description"] = desc
            hit["offending_value"] = value[mask].astype(str).values
            findings.append(hit)
    if not findings:
        return pd.DataFrame(columns=["source_row_num", "encounter_id", "rule_id",
                                     "rule_description", "offending_value"])
    return pd.concat(findings, ignore_index=True)


def record_dq(run_id, layer, name, severity, passed, failed_count, details):
    db.insert_rows(
        "audit.dq_check_results",
        ["run_id", "layer", "check_name", "severity", "passed", "failed_count", "details"],
        [(run_id, layer, name, severity, bool(passed), int(failed_count), details)],
    )


def run(run_id: str) -> dict:
    df = db.fetch_df("SELECT * FROM staging.encounters WHERE run_id=%s ORDER BY source_row_num",
                     (run_id,))
    total = len(df)
    if total == 0:
        raise RuntimeError("staging.encounters is empty for this run")

    findings = apply_rules(df)
    rejected_rows = set(findings["source_row_num"].astype(int))

    # --- rejected-record log (DB + CSV) -------------------------------------
    if len(findings):
        # keep the full record for traceability, but drop patient_nbr (privacy)
        drop = {"source_row_num", "patient_nbr", "run_id", "loaded_at", "dq_passed"}
        rec_cols = [c for c in df.columns if c not in drop]
        bad_df = df[df["source_row_num"].isin(rejected_rows)]
        records = bad_df.set_index("source_row_num")[rec_cols].astype(str).to_dict(orient="index")
        rows = [
            (run_id, "validate", int(r.source_row_num), r.encounter_id, r.rule_id,
             r.rule_description, r.offending_value,
             json.dumps(records[int(r.source_row_num)]))
            for r in findings.itertuples(index=False)
        ]
        db.insert_rows(
            "audit.rejected_records",
            ["run_id", "stage", "source_row_num", "encounter_id", "rule_id",
             "rule_description", "offending_value", "record"],
            rows,
        )
        config.REJECTED_DIR.mkdir(parents=True, exist_ok=True)
        out = config.REJECTED_DIR / f"{safe_run_folder(run_id)}_rejected.csv"
        findings.to_csv(out, index=False)
        log.info("Rejected-record log written to %s", out)

    # --- flag rows in staging ------------------------------------------------
    with db.connect() as conn, conn.cursor() as cur:
        cur.execute("UPDATE staging.encounters SET dq_passed = TRUE WHERE run_id=%s", (run_id,))
        if rejected_rows:
            cur.execute(
                "UPDATE staging.encounters SET dq_passed = FALSE "
                "WHERE run_id=%s AND source_row_num = ANY(%s)",
                (run_id, sorted(rejected_rows)),
            )

    # --- DQ results summary --------------------------------------------------
    per_rule = findings.groupby("rule_id").size().to_dict() if len(findings) else {}
    for rule_id, desc, _ in REJECT_RULES:
        n = per_rule.get(rule_id, 0)
        record_dq(run_id, "staging", f"{rule_id}: {desc}", "reject", n == 0, n,
                  f"{n} rows rejected")
    for name, col, missing in WARNING_CHECKS:
        n = int(df[col].isin(missing).sum())
        record_dq(run_id, "staging", name, "warning", n == 0, n,
                  f"{n / total:.1%} of rows missing {col}")

    n_rejected = len(rejected_rows)
    n_valid = total - n_rejected
    reject_rate = n_rejected / total
    ok = reject_rate <= MAX_REJECT_RATE
    record_dq(run_id, "staging", "reject_rate_below_threshold", "critical", ok, n_rejected,
              f"reject rate {reject_rate:.2%} (limit {MAX_REJECT_RATE:.0%})")

    db.execute("UPDATE audit.pipeline_runs SET rows_valid=%s, rows_rejected=%s WHERE run_id=%s",
               (n_valid, n_rejected, run_id))
    log.info("Validation: %s valid, %s rejected (%.2f%%)", n_valid, n_rejected, reject_rate * 100)
    if not ok:
        raise RuntimeError(f"Reject rate {reject_rate:.2%} exceeds {MAX_REJECT_RATE:.0%} - stopping")
    return {"valid": n_valid, "rejected": n_rejected, "by_rule": per_rule}
