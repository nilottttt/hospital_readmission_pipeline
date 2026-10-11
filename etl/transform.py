"""STEP 4 - TRANSFORM (staging -> clean)

For every row that passed validation:
* replaces special missing-value codes ('?', 'None', NULL/Not Mapped IDs)
* standardizes admission, discharge, diagnosis and medication fields
* creates patient-safe surrogate keys and drops the original IDs
* engineers length-of-stay, prior-admission, diagnosis-count and
  medication-count variables, readmission flags and a rule-based risk score
"""
import logging
import math
import re

import numpy as np
import pandas as pd

from etl import db
from etl.columns import (
    INTEGER_COLUMNS, MEDICATION_COLUMNS, MISSING_ADMISSION_SOURCE_IDS,
    MISSING_ADMISSION_TYPE_IDS, MISSING_DISCHARGE_IDS, VALID_AGE_BRACKETS,
)
from etl.privacy import surrogate_keys
from etl.validate import record_dq

log = logging.getLogger(__name__)

MISSING_TOKENS = {"?", "", "NULL", "null"}
UNKNOWN = "Unknown"


# ----------------------------------------------------------------------------
# Standardization helpers (pure functions - unit tested in tests/)
# ----------------------------------------------------------------------------
def icd9_category(code) -> str:
    """Group an ICD-9 code into a clinical category (Strack et al., 2014)."""
    if code is None or (isinstance(code, float) and math.isnan(code)):
        return "Missing"
    code = str(code).strip()
    if code in MISSING_TOKENS:
        return "Missing"
    if code[0] in ("V", "E"):          # supplementary / external-cause codes
        return "Other"
    try:
        num = float(code)
    except ValueError:
        return "Other"
    whole = int(num)
    if 390 <= num < 460 or whole == 785:
        return "Circulatory"
    if 460 <= num < 520 or whole == 786:
        return "Respiratory"
    if 520 <= num < 580 or whole == 787:
        return "Digestive"
    if whole == 250:
        return "Diabetes"
    if 800 <= num < 1000:
        return "Injury"
    if 710 <= num < 740:
        return "Musculoskeletal"
    if 580 <= num < 630 or whole == 788:
        return "Genitourinary"
    if 140 <= num < 240:
        return "Neoplasms"
    return "Other"


def specialty_label(raw) -> str:
    """'InternalMedicine' -> 'Internal Medicine', '?' -> 'Unknown'."""
    if raw is None or pd.isna(raw) or str(raw).strip() in MISSING_TOKENS:
        return UNKNOWN
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", str(raw).strip())
    return s.replace("&", " & ").replace("  ", " ")


def department_for(raw) -> str:
    """Group the ~70 medical specialties into ~15 departments."""
    if raw is None or pd.isna(raw) or str(raw).strip() in MISSING_TOKENS:
        return UNKNOWN
    s = str(raw).strip()
    rules = [
        (("InternalMedicine",), "Internal Medicine"),
        (("Emergency",), "Emergency/Trauma"),
        (("Family/GeneralPractice",), "Family/General Practice"),
        (("Cardiology",), "Cardiology"),
        (("Surg",), "Surgery"),
        (("Orthopedics",), "Orthopedics"),
        (("Nephrology",), "Nephrology"),
        (("Pulmonology",), "Pulmonology"),
        (("Gastroenterology",), "Gastroenterology"),
        (("Psychiatry", "Psychology"), "Psychiatry"),
        (("Oncology", "Hematology"), "Oncology/Hematology"),
        (("Obstetric", "Gynecology"), "Obstetrics & Gynecology"),
        (("Pediatrics",), "Pediatrics"),
        (("Radiolog",), "Radiology"),
        (("Neurology",), "Neurology"),
        (("Urology",), "Urology"),
    ]
    for prefixes, dept in rules:
        if any(s.startswith(p) for p in prefixes):
            return dept
    return "Other"


def discharge_group(code: int) -> str:
    if code in MISSING_DISCHARGE_IDS:
        return UNKNOWN
    if code == 1:
        return "Home"
    if code in (6, 8):
        return "Home with health service"
    if code == 7:
        return "Left against medical advice"
    if code in (2, 3, 4, 5, 10, 15, 22, 23, 24, 27, 28, 29, 30):
        return "Transferred to another facility"
    return "Other"


def admission_source_group(code: int) -> str:
    if code in MISSING_ADMISSION_SOURCE_IDS:
        return UNKNOWN
    if code == 7:
        return "Emergency Room"
    if code in (1, 2, 3):
        return "Referral"
    if code in (4, 5, 6, 10, 18, 22, 25, 26):
        return "Transfer"
    if code in (11, 12, 13, 14, 23, 24):
        return "Birth"
    return "Other"


def los_band(days: int) -> str:
    if days <= 2:
        return "1-2 days"
    if days <= 4:
        return "3-4 days"
    if days <= 7:
        return "5-7 days"
    return "8-14 days"


def risk_points(df: pd.DataFrame) -> pd.Series:
    """Transparent rule-based risk score used for the high-risk cohort view.
    (This is NOT the ML model - that comes in Part 2.)"""
    score = (
        np.where(df["number_inpatient"] >= 2, 2, np.where(df["number_inpatient"] == 1, 1, 0))
        + (df["number_emergency"] >= 1).astype(int)
        + (df["length_of_stay_days"] >= 7).astype(int)
        + (df["num_medications"] >= 20).astype(int)
        + (df["number_diagnoses"] >= 9).astype(int)
        + (df["age_midpoint"] >= 70).astype(int)
        + (df["discharge_group"] == "Transferred to another facility").astype(int)
    )
    return pd.Series(score, index=df.index)


def risk_level(score: int) -> str:
    return "High" if score >= 5 else ("Medium" if score >= 3 else "Low")


# ----------------------------------------------------------------------------
# Lookups
# ----------------------------------------------------------------------------
def build_lookups() -> pd.DataFrame:
    raw = db.fetch_df("SELECT lookup_type, code, description FROM raw.lookup_ids")
    missing_sets = {
        "admission_type_id": MISSING_ADMISSION_TYPE_IDS,
        "discharge_disposition_id": MISSING_DISCHARGE_IDS,
        "admission_source_id": MISSING_ADMISSION_SOURCE_IDS,
    }
    raw["description"] = raw["description"].fillna("").str.strip().str.rstrip(" .")
    raw["is_missing_code"] = raw.apply(
        lambda r: r["code"] in missing_sets.get(r["lookup_type"], set())
        or r["description"] in ("", "NULL", "Not Available", "Not Mapped", "Unknown/Invalid"),
        axis=1,
    )
    raw.loc[raw["is_missing_code"], "description"] = UNKNOWN
    # fix the one typo in the source file
    raw["description"] = raw["description"].str.replace("critial", "critical", regex=False)
    return raw


def map_code(series: pd.Series, lookups: pd.DataFrame, lookup_type: str) -> pd.Series:
    table = lookups[lookups["lookup_type"] == lookup_type].set_index("code")["description"]
    return series.map(table).fillna(UNKNOWN)


# ----------------------------------------------------------------------------
# Main transformation
# ----------------------------------------------------------------------------
def transform(df: pd.DataFrame, lookups: pd.DataFrame, run_id: str) -> pd.DataFrame:
    df = df.copy()

    # 1) special missing-value codes -> NaN
    for c in ["race", "weight", "payer_code", "medical_specialty", "diag_1", "diag_2", "diag_3"]:
        df[c] = df[c].where(~df[c].isin(MISSING_TOKENS), np.nan)
    for c in INTEGER_COLUMNS:
        df[c] = pd.to_numeric(df[c]).astype("int64")

    # 2) ordering within patient (encounter_id increases over time in this dataset)
    df = df.sort_values(["patient_nbr", "encounter_id"])
    out = pd.DataFrame(index=df.index)
    seq = df.groupby("patient_nbr").cumcount() + 1

    # 3) surrogate keys (original IDs are NOT carried forward)
    out["encounter_sk"] = surrogate_keys(df["encounter_id"], "enc")
    out["patient_sk"] = surrogate_keys(df["patient_nbr"], "pat")
    out["encounter_seq"] = seq
    # global time-order proxy (1 = oldest encounter). Used for the chronological
    # train/validation/test split in Part 2. A rank reveals no original ID.
    out["encounter_order"] = df["encounter_id"].rank(method="first").astype("int64")
    out["run_id"] = run_id

    # 4) demographics
    out["race"] = df["race"].fillna(UNKNOWN).replace({"AfricanAmerican": "African American"})
    out["gender"] = df["gender"]
    age_order = {b: i + 1 for i, b in enumerate(VALID_AGE_BRACKETS)}
    out["age_group"] = df["age"].str.strip("[)")
    out["age_group_order"] = df["age"].map(age_order)
    out["age_midpoint"] = df["age"].map(lambda a: int(a.strip("[)").split("-")[0]) + 5)
    out["weight_recorded"] = df["weight"].notna()

    # 5) admission / discharge standardization
    out["admission_type_id"] = df["admission_type_id"]
    out["admission_type"] = map_code(df["admission_type_id"], lookups, "admission_type_id")
    out["discharge_disposition_id"] = df["discharge_disposition_id"]
    out["discharge_disposition"] = map_code(df["discharge_disposition_id"], lookups,
                                            "discharge_disposition_id")
    out["discharge_group"] = df["discharge_disposition_id"].map(discharge_group)
    out["admission_source_id"] = df["admission_source_id"]
    out["admission_source"] = map_code(df["admission_source_id"], lookups, "admission_source_id")
    out["admission_source_group"] = df["admission_source_id"].map(admission_source_group)
    out["payer_code"] = df["payer_code"].fillna(UNKNOWN)
    out["medical_specialty"] = df["medical_specialty"].map(specialty_label)
    out["department"] = df["medical_specialty"].map(department_for)

    # 6) length of stay and utilisation
    out["length_of_stay_days"] = df["time_in_hospital"]
    out["los_band"] = df["time_in_hospital"].map(los_band)
    for c in ["num_lab_procedures", "num_procedures", "num_medications",
              "number_outpatient", "number_emergency", "number_inpatient"]:
        out[c] = df[c]
    out["prior_admissions"] = df["number_outpatient"] + df["number_emergency"] + df["number_inpatient"]
    out["prior_encounters_in_data"] = seq - 1

    # 7) diagnoses
    cat_cache = {}
    for i in (1, 2, 3):
        col = f"diag_{i}"
        out[col] = df[col].str.strip()
        out[f"{col}_category"] = df[col].map(
            lambda c: cat_cache.setdefault(c, icd9_category(c)))
    out["number_diagnoses"] = df["number_diagnoses"]
    out["diagnosis_count"] = df[["diag_1", "diag_2", "diag_3"]].notna().sum(axis=1)

    # 8) lab results ('None' in the source means "test not done")
    out["max_glu_serum"] = df["max_glu_serum"].replace({"None": "Not measured", "": "Not measured"})
    out["a1c_result"] = df["a1c_result"].replace({"None": "Not measured", "": "Not measured"})

    # 9) medications
    meds = df[MEDICATION_COLUMNS]
    out["medication_count"] = (meds != "No").sum(axis=1)
    out["medication_change_count"] = meds.isin(["Up", "Down"]).sum(axis=1)
    out["insulin"] = df["insulin"]
    out["med_changed"] = df["change"].eq("Ch")
    out["diabetes_med"] = df["diabetes_med"].eq("Yes")

    # 10) readmission target
    out["readmitted"] = df["readmitted"]
    out["readmitted_30d"] = df["readmitted"].eq("<30")
    out["readmitted_any"] = df["readmitted"].isin(["<30", ">30"])

    # 11) rule-based risk score for the high-risk cohort dashboard
    out["risk_score"] = risk_points(out)
    out["risk_level"] = out["risk_score"].map(risk_level)

    for c in MEDICATION_COLUMNS:
        if c != "insulin":
            out[c] = df[c]
    return out.reset_index(drop=True)


def run(run_id: str) -> int:
    staged = db.fetch_df(
        "SELECT * FROM staging.encounters WHERE run_id=%s AND dq_passed ORDER BY source_row_num",
        (run_id,))
    lookups = build_lookups()
    clean = transform(staged, lookups, run_id)

    # post-transform critical checks
    checks = [
        ("encounter_sk_unique", clean["encounter_sk"].duplicated().sum()),
        ("encounter_sk_not_null", clean["encounter_sk"].isna().sum()),
        ("row_count_matches_valid_rows", abs(len(clean) - len(staged))),
        ("no_nulls_in_required_columns",
         clean.drop(columns=["diag_1", "diag_2", "diag_3"]).isna().any(axis=1).sum()),
    ]
    for name, failed in checks:
        record_dq(run_id, "clean", name, "critical", failed == 0, failed, f"{failed} failures")
        if failed:
            raise RuntimeError(f"Critical data-quality check failed: {name} ({failed})")

    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute("TRUNCATE clean.lookup_codes, clean.encounters")
        db.copy_df(lookups[["lookup_type", "code", "description", "is_missing_code"]],
                   "clean.lookup_codes", conn=conn)
        db.copy_df(clean, "clean.encounters", conn=conn)

    db.execute("UPDATE audit.pipeline_runs SET rows_clean=%s WHERE run_id=%s", (len(clean), run_id))
    log.info("Clean layer written: %s rows", len(clean))
    return len(clean)
