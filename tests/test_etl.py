"""Unit tests for the pure ETL logic (no database needed).
Run:  pytest -q
"""
import os
from pathlib import Path

import pandas as pd
import pytest

os.environ.setdefault("HASH_SALT", "unit-test-salt")

from etl.columns import MEDICATION_COLUMNS, STAGING_COLUMNS  # noqa: E402
from etl.extract import parse_mapping_file  # noqa: E402
from etl.privacy import surrogate_key  # noqa: E402
from etl.transform import (  # noqa: E402
    admission_source_group, department_for, discharge_group, icd9_category,
    los_band, specialty_label, transform,
)
from etl.validate import apply_rules  # noqa: E402


@pytest.mark.parametrize("code,expected", [
    ("428", "Circulatory"), ("785", "Circulatory"), ("486", "Respiratory"),
    ("250.83", "Diabetes"), ("250", "Diabetes"), ("996", "Injury"),
    ("715", "Musculoskeletal"), ("599", "Genitourinary"), ("153", "Neoplasms"),
    ("V57", "Other"), ("E888", "Other"), ("?", "Missing"), (None, "Missing"),
    ("562", "Digestive"), ("038", "Other"),
])
def test_icd9_category(code, expected):
    assert icd9_category(code) == expected


def test_specialty_and_department():
    assert specialty_label("InternalMedicine") == "Internal Medicine"
    assert specialty_label("?") == "Unknown"
    assert department_for("Surgery-Cardiovascular/Thoracic") == "Surgery"
    assert department_for("Orthopedics-Reconstructive") == "Orthopedics"
    assert department_for("Dermatology") == "Other"
    assert department_for("?") == "Unknown"


def test_code_groups():
    assert discharge_group(1) == "Home"
    assert discharge_group(3) == "Transferred to another facility"
    assert discharge_group(18) == "Unknown"
    assert admission_source_group(7) == "Emergency Room"
    assert admission_source_group(17) == "Unknown"
    assert [los_band(d) for d in (1, 3, 6, 10)] == ["1-2 days", "3-4 days", "5-7 days", "8-14 days"]


def test_surrogate_key_is_stable_and_hides_id():
    a, b = surrogate_key(12345, "pat"), surrogate_key(12345, "pat")
    assert a == b and "12345" not in a and a.startswith("pat_")
    assert surrogate_key(12346, "pat") != a


def test_parse_mapping_file(tmp_path: Path):
    f = tmp_path / "IDs_mapping.csv"
    f.write_text("admission_type_id,description\n1,Emergency\n6,NULL\n,\n"
                 "discharge_disposition_id,description\n1,Discharged to home\n"
                 '19,"Expired at home. Medicaid only, hospice."\n,\n'
                 "admission_source_id,description\n7, Emergency Room\n")
    rows = parse_mapping_file(f)
    assert ("admission_type_id", 1, "Emergency") in rows
    assert ("discharge_disposition_id", 19, "Expired at home. Medicaid only, hospice.") in rows
    assert ("admission_source_id", 7, "Emergency Room") in rows
    assert len(rows) == 5


def _row(**overrides):
    base = {c: "No" for c in MEDICATION_COLUMNS}
    base.update({
        "source_row_num": 2, "encounter_id": "100", "patient_nbr": "555", "race": "Caucasian",
        "gender": "Female", "age": "[70-80)", "weight": "?", "admission_type_id": "1",
        "discharge_disposition_id": "1", "admission_source_id": "7", "time_in_hospital": "8",
        "payer_code": "?", "medical_specialty": "Cardiology", "num_lab_procedures": "40",
        "num_procedures": "1", "num_medications": "22", "number_outpatient": "0",
        "number_emergency": "1", "number_inpatient": "2", "diag_1": "428", "diag_2": "250.01",
        "diag_3": "?", "number_diagnoses": "9", "max_glu_serum": "None", "a1c_result": ">8",
        "insulin": "Up", "change": "Ch", "diabetes_med": "Yes", "readmitted": "<30",
    })
    base.update(overrides)
    return base


def test_validation_rules():
    df = pd.DataFrame([
        _row(source_row_num=2),
        _row(source_row_num=3, encounter_id="100"),                  # duplicate
        _row(source_row_num=4, encounter_id="101", gender="Unknown/Invalid"),
        _row(source_row_num=5, encounter_id="102", discharge_disposition_id="11"),  # expired
        _row(source_row_num=6, encounter_id="103", time_in_hospital="0"),
        _row(source_row_num=7, encounter_id="104"),                  # clean
    ])
    found = apply_rules(df)
    assert set(found["rule_id"]) == {"V02", "V04", "V10", "V06"}
    assert set(found["source_row_num"]) == {3, 4, 5, 6}


def test_transform_features():
    lookups = pd.DataFrame([
        ("admission_type_id", 1, "Emergency", False),
        ("discharge_disposition_id", 1, "Discharged to home", False),
        ("admission_source_id", 7, "Emergency Room", False),
    ], columns=["lookup_type", "code", "description", "is_missing_code"])
    df = pd.DataFrame([_row(), _row(source_row_num=3, encounter_id="200", readmitted="NO")])
    out = transform(df, lookups, "test-run")
    first = out.iloc[0]
    assert set(STAGING_COLUMNS) - set(df.columns) == set()
    assert first["age_group"] == "70-80" and first["age_midpoint"] == 75
    assert first["prior_admissions"] == 3
    assert first["diagnosis_count"] == 2              # diag_3 was '?'
    assert first["diag_1_category"] == "Circulatory"
    assert first["medication_count"] == 1 and first["medication_change_count"] == 1
    assert first["max_glu_serum"] == "Not measured"
    assert bool(first["readmitted_30d"]) is True
    assert first["risk_score"] == 2 + 1 + 1 + 1 + 1 + 1   # inpatient>=2, ER, LOS>=7, meds>=20, dx>=9, age>=70
    assert first["risk_level"] == "High"
    assert list(out["encounter_seq"]) == [1, 2]
    assert "patient_nbr" not in out.columns and "encounter_id" not in out.columns
