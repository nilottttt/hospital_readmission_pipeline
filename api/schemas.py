"""Request / response schemas. Pydantic rejects impossible values (HTTP 422);
values that are possible but unusual are accepted and flagged (dq_flags)."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

DISCLAIMER = ("Academic decision-support prototype trained on 1999-2008 public data. "
              "Not validated for clinical use; must not replace clinical judgement.")


class Encounter(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "age_midpoint": 75, "length_of_stay_days": 6, "num_lab_procedures": 45,
        "num_procedures": 1, "num_medications": 18, "number_outpatient": 0,
        "number_emergency": 1, "number_inpatient": 2, "prior_admissions": 3,
        "prior_encounters_in_data": 1, "number_diagnoses": 9, "diagnosis_count": 3,
        "medication_count": 2, "medication_change_count": 1, "med_changed": 1, "diabetes_med": 1,
        "gender": "Female", "admission_type": "Emergency", "admission_source_group": "Emergency Room",
        "discharge_group": "Transferred to another facility", "department": "Internal Medicine",
        "primary_diagnosis_category": "Circulatory", "diag_2_category": "Diabetes",
        "diag_3_category": "Circulatory", "max_glu_serum": "Not measured", "a1c_result": ">8",
        "insulin": "Up"}})

    # numeric
    age_midpoint: float = Field(..., ge=0, le=100, description="Age bracket midpoint, e.g. 75 for 70-80")
    length_of_stay_days: float = Field(..., ge=1, le=14)
    num_lab_procedures: float = Field(..., ge=0, le=200)
    num_procedures: float = Field(..., ge=0, le=20)
    num_medications: float = Field(..., ge=0, le=100)
    number_outpatient: float = Field(..., ge=0, le=100)
    number_emergency: float = Field(..., ge=0, le=100)
    number_inpatient: float = Field(..., ge=0, le=100)
    prior_admissions: float = Field(..., ge=0, le=300)
    prior_encounters_in_data: float = Field(0, ge=0, le=100)
    number_diagnoses: float = Field(..., ge=1, le=20)
    diagnosis_count: float = Field(..., ge=0, le=3)
    medication_count: float = Field(..., ge=0, le=23)
    medication_change_count: float = Field(0, ge=0, le=23)
    med_changed: int = Field(..., ge=0, le=1)
    diabetes_med: int = Field(..., ge=0, le=1)
    # categorical
    gender: str
    admission_type: str
    admission_source_group: str
    discharge_group: str
    department: str
    primary_diagnosis_category: str
    diag_2_category: str = "Missing"
    diag_3_category: str = "Missing"
    max_glu_serum: str = "Not measured"
    a1c_result: str = "Not measured"
    insulin: str = "No"
    # optional link to a known (de-identified) encounter, used for performance monitoring
    encounter_sk: Optional[str] = None


class PredictRequest(BaseModel):
    encounter: Encounter
    explain: bool = True
    source: str = "api"


class BatchRequest(BaseModel):
    encounters: list[Encounter] = Field(..., min_length=1, max_length=5000)
    explain: bool = False
    source: str = "batch"


class Factor(BaseModel):
    feature: str
    value: Any = None
    contribution: float


class Prediction(BaseModel):
    request_id: str
    encounter_sk: Optional[str] = None
    probability: float
    risk_band: str
    flagged: bool
    threshold: float
    model_name: str
    model_version: str
    top_factors: list[Factor] = []
    dq_flags: list[str] = []
    latency_ms: float
    disclaimer: str = DISCLAIMER


class BatchResponse(BaseModel):
    model_version: str
    count: int
    latency_ms: float
    predictions: list[Prediction]
