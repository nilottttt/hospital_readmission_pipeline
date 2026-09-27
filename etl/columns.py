"""Column definitions and code lists for the UCI Diabetes 130-US dataset."""

# The 23 medication columns in the source file (source spelling)
MEDICATION_SOURCE_COLUMNS = [
    "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
    "acetohexamide", "glipizide", "glyburide", "tolbutamide", "pioglitazone",
    "rosiglitazone", "acarbose", "miglitol", "troglitazone", "tolazamide",
    "examide", "citoglipton", "insulin", "glyburide-metformin",
    "glipizide-metformin", "glimepiride-pioglitazone", "metformin-rosiglitazone",
    "metformin-pioglitazone",
]

# All 50 columns expected in diabetic_data.csv, in source order
SOURCE_COLUMNS = [
    "encounter_id", "patient_nbr", "race", "gender", "age", "weight",
    "admission_type_id", "discharge_disposition_id", "admission_source_id",
    "time_in_hospital", "payer_code", "medical_specialty", "num_lab_procedures",
    "num_procedures", "num_medications", "number_outpatient", "number_emergency",
    "number_inpatient", "diag_1", "diag_2", "diag_3", "number_diagnoses",
    "max_glu_serum", "A1Cresult", *MEDICATION_SOURCE_COLUMNS,
    "change", "diabetesMed", "readmitted",
]


def to_snake(name: str) -> str:
    """'glyburide-metformin' -> 'glyburide_metformin', 'A1Cresult' -> 'a1c_result'."""
    special = {"A1Cresult": "a1c_result", "diabetesMed": "diabetes_med"}
    return special.get(name, name.lower().replace("-", "_"))


STAGING_COLUMNS = [to_snake(c) for c in SOURCE_COLUMNS]
MEDICATION_COLUMNS = [to_snake(c) for c in MEDICATION_SOURCE_COLUMNS]

INTEGER_COLUMNS = [
    "encounter_id", "patient_nbr", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "time_in_hospital", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "number_diagnoses",
]

VALID_AGE_BRACKETS = [
    "[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)",
    "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)",
]
VALID_GENDERS = {"Male", "Female"}
VALID_READMITTED = {"<30", ">30", "NO"}
VALID_MED_VALUES = {"No", "Steady", "Up", "Down"}

# Discharge dispositions meaning the patient died or went to hospice.
# These patients cannot be readmitted, so the encounters are excluded
# from readmission analysis (standard practice, Strack et al. 2014).
EXPIRED_OR_HOSPICE_DISCHARGE_IDS = {11, 13, 14, 19, 20, 21}

# Special "missing" codes used by the ID columns (see IDs_mapping.csv)
MISSING_ADMISSION_TYPE_IDS = {5, 6, 8}          # Not Available / NULL / Not Mapped
MISSING_DISCHARGE_IDS = {18, 25, 26}            # NULL / Not Mapped / Unknown/Invalid
MISSING_ADMISSION_SOURCE_IDS = {9, 15, 17, 20, 21}  # Not Available / NULL / Not Mapped / Unknown
