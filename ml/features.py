"""Feature definitions and the reproducible feature-engineering pipeline.

The raw features come from mart.v_ml_features (built by the Part 1 ETL).
The model-side transformations (scaling, one-hot encoding) live in a
scikit-learn ColumnTransformer that is saved INSIDE the model, so training
and serving always apply exactly the same steps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ml import config

NUMERIC_FEATURES = [
    "age_midpoint", "length_of_stay_days", "num_lab_procedures", "num_procedures",
    "num_medications", "number_outpatient", "number_emergency", "number_inpatient",
    "prior_admissions", "prior_encounters_in_data", "number_diagnoses",
    "diagnosis_count", "medication_count", "medication_change_count",
    "med_changed", "diabetes_med",
]
CATEGORICAL_FEATURES = [
    "gender", "admission_type", "admission_source_group", "discharge_group",
    "department", "primary_diagnosis_category", "diag_2_category", "diag_3_category",
    "max_glu_serum", "a1c_result", "insulin",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
LABEL = "label_readmitted_30d"
ORDER_COLUMN = "encounter_order"
# Used to check fairness / subgroup performance. Deliberately NOT model inputs
# (race is excluded from the model to avoid encoding a social construct as risk).
SUBGROUP_COLUMNS = ["gender", "race", "age_group"]


def load_feature_table() -> pd.DataFrame:
    from etl import db  # imported lazily so the API does not need the ETL code
    df = db.fetch_df("SELECT * FROM mart.v_ml_features ORDER BY encounter_order")
    return coerce_types(df)


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in NUMERIC_FEATURES:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)
    for c in CATEGORICAL_FEATURES:
        if c in df:
            df[c] = df[c].astype("string").fillna("Unknown").astype(object)
    if LABEL in df:
        df[LABEL] = df[LABEL].astype(int)
    return df


def chronological_split(df: pd.DataFrame):
    """Oldest 70 % train, next 15 % validation, newest 15 % test.
    encounter_order is the rank of encounter_id, which increases over time,
    so the test set always lies 'in the future' of the training data."""
    df = df.sort_values(ORDER_COLUMN).reset_index(drop=True)
    n = len(df)
    i_train = int(n * config.TRAIN_FRACTION)
    i_valid = int(n * (config.TRAIN_FRACTION + config.VALID_FRACTION))
    return df.iloc[:i_train], df.iloc[i_train:i_valid], df.iloc[i_valid:]


def build_preprocessor(scale_numeric: bool = True) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=30,
                                  sparse_output=False), CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=True,
    )


def original_feature_of(transformed_name: str) -> str:
    """'cat__department_Cardiology' -> 'department', 'num__age_midpoint' -> 'age_midpoint'."""
    name = transformed_name.split("__", 1)[-1]
    if transformed_name.startswith("num__"):
        return name
    for c in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
        if name == c or name.startswith(c + "_"):
            return c
    return name


def feature_ranges(df: pd.DataFrame) -> dict:
    """Training-time ranges/categories, stored with the model for input-quality checks."""
    return {
        "numeric": {c: [float(np.nanmin(df[c])), float(np.nanmax(df[c]))] for c in NUMERIC_FEATURES},
        "categorical": {c: sorted(map(str, df[c].dropna().unique())) for c in CATEGORICAL_FEATURES},
    }
