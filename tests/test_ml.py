"""Unit tests for the Part 2 ML code (no database, MLflow or Docker needed).
Run:  pytest -q
"""
import os

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("API_PRELOAD", "0")
os.environ.setdefault("DISABLE_PREDICTION_LOG", "1")

from ml import config, evaluate  # noqa: E402
from ml.drift import build_profile, feature_psi, psi  # noqa: E402
from ml.features import (  # noqa: E402
    CATEGORICAL_FEATURES, FEATURES, LABEL, NUMERIC_FEATURES, chronological_split,
    feature_ranges, original_feature_of,
)
from ml.model import ReadmissionRiskModel  # noqa: E402
from ml.train import make_candidates, to_db_rows  # noqa: E402


def synthetic_features(n=3000, seed=0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 10, n).astype(float) for c in NUMERIC_FEATURES})
    df["age_midpoint"] = rng.choice([45, 55, 65, 75, 85], n).astype(float)
    df["length_of_stay_days"] = rng.integers(1, 15, n).astype(float)
    df["med_changed"] = rng.integers(0, 2, n).astype(float)
    df["diabetes_med"] = rng.integers(0, 2, n).astype(float)
    for c in CATEGORICAL_FEATURES:
        df[c] = rng.choice(["A", "B", "C"], n)
    df["gender"] = rng.choice(["Male", "Female"], n)
    df["race"] = rng.choice(["Caucasian", "African American"], n)
    df["age_group"] = df["age_midpoint"].map(lambda a: f"{int(a) - 5}-{int(a) + 5}")
    logit = -2.2 + 0.35 * df["number_inpatient"] / 3 + 0.1 * (df["length_of_stay_days"] > 7)
    df[LABEL] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["encounter_order"] = np.arange(1, n + 1)
    df["encounter_sk"] = [f"enc_{i}" for i in range(n)]
    return df


# ------------------------------------------------------------------ metrics & drift
def test_psi_zero_for_identical_and_large_for_shift():
    assert psi([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0, abs=1e-9)
    assert psi([0.9, 0.1], [0.1, 0.9]) > 1.0


def test_feature_psi_detects_shift():
    df = synthetic_features()
    prof = build_profile(df)
    stable = feature_psi(prof, df.sample(1000, random_state=1))
    assert (stable["psi"] < config.PSI_WARNING).all()
    shifted = df.copy()
    shifted["length_of_stay_days"] = 14.0
    out = feature_psi(prof, shifted).set_index("feature")
    assert out.loc["length_of_stay_days", "status"] == "drift"


def test_ece_and_threshold():
    y = np.array([0] * 90 + [1] * 10)
    assert evaluate.expected_calibration_error(y, np.full(100, 0.1)) == pytest.approx(0, abs=1e-9)
    assert evaluate.expected_calibration_error(y, np.full(100, 0.6)) == pytest.approx(0.5, abs=1e-9)
    p = np.r_[np.linspace(0, 0.4, 90), np.linspace(0.5, 1, 10)]
    thr = evaluate.best_f1_threshold(y, p)
    m = evaluate.classification_metrics(y, p, thr)
    assert m["recall"] == 1.0 and m["precision"] == 1.0


def test_chronological_split_respects_order():
    df = synthetic_features(1000).sample(frac=1, random_state=3)
    tr, va, te = chronological_split(df)
    assert len(tr) == 700 and len(va) == 150 and len(te) == 150
    assert tr["encounter_order"].max() < va["encounter_order"].min() < te["encounter_order"].min()


def test_original_feature_of():
    assert original_feature_of("num__age_midpoint") == "age_midpoint"
    assert original_feature_of("cat__department_Internal Medicine") == "department"
    assert original_feature_of("cat__diag_2_category_Diabetes") == "diag_2_category"


def test_to_db_rows_converts_numpy_and_nan():
    rows = to_db_rows(pd.DataFrame({"a": [np.int64(1)], "b": [np.nan]}))
    assert rows == [(1, None)] and type(rows[0][0]) is int


# ------------------------------------------------------------------ model wrapper
@pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest", "xgboost"])
def test_model_wrapper_predicts_and_explains(model_type):
    from sklearn.isotonic import IsotonicRegression
    df = synthetic_features()
    tr, va, te = chronological_split(df)
    pipe, _ = make_candidates(pos_weight=5.0)[model_type]
    if model_type == "random_forest":
        pipe.set_params(model__n_estimators=20)
    pipe.fit(tr[FEATURES], tr[LABEL])
    cal = IsotonicRegression(out_of_bounds="clip").fit(pipe.predict_proba(va[FEATURES])[:, 1], va[LABEL])
    means = pipe.named_steps["preprocess"].transform(tr[FEATURES]).mean(axis=0)
    model = ReadmissionRiskModel(pipe, cal, 0.2, float(tr[LABEL].mean()), model_type,
                                 feature_ranges(tr), transformed_means=means)
    out = model.predict(None, te[FEATURES].head(10))
    assert list(out.columns) == ["probability", "flagged", "risk_band"]
    assert out["probability"].between(0, 1).all()
    expl = model.explain(te[FEATURES].head(3), top_k=4)
    assert len(expl) == 3 and len(expl[0]) == 4
    assert {f["feature"] for f in expl[0]} <= set(FEATURES)
    odd = te[FEATURES].head(1).copy()
    odd["department"] = "Never seen"
    odd["length_of_stay_days"] = 99
    flags = model.input_quality_flags(odd)[0]
    assert "unseen_category:department" in flags and "out_of_range:length_of_stay_days" in flags


# ------------------------------------------------------------------ API
def test_api_predict_with_injected_model():
    from fastapi.testclient import TestClient
    from sklearn.isotonic import IsotonicRegression
    from api import main
    df = synthetic_features()
    tr, va, _ = chronological_split(df)
    pipe, _ = make_candidates(pos_weight=5.0)["logistic_regression"]
    pipe.fit(tr[FEATURES], tr[LABEL])
    cal = IsotonicRegression(out_of_bounds="clip").fit(pipe.predict_proba(va[FEATURES])[:, 1], va[LABEL])
    main.holder.model = ReadmissionRiskModel(pipe, cal, 0.2, 0.1, "logistic_regression", feature_ranges(tr))
    main.holder.version = "99"
    client = TestClient(main.app)
    enc = {c: float(tr.iloc[0][c]) for c in NUMERIC_FEATURES}
    enc.update({c: str(tr.iloc[0][c]) for c in CATEGORICAL_FEATURES})
    enc["diagnosis_count"] = 3
    enc["number_diagnoses"] = 9
    r = client.post("/predict", json={"encounter": enc})
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0 <= body["probability"] <= 1 and body["model_version"] == "99" and body["top_factors"]
    r = client.post("/predict/batch", json={"encounters": [enc, enc]})
    assert r.status_code == 200 and r.json()["count"] == 2
    bad = dict(enc, length_of_stay_days=40)
    assert client.post("/predict", json={"encounter": bad}).status_code == 422
    assert "predictions_total" in client.get("/metrics").text
    assert client.get("/health").json()["model_loaded"] is True
