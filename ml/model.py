"""The deployable model: preprocessing + classifier + calibrator + threshold,
packaged as one MLflow pyfunc model so the API loads a single artifact.

    probability = isotonic_calibrator( pipeline.predict_proba(X)[:, 1] )
    flagged     = probability >= threshold
    risk_band   = High (>= threshold) / Medium (>= training base rate) / Low
"""
from __future__ import annotations

import numpy as np
import pandas as pd

try:  # mlflow is needed for packaging, not for using the object directly
    import mlflow.pyfunc
    _Base = mlflow.pyfunc.PythonModel
except ImportError:  # pragma: no cover
    _Base = object

from ml.features import CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES, coerce_types, original_feature_of


class ReadmissionRiskModel(_Base):
    def __init__(self, pipeline, calibrator, threshold: float, base_rate: float,
                 model_type: str, feature_ranges: dict, transformed_means=None):
        self.pipeline = pipeline            # sklearn Pipeline(preprocess, clf)
        self.calibrator = calibrator        # IsotonicRegression or None
        self.threshold = float(threshold)
        self.base_rate = float(base_rate)
        self.model_type = model_type        # logistic_regression / random_forest / xgboost
        self.feature_ranges = feature_ranges
        self.transformed_means = transformed_means  # for linear-model explanations

    # ------------------------------------------------------------------ scoring
    def raw_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(coerce_types(X)[FEATURES])[:, 1]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p = self.raw_proba(X)
        if self.calibrator is not None:
            p = self.calibrator.predict(p)
        return np.clip(p, 0.0, 1.0)

    def band(self, p: float) -> str:
        if p >= self.threshold:
            return "High"
        return "Medium" if p >= self.base_rate else "Low"

    def predict(self, context, model_input: pd.DataFrame, params=None) -> pd.DataFrame:  # pyfunc API
        p = self.predict_proba(model_input)
        return pd.DataFrame({
            "probability": p,
            "flagged": p >= self.threshold,
            "risk_band": [self.band(x) for x in p],
        })

    # ------------------------------------------------------------------ explanations
    def explain(self, X: pd.DataFrame, top_k: int = 5) -> list[list[dict]]:
        """Per-row top contributing features (SHAP values; log-odds units for
        XGBoost / logistic regression, probability units for random forest).
        Positive = pushes risk up. One-hot columns are summed back to their
        original feature."""
        X = coerce_types(X)[FEATURES]
        pre = self.pipeline.named_steps["preprocess"]
        clf = self.pipeline.named_steps["model"]
        Xt = pre.transform(X)
        names = list(pre.get_feature_names_out())

        if self.model_type == "xgboost":
            import xgboost as xgb
            contrib = clf.get_booster().predict(xgb.DMatrix(Xt, feature_names=None),
                                                pred_contribs=True)[:, :-1]
        elif self.model_type == "logistic_regression":
            means = self.transformed_means if self.transformed_means is not None else Xt.mean(axis=0)
            contrib = (Xt - means) * clf.coef_[0]
        else:
            try:
                import shap
                if getattr(self, "_tree_explainer", None) is None:   # cached, built once
                    self._tree_explainer = shap.TreeExplainer(clf)
                sv = self._tree_explainer.shap_values(Xt)
                contrib = sv[1] if isinstance(sv, list) else (sv[..., 1] if sv.ndim == 3 else sv)
            except Exception:
                return [[] for _ in range(len(X))]

        groups = pd.Series([original_feature_of(n) for n in names])
        out = []
        for i in range(len(X)):
            s = pd.Series(contrib[i]).groupby(groups.values).sum()
            s = s.reindex(s.abs().sort_values(ascending=False).index)[:top_k]
            out.append([{"feature": f, "value": _jsonable(X.iloc[i][f]),
                         "contribution": round(float(v), 4)} for f, v in s.items()])
        return out

    # ------------------------------------------------------------------ input checks
    def input_quality_flags(self, X: pd.DataFrame) -> list[list[str]]:
        """Values outside the training range or categories never seen in training."""
        flags = []
        X = coerce_types(X)
        for _, row in X.iterrows():
            f = []
            for c in NUMERIC_FEATURES:
                lo, hi = self.feature_ranges["numeric"].get(c, [None, None])
                v = row.get(c)
                if v is None or pd.isna(v):
                    f.append(f"missing:{c}")
                elif lo is not None and (v < lo or v > hi):
                    f.append(f"out_of_range:{c}")
            for c in CATEGORICAL_FEATURES:
                known = self.feature_ranges["categorical"].get(c)
                if known and str(row.get(c)) not in known:
                    f.append(f"unseen_category:{c}")
            flags.append(f)
        return flags


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v
