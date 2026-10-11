"""Reference profiles and Population Stability Index (PSI) for drift detection.

PSI = sum( (cur% - ref%) * ln(cur% / ref%) ) over bins
  < 0.10  stable   |   0.10-0.25  moderate shift (warning)   |   > 0.25  significant drift
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ml import config
from ml.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

EPS = 1e-4


def build_profile(df: pd.DataFrame) -> dict:
    """Bin edges / category shares of every feature in the training data."""
    profile = {}
    for c in NUMERIC_FEATURES:
        x = pd.to_numeric(df[c], errors="coerce").dropna().to_numpy()
        edges = np.unique(np.quantile(x, np.linspace(0, 1, 11)))
        if len(edges) < 2:
            edges = np.array([x.min() - 0.5, x.max() + 0.5])
        inner = edges[1:-1].tolist()                 # open-ended outer bins
        props = _numeric_props(x, inner)
        profile[c] = {"kind": "numeric", "edges": inner, "proportions": props}
    for c in CATEGORICAL_FEATURES:
        shares = df[c].astype(str).value_counts(normalize=True)
        profile[c] = {"kind": "categorical", "categories": {k: float(v) for k, v in shares.items()}}
    return profile


def _numeric_props(x: np.ndarray, inner_edges: list) -> list:
    idx = np.searchsorted(np.asarray(inner_edges), x, side="right")
    counts = np.bincount(idx, minlength=len(inner_edges) + 1)
    return (counts / max(len(x), 1)).tolist()


def psi(ref: np.ndarray, cur: np.ndarray) -> float:
    ref = np.clip(np.asarray(ref, float), EPS, None)
    cur = np.clip(np.asarray(cur, float), EPS, None)
    ref, cur = ref / ref.sum(), cur / cur.sum()
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def feature_psi(profile: dict, current: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for feat, p in profile.items():
        if feat not in current or current[feat].dropna().empty:
            continue
        if p["kind"] == "numeric":
            x = pd.to_numeric(current[feat], errors="coerce").dropna().to_numpy()
            value = psi(p["proportions"], _numeric_props(x, p["edges"]))
        else:
            ref = p["categories"]
            cur = current[feat].astype(str).value_counts(normalize=True)
            cats = sorted(set(ref) | set(cur.index))
            value = psi([ref.get(k, 0.0) for k in cats], [cur.get(k, 0.0) for k in cats])
        status = "drift" if value >= config.PSI_ALERT else ("warning" if value >= config.PSI_WARNING else "stable")
        rows.append({"feature": feat, "kind": p["kind"], "psi": round(value, 5), "status": status})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
