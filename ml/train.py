"""Training pipeline: compare Logistic Regression, Random Forest and XGBoost,
track everything in MLflow, calibrate, explain, register and (maybe) promote.

    python -m ml.train            (or the Airflow DAG readmission_model_training)

Steps
  1. Load mart.v_ml_features and split chronologically (70 / 15 / 15).
  2. Fit each candidate on TRAIN; score VALIDATION.
  3. Fit an isotonic calibrator on VALIDATION scores; choose the decision
     threshold that maximises F1 on calibrated VALIDATION probabilities.
  4. Pick the candidate with the best VALIDATION ROC-AUC.
  5. Evaluate every candidate once on the untouched TEST set (reported, not used to choose).
  6. Log params / metrics / plots / SHAP / model to MLflow; register the selected model.
  7. Promote to alias 'champion' only if it is at least as good as the current
     champion on the same test set (old champion keeps alias 'previous').
"""
from __future__ import annotations

import json
import logging
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from ml import config, evaluate
from ml.drift import build_profile
from ml.features import (
    FEATURES, LABEL, SUBGROUP_COLUMNS, build_preprocessor, chronological_split,
    feature_ranges, load_feature_table, original_feature_of,
)
from ml.model import ReadmissionRiskModel

log = logging.getLogger(__name__)
ML_DIR = Path(__file__).resolve().parent


# ----------------------------------------------------------------------------
# candidates
# ----------------------------------------------------------------------------
def make_candidates(pos_weight: float) -> dict:
    return {
        "logistic_regression": (
            Pipeline([("preprocess", build_preprocessor(scale_numeric=True)),
                      ("model", LogisticRegression(C=0.5, class_weight="balanced", max_iter=3000,
                                                   random_state=config.RANDOM_STATE))]),
            {"C": 0.5, "class_weight": "balanced", "penalty": "l2"},
        ),
        "random_forest": (
            Pipeline([("preprocess", build_preprocessor(scale_numeric=False)),
                      ("model", RandomForestClassifier(
                          n_estimators=300, max_depth=12, min_samples_leaf=20,
                          max_features="sqrt", class_weight="balanced_subsample",
                          n_jobs=2, random_state=config.RANDOM_STATE))]),
            {"n_estimators": 300, "max_depth": 12, "min_samples_leaf": 20,
             "max_features": "sqrt", "class_weight": "balanced_subsample"},
        ),
        "xgboost": (
            Pipeline([("preprocess", build_preprocessor(scale_numeric=False)),
                      ("model", XGBClassifier(
                          n_estimators=350, learning_rate=0.05, max_depth=4,
                          subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                          reg_lambda=1.0, scale_pos_weight=pos_weight,
                          tree_method="hist", eval_metric="logloss", n_jobs=2,
                          random_state=config.RANDOM_STATE))]),
            {"n_estimators": 350, "learning_rate": 0.05, "max_depth": 4, "subsample": 0.8,
             "colsample_bytree": 0.8, "min_child_weight": 5, "scale_pos_weight": round(pos_weight, 3)},
        ),
    }


def global_importance(model_type: str, pipeline: Pipeline) -> pd.Series:
    pre, clf = pipeline.named_steps["preprocess"], pipeline.named_steps["model"]
    names = pre.get_feature_names_out()
    if model_type == "logistic_regression":
        raw = np.abs(clf.coef_[0])
    else:
        raw = clf.feature_importances_
    s = pd.Series(raw, index=[original_feature_of(n) for n in names]).groupby(level=0).sum()
    return (s / s.sum()).sort_values(ascending=False)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def run(triggered_by: str = "cli") -> dict:
    import mlflow
    from mlflow import MlflowClient
    from mlflow.models import infer_signature

    from etl import db

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    client = MlflowClient()

    df = load_feature_table()
    if len(df) < config.MIN_TRAINING_ROWS:
        raise RuntimeError(f"Only {len(df)} rows in mart.v_ml_features - run the ETL DAG first")
    train, valid, test = chronological_split(df)
    X_tr, y_tr = train[FEATURES], train[LABEL].to_numpy()
    X_va, y_va = valid[FEATURES], valid[LABEL].to_numpy()
    X_te, y_te = test[FEATURES], test[LABEL].to_numpy()
    base_rate = float(y_tr.mean())
    pos_weight = (1 - base_rate) / base_rate
    ranges = feature_ranges(train)
    log.info("Split: train=%s valid=%s test=%s | positive rate train=%.3f valid=%.3f test=%.3f",
             len(train), len(valid), len(test), y_tr.mean(), y_va.mean(), y_te.mean())

    out_dir = Path(tempfile.mkdtemp(prefix="training_"))
    results, models, test_curves = {}, {}, {}

    with mlflow.start_run(run_name=f"training_{datetime.now(timezone.utc):%Y%m%dT%H%M%S}") as parent:
        mlflow.set_tags({"triggered_by": triggered_by, "split": "chronological 70/15/15",
                         "target": "readmitted < 30 days"})
        mlflow.log_params({"n_train": len(train), "n_valid": len(valid), "n_test": len(test),
                           "n_features": len(FEATURES), "train_positive_rate": round(base_rate, 4)})
        mlflow.log_dict({"features": FEATURES, "label": LABEL}, "feature_list.json")

        for name, (pipe, params) in make_candidates(pos_weight).items():
            with mlflow.start_run(run_name=name, nested=True) as child:
                t0 = time.time()
                pipe.fit(X_tr, y_tr)
                fit_seconds = time.time() - t0

                p_va_raw = pipe.predict_proba(X_va)[:, 1]
                calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_va_raw, y_va)
                p_va = calibrator.predict(p_va_raw)
                threshold = evaluate.best_f1_threshold(y_va, p_va)

                p_te_raw = pipe.predict_proba(X_te)[:, 1]
                p_te = calibrator.predict(p_te_raw)
                val_m = evaluate.classification_metrics(y_va, p_va, threshold)
                val_m["roc_auc"] = evaluate.classification_metrics(y_va, p_va_raw, 0.5)["roc_auc"]
                test_m = evaluate.classification_metrics(y_te, p_te, threshold)
                test_raw_m = evaluate.classification_metrics(y_te, p_te_raw, 0.5)

                mlflow.set_tag("model_type", name)
                mlflow.log_params({**params, "threshold_rule": "max F1 on calibrated validation",
                                   "calibration": "isotonic (validation set)"})
                mlflow.log_metrics({"fit_seconds": fit_seconds, "threshold": threshold,
                                    **{f"val_{k}": v for k, v in val_m.items()},
                                    **{f"test_{k}": v for k, v in test_m.items()},
                                    "test_ece_uncalibrated": test_raw_m["ece"],
                                    "test_brier_uncalibrated": test_raw_m["brier"]})
                imp = global_importance(name, pipe)
                imp_path = evaluate.plot_importance(imp, f"Feature importance - {name}",
                                                    out_dir / f"importance_{name}.png")
                mlflow.log_artifact(str(imp_path), "explainability")
                mlflow.log_dict(imp.round(5).to_dict(), "explainability/feature_importance.json")
                cal_path = evaluate.plot_calibration(
                    y_te, {"uncalibrated": p_te_raw, "isotonic calibrated": p_te}, out_dir,
                    name=f"calibration_{name}.png")
                mlflow.log_artifact(str(cal_path), "calibration")

                pre = pipe.named_steps["preprocess"]
                means = pre.transform(X_tr).mean(axis=0) if name == "logistic_regression" else None
                models[name] = ReadmissionRiskModel(pipe, calibrator, threshold, base_rate, name,
                                                    ranges, transformed_means=means)
                results[name] = {"run_id": child.info.run_id, "val": val_m, "test": test_m,
                                 "test_raw": test_raw_m, "threshold": threshold, "p_te": p_te}
                test_curves[name] = (y_te, p_te_raw)
                log.info("%-20s val AUC %.4f | test AUC %.4f  recall %.3f  precision %.3f  ECE %.4f",
                         name, val_m["roc_auc"], test_m["roc_auc"], test_m["recall"],
                         test_m["precision"], test_m["ece"])

        # ---------------- selection (validation only) ----------------
        best = max(results, key=lambda k: results[k]["val"]["roc_auc"])
        sel, model = results[best], models[best]
        log.info("Selected model: %s", best)

        comparison = pd.DataFrame({k: {"val_roc_auc": v["val"]["roc_auc"], **{f"test_{m}": v["test"][m] for m in
                                       ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier", "ece"]}}
                                   for k, v in results.items()}).T.round(4)
        comparison.to_csv(out_dir / "model_comparison.csv")
        for p in evaluate.plot_roc_pr(test_curves, out_dir):
            mlflow.log_artifact(str(p), "comparison")
        mlflow.log_artifact(str(out_dir / "model_comparison.csv"), "comparison")
        mlflow.set_tag("selected_model", best)
        mlflow.log_metrics({f"selected_test_{k}": sel["test"][k] for k in
                            ["roc_auc", "pr_auc", "precision", "recall", "f1", "brier", "ece"]})

        # ---------------- subgroup performance on test ----------------
        te = test.assign(prob=sel["p_te"])
        sub = evaluate.subgroup_metrics(te, LABEL, "prob", sel["threshold"], SUBGROUP_COLUMNS,
                                       min_n=config.SUBGROUP_MIN_N)

        # ---------------- log + register the selected model ----------------
        with mlflow.start_run(run_id=sel["run_id"], nested=True):
            cm_path = evaluate.plot_confusion(sel["test"], out_dir)
            mlflow.log_artifact(str(cm_path), "evaluation")
            sub.to_csv(out_dir / "subgroup_metrics_test.csv", index=False)
            mlflow.log_artifact(str(out_dir / "subgroup_metrics_test.csv"), "fairness")
            shap_paths = shap_summary(model, X_te, out_dir)
            for p in shap_paths:
                mlflow.log_artifact(str(p), "explainability")
            profile = build_profile(train)
            mlflow.log_dict(profile, "monitoring/reference_profile.json")

            sample = X_te.head(5)
            signature = infer_signature(sample, model.predict(None, sample))
            info = mlflow.pyfunc.log_model(
                artifact_path="model", python_model=model, code_paths=[str(ML_DIR)],
                signature=signature, input_example=sample,
                registered_model_name=config.MODEL_NAME,
                pip_requirements=_pip_requirements(),
            )
        version = str(getattr(info, "registered_model_version", None) or
                      client.search_model_versions(f"run_id='{sel['run_id']}'")[0].version)
        client.update_model_version(
            config.MODEL_NAME, version,
            description=(f"{best} | test ROC-AUC {sel['test']['roc_auc']:.3f}, recall "
                         f"{sel['test']['recall']:.3f}, precision {sel['test']['precision']:.3f}, "
                         f"ECE {sel['test']['ece']:.3f}. Academic decision-support prototype."))
        for k, v in {"model_type": best, "threshold": f"{sel['threshold']:.4f}",
                     "base_rate": f"{base_rate:.4f}", "training_id": parent.info.run_id,
                     "test_roc_auc": f"{sel['test']['roc_auc']:.4f}"}.items():
            client.set_model_version_tag(config.MODEL_NAME, version, k, v)

        # ---------------- promotion decision ----------------
        promoted, decision = decide_promotion(client, model, X_te, y_te, sel["test"]["roc_auc"])
        if promoted:
            try:
                old = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
                client.set_registered_model_alias(config.MODEL_NAME, config.PREVIOUS_ALIAS, old.version)
            except Exception:
                pass
            client.set_registered_model_alias(config.MODEL_NAME, config.CHAMPION_ALIAS, version)
        client.set_model_version_tag(config.MODEL_NAME, version, "promotion_decision", decision)
        mlflow.set_tags({"registered_version": version, "promoted": str(promoted),
                         "promotion_decision": decision})
        log.info("Registered %s v%s - %s", config.MODEL_NAME, version, decision)

    # ---------------- persist to PostgreSQL for the dashboard / monitoring ----------------
    db.run_sql_file(Path(__file__).resolve().parent.parent / "sql" / "init_mlops.sql")
    rows = []
    for name, r in results.items():
        rows.append((parent.info.run_id, r["run_id"], name, len(train), len(valid), len(test), base_rate,
                     r["threshold"], r["val"]["roc_auc"], r["val"]["pr_auc"], r["test"]["roc_auc"],
                     r["test"]["pr_auc"], r["test"]["precision"], r["test"]["recall"], r["test"]["f1"],
                     r["test"]["brier"], r["test"]["ece"], r["test_raw"]["ece"], name == best,
                     version if name == best else None, promoted and name == best,
                     decision if name == best else "not selected (lower validation ROC-AUC)"))
    db.insert_rows("mlops.training_runs",
                   ["training_id", "mlflow_run_id", "model_type", "n_train", "n_valid", "n_test",
                    "train_positive_rate", "threshold", "val_roc_auc", "val_pr_auc", "test_roc_auc",
                    "test_pr_auc", "test_precision", "test_recall", "test_f1", "test_brier", "test_ece",
                    "test_ece_uncalibrated", "selected", "registered_version", "promoted", "decision"],
                   rows)
    db.execute("DELETE FROM mlops.reference_profile WHERE model_version=%s", (version,))
    db.insert_rows("mlops.reference_profile", ["model_version", "feature", "kind", "profile"],
                   [(version, f, p["kind"], json.dumps(p)) for f, p in profile.items()])
    sub_id = f"training:v{version}"
    db.execute("DELETE FROM mlops.subgroup_metrics WHERE monitoring_id=%s", (sub_id,))
    db.insert_rows("mlops.subgroup_metrics",
                   ["monitoring_id", "group_type", "group_value", "n", "positives", "roc_auc", "recall",
                    "precision", "flag_rate", "mean_predicted", "observed_rate"],
                   [(sub_id, *row) for row in to_db_rows(
                       sub[["group_type", "group_value", "n", "positives", "roc_auc", "recall",
                            "precision", "flag_rate", "mean_predicted", "observed_rate"]])])
    return {"training_id": parent.info.run_id, "selected": best, "version": version,
            "promoted": promoted, "decision": decision,
            "comparison": comparison.to_dict(orient="index")}


def decide_promotion(client, model, X_te, y_te, new_auc: float) -> tuple[bool, str]:
    import mlflow
    from sklearn.metrics import roc_auc_score
    try:
        champ = client.get_model_version_by_alias(config.MODEL_NAME, config.CHAMPION_ALIAS)
    except Exception:
        return True, "promoted: no existing champion"
    try:
        loaded = mlflow.pyfunc.load_model(f"models:/{config.MODEL_NAME}@{config.CHAMPION_ALIAS}")
        champ_auc = roc_auc_score(y_te, loaded.unwrap_python_model().predict_proba(X_te))
    except Exception as exc:
        return True, f"promoted: champion v{champ.version} could not be evaluated ({exc.__class__.__name__})"
    if new_auc >= champ_auc - config.PROMOTION_TOLERANCE:
        return True, f"promoted: test AUC {new_auc:.4f} vs champion v{champ.version} {champ_auc:.4f}"
    return False, f"kept champion v{champ.version}: test AUC {new_auc:.4f} < {champ_auc:.4f}"


def shap_summary(model: ReadmissionRiskModel, X: pd.DataFrame, out_dir: Path) -> list[Path]:
    """Global SHAP explanation of the selected model on a test sample."""
    import matplotlib.pyplot as plt
    try:
        import shap
    except ImportError:
        return []
    n = 500 if model.model_type == "random_forest" else 2000   # RF SHAP is slow
    sample = X.sample(min(n, len(X)), random_state=config.RANDOM_STATE)
    pre = model.pipeline.named_steps["preprocess"]
    clf = model.pipeline.named_steps["model"]
    Xt = pd.DataFrame(pre.transform(sample), columns=pre.get_feature_names_out())
    try:
        if model.model_type == "logistic_regression":
            explainer = shap.LinearExplainer(clf, Xt)
            sv = explainer(Xt)
        else:
            explainer = shap.TreeExplainer(clf)
            sv = explainer(Xt)
            if sv.values.ndim == 3:            # random forest: (n, features, classes)
                sv = sv[..., 1]
    except Exception as exc:  # explainability must never break training
        log.warning("SHAP failed: %s", exc)
        return []
    paths = []
    plt.figure()
    shap.plots.beeswarm(sv, max_display=15, show=False)
    plt.title("SHAP values (test sample)")
    plt.tight_layout(); plt.savefig(out_dir / "shap_beeswarm.png", dpi=130); plt.close("all")
    paths.append(out_dir / "shap_beeswarm.png")
    grouped = (pd.DataFrame(np.abs(sv.values), columns=Xt.columns)
               .T.groupby(lambda n: original_feature_of(n)).sum().T.mean())
    paths.append(evaluate.plot_importance(grouped / grouped.sum(), "Mean |SHAP| by feature",
                                          out_dir / "shap_importance.png"))
    return paths


def to_db_rows(df: pd.DataFrame) -> list[tuple]:
    """DataFrame -> list of tuples with plain Python types (NaN -> None) for psycopg2."""
    clean = df.astype(object).where(pd.notna(df), None)
    return [tuple(v.item() if hasattr(v, "item") else v for v in row)
            for row in clean.itertuples(index=False)]


def _pip_requirements() -> list[str]:
    import sklearn
    import xgboost
    return [f"scikit-learn=={sklearn.__version__}", f"xgboost-cpu=={xgboost.__version__}",
            f"pandas=={pd.__version__}", f"numpy=={np.__version__}", "mlflow-skinny"]


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    print(json.dumps(run(), indent=2, default=str))
