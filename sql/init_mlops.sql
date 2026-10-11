-- =====================================================================
-- MLOPS schema (Part 2): training history, served predictions, monitoring
-- Safe to run many times.
-- =====================================================================
CREATE SCHEMA IF NOT EXISTS mlops;

-- One row per candidate model per training run (also tracked in MLflow)
CREATE TABLE IF NOT EXISTS mlops.training_runs (
    id                 BIGSERIAL PRIMARY KEY,
    training_id        TEXT NOT NULL,          -- MLflow parent run id
    mlflow_run_id      TEXT NOT NULL,          -- MLflow child run id of this candidate
    model_type         TEXT NOT NULL,          -- logistic_regression / random_forest / xgboost
    trained_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    n_train INTEGER, n_valid INTEGER, n_test INTEGER,
    train_positive_rate DOUBLE PRECISION,
    threshold          DOUBLE PRECISION,
    val_roc_auc DOUBLE PRECISION, val_pr_auc DOUBLE PRECISION,
    test_roc_auc DOUBLE PRECISION, test_pr_auc DOUBLE PRECISION,
    test_precision DOUBLE PRECISION, test_recall DOUBLE PRECISION, test_f1 DOUBLE PRECISION,
    test_brier DOUBLE PRECISION, test_ece DOUBLE PRECISION, test_ece_uncalibrated DOUBLE PRECISION,
    selected           BOOLEAN NOT NULL DEFAULT FALSE,
    registered_version TEXT,
    promoted           BOOLEAN NOT NULL DEFAULT FALSE,
    decision           TEXT
);

-- Distribution of every feature in the training data of a model version
-- (the reference for drift detection)
CREATE TABLE IF NOT EXISTS mlops.reference_profile (
    model_version TEXT NOT NULL,
    feature       TEXT NOT NULL,
    kind          TEXT NOT NULL,      -- numeric / categorical
    profile       JSONB NOT NULL,     -- {"edges": [...], "proportions": [...]} or {"categories": {...}}
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (model_version, feature)
);

-- Every prediction served by the API (inputs, output, latency, status)
CREATE TABLE IF NOT EXISTS mlops.prediction_log (
    id            BIGSERIAL PRIMARY KEY,
    request_id    TEXT NOT NULL,
    predicted_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    source        TEXT NOT NULL DEFAULT 'api',   -- api / batch / simulation / dashboard
    model_name    TEXT,
    model_version TEXT,
    encounter_sk  TEXT,                           -- set when scoring a known encounter
    features      JSONB,
    probability   DOUBLE PRECISION,
    risk_band     TEXT,
    flagged       BOOLEAN,
    threshold     DOUBLE PRECISION,
    latency_ms    DOUBLE PRECISION,
    status        TEXT NOT NULL,                  -- ok / error / rejected
    error         TEXT,
    dq_flags      TEXT[]
);
CREATE INDEX IF NOT EXISTS ix_predlog_time ON mlops.prediction_log (predicted_at);
CREATE INDEX IF NOT EXISTS ix_predlog_enc ON mlops.prediction_log (encounter_sk);

-- One row per monitoring run
CREATE TABLE IF NOT EXISTS mlops.monitoring_runs (
    monitoring_id        TEXT PRIMARY KEY,
    run_at               TIMESTAMPTZ NOT NULL DEFAULT now(),
    model_version        TEXT,
    window_start         TIMESTAMPTZ,
    window_end           TIMESTAMPTZ,
    n_predictions        INTEGER,
    n_labeled            INTEGER,
    error_rate           DOUBLE PRECISION,
    p50_latency_ms       DOUBLE PRECISION,
    p95_latency_ms       DOUBLE PRECISION,
    dq_flag_rate         DOUBLE PRECISION,
    max_psi              DOUBLE PRECISION,
    drifted_features     TEXT[],
    roc_auc              DOUBLE PRECISION,
    reference_roc_auc    DOUBLE PRECISION,
    precision            DOUBLE PRECISION,
    recall               DOUBLE PRECISION,
    f1                   DOUBLE PRECISION,
    brier                DOUBLE PRECISION,
    ece                  DOUBLE PRECISION,
    max_subgroup_recall_gap DOUBLE PRECISION,
    retrain_recommended  BOOLEAN NOT NULL DEFAULT FALSE,
    alerts               TEXT[],
    retrain_triggered    BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE IF NOT EXISTS mlops.drift_metrics (
    monitoring_id TEXT NOT NULL,
    feature       TEXT NOT NULL,
    kind          TEXT NOT NULL,
    psi           DOUBLE PRECISION,
    status        TEXT NOT NULL,       -- stable / warning / drift
    PRIMARY KEY (monitoring_id, feature)
);

CREATE TABLE IF NOT EXISTS mlops.subgroup_metrics (
    monitoring_id TEXT NOT NULL,       -- 'training:<version>' for the test-set baseline
    group_type    TEXT NOT NULL,
    group_value   TEXT NOT NULL,
    n             INTEGER,
    positives     INTEGER,
    roc_auc       DOUBLE PRECISION,
    recall        DOUBLE PRECISION,
    precision     DOUBLE PRECISION,
    flag_rate     DOUBLE PRECISION,
    mean_predicted DOUBLE PRECISION,
    observed_rate DOUBLE PRECISION,
    PRIMARY KEY (monitoring_id, group_type, group_value)
);
