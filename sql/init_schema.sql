-- =====================================================================
-- Hospital Readmission Pipeline - database layers
--   raw      : files in data/raw/ + raw.lookup_ids (mapping file as-is)
--   staging  : source rows loaded as TEXT, untouched
--   clean    : validated, standardized, de-identified rows
--   mart     : analytical tables and views used by the dashboard
--   audit    : pipeline runs, ingestion log, data-quality results, rejects
-- Safe to run many times (IF NOT EXISTS everywhere).
-- =====================================================================

CREATE SCHEMA IF NOT EXISTS raw;
CREATE SCHEMA IF NOT EXISTS staging;
CREATE SCHEMA IF NOT EXISTS clean;
CREATE SCHEMA IF NOT EXISTS mart;
CREATE SCHEMA IF NOT EXISTS audit;

-- ---------------------------------------------------------------------
-- AUDIT
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS audit.pipeline_runs (
    run_id          TEXT PRIMARY KEY,
    started_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at     TIMESTAMPTZ,
    status          TEXT NOT NULL DEFAULT 'running',   -- running / success / failed
    triggered_by    TEXT,                                -- airflow / cli
    rows_extracted  INTEGER,
    rows_staged     INTEGER,
    rows_valid      INTEGER,
    rows_rejected   INTEGER,
    rows_clean      INTEGER,
    error_message   TEXT
);

CREATE TABLE IF NOT EXISTS audit.ingestion_log (
    log_id          BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,
    extracted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    source_name     TEXT NOT NULL,
    source_url      TEXT,
    file_name       TEXT,
    file_path       TEXT,
    file_size_bytes BIGINT,
    sha256          TEXT,
    status          TEXT NOT NULL,     -- downloaded / manual_fallback / failed
    http_status     INTEGER,
    row_count       INTEGER,
    bad_line_count  INTEGER DEFAULT 0,
    message         TEXT
);

CREATE TABLE IF NOT EXISTS audit.rejected_records (
    reject_id       BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,
    stage           TEXT NOT NULL,     -- extract / validate
    source_row_num  INTEGER,
    encounter_id    TEXT,
    rule_id         TEXT NOT NULL,
    rule_description TEXT NOT NULL,
    offending_value TEXT,
    record          JSONB,
    rejected_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_rejected_run ON audit.rejected_records (run_id);

CREATE TABLE IF NOT EXISTS audit.dq_check_results (
    check_id        BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL,
    layer           TEXT NOT NULL,     -- staging / clean / mart
    check_name      TEXT NOT NULL,
    severity        TEXT NOT NULL,     -- critical / warning
    passed          BOOLEAN NOT NULL,
    failed_count    INTEGER,
    details         TEXT,
    checked_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ---------------------------------------------------------------------
-- RAW: lookup file loaded exactly as published
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS raw.lookup_ids (
    lookup_type     TEXT NOT NULL,     -- admission_type_id / discharge_disposition_id / admission_source_id
    code            INTEGER NOT NULL,
    description     TEXT,
    PRIMARY KEY (lookup_type, code)
);

-- ---------------------------------------------------------------------
-- STAGING: one row per line of diabetic_data.csv, every column as TEXT
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS staging.encounters (
    source_row_num   INTEGER PRIMARY KEY,   -- line number in the source file
    run_id           TEXT NOT NULL,
    loaded_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    dq_passed        BOOLEAN,               -- set by the validate step
    encounter_id TEXT, patient_nbr TEXT, race TEXT, gender TEXT, age TEXT, weight TEXT,
    admission_type_id TEXT, discharge_disposition_id TEXT, admission_source_id TEXT,
    time_in_hospital TEXT, payer_code TEXT, medical_specialty TEXT,
    num_lab_procedures TEXT, num_procedures TEXT, num_medications TEXT,
    number_outpatient TEXT, number_emergency TEXT, number_inpatient TEXT,
    diag_1 TEXT, diag_2 TEXT, diag_3 TEXT, number_diagnoses TEXT,
    max_glu_serum TEXT, a1c_result TEXT,
    metformin TEXT, repaglinide TEXT, nateglinide TEXT, chlorpropamide TEXT,
    glimepiride TEXT, acetohexamide TEXT, glipizide TEXT, glyburide TEXT,
    tolbutamide TEXT, pioglitazone TEXT, rosiglitazone TEXT, acarbose TEXT,
    miglitol TEXT, troglitazone TEXT, tolazamide TEXT, examide TEXT,
    citoglipton TEXT, insulin TEXT, glyburide_metformin TEXT,
    glipizide_metformin TEXT, glimepiride_pioglitazone TEXT,
    metformin_rosiglitazone TEXT, metformin_pioglitazone TEXT,
    change TEXT, diabetes_med TEXT, readmitted TEXT
);

-- ---------------------------------------------------------------------
-- CLEAN: validated, typed, standardized, de-identified
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS clean.lookup_codes (
    lookup_type     TEXT NOT NULL,
    code            INTEGER NOT NULL,
    description     TEXT NOT NULL,
    is_missing_code BOOLEAN NOT NULL,
    PRIMARY KEY (lookup_type, code)
);

CREATE TABLE IF NOT EXISTS clean.encounters (
    encounter_sk             TEXT PRIMARY KEY,   -- salted hash of encounter_id
    patient_sk               TEXT NOT NULL,      -- salted hash of patient_nbr
    encounter_seq            INTEGER NOT NULL,   -- 1 = patient's first encounter in the data
    run_id                   TEXT NOT NULL,
    race                     TEXT NOT NULL,
    gender                   TEXT NOT NULL,
    age_group                TEXT NOT NULL,      -- '0-10' ... '90-100'
    age_group_order          SMALLINT NOT NULL,
    age_midpoint             SMALLINT NOT NULL,
    weight_recorded          BOOLEAN NOT NULL,
    admission_type_id        INTEGER NOT NULL,
    admission_type           TEXT NOT NULL,
    discharge_disposition_id INTEGER NOT NULL,
    discharge_disposition    TEXT NOT NULL,
    discharge_group          TEXT NOT NULL,
    admission_source_id      INTEGER NOT NULL,
    admission_source         TEXT NOT NULL,
    admission_source_group   TEXT NOT NULL,
    payer_code               TEXT NOT NULL,
    medical_specialty        TEXT NOT NULL,
    department               TEXT NOT NULL,      -- top specialties, rest = 'Other'
    length_of_stay_days      SMALLINT NOT NULL,
    los_band                 TEXT NOT NULL,
    num_lab_procedures       SMALLINT NOT NULL,
    num_procedures           SMALLINT NOT NULL,
    num_medications          SMALLINT NOT NULL,
    number_outpatient        SMALLINT NOT NULL,
    number_emergency         SMALLINT NOT NULL,
    number_inpatient         SMALLINT NOT NULL,
    prior_admissions         SMALLINT NOT NULL,  -- outpatient + emergency + inpatient (previous year)
    prior_encounters_in_data SMALLINT NOT NULL,
    diag_1 TEXT, diag_2 TEXT, diag_3 TEXT,
    diag_1_category          TEXT NOT NULL,
    diag_2_category          TEXT NOT NULL,
    diag_3_category          TEXT NOT NULL,
    number_diagnoses         SMALLINT NOT NULL,
    diagnosis_count          SMALLINT NOT NULL,  -- non-missing diag_1..diag_3
    max_glu_serum            TEXT NOT NULL,
    a1c_result               TEXT NOT NULL,
    medication_count         SMALLINT NOT NULL,  -- diabetes drugs prescribed (not 'No')
    medication_change_count  SMALLINT NOT NULL,  -- dosage Up/Down
    insulin                  TEXT NOT NULL,
    med_changed              BOOLEAN NOT NULL,
    diabetes_med             BOOLEAN NOT NULL,
    readmitted               TEXT NOT NULL,      -- '<30' / '>30' / 'NO'
    readmitted_30d           BOOLEAN NOT NULL,
    readmitted_any           BOOLEAN NOT NULL,
    risk_score               SMALLINT NOT NULL,
    risk_level               TEXT NOT NULL,      -- Low / Medium / High
    metformin TEXT, repaglinide TEXT, nateglinide TEXT, chlorpropamide TEXT,
    glimepiride TEXT, acetohexamide TEXT, glipizide TEXT, glyburide TEXT,
    tolbutamide TEXT, pioglitazone TEXT, rosiglitazone TEXT, acarbose TEXT,
    miglitol TEXT, troglitazone TEXT, tolazamide TEXT, examide TEXT,
    citoglipton TEXT, glyburide_metformin TEXT,
    glipizide_metformin TEXT, glimepiride_pioglitazone TEXT,
    metformin_rosiglitazone TEXT, metformin_pioglitazone TEXT
);

-- ---------------------------------------------------------------------
-- Part 2 additions (safe on databases created by Part 1)
-- ---------------------------------------------------------------------
-- Global chronological order of encounters (rank of encounter_id), used
-- for the chronological train/validation/test split of the ML model.
ALTER TABLE clean.encounters ADD COLUMN IF NOT EXISTS encounter_order INTEGER;
