-- =====================================================================
-- MART LAYER - rebuilt from clean.encounters on every run
-- Star schema:
--   fact_admission  (1 row per encounter)  --> dim_patient, dim_admission_type,
--                                              dim_discharge_disposition, dim_admission_source
--   fact_diagnosis  (1 row per encounter x diagnosis position)
--   fact_readmission(1 row per encounter, outcome-focused)
--   dim_patient     = patient summary (1 row per patient)
-- plus reporting views (v_*) used by the dashboard.
-- =====================================================================

DROP TABLE IF EXISTS mart.fact_diagnosis, mart.fact_readmission, mart.fact_admission,
    mart.dim_patient, mart.dim_admission_type, mart.dim_discharge_disposition,
    mart.dim_admission_source CASCADE;

-- ---------------- dimensions ----------------------------------------
CREATE TABLE mart.dim_admission_type AS
SELECT code AS admission_type_id, description AS admission_type, is_missing_code
FROM clean.lookup_codes WHERE lookup_type = 'admission_type_id';
ALTER TABLE mart.dim_admission_type ADD PRIMARY KEY (admission_type_id);

CREATE TABLE mart.dim_discharge_disposition AS
SELECT code AS discharge_disposition_id, description AS discharge_disposition, is_missing_code
FROM clean.lookup_codes WHERE lookup_type = 'discharge_disposition_id';
ALTER TABLE mart.dim_discharge_disposition ADD PRIMARY KEY (discharge_disposition_id);

CREATE TABLE mart.dim_admission_source AS
SELECT code AS admission_source_id, description AS admission_source, is_missing_code
FROM clean.lookup_codes WHERE lookup_type = 'admission_source_id';
ALTER TABLE mart.dim_admission_source ADD PRIMARY KEY (admission_source_id);

-- Patient summary (patient-level dimension)
CREATE TABLE mart.dim_patient AS
SELECT
    patient_sk,
    MIN(gender)                                   AS gender,
    MIN(race)                                     AS race,
    (ARRAY_AGG(age_group ORDER BY encounter_seq))[1]      AS first_age_group,
    (ARRAY_AGG(age_group ORDER BY encounter_seq DESC))[1] AS last_age_group,
    COUNT(*)                                      AS encounter_count,
    SUM(length_of_stay_days)                      AS total_los_days,
    ROUND(AVG(length_of_stay_days), 2)            AS avg_los_days,
    ROUND(AVG(num_medications), 2)                AS avg_num_medications,
    MAX(number_diagnoses)                         AS max_number_diagnoses,
    SUM(readmitted_30d::int)                      AS readmissions_30d,
    SUM(readmitted_any::int)                      AS readmissions_any,
    BOOL_OR(readmitted_30d)                       AS ever_readmitted_30d,
    MAX(risk_score)                               AS max_risk_score,
    CASE WHEN MAX(risk_score) >= 5 THEN 'High'
         WHEN MAX(risk_score) >= 3 THEN 'Medium' ELSE 'Low' END AS max_risk_level
FROM clean.encounters
GROUP BY patient_sk;
ALTER TABLE mart.dim_patient ADD PRIMARY KEY (patient_sk);

-- ---------------- facts ---------------------------------------------
-- Admission fact
CREATE TABLE mart.fact_admission AS
SELECT
    encounter_sk, patient_sk, encounter_seq,
    admission_type_id, discharge_disposition_id, admission_source_id,
    age_group, age_group_order, age_midpoint, gender, race, payer_code,
    medical_specialty, department,
    admission_type, admission_source_group, discharge_group,
    length_of_stay_days, los_band,
    num_lab_procedures, num_procedures, num_medications,
    number_outpatient, number_emergency, number_inpatient,
    prior_admissions, prior_encounters_in_data,
    diag_1_category AS primary_diagnosis_category,
    number_diagnoses, diagnosis_count,
    max_glu_serum, a1c_result,
    medication_count, medication_change_count, insulin, med_changed, diabetes_med,
    readmitted, readmitted_30d, readmitted_any,
    risk_score, risk_level
FROM clean.encounters;
ALTER TABLE mart.fact_admission ADD PRIMARY KEY (encounter_sk);
ALTER TABLE mart.fact_admission
    ADD FOREIGN KEY (patient_sk) REFERENCES mart.dim_patient (patient_sk),
    ADD FOREIGN KEY (admission_type_id) REFERENCES mart.dim_admission_type (admission_type_id),
    ADD FOREIGN KEY (discharge_disposition_id) REFERENCES mart.dim_discharge_disposition (discharge_disposition_id),
    ADD FOREIGN KEY (admission_source_id) REFERENCES mart.dim_admission_source (admission_source_id);
CREATE INDEX ix_fact_adm_patient ON mart.fact_admission (patient_sk);

-- Diagnosis fact (long format: up to 3 rows per encounter)
CREATE TABLE mart.fact_diagnosis AS
SELECT encounter_sk, patient_sk, 1 AS diag_position, diag_1 AS icd9_code, diag_1_category AS diagnosis_category, readmitted_30d
FROM clean.encounters WHERE diag_1 IS NOT NULL
UNION ALL
SELECT encounter_sk, patient_sk, 2, diag_2, diag_2_category, readmitted_30d
FROM clean.encounters WHERE diag_2 IS NOT NULL
UNION ALL
SELECT encounter_sk, patient_sk, 3, diag_3, diag_3_category, readmitted_30d
FROM clean.encounters WHERE diag_3 IS NOT NULL;
ALTER TABLE mart.fact_diagnosis ADD PRIMARY KEY (encounter_sk, diag_position);
ALTER TABLE mart.fact_diagnosis
    ADD FOREIGN KEY (encounter_sk) REFERENCES mart.fact_admission (encounter_sk);

-- Readmission fact (outcome table - the Part 2 model target lives here)
CREATE TABLE mart.fact_readmission AS
SELECT
    encounter_sk, patient_sk, encounter_seq,
    readmitted            AS readmission_class,
    readmitted_30d,
    readmitted_any,
    CASE readmitted WHEN '<30' THEN 'Within 30 days'
                    WHEN '>30' THEN 'After 30 days'
                    ELSE 'Not readmitted' END AS readmission_label,
    risk_score, risk_level
FROM clean.encounters;
ALTER TABLE mart.fact_readmission ADD PRIMARY KEY (encounter_sk);
ALTER TABLE mart.fact_readmission
    ADD FOREIGN KEY (encounter_sk) REFERENCES mart.fact_admission (encounter_sk);

-- ---------------- reporting views -----------------------------------
CREATE OR REPLACE VIEW mart.v_kpis AS
SELECT COUNT(*)                                  AS encounters,
       COUNT(DISTINCT patient_sk)                AS patients,
       ROUND(100.0 * AVG(readmitted_30d::int), 2) AS readmit_30d_pct,
       ROUND(100.0 * AVG(readmitted_any::int), 2) AS readmit_any_pct,
       ROUND(AVG(length_of_stay_days), 2)        AS avg_los_days,
       ROUND(100.0 * AVG((risk_level = 'High')::int), 2) AS high_risk_pct
FROM mart.fact_admission;

CREATE OR REPLACE VIEW mart.v_readmission_by_age AS
SELECT age_group, age_group_order, COUNT(*) AS encounters,
       ROUND(100.0 * AVG(readmitted_30d::int), 2) AS readmit_30d_pct,
       ROUND(100.0 * AVG(readmitted_any::int), 2) AS readmit_any_pct
FROM mart.fact_admission GROUP BY age_group, age_group_order ORDER BY age_group_order;

CREATE OR REPLACE VIEW mart.v_readmission_by_diagnosis AS
SELECT primary_diagnosis_category AS diagnosis_category, COUNT(*) AS encounters,
       ROUND(100.0 * AVG(readmitted_30d::int), 2) AS readmit_30d_pct
FROM mart.fact_admission GROUP BY 1 ORDER BY encounters DESC;

CREATE OR REPLACE VIEW mart.v_readmission_by_department AS
SELECT department, COUNT(*) AS encounters,
       ROUND(100.0 * AVG(readmitted_30d::int), 2) AS readmit_30d_pct,
       ROUND(AVG(length_of_stay_days), 2) AS avg_los_days
FROM mart.fact_admission GROUP BY 1 ORDER BY encounters DESC;

CREATE OR REPLACE VIEW mart.v_los_summary AS
SELECT los_band, length_of_stay_days, COUNT(*) AS encounters,
       ROUND(100.0 * AVG(readmitted_30d::int), 2) AS readmit_30d_pct
FROM mart.fact_admission GROUP BY 1, 2 ORDER BY 2;

CREATE OR REPLACE VIEW mart.v_admission_discharge AS
SELECT admission_type, admission_source_group, discharge_group, COUNT(*) AS encounters,
       ROUND(100.0 * AVG(readmitted_30d::int), 2) AS readmit_30d_pct
FROM mart.fact_admission GROUP BY 1, 2, 3;

CREATE OR REPLACE VIEW mart.v_high_risk_cohort AS
SELECT f.encounter_sk, f.patient_sk, f.age_group, f.gender, f.department,
       f.primary_diagnosis_category, f.length_of_stay_days, f.number_inpatient,
       f.number_emergency, f.num_medications, f.number_diagnoses, f.discharge_group,
       f.risk_score, f.readmitted
FROM mart.fact_admission f
WHERE f.risk_level = 'High';
