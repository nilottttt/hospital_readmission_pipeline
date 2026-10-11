# Data dictionary

## Database layers (schemas in the `healthcare` database)

| Schema | Tables | Purpose |
|---|---|---|
| (files) | `data/raw/<run_id>/source.zip`, `diabetic_data.csv`, `IDs_mapping.csv` | Raw layer: untouched copy of each extraction (read-only) |
| `raw` | `lookup_ids` | `IDs_mapping.csv` loaded exactly as published |
| `staging` | `encounters` | Every source line as TEXT + `source_row_num`, `run_id`, `dq_passed` |
| `clean` | `encounters`, `lookup_codes` | Validated, typed, standardized, de-identified rows |
| `mart` | `fact_admission`, `fact_diagnosis`, `fact_readmission`, `dim_patient`, `dim_admission_type`, `dim_discharge_disposition`, `dim_admission_source`, `v_*` views | Star schema for analytics and the dashboard |
| `audit` | `pipeline_runs`, `ingestion_log`, `rejected_records`, `dq_check_results` | Run history, extraction metadata, error log, data-quality results |

## Source columns (`diabetic_data.csv` → `staging.encounters`)

| Source column | Staging column | Meaning |
|---|---|---|
| encounter_id | encounter_id | Unique ID of the hospital stay (replaced by `encounter_sk` in clean) |
| patient_nbr | patient_nbr | Patient ID (replaced by `patient_sk` in clean) |
| race | race | Caucasian, AfricanAmerican, Asian, Hispanic, Other, `?` |
| gender | gender | Male, Female, Unknown/Invalid |
| age | age | 10-year bracket, e.g. `[70-80)` |
| weight | weight | Weight bracket in pounds; ≈97 % `?` |
| admission_type_id | admission_type_id | Code → IDs_mapping (Emergency, Urgent, Elective …) |
| discharge_disposition_id | discharge_disposition_id | Code → IDs_mapping (Home, SNF, Expired …) |
| admission_source_id | admission_source_id | Code → IDs_mapping (Emergency Room, Referral …) |
| time_in_hospital | time_in_hospital | Days between admission and discharge (1–14) |
| payer_code | payer_code | Payer (MC = Medicare, etc.), `?` = unknown |
| medical_specialty | medical_specialty | Specialty of the admitting physician (≈70 values) |
| num_lab_procedures | num_lab_procedures | Lab tests during the stay |
| num_procedures | num_procedures | Non-lab procedures during the stay |
| num_medications | num_medications | Distinct generic medicines given |
| number_outpatient / number_emergency / number_inpatient | same | Visits of each type in the year **before** the encounter |
| diag_1, diag_2, diag_3 | same | Primary / secondary / additional ICD-9 diagnosis codes |
| number_diagnoses | number_diagnoses | Diagnoses entered in the system |
| max_glu_serum | max_glu_serum | Glucose serum test: `>200`, `>300`, `Norm`, `None` (not taken) |
| A1Cresult | a1c_result | HbA1c test: `>7`, `>8`, `Norm`, `None` (not taken) |
| metformin … metformin-pioglitazone (23 cols) | snake_case names | Diabetes drug: `No`, `Steady`, `Up` (dose raised), `Down` (dose lowered) |
| change | change | `Ch` if any diabetes medication changed |
| diabetesMed | diabetes_med | `Yes` if any diabetes medication prescribed |
| readmitted | readmitted | `<30` days, `>30` days, `NO` |

## `clean.encounters` (engineered / standardized columns)

| Column | Type | Derivation |
|---|---|---|
| encounter_sk | text | `enc_` + first 20 hex chars of SHA-256(salt · encounter_id) |
| patient_sk | text | `pat_` + first 20 hex chars of SHA-256(salt · patient_nbr) |
| encounter_seq | int | 1, 2, 3… order of the patient's encounters (by encounter_id) |
| race | text | `?` → Unknown, AfricanAmerican → African American |
| age_group / age_group_order / age_midpoint | text / int / int | `[70-80)` → `70-80` / 8 / 75 |
| weight_recorded | bool | weight was not `?` |
| admission_type, discharge_disposition, admission_source | text | Description from IDs_mapping; missing codes → Unknown |
| discharge_group | text | Home · Home with health service · Transferred to another facility · Left against medical advice · Other · Unknown |
| admission_source_group | text | Emergency Room · Referral · Transfer · Birth · Other · Unknown |
| medical_specialty | text | `InternalMedicine` → `Internal Medicine`, `?` → Unknown |
| department | text | Specialty grouped into ~16 departments (e.g. all `Surgery-*` → Surgery) |
| length_of_stay_days | int | = time_in_hospital |
| los_band | text | 1-2 · 3-4 · 5-7 · 8-14 days |
| prior_admissions | int | number_outpatient + number_emergency + number_inpatient |
| prior_encounters_in_data | int | encounter_seq − 1 |
| diag_N_category | text | ICD-9 group: Circulatory (390–459, 785), Respiratory (460–519, 786), Digestive (520–579, 787), Diabetes (250.xx), Injury (800–999), Musculoskeletal (710–739), Genitourinary (580–629, 788), Neoplasms (140–239), Other (incl. V/E codes), Missing |
| diagnosis_count | int | Non-missing diag_1..diag_3 (0–3) |
| max_glu_serum, a1c_result | text | `None` → Not measured |
| medication_count | int | Number of the 23 diabetes drugs not equal to `No` |
| medication_change_count | int | Number of drugs with `Up` or `Down` |
| med_changed, diabetes_med | bool | change = `Ch`, diabetesMed = `Yes` |
| readmitted_30d | bool | readmitted = `<30` (main target for Part 2) |
| readmitted_any | bool | readmitted in (`<30`, `>30`) |
| risk_score | int (0–8) | +2 if ≥2 prior inpatient visits (+1 if exactly 1), +1 if ≥1 emergency visit, +1 if LOS ≥ 7, +1 if ≥20 medications, +1 if ≥9 diagnoses, +1 if age ≥ 70, +1 if transferred to another facility |
| risk_level | text | Low (0–2), Medium (3–4), High (5+) — descriptive rule, not an ML model |

## Mart tables

| Table | Grain | Key | Main columns |
|---|---|---|---|
| mart.fact_admission | 1 row per encounter | encounter_sk (FK patient_sk, admission_type_id, discharge_disposition_id, admission_source_id) | demographics, department, LOS, utilisation, primary_diagnosis_category, medication counts, readmission flags, risk |
| mart.fact_diagnosis | 1 row per encounter × diagnosis position | (encounter_sk, diag_position) | icd9_code, diagnosis_category, readmitted_30d |
| mart.fact_readmission | 1 row per encounter | encounter_sk | readmission_class, readmission_label, readmitted_30d, readmitted_any, risk_score, risk_level |
| mart.dim_patient | 1 row per patient (patient summary) | patient_sk | gender, race, first/last age group, encounter_count, total/avg LOS, readmissions_30d, ever_readmitted_30d, max_risk_level |
| mart.dim_admission_type / dim_discharge_disposition / dim_admission_source | 1 row per code | code id | description, is_missing_code |

## Reporting views
`v_kpis`, `v_readmission_by_age`, `v_readmission_by_diagnosis`, `v_readmission_by_department`, `v_los_summary`, `v_admission_discharge`, `v_high_risk_cohort` — ready-made aggregates (useful for Tableau or for screenshots in the report).

## Audit tables

| Table | One row per | Columns |
|---|---|---|
| audit.pipeline_runs | pipeline run | run_id, triggered_by, status, started/finished_at, rows_extracted/staged/valid/rejected/clean, error_message |
| audit.ingestion_log | file per extraction attempt | extracted_at, source_name, source_url, file_name, file_path, size, sha256, status (downloaded / manual_fallback / failed), http_status, row_count, bad_line_count, message |
| audit.rejected_records | broken rule per record | run_id, stage, source_row_num, encounter_id, rule_id, rule_description, offending_value, record (JSON, patient_nbr removed) |
| audit.dq_check_results | check per run | layer, check_name, severity (reject / warning / critical), passed, failed_count, details |

---

# Part 2 additions

## New columns
| Table | Column | Meaning |
|---|---|---|
| clean.encounters, mart.fact_admission | encounter_order | Rank of the original `encounter_id` over all valid encounters (1 = oldest). Used as the time order for the chronological train/validation/test split. A rank reveals no original ID. |
| mart.fact_admission | diag_2_category, diag_3_category | ICD-9 group of the secondary / additional diagnosis |

## `mart.v_ml_features` (view – the only input of the training pipeline)
| Group | Columns |
|---|---|
| Keys / order | encounter_sk, patient_sk, encounter_order |
| Numeric features | age_midpoint, length_of_stay_days, num_lab_procedures, num_procedures, num_medications, number_outpatient, number_emergency, number_inpatient, prior_admissions, prior_encounters_in_data, number_diagnoses, diagnosis_count, medication_count, medication_change_count, med_changed (0/1), diabetes_med (0/1) |
| Categorical features | gender, admission_type, admission_source_group, discharge_group, department, primary_diagnosis_category, diag_2_category, diag_3_category, max_glu_serum, a1c_result, insulin |
| Subgroup columns (not model inputs) | race, age_group |
| Label | label_readmitted_30d (1 = readmitted within 30 days) |

## `mlops` schema
| Table | One row per | Main columns |
|---|---|---|
| mlops.training_runs | candidate model per training run | training_id / mlflow_run_id (MLflow), model_type, sizes, threshold, val/test ROC-AUC, PR-AUC, precision, recall, F1, Brier, ECE before/after calibration, selected, registered_version, promoted, decision |
| mlops.reference_profile | feature per model version | bin edges & proportions (numeric) or category shares (categorical) of the training data – the drift reference |
| mlops.prediction_log | API request (per encounter) | request_id, predicted_at, source (api / batch / simulation / dashboard), model_version, encounter_sk (if known), features (JSON), probability, risk_band, flagged, threshold, latency_ms, status (ok / error / rejected), error, dq_flags |
| mlops.monitoring_runs | monitoring run | window, n_predictions, n_labeled, error_rate, p50/p95 latency, dq_flag_rate, max_psi, drifted_features, live ROC-AUC vs reference, precision, recall, F1, Brier, ECE, max_subgroup_recall_gap, retrain_recommended, alerts, retrain_triggered |
| mlops.drift_metrics | feature per monitoring run | psi, status (stable < 0.10 ≤ warning < 0.25 ≤ drift) |
| mlops.subgroup_metrics | group value per monitoring run (`training:v<N>` = test-set baseline) | group_type (gender / race / age_group), n, positives, ROC-AUC, recall, precision, flag_rate, mean_predicted, observed_rate |

## API prediction output
| Field | Meaning |
|---|---|
| probability | Calibrated probability of readmission within 30 days |
| risk_band | High (≥ threshold) · Medium (≥ training base rate) · Low |
| flagged | probability ≥ threshold (threshold = max-F1 point on validation) |
| top_factors | Up to 5 features with the largest SHAP contribution (positive = raises risk) |
| dq_flags | `out_of_range:<feature>`, `unseen_category:<feature>`, `missing:<feature>` |
