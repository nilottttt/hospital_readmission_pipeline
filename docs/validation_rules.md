# Validation rules and data-quality checks

The pipeline applies three kinds of checks. All results are written to
`audit.dq_check_results`; every rejected record is written to
`audit.rejected_records` **and** to `data/rejected/<run_id>_rejected.csv`.

## 1. Reject rules (row-level; a failing row is kept out of the clean layer)

| Rule | Stage | Check | Why |
|---|---|---|---|
| E01 | extract | The CSV line has the same number of fields as the header | Malformed/truncated lines cannot be parsed safely |
| V01 | validate | `encounter_id` is present and a whole number | Primary identifier of the encounter |
| V02 | validate | `encounter_id` is unique (first occurrence kept) | Removes duplicate encounters |
| V03 | validate | `patient_nbr` is present and a whole number | Needed for the patient surrogate key |
| V04 | validate | `gender` is `Male` or `Female` | `Unknown/Invalid` (3 rows in the real data) cannot be analysed |
| V05 | validate | `age` is one of the 10 brackets `[0-10)` … `[90-100)` | Age group is a key dimension |
| V06 | validate | `time_in_hospital` is a whole number between 1 and 14 | Dataset inclusion criterion (1–14 days) |
| V07 | validate | Count fields (`num_lab_procedures`, `num_procedures`, `num_medications`, `number_outpatient`, `number_emergency`, `number_inpatient`, `number_diagnoses`) are non-negative whole numbers | Needed for feature engineering |
| V08 | validate | `admission_type_id`, `discharge_disposition_id`, `admission_source_id` are whole numbers | Needed for the code lookups |
| V09 | validate | `readmitted` is `<30`, `>30` or `NO` | The readmission label |
| V10 | validate | Discharge disposition is **not** expired or hospice (IDs 11, 13, 14, 19, 20, 21) | A patient who died or went to hospice cannot be readmitted, so these encounters are invalid for readmission analysis |
| V11 | validate | Every medication field is `No`, `Steady`, `Up` or `Down` | Needed for the medication counts |

## 2. Warning checks (column-level; logged, rows are kept)

| Check | Column | Expected in the real data |
|---|---|---|
| race_missing | `race = '?'` | ≈ 2 % |
| weight_missing | `weight = '?'` | ≈ 97 % → column reduced to a `weight_recorded` flag |
| payer_code_missing | `payer_code = '?'` | ≈ 40 % → `Unknown` |
| medical_specialty_missing | `medical_specialty = '?'` | ≈ 49 % → `Unknown` |
| diag_1_missing | `diag_1 = '?'` | < 0.1 % |

## 3. Critical checks (pipeline stops if any fails)

| Layer | Check |
|---|---|
| staging | Reject rate ≤ 10 % (a higher rate means the source changed or is corrupt) |
| clean | `encounter_sk` unique and not null |
| clean | Clean row count = number of rows that passed validation |
| clean | No nulls in required columns |
| mart | `fact_admission` and `fact_readmission` row counts = clean row count |
| mart | `dim_patient` row count = number of distinct patients in clean |

## Special missing-value codes and how they are replaced

| Source value | Where | Replaced with |
|---|---|---|
| `?` | race, payer_code, medical_specialty | `Unknown` |
| `?` | weight | `weight_recorded = false` |
| `?` | diag_1, diag_2, diag_3 | NULL, category `Missing` |
| `None` | max_glu_serum, A1Cresult | `Not measured` (test not done – not a missing value) |
| IDs 5, 6, 8 | admission_type_id (Not Available / NULL / Not Mapped) | `Unknown` |
| IDs 18, 25, 26 | discharge_disposition_id (NULL / Not Mapped / Unknown/Invalid) | `Unknown` |
| IDs 9, 15, 17, 20, 21 | admission_source_id (Not Available / NULL / Not Mapped / Unknown/Invalid) | `Unknown` |
