# Model card – 30-day readmission risk (`readmission_risk_model`)

> **Academic decision-support prototype.** Trained on public 1999–2008 data. It is not
> validated for clinical use and must never replace clinical judgement.

## Intended use
* **Task:** estimate the probability that a diabetic inpatient encounter is followed by a readmission **within 30 days** (`readmitted = "<30"`).
* **Intended users:** students and instructors, to demonstrate an end-to-end MLOps workflow. A real deployment would, for example, help care coordinators prioritise discharge follow-up calls.
* **Out of scope:** individual clinical decisions, insurance or billing decisions, any use on non-diabetic or post-2008 populations without revalidation.

## Data
* **Source:** UCI Diabetes 130-US Hospitals (1999–2008), cleaned by the Part 1 pipeline (`mart.v_ml_features`).
* **Rows:** 99,340 encounters, after removing patients who died or went to hospice. 30-day readmission rate ≈ 11.4 %.
* **Split (chronological, by encounter order):** train 69,538 (oldest 70 %) · validation 14,901 · test 14,901 (newest 15 %). Positive rate: 11.6 % / 11.1 % / 10.8 %.
* **Features (27):**
  * *numeric:* age (bracket midpoint), length of stay, lab procedures, procedures, medications, outpatient/emergency/inpatient visits in the previous year, prior admissions, earlier encounters in the system, number of diagnoses, diagnosis count, diabetes drugs prescribed and changed, medication-changed flag, on-diabetes-medication flag.
  * *categorical:* gender, admission type, admission source group, discharge group, department, primary/secondary/additional diagnosis category (ICD-9 groups), HbA1c result, max glucose serum, insulin.
* **Deliberately excluded:** race (used only to check fairness), payer code (socio-economic proxy), weight (97 % missing), the rule-based risk score from Part 1 (it would duplicate the features).

## Training procedure
| Step | Detail |
|---|---|
| Preprocessing | `ColumnTransformer`: median imputation (+ standard scaling for logistic regression) and one-hot encoding (categories with < 30 rows grouped as "infrequent"). Saved inside the model, so serving applies exactly the same steps. |
| Candidates | Logistic Regression (L2, C = 0.5, balanced class weights); Random Forest (300 trees, depth 12, min leaf 20, balanced subsample); XGBoost (350 trees, depth 4, learning rate 0.05, `scale_pos_weight` = neg/pos) |
| Class imbalance | Class weighting / `scale_pos_weight` |
| Calibration | Isotonic regression fitted on the **validation** set (class weighting inflates the raw probabilities) |
| Threshold | The value that maximises F1 on calibrated validation probabilities |
| Selection | Highest **validation** ROC-AUC. The test set is used only for the final report. |
| Tracking | MLflow experiment `hospital_readmission_30d`: one parent run per training, one child run per model. Params, metrics, ROC/PR/calibration/confusion plots, feature importance, SHAP plots, subgroup table and the reference profile are logged. |
| Registry | The selected model is registered as `readmission_risk_model`. It gets alias `champion` only if it is at least as good as the current champion on the same test set; the old champion keeps alias `previous`. |

## Results (verification run, test set = newest 14,901 encounters)

| Model | Val ROC-AUC | Test ROC-AUC | PR-AUC | Recall | Precision | F1 | Brier | ECE raw → calibrated |
|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.642 | 0.650 | 0.188 | 0.509 | 0.171 | 0.256 | 0.0928 | 0.381 → 0.008 |
| **Random Forest (selected)** | **0.662** | **0.669** | **0.197** | 0.373 | **0.193** | 0.254 | **0.0922** | 0.359 → 0.008 |
| XGBoost | 0.657 | 0.659 | 0.191 | 0.355 | 0.187 | 0.245 | 0.0926 | 0.366 → 0.011 |

* Decision threshold 13.6 % (training base rate 11.6 %). About 1 in 5 flagged encounters is readmitted within 30 days, versus 1 in 9 overall.
* A ROC-AUC of about 0.65–0.67 matches published results on this dataset. Readmission depends heavily on factors that aren't recorded here (social support, adherence, post-discharge care).
* **Most influential features (SHAP):** prior inpatient admissions, discharge destination (home vs. transfer), prior admissions, earlier encounters, length of stay, number of diagnoses.

## Fairness (test set, recall at the chosen threshold)
| Group | Recall | ROC-AUC |
|---|---|---|
| Female / Male | 0.397 / 0.346 | 0.674 / 0.663 |
| Caucasian / African American | 0.386 / 0.361 | 0.670 / 0.665 |
| Hispanic / Asian / Other (n < 500 each) | 0.24 / 0.24 / 0.23 | 0.70 / 0.71 / 0.64 |

Recall is lower for the small race groups. Their samples are small (Asian n = 182), so the estimates are noisy, but the monitoring DAG tracks the gap and raises a *fairness review* alert when it exceeds 0.20.

## Limitations and risks
* The data is 15+ years old and US-only. Coding practices have changed since then: the monitoring DAG measures strong drift in `department` (missing specialty rose from 41 % to 67 %) and `number_diagnoses` (the maximum went from 9 to 16 after 2008) between the training and test periods.
* There are no real dates. `encounter_id` order is used as a time proxy.
* With moderate discrimination, many readmissions are missed (recall ≈ 0.37) and most flags are false positives. The model is suitable only as one input to prioritisation.
* SHAP explanations describe the model, not causes.

## Serving
FastAPI (`/predict`, `/predict/batch`) in a Docker container. Every response includes the probability, risk band, flag, top factors, input-quality flags and a disclaimer. Every request is logged for monitoring (`mlops.prediction_log`).
