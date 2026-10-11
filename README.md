# Hospital Readmission Analytics and Prediction

A privacy-conscious healthcare data pipeline (Part 1) extended into an end-to-end
MLOps solution (Part 2), built on the public, de-identified
**UCI Diabetes 130-US Hospitals (1999–2008)** dataset.

* **Part 1 – Data pipeline:** **Apache Airflow** runs the Python/Pandas ETL. The data moves through
  **raw → staging → clean → mart** layers in **PostgreSQL**, and a **Streamlit** dashboard shows
  admission and readmission patterns.
* **Part 2 – MLOps:** predicts **30-day readmission**. It compares Logistic Regression, Random Forest and
  XGBoost, tracks experiments in **MLflow**, registers and versions the model, serves it with
  **FastAPI** in **Docker**, explains predictions with **SHAP**, and monitors drift, calibration,
  subgroup performance, latency and failures, with automatic retraining.

Everything runs in **Docker**. Part 2 is described in [section 11](#11-part-2--mlops-pipeline).

![Architecture](docs/architecture_diagram.png)

---

## 1. Prerequisites

| Tool | Why |
|---|---|
| Docker Desktop (WSL 2 engine on Windows), with at least **6 GB** of memory under Settings → Resources (Part 2 adds model training) | Runs PostgreSQL, Airflow, MLflow, the API and Streamlit |
| Git | Version control |
| Python 3.11+ (optional) | Only needed to run the unit tests or the pipeline outside Docker |

## 2. Setup (one time)

```powershell
cd C:\Users\<you>\Hospital_Readmission_Pipeline
copy .env.example .env
```
Open `.env` and change the passwords, `AIRFLOW_SECRET_KEY` and `HASH_SALT`.
Choose `HASH_SALT` once and don't change it later, because the surrogate keys are derived from it.

```powershell
docker compose up -d --build
```
The first build downloads the images and takes 5–10 minutes. Then check the status:
```powershell
docker compose ps
```
`airflow-init` should show **Exited (0)**. The other services should show **running / healthy**.

| URL | What | Login |
|---|---|---|
| http://localhost:8080 | Airflow UI | `AIRFLOW_ADMIN_USER` / `AIRFLOW_ADMIN_PASSWORD` from `.env` |
| http://localhost:8501 | Streamlit dashboard | – |
| localhost:5432 | PostgreSQL (DBeaver / pgAdmin) | `POSTGRES_USER` / `POSTGRES_PASSWORD`, database `healthcare` |
| http://localhost:5000 | MLflow (experiments + model registry) – Part 2 | – |
| http://localhost:8000/docs | Prediction API (Swagger UI) – Part 2 | – |

## 3. Run the pipeline

1. Open the Airflow UI and find **`hospital_readmission_etl`**.
2. Switch the DAG **on** (toggle on the left). New DAGs start paused.
3. Click **▶ Trigger DAG**.
4. Open the run and watch the **Graph** view. All 8 tasks should turn green in about 1–2 minutes.
5. Open the dashboard at http://localhost:8501 and refresh it.

After the first run, the DAG also runs automatically every week (`@weekly`).

### If the download is blocked
The `extract` task tries the UCI URL 3 times. If your network blocks it:

1. Download the zip from https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008.
2. Put the zip in `data/manual/`.
3. Trigger the DAG again. The run is recorded as `manual_fallback` in `audit.ingestion_log`.

## 4. What the pipeline does

| Task | What happens |
|---|---|
| `init_database` | Creates the `raw`, `staging`, `clean`, `mart` and `audit` schemas and tables if they don't exist |
| `start_run` | Records the run in `audit.pipeline_runs` |
| `extract` | Downloads the zip, keeps an untouched read-only copy in `data/raw/<run_id>/`, checks every CSV line and logs date, source, status, checksum and row count to `audit.ingestion_log` |
| `stage` | Loads the CSV as-is (all TEXT) into `staging.encounters` |
| `validate` | Applies 11 reject rules (duplicates, invalid values, expired/hospice patients …). Failures go to `audit.rejected_records` and `data/rejected/` |
| `transform` | Replaces missing-value codes, standardizes codes/specialties/ICD-9 groups, creates hashed surrogate keys and engineers length-of-stay, prior-admission, diagnosis-count, medication-count and risk features |
| `build_mart` | Rebuilds the star schema: `fact_admission`, `fact_diagnosis`, `fact_readmission`, `dim_patient` (patient summary), code dimensions, reporting views, and reconciles row counts |
| `finish_run` | Marks the run successful |

Details are in [docs/architecture.md](docs/architecture.md), [docs/validation_rules.md](docs/validation_rules.md) and [docs/data_dictionary.md](docs/data_dictionary.md).

## 5. Dashboard views

Filters: age group, gender, admission type and department. The KPI row shows encounters, patients, 30-day readmission rate, average length of stay and high-risk share.

1. **Readmission by age group:** 30-day rate and the full outcome mix.
2. **Diagnosis & department:** rate by ICD-9 category, by department, and a department × diagnosis heatmap.
3. **Length of stay:** LOS distribution, readmission rate by LOS, and LOS by department.
4. **Admission & discharge:** admission types, readmission by discharge destination, and a source → destination heatmap.
5. **High-risk cohort:** a rule-based risk level with its readmission rate, a cohort table and a CSV download.
6. **Data quality & pipeline:** run history, ingestion log, rejects by rule and all data-quality check results.

## 6. Project structure

```
├── dags/readmission_etl_dag.py     Airflow DAG: ETL (Part 1)
├── dags/readmission_ml_training_dag.py   Airflow DAG: train / compare / register (Part 2)
├── dags/readmission_monitoring_dag.py    Airflow DAG: monitoring + retraining trigger (Part 2)
├── etl/                            ETL package (all logic)
│   ├── config.py                   settings from environment variables
│   ├── db.py                       PostgreSQL helpers (COPY bulk load)
│   ├── columns.py                  source columns and code lists
│   ├── extract.py                  download / raw copy / ingestion log
│   ├── stage.py                    raw CSV -> staging
│   ├── validate.py                 reject rules, warnings, critical checks
│   ├── privacy.py                  salted-hash surrogate keys
│   ├── transform.py                cleaning + feature engineering
│   ├── load_mart.py                runs sql/build_mart.sql + reconciliation
│   ├── run_log.py                  audit.pipeline_runs bookkeeping
│   └── run_pipeline.py             run everything without Airflow
├── ml/                             Part 2: features, training, model class, evaluation, drift, monitoring
├── api/                            Part 2: FastAPI inference service
├── sql/init_mlops.sql              Part 2: mlops schema (training runs, prediction log, monitoring)
├── sql/init_schema.sql             schemas, staging/clean/audit tables
├── sql/build_mart.sql              star schema + reporting views
├── dashboard/app.py                Streamlit dashboard (8 tabs)
├── docker/                         Dockerfiles + postgres init script
├── docker-compose.yml
├── tests/                          unit tests (pytest): ETL, ML, API
├── scripts/generate_sample_data.py synthetic data for OFFLINE testing only
├── scripts/simulate_traffic.py     Part 2: send normal / drifted requests to the API
├── data/raw | manual | rejected    raw layer, manual fallback, reject log CSVs
└── docs/                           architecture, data dictionary, rules, source
```

## 7. Useful commands

```powershell
docker compose logs -f airflow-scheduler          # follow task logs
docker compose exec postgres psql -U hc_admin -d healthcare   # SQL shell
docker compose restart dashboard                  # after editing app.py
docker compose up -d --build api                  # after editing api/ or ml/ (the API image copies the code)
docker compose down                               # stop (data is kept)
docker compose down -v                            # stop and DELETE the database
```

Example queries for the report:
```sql
SELECT * FROM audit.pipeline_runs ORDER BY started_at DESC;
SELECT * FROM audit.ingestion_log ORDER BY log_id DESC LIMIT 5;
SELECT rule_id, COUNT(*) FROM audit.rejected_records GROUP BY 1 ORDER BY 1;
SELECT * FROM mart.v_kpis;
SELECT * FROM mart.fact_admission LIMIT 10;
```

## 8. Tests and running without Airflow

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                   # 31 unit tests (ETL, ML, API), no database needed
```
To run the full pipeline from your laptop against the Docker database, set the
environment variables `HC_DB_HOST=localhost`, `HC_DB_USER`, `HC_DB_PASSWORD` and
`HASH_SALT` first, then run:
```powershell
python -m etl.run_pipeline
```

## 9. Execution evidence checklist (for the submission)
- [ ] `docker compose ps` showing all services healthy
- [ ] Airflow Graph view of a successful run (all tasks green)
- [ ] Airflow task log of `validate` (valid vs rejected counts)
- [ ] `audit.pipeline_runs`, `audit.ingestion_log` and `audit.rejected_records` query results
- [ ] Sample rows from `mart.fact_admission` and `mart.dim_patient`
- [ ] Each dashboard tab

## 10. Troubleshooting
| Problem | Fix |
|---|---|
| `failed to connect to the docker API` | Start Docker Desktop and wait for "Engine running" |
| Port 5432 already in use | Set `POSTGRES_HOST_PORT=5433` in `.env` and run `docker compose up -d` |
| `airflow-init` exits with an error | `docker compose logs airflow-init`. Usually a wrong password in `.env`. Fix it, then run `docker compose down -v` and `docker compose up -d` |
| DAG not visible in Airflow | Wait about 30 seconds, then check `docker compose logs airflow-scheduler` for import errors |
| `extract` fails | Download blocked. Use the manual fallback (section 3) |
| Dashboard says "mart tables do not exist" | Run the DAG first |
| Changed `.env` passwords after the first start | The DB keeps the old ones. Run `docker compose down -v` to reset |

## 11. Part 2 – MLOps pipeline

### 11.1 What was added
| Requirement | Where |
|---|---|
| Predict 30-day readmission | Target `readmitted_30d`, features in `mart.v_ml_features` (built by the ETL) |
| Compare Logistic Regression, Random Forest, XGBoost | `ml/train.py` (one MLflow child run per model) |
| Track recall, precision, F1, ROC-AUC, calibration with MLflow | All logged per model for validation and test, plus PR-AUC, Brier, ECE, calibration plots |
| Explainability | Feature importance for every model, SHAP beeswarm and importance for the selected model, per-prediction top factors in the API |
| Reproducible feature engineering and training pipelines | `ColumnTransformer` saved inside the model; fixed seeds; Airflow DAG `readmission_model_training` |
| Chronological split | Oldest 70 % train / next 15 % validation / newest 15 % test (by encounter order) |
| Register and version the model | MLflow registry `readmission_risk_model`, aliases `champion` / `previous`, promotion gate |
| Serve with FastAPI | `api/main.py` – `/predict`, `/predict/batch`, `/model-info`, `/health`, `/metrics`, `/reload` |
| Containerize the inference service | `docker/api/Dockerfile` (service `api`) |
| Integrate into Streamlit | Tab 7 *Risk prediction (ML)* and tab 8 *Model performance & monitoring* |
| Monitor data quality, drift, performance, latency, failures | DAG `readmission_model_monitoring`, `ml/monitoring.py`, `mlops.*` tables |
| Retraining criteria and model lifecycle | [docs/model_lifecycle.md](docs/model_lifecycle.md), [docs/model_card.md](docs/model_card.md) |
| Academic decision-support prototype only | Disclaimer in every API response, in the dashboard and in the model card |

### 11.2 Upgrade an existing Part 1 installation
```powershell
docker compose up -d --build        # builds the new airflow, mlflow and api images (10-15 min the first time)
docker compose ps                   # mlflow and api should become healthy
```
Your Part 1 data stays in place. Then, in the Airflow UI:

1. **Trigger `hospital_readmission_etl` once more.** It adds the Part 2 columns (`encounter_order`, `mart.v_ml_features`).
2. **Switch on and trigger `readmission_model_training`** (about 2–4 minutes). All 4 tasks should turn green. The last one tells the API to load the new model.
3. Open **MLflow** (http://localhost:5000): experiment `hospital_readmission_30d` → the training run → 3 child runs. *Models* → `readmission_risk_model` → version 1 with alias `champion`.
4. Open the **API docs** (http://localhost:8000/docs) → `POST /predict` → *Try it out* → *Execute*. The example body is pre-filled.
5. **Switch on and trigger `readmission_model_monitoring`.** It scores 2,000 recent encounters through the API and computes the monitoring report.
6. Open the **dashboard** tabs 7 and 8.

### 11.3 Calling the API
```powershell
curl http://localhost:8000/health
curl http://localhost:8000/model-info
```
PowerShell example:
```powershell
$body = @{ encounter = @{ age_midpoint=75; length_of_stay_days=6; num_lab_procedures=45; num_procedures=1;
  num_medications=18; number_outpatient=0; number_emergency=1; number_inpatient=2; prior_admissions=3;
  prior_encounters_in_data=1; number_diagnoses=9; diagnosis_count=3; medication_count=2;
  medication_change_count=1; med_changed=1; diabetes_med=1; gender="Female"; admission_type="Emergency";
  admission_source_group="Emergency Room"; discharge_group="Transferred to another facility";
  department="Internal Medicine"; primary_diagnosis_category="Circulatory"; diag_2_category="Diabetes";
  diag_3_category="Circulatory"; max_glu_serum="Not measured"; a1c_result=">8"; insulin="Up" } } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/predict -Method Post -Body $body -ContentType "application/json"
```
The response contains `probability`, `risk_band` (Low / Medium / High), `flagged`, `threshold`,
`top_factors` (SHAP), `dq_flags` (input-quality warnings), `model_version`, `latency_ms` and a disclaimer.
Impossible values (e.g. `length_of_stay_days = 40`) are rejected with HTTP 422, and the rejection is logged.

### 11.4 Demonstrating monitoring and retraining
```powershell
docker compose exec airflow-scheduler python /opt/airflow/scripts/simulate_traffic.py --n 300            # normal traffic
docker compose exec airflow-scheduler python /opt/airflow/scripts/simulate_traffic.py --n 2000 --drift   # drifted traffic
```
Then trigger `readmission_model_monitoring` again and look at tab 8 (PSI chart, alerts, trends).
When a retraining criterion is met and the last training is older than the 7-day cool-down, the `decide`
task branches to `trigger_retraining`, which starts `readmission_model_training`.
See [docs/model_lifecycle.md](docs/model_lifecycle.md) for all thresholds and how to demo the trigger immediately.

### 11.5 Results from the verification run (test set = newest 14,901 encounters)
| Model | Test ROC-AUC | PR-AUC | Recall | Precision | F1 | ECE (calibrated) |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.650 | 0.188 | 0.509 | 0.171 | 0.256 | 0.008 |
| **Random Forest (selected)** | **0.669** | **0.197** | 0.373 | 0.193 | 0.254 | 0.008 |
| XGBoost | 0.659 | 0.191 | 0.355 | 0.187 | 0.245 | 0.011 |

Your numbers should be identical or very close (fixed seeds, same data). Details, fairness results and limitations are in the [model card](docs/model_card.md).

### 11.6 Part 2 evidence checklist
- [ ] Airflow: green runs of `readmission_model_training` and `readmission_model_monitoring` (Graph view)
- [ ] MLflow: experiment page comparing the 3 child runs (select them → *Compare*), artifacts of the selected run (calibration, SHAP, ROC/PR)
- [ ] MLflow: registered model with versions and the `champion` alias
- [ ] API: Swagger `/docs` with a successful `/predict` response; `/metrics` output
- [ ] `docker compose ps` showing the `api` and `mlflow` containers healthy
- [ ] Dashboard tab 7 (a prediction with its top factors) and tab 8 (model comparison, drift, subgroups, latency)
- [ ] `SELECT * FROM mlops.monitoring_runs ORDER BY run_at DESC;` showing alerts / retrain decision

### 11.7 Part 2 troubleshooting
| Problem | Fix |
|---|---|
| Build of `airflow` image fails downloading constraints | Needs internet access to raw.githubusercontent.com. Retry, or check proxy/firewall |
| `check_training_data` fails: *encounter_order is empty* | Re-run the `hospital_readmission_etl` DAG once |
| Training task killed / exit code -9 | Docker ran out of memory. Give Docker Desktop at least 6 GB |
| API `/predict` returns 503 | No champion yet. Run the training DAG, then `curl -X POST http://localhost:8000/reload` |
| Dashboard tab 7 says API not reachable | `docker compose ps api` / `docker compose logs api` |
| Monitoring DAG shows all tasks *skipped* | The API had no model loaded (short-circuit). Train first |

## Data and privacy
The data is published by UCI under CC BY 4.0 and is already de-identified. The
pipeline additionally replaces patient and encounter IDs with salted SHA-256
surrogate keys. Original IDs only exist in the raw files and the staging table.
No personal data is committed: `data/` and `.env` are git-ignored.

This is an academic prototype and is not for clinical decision-making. The prediction model is a
decision-support prototype only and must not replace clinical judgement.
