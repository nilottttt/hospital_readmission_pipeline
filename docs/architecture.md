# Architecture and pipeline flow

> Use this as the basis for your own diagram (draw.io / diagrams.net). GitHub and
> VS Code (with a Mermaid extension) render the diagrams below directly.

## Layered architecture

![Architecture diagram](architecture_diagram.png)

Mermaid source (edit and re-render, or redraw in draw.io):

```mermaid
flowchart LR
    subgraph SRC[Source layer]
        UCI[(UCI Diabetes 130-US<br/>zip over HTTPS)]
        MAN[/data/manual<br/>fallback copy/]
    end
    subgraph ING[Ingestion layer - Airflow + Python]
        EX[extract.py<br/>download, retry,<br/>checksum, row count]
    end
    subgraph RAW[Raw / staging layer]
        RF[/data/raw/run_id/<br/>read-only files/]
        RL[(raw.lookup_ids)]
        ST[(staging.encounters<br/>all TEXT)]
    end
    subgraph TR[Transformation layer - Pandas]
        VA[validate.py<br/>11 reject rules]
        TF[transform.py<br/>clean, standardize,<br/>surrogate keys, features]
    end
    subgraph STO[Storage layer - PostgreSQL]
        CL[(clean.encounters)]
        MA[(mart star schema<br/>facts + dims + views)]
        AU[(audit<br/>runs, ingestion log,<br/>rejects, DQ results)]
    end
    subgraph AN[Analytics layer]
        DB[Streamlit dashboard<br/>6 tabs]
    end
    subgraph ML[MLOps layer - Part 2]
        P2[MLflow · FastAPI · Docker<br/>monitoring]
    end

    UCI --> EX
    MAN -. if download fails .-> EX
    EX --> RF --> ST
    EX --> RL
    ST --> VA --> TF --> CL --> MA --> DB
    VA -- rejected rows --> AU
    EX -- ingestion log --> AU
    MA -. features / target .-> P2
    P2 -. predictions .-> DB
```

## Airflow DAG `hospital_readmission_etl` (schedule: weekly, or trigger manually)

![DAG flow](dag_flow.png)

```mermaid
flowchart LR
    A[init_database] --> B[start_run] --> C[extract] --> D[stage] --> E[validate] --> F[transform] --> G[build_mart] --> H[finish_run]
```

| Task | Reads | Writes |
|---|---|---|
| init_database | sql/init_schema.sql | schemas + tables (IF NOT EXISTS) |
| start_run | – | audit.pipeline_runs (status = running) |
| extract | UCI URL or data/manual | data/raw/<run_id>/, raw.lookup_ids, audit.ingestion_log, audit.rejected_records (E01) |
| stage | raw CSV | staging.encounters |
| validate | staging.encounters | staging.dq_passed, audit.rejected_records, data/rejected/*.csv, audit.dq_check_results |
| transform | staging (passed rows), raw.lookup_ids | clean.encounters, clean.lookup_codes |
| build_mart | clean.* | mart.* (rebuilt), audit.dq_check_results |
| finish_run | – | audit.pipeline_runs (status = success) |

Each task retries twice. If a task still fails, `on_failure_callback` marks the run `failed` in `audit.pipeline_runs` with the error message.

## Star schema (mart)

![Star schema](star_schema.png)

```mermaid
erDiagram
    dim_patient ||--o{ fact_admission : patient_sk
    dim_admission_type ||--o{ fact_admission : admission_type_id
    dim_discharge_disposition ||--o{ fact_admission : discharge_disposition_id
    dim_admission_source ||--o{ fact_admission : admission_source_id
    fact_admission ||--o{ fact_diagnosis : encounter_sk
    fact_admission ||--|| fact_readmission : encounter_sk
```

## Containers (docker-compose.yml)

| Service | Image | Port | Role |
|---|---|---|---|
| postgres | postgres:16 | 5432 | `airflow` DB (Airflow metadata) + `healthcare` DB (project data) |
| airflow-init | hrp-airflow (apache/airflow:2.10.5-python3.11 + pandas) | – | one-off: migrate Airflow DB, create admin user |
| airflow-webserver | hrp-airflow | 8080 | Airflow UI |
| airflow-scheduler | hrp-airflow | – | runs the DAG tasks (LocalExecutor) |
| dashboard | hrp-dashboard (python:3.11-slim + streamlit) | 8501 | Streamlit dashboard |

## Design decisions
* **Full refresh** each run: staging, clean and mart are rebuilt, so a rerun always gives the same result (idempotent). Audit tables accumulate history across runs.
* **psycopg2 + COPY** for loading: fast (~100k rows in about 2 seconds) and independent of SQLAlchemy versions.
* **ETL logic lives in the `etl` package**, not in the DAG file. The DAG only orchestrates, and the same code can run without Airflow (`python -m etl.run_pipeline`).
* **Privacy:** original IDs never leave the staging layer; clean and mart only hold salted hashes. The salt lives in `.env`.

---

# Part 2 – MLOps extension

![MLOps architecture](mlops_architecture.png)

```mermaid
flowchart LR
    subgraph DATA[Part 1 data platform]
        MART[(mart.v_ml_features<br/>features + label)]
    end
    subgraph ORCH[Airflow]
        TD[readmission_model_training<br/>monthly / on alert]
        MD[readmission_model_monitoring<br/>daily]
    end
    subgraph TRACK[MLflow server]
        EXP[(Experiments<br/>params · metrics · plots)]
        REG[(Model registry<br/>alias champion / previous)]
    end
    subgraph SERVE[Inference - Docker]
        API[FastAPI<br/>/predict · /predict/batch<br/>/health · /metrics]
    end
    subgraph OBS[Monitoring store - PostgreSQL mlops]
        PL[(prediction_log)]
        MR[(monitoring_runs · drift_metrics<br/>subgroup_metrics)]
        TR[(training_runs ·<br/>reference_profile)]
    end
    DASH[Streamlit<br/>prediction + monitoring tabs]

    MART --> TD
    TD -- LR / RF / XGBoost runs --> EXP
    TD -- register + promote --> REG
    TD --> TR
    TD -- POST /reload --> API
    REG -- load champion --> API
    API --> PL
    MD -- score recent batch --> API
    PL --> MD
    MART -- outcomes --> MD
    MD --> MR
    MD -. retraining criteria met .-> TD
    DASH -- predict --> API
    MR --> DASH
    TR --> DASH
```

## New Airflow DAGs

| DAG | Schedule | Tasks |
|---|---|---|
| `readmission_model_training` | `@monthly` + triggered by monitoring | init_mlops → check_training_data → train_compare_register → reload_api |
| `readmission_model_monitoring` | `@daily` | check_api → score_recent_batch → compute_monitoring → decide → trigger_retraining / no_retraining |

## New containers

| Service | Image | Port | Role |
|---|---|---|---|
| mlflow | hrp-mlflow (python:3.11-slim + mlflow 2.22) | 5000 | Tracking server + model registry (SQLite backend and artifacts in the `mlflow_data` volume, artifacts served over HTTP) |
| api | hrp-api (python:3.11-slim + FastAPI, scikit-learn, XGBoost, SHAP) | 8000 | Loads `models:/readmission_risk_model@champion`; logs every request |

## Code layout

| Path | Purpose |
|---|---|
| `ml/features.py` | Feature lists, chronological split, preprocessing `ColumnTransformer` |
| `ml/train.py` | Trains and compares the 3 models, calibration, threshold, SHAP, MLflow logging, registry and promotion |
| `ml/model.py` | `ReadmissionRiskModel`: the MLflow pyfunc packaging preprocessing, model, calibrator, threshold, explanations and input checks |
| `ml/evaluate.py` | Metrics (ROC-AUC, PR-AUC, precision, recall, F1, Brier, ECE), subgroup metrics, plots |
| `ml/drift.py` | Reference profiles and PSI |
| `ml/monitoring.py` | Batch scoring, monitoring metrics, retraining decision, cool-down |
| `api/` | FastAPI app, request/response schemas, prediction logging |
| `sql/init_mlops.sql` | `mlops` schema |

## Design decisions (Part 2)
* **One deployable artifact.** Preprocessing, classifier, isotonic calibrator, threshold and training ranges are packaged together as an MLflow pyfunc model, so training and serving can never use different transformations.
* **Same library versions** in the training image (Airflow) and the inference image (API): scikit-learn 1.6.1, xgboost 2.1.4, numpy 1.26.4, pandas 2.1.4, cloudpickle 2.2.1.
* **Chronological split** (by encounter order) instead of a random split. The test set lies "in the future" of the training data, which is how the model would actually be used.
* **Selection on validation, report on test.** The test set never influences any choice.
* **Promotion gate.** A new model replaces the champion only if it is at least as good on the same test set.
* **Monitoring data in PostgreSQL.** The dashboard reads it with plain SQL. MLflow remains the system of record for experiments and models.
