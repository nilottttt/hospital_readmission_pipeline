# Hospital Readmission Analytics – Part 1: Data Pipeline

A privacy-conscious healthcare data pipeline built on the public, de-identified
**UCI Diabetes 130-US Hospitals (1999–2008)** dataset.

**Apache Airflow** runs the Python/Pandas ETL on a schedule. The data moves through
**raw → staging → clean → mart** layers in **PostgreSQL**, and a **Streamlit**
dashboard presents admission and readmission patterns. Everything runs in
**Docker**.

![Architecture](docs/architecture_diagram.png)

---

## 1. Prerequisites

| Tool | Why |
|---|---|
| Docker Desktop (WSL 2 engine on Windows), with at least 4 GB of memory under Settings → Resources | Runs PostgreSQL, Airflow and Streamlit |
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
├── dags/readmission_etl_dag.py     Airflow DAG (orchestration only)
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
├── sql/init_schema.sql             schemas, staging/clean/audit tables
├── sql/build_mart.sql              star schema + reporting views
├── dashboard/app.py                Streamlit dashboard
├── docker/                         Dockerfiles + postgres init script
├── docker-compose.yml
├── tests/test_etl.py               unit tests (pytest)
├── scripts/generate_sample_data.py synthetic data for OFFLINE testing only
├── data/raw | manual | rejected    raw layer, manual fallback, reject log CSVs
└── docs/                           architecture, data dictionary, rules, source
```

## 7. Useful commands

```powershell
docker compose logs -f airflow-scheduler          # follow task logs
docker compose exec postgres psql -U hc_admin -d healthcare   # SQL shell
docker compose restart dashboard                  # after editing app.py
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
pytest -q                                   # unit tests, no database needed
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

## Data and privacy
The data is published by UCI under CC BY 4.0 and is already de-identified. The
pipeline additionally replaces patient and encounter IDs with salted SHA-256
surrogate keys. Original IDs only exist in the raw files and the staging table.
No personal data is committed: `data/` and `.env` are git-ignored.

This is an academic prototype and is not for clinical decision-making.
