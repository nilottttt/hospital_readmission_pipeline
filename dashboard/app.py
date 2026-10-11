"""
Hospital Readmission Analytics - Streamlit dashboard (Part 1)

Reads the mart.* and audit.* tables built by the Airflow pipeline.
Run inside Docker (http://localhost:8501) or locally:
    streamlit run dashboard/app.py
"""
import os

import requests

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine, text

st.set_page_config(page_title="Hospital Readmission Analytics", layout="wide")

# ---------------------------------------------------------------- colours
BLUE = "#2a78d6"                                    # single-series colour
CLASS_COLORS = {"Within 30 days": "#eb6834",        # categorical slots (fixed order)
                "After 30 days": "#2a78d6",
                "Not readmitted": "#1baf7a"}
RISK_COLORS = {"Low": "#86b6ef", "Medium": "#2a78d6", "High": "#104281"}  # ordinal blue ramp
SEQ_SCALE = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]       # sequential blue
AGE_ORDER = ["0-10", "10-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80-90", "90-100"]
RISK_ORDER = ["Low", "Medium", "High"]


# ---------------------------------------------------------------- data access
@st.cache_resource
def get_engine():
    url = (
        f"postgresql+psycopg2://{os.getenv('HC_DB_USER', os.getenv('POSTGRES_USER', 'hc_admin'))}:"
        f"{os.getenv('HC_DB_PASSWORD', os.getenv('POSTGRES_PASSWORD', ''))}@"
        f"{os.getenv('HC_DB_HOST', 'localhost')}:{os.getenv('HC_DB_PORT', '5432')}/"
        f"{os.getenv('HC_DB_NAME', 'healthcare')}"
    )
    return create_engine(url, pool_pre_ping=True)


@st.cache_data(ttl=600, show_spinner="Loading data from PostgreSQL...")
def query(sql: str) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


def mart_ready() -> bool:
    try:
        return bool(query("SELECT to_regclass('mart.fact_admission') IS NOT NULL AS ok")["ok"][0])
    except Exception as exc:  # DB not reachable
        st.error(f"Cannot connect to the database: {exc}")
        st.stop()


def style(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10),
                      legend_title_text="", bargap=0.25, hoverlabel=dict(namelength=-1))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    return fig


def readmit_rate(df: pd.DataFrame, by, extra=None) -> pd.DataFrame:
    agg = dict(encounters=("encounter_sk", "size"), readmit_30d_pct=("readmitted_30d", "mean"))
    if extra:
        agg.update(extra)
    out = df.groupby(by, observed=True).agg(**agg).reset_index()
    out["readmit_30d_pct"] = (out["readmit_30d_pct"] * 100).round(2)
    return out


# ---------------------------------------------------------------- page
st.title("Hospital Readmission Analytics")
st.caption("UCI Diabetes 130-US Hospitals (1999-2008) · de-identified encounter data · "
           "academic prototype, not for clinical use")

if not mart_ready():
    st.warning("The mart tables do not exist yet. Trigger the **hospital_readmission_etl** DAG "
               "in Airflow (http://localhost:8080), then refresh this page.")
    st.stop()

fact = query("""
    SELECT encounter_sk, patient_sk, age_group, gender, race, department,
           admission_type, admission_source_group, discharge_group,
           length_of_stay_days, los_band, num_medications, number_inpatient,
           number_emergency, number_diagnoses, prior_admissions, medication_count,
           primary_diagnosis_category, a1c_result, insulin,
           readmitted, readmitted_30d, risk_score, risk_level
    FROM mart.fact_admission
""")
fact["readmission_label"] = fact["readmitted"].map(
    {"<30": "Within 30 days", ">30": "After 30 days", "NO": "Not readmitted"})
fact["age_group"] = pd.Categorical(fact["age_group"], AGE_ORDER, ordered=True)
fact["risk_level"] = pd.Categorical(fact["risk_level"], RISK_ORDER, ordered=True)

# ---------------------------------------------------------------- filters
with st.sidebar:
    st.header("Filters")
    ages = st.multiselect("Age group", AGE_ORDER, default=[])
    genders = st.multiselect("Gender", sorted(fact["gender"].unique()), default=[])
    adm_types = st.multiselect("Admission type", sorted(fact["admission_type"].unique()), default=[])
    depts = st.multiselect("Department", sorted(fact["department"].unique()), default=[])
    st.caption("Leave a filter empty to include everything.")
    last = query("SELECT run_id, finished_at FROM audit.pipeline_runs "
                 "WHERE status='success' ORDER BY finished_at DESC LIMIT 1")
    if len(last):
        st.divider()
        st.caption(f"Last successful pipeline run:\n\n`{last.run_id[0]}`\n\n"
                   f"{pd.to_datetime(last.finished_at[0]):%Y-%m-%d %H:%M} UTC")

df = fact
if ages:
    df = df[df["age_group"].isin(ages)]
if genders:
    df = df[df["gender"].isin(genders)]
if adm_types:
    df = df[df["admission_type"].isin(adm_types)]
if depts:
    df = df[df["department"].isin(depts)]

if df.empty:
    st.info("No encounters match the selected filters.")
    st.stop()

# ---------------------------------------------------------------- KPI row
k = st.columns(5)
k[0].metric("Encounters", f"{len(df):,}")
k[1].metric("Patients", f"{df['patient_sk'].nunique():,}")
k[2].metric("30-day readmission rate", f"{df['readmitted_30d'].mean():.1%}")
k[3].metric("Average length of stay", f"{df['length_of_stay_days'].mean():.1f} days")
k[4].metric("High-risk encounters", f"{(df['risk_level'] == 'High').mean():.1%}")

tabs = st.tabs(["1 · Readmission by age", "2 · Diagnosis & department", "3 · Length of stay",
                "4 · Admission & discharge", "5 · High-risk cohort", "6 · Data quality & pipeline",
                "7 · Risk prediction (ML)", "8 · Model performance & monitoring"])

# ---------------------------------------------------------------- 1. age
with tabs[0]:
    c1, c2 = st.columns(2)
    by_age = readmit_rate(df, "age_group")
    fig = px.bar(by_age, x="age_group", y="readmit_30d_pct", color_discrete_sequence=[BLUE],
                 labels={"age_group": "Age group", "readmit_30d_pct": "30-day readmission rate (%)"},
                 hover_data={"encounters": ":,"}, title="30-day readmission rate by age group")
    c1.plotly_chart(style(fig), use_container_width=True)

    mix = (df.groupby(["age_group", "readmission_label"], observed=True).size()
             .rename("encounters").reset_index())
    mix["share_pct"] = (mix["encounters"] / mix.groupby("age_group", observed=True)["encounters"]
                        .transform("sum") * 100).round(1)
    fig = px.bar(mix, x="age_group", y="share_pct", color="readmission_label",
                 color_discrete_map=CLASS_COLORS,
                 category_orders={"readmission_label": list(CLASS_COLORS), "age_group": AGE_ORDER},
                 labels={"age_group": "Age group", "share_pct": "Share of encounters (%)"},
                 hover_data={"encounters": ":,"}, title="Readmission outcome mix by age group")
    c2.plotly_chart(style(fig), use_container_width=True)
    st.markdown("**Reading it:** older patients are admitted far more often; compare the rate "
                "(left) with the outcome mix (right) to see whether risk rises with age.")

# ---------------------------------------------------------------- 2. diagnosis & dept
with tabs[1]:
    c1, c2 = st.columns(2)
    by_diag = readmit_rate(df, "primary_diagnosis_category").sort_values("readmit_30d_pct")
    fig = px.bar(by_diag, y="primary_diagnosis_category", x="readmit_30d_pct", orientation="h",
                 color_discrete_sequence=[BLUE], hover_data={"encounters": ":,"},
                 labels={"primary_diagnosis_category": "", "readmit_30d_pct": "30-day readmission rate (%)"},
                 title="By primary diagnosis category (ICD-9 group)")
    c1.plotly_chart(style(fig, 420), use_container_width=True)

    by_dept = readmit_rate(df, "department", {"avg_los": ("length_of_stay_days", "mean")})
    by_dept = by_dept[by_dept["encounters"] >= 50].sort_values("readmit_30d_pct")
    by_dept["avg_los"] = by_dept["avg_los"].round(2)
    fig = px.bar(by_dept, y="department", x="readmit_30d_pct", orientation="h",
                 color_discrete_sequence=[BLUE], hover_data={"encounters": ":,", "avg_los": True},
                 labels={"department": "", "readmit_30d_pct": "30-day readmission rate (%)",
                         "avg_los": "Avg LOS (days)"},
                 title="By department (medical specialty, ≥50 encounters)")
    c2.plotly_chart(style(fig, 420), use_container_width=True)

    heat = readmit_rate(df, ["department", "primary_diagnosis_category"])
    heat = heat[heat["encounters"] >= 30]
    pivot = heat.pivot(index="department", columns="primary_diagnosis_category", values="readmit_30d_pct")
    fig = px.imshow(pivot, color_continuous_scale=SEQ_SCALE, aspect="auto", text_auto=".1f",
                    labels=dict(color="30-day %", x="Primary diagnosis", y="Department"),
                    title="30-day readmission rate (%): department × diagnosis (cells with ≥30 encounters)")
    st.plotly_chart(style(fig, 520), use_container_width=True)

# ---------------------------------------------------------------- 3. LOS
with tabs[2]:
    c1, c2 = st.columns(2)
    los = readmit_rate(df, "length_of_stay_days")
    fig = px.bar(los, x="length_of_stay_days", y="encounters", color_discrete_sequence=[BLUE],
                 labels={"length_of_stay_days": "Length of stay (days)", "encounters": "Encounters"},
                 title="Distribution of length of stay")
    c1.plotly_chart(style(fig), use_container_width=True)
    fig = px.line(los, x="length_of_stay_days", y="readmit_30d_pct", markers=True,
                  color_discrete_sequence=[BLUE], hover_data={"encounters": ":,"},
                  labels={"length_of_stay_days": "Length of stay (days)",
                          "readmit_30d_pct": "30-day readmission rate (%)"},
                  title="30-day readmission rate by length of stay")
    fig.update_traces(line_width=2, marker_size=8)
    c2.plotly_chart(style(fig), use_container_width=True)

    by_dept_los = (df.groupby("department")["length_of_stay_days"].agg(["mean", "median", "size"])
                     .query("size >= 50").sort_values("mean").reset_index())
    fig = px.bar(by_dept_los, y="department", x="mean", orientation="h", color_discrete_sequence=[BLUE],
                 hover_data={"median": True, "size": ":,"},
                 labels={"department": "", "mean": "Average length of stay (days)", "size": "Encounters"},
                 title="Average length of stay by department")
    st.plotly_chart(style(fig, 460), use_container_width=True)

# ---------------------------------------------------------------- 4. admission & discharge
with tabs[3]:
    c1, c2 = st.columns(2)
    adm = readmit_rate(df, "admission_type").sort_values("encounters")
    fig = px.bar(adm, y="admission_type", x="encounters", orientation="h", color_discrete_sequence=[BLUE],
                 hover_data={"readmit_30d_pct": True},
                 labels={"admission_type": "", "encounters": "Encounters",
                         "readmit_30d_pct": "30-day readmission rate (%)"},
                 title="Admissions by admission type")
    c1.plotly_chart(style(fig), use_container_width=True)

    dis = readmit_rate(df, "discharge_group").sort_values("readmit_30d_pct")
    fig = px.bar(dis, y="discharge_group", x="readmit_30d_pct", orientation="h",
                 color_discrete_sequence=[BLUE], hover_data={"encounters": ":,"},
                 labels={"discharge_group": "", "readmit_30d_pct": "30-day readmission rate (%)"},
                 title="30-day readmission rate by discharge destination")
    c2.plotly_chart(style(fig), use_container_width=True)

    flow = df.groupby(["admission_source_group", "discharge_group"]).size().rename("encounters").reset_index()
    pivot = flow.pivot(index="admission_source_group", columns="discharge_group", values="encounters").fillna(0)
    fig = px.imshow(pivot, color_continuous_scale=SEQ_SCALE, aspect="auto", text_auto=",.0f",
                    labels=dict(color="Encounters", x="Discharged to", y="Admitted from"),
                    title="Admission source → discharge destination (encounter counts)")
    st.plotly_chart(style(fig, 420), use_container_width=True)

# ---------------------------------------------------------------- 5. high-risk cohort
with tabs[4]:
    st.markdown(
        "Risk level comes from a **transparent rule-based score** built in the ETL "
        "(prior inpatient/emergency visits, long stay, many medications, many diagnoses, age ≥ 70, "
        "transfer at discharge). 0–2 = Low, 3–4 = Medium, 5+ = High. "
        "It is a descriptive cohort flag, not the Part 2 machine-learning model.")
    c1, c2 = st.columns(2)
    risk = readmit_rate(df, "risk_level")
    fig = px.bar(risk, x="risk_level", y="readmit_30d_pct", color="risk_level",
                 color_discrete_map=RISK_COLORS, category_orders={"risk_level": RISK_ORDER},
                 hover_data={"encounters": ":,"},
                 labels={"risk_level": "Risk level", "readmit_30d_pct": "30-day readmission rate (%)"},
                 title="Readmission rate by risk level")
    fig.update_layout(showlegend=False)
    c1.plotly_chart(style(fig), use_container_width=True)

    score = readmit_rate(df, "risk_score")
    fig = px.bar(score, x="risk_score", y="encounters", color_discrete_sequence=[BLUE],
                 hover_data={"readmit_30d_pct": True},
                 labels={"risk_score": "Risk score", "encounters": "Encounters",
                         "readmit_30d_pct": "30-day readmission rate (%)"},
                 title="Distribution of risk score")
    c2.plotly_chart(style(fig), use_container_width=True)

    high = df[df["risk_level"] == "High"]
    h = st.columns(4)
    h[0].metric("High-risk encounters", f"{len(high):,}")
    h[1].metric("Their 30-day readmission rate",
                f"{high['readmitted_30d'].mean():.1%}" if len(high) else "–")
    h[2].metric("Average prior inpatient visits",
                f"{high['number_inpatient'].mean():.1f}" if len(high) else "–")
    h[3].metric("Average length of stay",
                f"{high['length_of_stay_days'].mean():.1f} days" if len(high) else "–")

    cols = ["encounter_sk", "patient_sk", "age_group", "gender", "department",
            "primary_diagnosis_category", "length_of_stay_days", "number_inpatient",
            "number_emergency", "num_medications", "number_diagnoses", "discharge_group",
            "risk_score", "readmitted"]
    table = high.sort_values("risk_score", ascending=False)[cols]
    st.dataframe(table.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download high-risk cohort (CSV)", table.to_csv(index=False),
                       "high_risk_cohort.csv", "text/csv")

# ---------------------------------------------------------------- 6. data quality
with tabs[5]:
    runs = query("""SELECT run_id, triggered_by, status, started_at, finished_at, rows_extracted,
                           rows_staged, rows_valid, rows_rejected, rows_clean
                    FROM audit.pipeline_runs ORDER BY started_at DESC LIMIT 20""")
    st.subheader("Pipeline runs")
    st.dataframe(runs, use_container_width=True, hide_index=True)

    latest = runs.loc[runs["status"] == "success", "run_id"]
    if len(latest):
        rid = latest.iloc[0]
        c1, c2 = st.columns(2)
        rej = query(f"""SELECT rule_id AS rule, MIN(rule_description) AS description, COUNT(*) AS records
                        FROM audit.rejected_records WHERE run_id = '{rid.replace("'", "''")}'
                        GROUP BY rule_id ORDER BY records""")
        if len(rej):
            fig = px.bar(rej, y="rule", x="records", orientation="h", color_discrete_sequence=[BLUE],
                         text="records", hover_data={"description": True},
                         labels={"rule": "Rule", "records": "Rejected records", "description": "Rule"},
                         title=f"Rejected records by rule (run {rid})")
            fig.update_traces(textposition="outside", cliponaxis=False)
            c1.plotly_chart(style(fig, 360), use_container_width=True)
            c1.caption("Hover a bar for the rule description - full list in docs/validation_rules.md")
        else:
            c1.success("No records were rejected in the latest run.")
        ing = query(f"""SELECT extracted_at, status, file_name, row_count, bad_line_count, message
                        FROM audit.ingestion_log WHERE run_id = '{rid.replace("'", "''")}'
                        ORDER BY log_id""")
        c2.markdown("**Ingestion log (latest run)**")
        c2.dataframe(ing, use_container_width=True, hide_index=True)

        dq = query(f"""SELECT layer, check_name, severity,
                              CASE WHEN passed THEN '✅ pass' ELSE '⚠️ fail' END AS result,
                              failed_count, details
                       FROM audit.dq_check_results WHERE run_id = '{rid.replace("'", "''")}'
                       ORDER BY check_id""")
        st.subheader("Data-quality checks (latest successful run)")
        st.dataframe(dq, use_container_width=True, hide_index=True)

# =====================================================================
# PART 2 - MLOps: prediction service and model monitoring
# =====================================================================
API_URL = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_UI_URL = os.getenv("MLFLOW_UI_URL", "http://localhost:5000")
MODEL_COLORS = {"logistic_regression": "#2a78d6", "random_forest": "#eb6834", "xgboost": "#1baf7a"}
MODEL_LABELS = {"logistic_regression": "Logistic Regression", "random_forest": "Random Forest",
                "xgboost": "XGBoost"}


@st.cache_data(ttl=60, show_spinner=False)
def query_fresh(sql: str) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn)


@st.cache_data(ttl=30, show_spinner=False)
def api_model_info():
    try:
        r = requests.get(f"{API_URL}/model-info", timeout=10)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def table_exists(name: str) -> bool:
    return bool(query_fresh(f"SELECT to_regclass('{name}') IS NOT NULL AS ok")["ok"][0])


# ---------------------------------------------------------------- 7. prediction
with tabs[6]:
    info = api_model_info()
    if info is None:
        st.warning(f"The prediction API ({API_URL}) is not reachable or has no champion model yet. "
                   "Run the **readmission_model_training** DAG in Airflow first.")
    else:
        m = info.get("test_metrics", {})
        c = st.columns(5)
        c[0].metric(f"Served model (v{info['version']})", MODEL_LABELS.get(info["model_type"], info["model_type"]))
        c[1].metric("Test ROC-AUC", f"{m.get('roc_auc', float('nan')):.3f}")
        c[2].metric("Test recall", f"{m.get('recall', float('nan')):.1%}")
        c[3].metric("Test precision", f"{m.get('precision', float('nan')):.1%}")
        c[4].metric("Decision threshold", f"{info['threshold']:.1%}")
        st.caption("⚠️ Academic decision-support prototype trained on 1999–2008 public data. "
                   "Not validated for clinical use; it must not replace clinical judgement.")

        cats = info.get("categories", {})

        def opts(name, default=None):
            values = cats.get(name) or [default or "Unknown"]
            return values, (values.index(default) if default in values else 0)

        with st.form("predict_form"):
            st.markdown("**Describe the encounter**")
            a, b, cc = st.columns(3)
            with a:
                st.markdown("*Patient & admission*")
                age = st.selectbox("Age group", AGE_ORDER, index=7)
                gender = st.selectbox("Gender", *opts("gender", "Female"))
                adm_type = st.selectbox("Admission type", *opts("admission_type", "Emergency"))
                adm_src = st.selectbox("Admitted from", *opts("admission_source_group", "Emergency Room"))
                dept = st.selectbox("Department", *opts("department", "Internal Medicine"))
                disch = st.selectbox("Discharge destination", *opts("discharge_group", "Home"))
            with b:
                st.markdown("*Stay & diagnoses*")
                los = st.slider("Length of stay (days)", 1, 14, 5)
                labs = st.number_input("Lab procedures", 0, 150, 45)
                procs = st.number_input("Other procedures", 0, 6, 1)
                n_meds = st.number_input("Distinct medications given", 1, 80, 16)
                n_diag = st.slider("Number of diagnoses recorded", 1, 16, 8)
                diag1 = st.selectbox("Primary diagnosis", *opts("primary_diagnosis_category", "Circulatory"))
                diag2 = st.selectbox("Secondary diagnosis", *opts("diag_2_category", "Diabetes"))
                diag3 = st.selectbox("Additional diagnosis", *opts("diag_3_category", "Circulatory"))
            with cc:
                st.markdown("*History & diabetes care*")
                outp = st.number_input("Outpatient visits (past year)", 0, 40, 0)
                emer = st.number_input("Emergency visits (past year)", 0, 40, 0)
                inp = st.number_input("Inpatient admissions (past year)", 0, 20, 1)
                prior_enc = st.number_input("Earlier encounters in this system", 0, 40, 0)
                a1c = st.selectbox("HbA1c result", *opts("a1c_result", "Not measured"))
                glu = st.selectbox("Max glucose serum", *opts("max_glu_serum", "Not measured"))
                insulin = st.selectbox("Insulin", *opts("insulin", "Steady"))
                med_count = st.number_input("Diabetes drugs prescribed", 0, 23, 1)
                med_changes = st.number_input("Diabetes drug dose changes", 0, 23, 0)
                diabetes_med = st.checkbox("On diabetes medication", True)
            submitted = st.form_submit_button("Predict 30-day readmission risk", type="primary")

        if submitted:
            encounter = {
                "age_midpoint": int(age.split("-")[0]) + 5, "length_of_stay_days": los,
                "num_lab_procedures": labs, "num_procedures": procs, "num_medications": n_meds,
                "number_outpatient": outp, "number_emergency": emer, "number_inpatient": inp,
                "prior_admissions": outp + emer + inp, "prior_encounters_in_data": prior_enc,
                "number_diagnoses": n_diag, "diagnosis_count": sum(d != "Missing" for d in (diag1, diag2, diag3)),
                "medication_count": med_count, "medication_change_count": med_changes,
                "med_changed": int(med_changes > 0), "diabetes_med": int(diabetes_med),
                "gender": gender, "admission_type": adm_type, "admission_source_group": adm_src,
                "discharge_group": disch, "department": dept, "primary_diagnosis_category": diag1,
                "diag_2_category": diag2, "diag_3_category": diag3, "max_glu_serum": glu,
                "a1c_result": a1c, "insulin": insulin}
            try:
                r = requests.post(f"{API_URL}/predict", json={"encounter": encounter, "explain": True,
                                                              "source": "dashboard"}, timeout=30)
                r.raise_for_status()
                pred = r.json()
            except requests.RequestException as exc:
                st.error(f"Prediction failed: {exc}")
                pred = None
            if pred:
                icon = {"High": "🔴", "Medium": "🟠", "Low": "🟢"}[pred["risk_band"]]
                r1 = st.columns(4)
                r1[0].metric("Predicted 30-day readmission risk", f"{pred['probability']:.1%}")
                r1[1].metric("Risk band", f"{icon} {pred['risk_band']}")
                r1[2].metric("Average risk (training data)", f"{info['base_rate']:.1%}")
                r1[3].metric(f"Flag for follow-up? (threshold {pred['threshold']:.1%})",
                             "Yes" if pred["flagged"] else "No")
                if pred["dq_flags"]:
                    st.warning("Input-quality flags (outside the training data): " + ", ".join(pred["dq_flags"]))
                fac = pd.DataFrame(pred["top_factors"])
                if len(fac):
                    fac["effect"] = fac["contribution"].map(lambda v: "raises risk" if v > 0 else "lowers risk")
                    fac["label"] = fac.apply(lambda r: f"{r['feature']} = {r['value']}", axis=1)
                    fac = fac.iloc[::-1]
                    fig = px.bar(fac, x="contribution", y="label", orientation="h", color="effect",
                                 color_discrete_map={"raises risk": "#eb6834", "lowers risk": "#2a78d6"},
                                 labels={"contribution": "Contribution to the prediction (SHAP value)", "label": ""},
                                 title="Why? Top factors behind this prediction")
                    st.plotly_chart(style(fig, 320), use_container_width=True)
                st.caption(f"Model {pred['model_name']} v{pred['model_version']} · request {pred['request_id']} · "
                           f"{pred['latency_ms']:.0f} ms · logged for monitoring")

# ---------------------------------------------------------------- 8. performance & monitoring
with tabs[7]:
    if not table_exists("mlops.training_runs"):
        st.info("No model has been trained yet. Run the **readmission_model_training** DAG in Airflow.")
        st.stop()
    runs = query_fresh("SELECT * FROM mlops.training_runs ORDER BY trained_at DESC, id")
    latest_id = runs["training_id"].iloc[0]
    latest = runs[runs["training_id"] == latest_id].copy()
    latest["model"] = latest["model_type"].map(MODEL_LABELS)
    sel = latest[latest["selected"]].iloc[0]

    st.subheader("Model comparison (latest training run)")
    st.caption(f"Chronological split: train {int(sel.n_train):,} · validation {int(sel.n_valid):,} · "
               f"test {int(sel.n_test):,} encounters. Selected by validation ROC-AUC; numbers below are on "
               f"the untouched test set. [Open MLflow]({MLFLOW_UI_URL}) for every run, plot and model version.")
    c1, c2 = st.columns([3, 2])
    long = latest.melt(id_vars=["model", "model_type"],
                       value_vars=["test_roc_auc", "test_pr_auc", "test_recall", "test_precision", "test_f1"],
                       var_name="metric", value_name="value")
    long["metric"] = long["metric"].str.replace("test_", "").str.upper().str.replace("_", "-")
    fig = px.bar(long, x="metric", y="value", color="model", barmode="group", text_auto=".2f",
                 color_discrete_map={MODEL_LABELS[k]: v for k, v in MODEL_COLORS.items()},
                 category_orders={"model": list(MODEL_LABELS.values())},
                 labels={"metric": "", "value": "Score (test set)", "model": ""},
                 title="Test-set metrics by model")
    fig.update_traces(textposition="outside", cliponaxis=False)
    c1.plotly_chart(style(fig, 400), use_container_width=True)
    tbl = latest[["model", "val_roc_auc", "test_roc_auc", "test_pr_auc", "test_recall", "test_precision",
                  "test_f1", "test_brier", "test_ece_uncalibrated", "test_ece", "threshold", "selected"]].copy()
    tbl.columns = ["Model", "Val AUC", "Test AUC", "Test PR-AUC", "Recall", "Precision", "F1", "Brier",
                   "ECE before calib.", "ECE after calib.", "Threshold", "Selected"]
    c2.dataframe(tbl.round(4), hide_index=True, use_container_width=True)
    c2.markdown(f"**Registry:** `{sel.model_type}` registered as version **{sel.registered_version}** — "
                f"{sel.decision}")
    c2.caption("Class weighting inflates raw probabilities; isotonic calibration on the validation set "
               "brings the expected calibration error (ECE) close to zero.")

    st.divider()
    st.subheader("Production monitoring")
    mon = query_fresh("SELECT * FROM mlops.monitoring_runs ORDER BY run_at") if table_exists("mlops.monitoring_runs") else pd.DataFrame()
    if mon.empty:
        st.info("No monitoring run yet. Trigger the **readmission_model_monitoring** DAG in Airflow.")
        st.stop()
    last = mon.iloc[-1]
    k = st.columns(6)
    k[0].metric("Predictions (window)", f"{int(last.n_predictions or 0):,}")
    k[1].metric("With known outcome", f"{int(last.n_labeled or 0):,}")
    if pd.notna(last.roc_auc):
        k[2].metric("ROC-AUC (live)", f"{last.roc_auc:.3f}",
                    f"{last.roc_auc - last.reference_roc_auc:+.3f} vs test" if pd.notna(last.reference_roc_auc) else None)
        k[3].metric("Calibration ECE", f"{last.ece:.3f}")
    k[4].metric("Max feature PSI", f"{last.max_psi:.2f}" if pd.notna(last.max_psi) else "–")
    k[5].metric("p95 latency · error rate",
                (f"{last.p95_latency_ms:.0f} ms · {last.error_rate:.1%}" if pd.notna(last.p95_latency_ms) else "–"))
    alerts = last.alerts if isinstance(last.alerts, list) else []
    if last.retrain_recommended:
        st.error("🔁 **Retraining recommended**" + (" — training DAG triggered." if last.retrain_triggered
                                                       else " (not triggered: cool-down period or manual decision)."))
    for a in alerts:
        st.warning(f"⚠️ {a}")
    if not alerts:
        st.success("✅ No alerts in the latest monitoring run.")

    c1, c2 = st.columns(2)
    drift = query_fresh(f"SELECT feature, psi, status FROM mlops.drift_metrics "
                        f"WHERE monitoring_id = '{last.monitoring_id}' ORDER BY psi")
    if len(drift):
        fig = px.bar(drift, x="psi", y="feature", orientation="h", color_discrete_sequence=[BLUE],
                     hover_data={"status": True}, labels={"psi": "Population Stability Index", "feature": ""},
                     title="Input drift vs. training data (PSI)")
        fig.add_vline(x=0.10, line_dash="dot", line_color="grey", annotation_text="0.10 warning",
                      annotation_position="bottom right", annotation_font_size=10)
        fig.add_vline(x=0.25, line_dash="dash", line_color="grey", annotation_text="0.25 drift",
                      annotation_position="bottom right", annotation_font_size=10)
        c1.plotly_chart(style(fig, 560), use_container_width=True)
    if len(mon) > 1:
        specs = [("roc_auc", "Live ROC-AUC", [0.5, 0.85]),
                 ("ece", "Calibration error (ECE)", [0, max(0.15, float(mon["ece"].max() or 0) * 1.2)]),
                 ("max_psi", "Largest feature drift (PSI)", [0, max(0.6, float(mon["max_psi"].max() or 0) * 1.2)])]
        for measure, title, yr in specs:
            part = mon[["run_at", measure]].dropna()
            if len(part):
                fig = px.line(part, x="run_at", y=measure, markers=True, color_discrete_sequence=[BLUE],
                              labels={"run_at": "", measure: ""}, title=f"{title} per monitoring run")
                fig.update_traces(line_width=2, marker_size=8)
                fig.update_yaxes(range=yr)
                c2.plotly_chart(style(fig, 180), use_container_width=True)
    else:
        c2.info("Trends appear after the second monitoring run.")

    st.markdown("**Subgroup performance (fairness check)** — live window vs. test-set baseline")
    sub = query_fresh(f"""SELECT monitoring_id, group_type, group_value, n, positives, roc_auc, recall, precision,
                                 flag_rate, mean_predicted, observed_rate
                          FROM mlops.subgroup_metrics
                          WHERE monitoring_id IN ('{last.monitoring_id}', 'training:v{last.model_version}')""")
    if len(sub):
        sub["period"] = sub["monitoring_id"].map(lambda m: "test-set baseline" if m.startswith("training") else "live window")
        gt = st.radio("Group by", ["age_group", "gender", "race"], horizontal=True)
        part = sub[sub["group_type"] == gt].sort_values("group_value")
        fig = px.bar(part, x="group_value", y="recall", color="period", barmode="group",
                     color_discrete_map={"test-set baseline": "#86b6ef", "live window": "#2a78d6"},
                     hover_data={"n": ":,", "positives": True, "precision": ":.2f", "roc_auc": ":.3f"},
                     labels={"group_value": gt.replace("_", " ").title(), "recall": "Recall (share of readmissions caught)",
                             "period": ""}, title=f"Recall by {gt.replace('_', ' ')}")
        st.plotly_chart(style(fig, 360), use_container_width=True)
        st.dataframe(part.drop(columns=["monitoring_id"]).round(3), hide_index=True, use_container_width=True)

    vol = query_fresh("""SELECT date_trunc('hour', predicted_at) AS hour, source, status, COUNT(*) AS requests,
                                percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_ms
                         FROM mlops.prediction_log GROUP BY 1, 2, 3 ORDER BY 1""")
    if len(vol):
        st.markdown("**API traffic** (every request is logged in `mlops.prediction_log`)")
        v1, v2 = st.columns(2)
        fig = px.bar(vol, x="hour", y="requests", color="source", labels={"hour": "", "requests": "Requests"},
                     color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], title="Requests by source")
        v1.plotly_chart(style(fig, 300), use_container_width=True)
        lat = vol[(vol["status"] == "ok") & (vol["source"] != "batch")]
        if len(lat):
            fig = px.line(lat, x="hour", y="p95_ms", color="source", markers=True,
                          color_discrete_sequence=["#2a78d6", "#eb6834", "#1baf7a"],
                          labels={"hour": "", "p95_ms": "p95 latency (ms)"}, title="Interactive request latency (p95)")
            fig.add_hline(y=500, line_dash="dash", line_color="grey", annotation_text="500 ms alert")
            v2.plotly_chart(style(fig, 300), use_container_width=True)
