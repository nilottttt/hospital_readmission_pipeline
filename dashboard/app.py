"""
Hospital Readmission Analytics - Streamlit dashboard (Part 1)

Reads the mart.* and audit.* tables built by the Airflow pipeline.
Run inside Docker (http://localhost:8501) or locally:
    streamlit run dashboard/app.py
"""
import os

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
                "4 · Admission & discharge", "5 · High-risk cohort", "6 · Data quality & pipeline"])

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
