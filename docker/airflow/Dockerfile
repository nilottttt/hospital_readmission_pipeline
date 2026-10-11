# Airflow image with the extra Python libraries our ETL and ML pipelines need.
FROM apache/airflow:2.10.5-python3.11

ARG AIRFLOW_VERSION=2.10.5
ARG PYTHON_VERSION=3.11
COPY requirements-airflow.txt /requirements-airflow.txt

# The official constraints file pins every Airflow dependency to a tested version;
# installing against it stops pip from up/downgrading anything Airflow relies on.
# Pinning apache-airflow in the same command keeps Airflow itself unchanged.
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements-airflow.txt \
      --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYTHON_VERSION}.txt"
