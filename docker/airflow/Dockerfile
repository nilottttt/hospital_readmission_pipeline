# Airflow image with the extra Python libraries our ETL needs.
FROM apache/airflow:2.10.5-python3.11

COPY requirements-airflow.txt /requirements-airflow.txt

# Pinning apache-airflow in the same command stops pip from
# accidentally upgrading/downgrading Airflow itself.
RUN pip install --no-cache-dir "apache-airflow==2.10.5" -r /requirements-airflow.txt
