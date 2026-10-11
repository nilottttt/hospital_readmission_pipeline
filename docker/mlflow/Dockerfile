# MLflow tracking server + model registry
#   backend store : SQLite file in the mlflow_data volume (runs, metrics, registry)
#   artifacts     : /mlflow/artifacts in the same volume, served over HTTP (--serve-artifacts)
FROM python:3.11-slim

RUN pip install --no-cache-dir "mlflow==2.22.0" "sqlalchemy>=2.0.30,<2.1"

EXPOSE 5000
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts"]
