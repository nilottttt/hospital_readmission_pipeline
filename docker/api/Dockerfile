# Readmission-risk inference service (FastAPI + MLflow model registry)
# Build context = project root (see docker-compose.yml)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba_cache MPLCONFIGDIR=/tmp/matplotlib
WORKDIR /app

# libgomp is needed by xgboost / scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY docker/api/requirements-api.txt /tmp/requirements-api.txt
RUN pip install --no-cache-dir -r /tmp/requirements-api.txt

# Application code: the API and the ml package (model class + feature definitions)
COPY api/ /app/api/
COPY ml/ /app/ml/
COPY sql/init_mlops.sql /app/sql/init_mlops.sql

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
