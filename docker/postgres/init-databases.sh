#!/bin/bash
# Runs ONCE, the first time the postgres container starts with an empty volume.
# Creates the two databases used by the project:
#   - $AIRFLOW_DB_NAME : Airflow's own metadata
#   - $HC_DB_NAME      : the healthcare data (raw/staging/clean/mart/audit)
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres <<-EOSQL
    SELECT 'CREATE DATABASE ${AIRFLOW_DB_NAME}'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${AIRFLOW_DB_NAME}')\gexec
    SELECT 'CREATE DATABASE ${HC_DB_NAME}'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${HC_DB_NAME}')\gexec
EOSQL

echo "Databases ${AIRFLOW_DB_NAME} and ${HC_DB_NAME} are ready."
