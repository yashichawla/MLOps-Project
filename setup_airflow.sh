#!/usr/bin/env bash
set -euo pipefail

# --- paths ---
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# --- venv ---
if [[ ! -d "venv" ]]; then
  echo "➜ creating venv/ ..."
  python3 -m venv venv
fi
source venv/bin/activate
python -m pip install --upgrade pip setuptools wheel

# --- detect versions ---
PYVER=$(python -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
AIRFLOW_VERSION="${AIRFLOW_VERSION:-2.9.3}"

# --- ensure artifacts folders exist ---
mkdir -p airflow_artifacts/logs airflow_artifacts/scheduler data/tmp data/processed

# --- export env for this session ---
set -a
source ./.airflow.env
set +a

# --- install Airflow with constraints (always do this separately) ---
echo "➜ installing apache-airflow==$AIRFLOW_VERSION (python $PYVER) with constraints ..."
pip install "apache-airflow==${AIRFLOW_VERSION}" \
  --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-${AIRFLOW_VERSION}/constraints-${PYVER}.txt"

# --- install project deps ---
echo "➜ installing project requirements ..."
pip install -r requirements.txt || true

# --- install repo as editable package (so 'from scripts...' works everywhere) ---
if [[ -f "pyproject.toml" ]]; then
  echo "➜ pip install -e ."
  pip install -e .
fi

# --- init / migrate DB and create admin user ---
echo "➜ migrating Airflow DB ..."
airflow db migrate

echo "➜ creating admin user (idempotent) ..."
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin || true

echo "✅ setup complete."
echo "Next: run ./start_airflow.sh"
