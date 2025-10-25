#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR"

# venv + env
source venv/bin/activate
set -a
source ./.airflow.env
set +a

# make sure folders exist
mkdir -p "$AIRFLOW__LOGGING__BASE_LOG_FOLDER" "$AIRFLOW__SCHEDULER__child_process_log_directory"

# start services in background
airflow webserver -p 8080 > airflow_artifacts/logs/webserver.out 2>&1 &
WS_PID=$!
airflow scheduler > airflow_artifacts/logs/scheduler.out 2>&1 &
SCH_PID=$!

echo "✅ Airflow webserver (pid $WS_PID) & scheduler (pid $SCH_PID) started."
echo "UI: http://localhost:8080  (login: admin / admin)"
echo "Logs: airflow_artifacts/logs/"
