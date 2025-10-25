#!/usr/bin/env bash
set -euo pipefail
pkill -f "airflow" || true
echo "🛑 Airflow processes stopped."
