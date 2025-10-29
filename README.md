# Break The Bot

Find a Project Brief and Overview, see [project_overview.md](./project_overview.md)
## 1. Quick Setup & Run Instructions

Follow these steps in order to set up and run the complete project end-to-end.

1.1. Clone Repository
```bash
git clone https://github.com/yashichawla/MLOps-Project
cd MLOps-Project
```

1.2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

1.3. Create Environment Files
a) SMTP .env file (for email notifications)

Generate a new Google App Password for your account. 

Create a file named .env in the project root:
```bash
AIRFLOW_SMTP_USER=your_email@gmail.com
AIRFLOW_SMTP_PASSWORD=your_gmail_app_password   # 16-digit Google App Password
```

b) GCP Secrets JSON (for DVC)

Place the service account key JSON (provided by us in Canvas Submission Comments) inside:
```bash
.secrets/gcp-key.json
```

1.4. Initialize and Run Airflow with Docker
```bash
# Create logs folder (recommended)
mkdir -p airflow_artifacts/logs

# Initialize Airflow DB and create admin user
docker compose run --rm airflow-init

# Start Airflow services (Webserver + Scheduler + Postgres)
docker compose up -d webserver scheduler
```

Open the Airflow UI:

📍 http://localhost:8080

👤 Username: admin

🔑 Password: admin

You can update credentials in docker-compose.yml later.

Stop services but keep data:
```bash
docker compose down
```

Completely remove containers, logs, and database volumes:
```bash
docker compose down -v
rm -rf airflow_artifacts/logs/*
```

1.5. (Optional) Test Mode

If you want to skip preprocessing and only validate a CSV:

In Airflow UI → Admin → Variables

Set TEST_MODE = true

## 2. Repository Structure

```plaintext
MLOps-Project/
├── .dvc/
│   ├── .gitignore                     # Ignore DVC cache or temp files from version control
│   └── config                         # DVC configuration file defining remote storage (e.g., GCS bucket)
├── dags/                              # Airflow DAGs
│   └── salad_preprocess_dag.py        # Main DAG (preprocessing + single validation + email alerts)
├── scripts/
│   ├── preprocess_salad.py            # Data preprocessing pipeline
│   ├── ge_runner.py                   # Great Expectations Validator
│   └── utils/                         # Shared helper modules (if any)
├── config/
│   └── data_sources.json              # Config file for multi-source data ingestion
├── data/
│   ├── processed/                     # Output CSV (processed_data.csv)
│   ├── metrics/                       # Stats + validation results (used by Airflow + GE)
│   └── test_validation/               # Test CSVs for test-mode runs
├── documents/                         # Documents related to assignment submissions
├── tests/                             # Unit Test scripts for all components
├── .airflow.env                       # Environment variables for local Airflow setup (e.g., connections, paths)
├── .dockerignore                      # Files and folders excluded from the Docker build context
├── .dvcignore                         # Files and folders excluded from DVC tracking
├── .gitignore                         # Files and folders excluded from Git tracking
├── pyproject.toml                     # Project metadata and dependency management configuration
├── docker-compose.yml                 # Airflow + Postgres stack
├── dvc.lock                           # Auto-generated file tracking exact data and pipeline versions
├── dvc.yaml                           # Defines the DVC pipeline stages (data processing, training, etc.)
├── requirements.txt                   # Dev dependencies (includes pandas, airflow, etc.)
├── requirements-docker.txt            # Installed inside Docker containers
└── README.md
```


## 3. DAG flow

![DAG Pipeline Architecture](documents/DAG_Pipeline.jpg)

## 4. Email Notifications

The DAG now uses the unified validator’s XCom output for all emails:

| Trigger    | Email                 | Contents                         |
| ---------- | --------------------- | -------------------------------- |
| Always     | **Validation Report** | JSON report + anomalies attached |
| On Success | **✅ DAG Succeeded**  | Summary of counts and ranges     |
| On Failure | **❌ DAG Failed**     | Hard-fail reasons + report paths |

Recipients are configured in `salad_preprocess_dag.py` under each `EmailOperator`.

To add more recipients, edit in salad_preprocess_dag.py:

```bash
to=["yashi.chawla1@gmail.com", "...", "..."]
```

## 5. Validation Source of Truth

- `scripts/ge_runner.py` is the single validator used by the Airflow DAG.
- The DAG invokes:
  - `python scripts/ge_runner.py baseline --input <csv> --date YYYYMMDD` (creates `data/metrics/schema/baseline/schema.json` if missing)
  - `python scripts/ge_runner.py validate --input <csv> --baseline_schema <path> --date YYYYMMDD`
- Validation artifacts (source of truth):
  - `data/metrics/stats/YYYYMMDD/stats.json` (includes row_count, null/dup counts, unknown_category_rate, text_len_min/max, size_label_mismatch_count)
  - `data/metrics/validation/YYYYMMDD/anomalies.json` (hard_fail, soft_warn, info)
- Airflow reads these files to construct the XCom metrics used for gating and email reports.

## 6. DVC Usage Guide

This repository integrates DVC (Data Version Control) with Google Cloud Storage (GCS) to version datasets and validation artifacts generated by the Salad data pipeline.

DVC is used to track and version the following pipeline outputs:

```text
data/processed/processed_data.csv
data/stats/
data/validation/
```

The Airflow DAG (salad_preprocess_dag.py) orchestrates:

- dvc pull at pipeline start — ensures the latest data version is fetched.
- Preprocessing & validation tasks.
- dvc push after completion — uploads new artifacts to the GCS remote

All commands run automatically inside the Airflow Docker containers — no CLI interaction is needed.

For Local Debugging (Optional)

If you wish to run DVC manually outside Docker:

```bash
pip install -r requirements.txt
$env:GOOGLE_APPLICATION_CREDENTIALS = "D:\MLOps-Project\.secrets\gcp-key.json"
dvc pull       # fetch data from GCS
dvc repro      # rebuild pipeline
dvc push       # upload results
```

### 6.1 When You Modify the Pipeline or Data

```bash
dvc repro
git add dvc.yaml dvc.lock
git commit -m "Update DVC pipeline or data sources"
dvc push
git push
```

### 6.2 Remote Storage Details

```text
GCS Bucket: gs://mlops-project-dvc
GCP Project ID: break-the-bot
```

## 7. Bias Detection & Mitigation Document

Located in /documents/bias_detection_mitigation.md — explains bias definition, detection via data slicing, mitigation strategies, and fairness calibration.

## 8. DAG Execution Timeline (Gannt Chart Overview)

- The DAG starts with dvc_pull, which is the longest-running task (~15s) since it fetches tracked data from remote storage.
- Set up tasks like ensure_dirs and ensure_config complete quickly (a few seconds each).
- preprocess_input_csv and validate_output are moderate in duration, taking several seconds depending on the dataset size.
- Validation follow-ups (report_validation_status, enforce_validation_policy) run almost instantly after validation completes.
- dvc_push is another longer task (~10–12s) as it uploads outputs and validation reports back to remote storage.
- Notification tasks (email_validation_report, email_success, email_failure) are short and run in parallel depending on pipeline status.


![Airflow DAG Gantt Chart](documents/airflow_gantt.jpeg)

## 9. Tests
- Covers preprocessing, validation, and DAG structure under `MLOps-Project/tests`.
- Run:
  - From repo root: `pytest -q` (uses `pytest.ini` with `testpaths = tests`).
  - Single file: `pytest MLOps-Project/tests/test_preprocess_salad.py -q`.
- Artifacts:
  - GE validation tests write JSON to `data/metrics/stats/<date>/stats.json` and `data/metrics/validation/<date>/anomalies.json`.


