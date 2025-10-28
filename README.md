# Break The Bot - MLOps Project

## 📌 Introduction

Large Language Models (LLMs) are increasingly deployed in real-world applications, but they remain vulnerable to jailbreaks and prompt-injection attacks.  
Our project, **Break The Bot**, aims to build an automated MLOps pipeline for continuous safety evaluation of LLMs.

### System will:

- Preprocess and run adversarial prompts
- Measure **Attack Success Rate (ASR)** and **Refusal Quality**
- Use **LLM-as-a-Judge** for automated scoring
- Store and visualize results on dashboards
- Integrate with **CI/CD pipelines** to block unsafe releases

### Team Members:

1. Anjali Pai
2. Atharv Talnikar
3. Nitya Ravi
4. Rahul Kulkarni
5. Taniksha Datar
6. Yashi Chawla

### Repository Structure

```plaintext
MLOps-Project/
├── dags/                              # Airflow DAGs
│   └── salad_preprocess_dag.py        # Main DAG (preprocessing + single validation + email alerts)
├── scripts/
│   ├── preprocess_salad.py            # Data preprocessing pipeline
│   ├── validator.py                   # Pandas-based data validator (main source of truth)
│   ├── ge_runner.py                   # Optional Great Expectations sidecar (baseline + schema.json)
│   └── utils/                         # Shared helper modules (if any)
├── config/
│   └── data_sources.json              # Config file for multi-source data ingestion
├── data/
│   ├── processed/                     # Output CSV (processed_data.csv)
│   ├── metrics/                       # Stats + validation results (used by Airflow + GE)
│   └── test_validation/               # Test CSVs for test-mode runs
├── airflow_artifacts/
│   └── logs/                          # Mounted Airflow logs
├── docker-compose.yml                 # Airflow + Postgres stack
├── requirements.txt                   # Dev dependencies (includes pandas, airflow, etc.)
├── requirements-docker.txt            # Installed inside Docker containers
├── .env                               # Stores AIRFLOW_SMTP_USER & AIRFLOW_SMTP_PASSWORD
├── setup_airflow.sh                   # One-time DB/user setup
├── start_airflow.sh                   # Start Airflow (webserver + scheduler)
├── stop_airflow.sh                    # Stop Airflow services
└── README.md
```

### Setup Instructions:

1. Clone Repository

```bash
git clone https://github.com/yashichawla/MLOps-Project
cd MLOps-Project
```

2. Create Virtual Environment

```bash
   python -m venv venv
   source venv/bin/activate # On Mac/Linux
   venv\Scripts\activate # On Windows
```

3. Install Dependencies

```bash
   pip install -r requirements.txt
```

4. Run with Docker

```bash
   docker-compose up --build
```

### Features:

- Prompt Generator: Generates adversarial prompts using attacker LLMs.
- Evaluator Service: Runs prompts against target models and logs results.
- Judge Service: Scores responses for safety and refusal quality.
- Dashboards: Grafana visualizations for safety trends, ASR, and alerts.
- CI/CD Integration: Blocks unsafe deployments if safety metrics fail.
- Failure Analysis: Clustering and regression testing of jailbreak cases.

### Key Metrics:

- Attack Success Rate (ASR) - % of successful jailbreaks.
- Refusal Quality - judged clarity and robustness of refusals.
- Coverage Metrics - number and diversity of tested adversarial prompts.

### Project Timeline:

- Week 1-2: Repo setup, governance policy, seed prompt generation.
- Week 3-4: Prompt generator + evaluator API.
- Week 5-6: Judge API + calibration with human labels.
- Week 7-8: Dashboards, monitoring, failure analysis.
- Week 9-10: CI/CD gates, final validation, and reporting.

### Bias Detection & Mitigation Document

This repository also includes a Bias Detection and Mitigation Report (/documents/bias_detection_mitigation.md).
This document was created specifically for the Data Pipeline assignment submission and explains:

- What "bias" means in the context of this project (LLMs being more vulnerable to certain adversarial categories)

- How we plan to detect bias using data slicing (category-wise performance evaluation)

- Future integration of bias analysis into the LLM evaluation pipeline

- Possible mitigation strategies such as rebalancing prompts, fairness-aware evaluation, and score calibration

### Setting up and running airflow w Docker (recommended)

```bash
# Navigate to project root
cd MLOps-Project

# (Optional but recommended) Create logs folder
mkdir -p airflow_artifacts/logs

# Initialize Airflow database + admin user in Docker
docker compose run --rm airflow-init
# Only for setup, after that just need to use compose up

#Start Airflow (Webserver + Scheduler + Postgres)
docker compose up -d webserver scheduler
```

Then open the UI:

📍 http://localhost:8080

👤 Username: admin
🔑 Password: admin

(This can be changed in docker-compose later if needed.)

To stop services but keep the database & logs:

```bash
docker compose down
```

To stop and remove everything (Postgres DB, logs, container volumes):

```bash
docker compose down -v       # removes DB volume
rm -rf airflow_artifacts/logs/*
```

### Airflow Test Mode (Validation-only workfflow)

Use this when you want to skip preprocessing and only validate a CSV.

In the Airflow UI, open Admin → Variables, and set TEST_MODE to true to activate test mode.

### Validation Pipeline Overview

We simplified the DAG to use a **single validation task** that integrates both:
- a **Pandas-based validator** (`scripts/validator.py`) — drives hard/soft checks and email reports
- an optional **Great Expectations sidecar** (`scripts/ge_runner.py`) — generates a baseline `schema.json`, per-run stats, and anomalies for auditing

**Key benefits:**
- No duplicate validation tasks
- Works on Python 3.12 (no TFDV)
- Same XCom metrics feed into all Airflow emails

#### DAG flow 

preprocess_input_csv
└── validate_output # runs validator + GE sidecar
      ├── report_validation_status (logs)
      ├── enforce_validation_policy (fails on hard errors)
   ├── email_validation_report (always)
            ├── email_success (if all pass)
            └── email_failure (if any fail)


#### What `validator.py` checks

| Level       | Check                                                                             | Severity  |
|-------------|-----------------------------------------------------------------------------------|-----------|
| File        | File must exist and ≥ 50 rows                                                     | Hard Fail |
| Schema      | Columns `prompt`, `category`, `prompt_id`, `text_length`, `size_label` must exist | Hard Fail |
| Prompt      | Null or empty prompts                                                             | Hard Fail |
| Prompt      | Duplicate prompts                                                                 | Soft Warn |
| Category    | Not in allowed list (if ALLOWED_CATEGORIES set)                                   | Hard Fail |
| Category    | "Unknown" fraction > 0.30                                                         | Soft Warn |
| Text Length | Record min/max for info                                                           | Info      |
| Text Length | Max > 8000                                                                        | Soft Warn |
| Size Label  | Mismatch with expected S/M/L bin                                                  | Soft Warn |

Outputs per run:
- data/metrics/stats/YYYYMMDD/stats.json
- data/metrics/validation/YYYYMMDD/anomalies.json

### ⚙️ Optional: Great Expectations Sidecar

Inside `validate_output`, we also invoke:
- `ge_runner.py baseline` → creates `data/metrics/schema/baseline/schema.json` if missing  
- `ge_runner.py validate` → writes additional `stats.json` and `anomalies.json`

These artifacts are for audit only and **do not affect pass/fail**.

### SMTP Setup (Gmail)

Create .env file in project root:

```in
AIRFLOW_SMTP_USER=your_email@gmail.com
AIRFLOW_SMTP_PASSWORD=your_gmail_app_password   # Not normal password
```

## 📧 Email Notifications (automatic)

The DAG now uses the unified validator’s XCom output for all emails:

| Trigger | Email | Contents |
|----------|--------|----------|
| Always | **Validation Report** | JSON report + anomalies attached |
| On Success | **✅ DAG Succeeded** | Summary of counts and ranges |
| On Failure | **❌ DAG Failed** | Hard-fail reasons + report paths |

Recipients are configured in `salad_preprocess_dag.py` under each `EmailOperator`.

To add more recipients, edit in salad_preprocess_dag.py:

```bash
to=["yashi.chawla1@gmail.com", "...", "..."]
```

#### 🧩 DVC Setup & Usage Guide

This repository uses DVC (Data Version Control) with Google Cloud Storage (GCS) to version processed datasets generated by the Salad preprocessing pipeline.

⚡ Quick Start

```bash
pip install -r requirements.txt
gcloud auth application-default login
dvc pull
```

🗂 Remote Storage Details

```text
GCS Bucket: gs://mlops-project-dvc
GCP Project ID: break-the-bot
```

⚙️ 1. Prerequisites

Install all dependencies from the project requirements (includes DVC + GCS plugins):

```bash
pip install -r requirements.txt
```

🪣 2. Authenticate with Google Cloud (use your own creds)

Each collaborator can authenticate with their own Google account—no shared keys required.

```bash
gcloud auth application-default login
gcloud auth list   # optional: verify your active account
```

DVC will automatically use these credentials to access:

```text
gs://mlops-project-dvc  (project: break-the-bot)
```

🚀 3. Run the Data Pipeline

Rebuild and version the processed dataset:

```bash
dvc repro
```

This executes:

scripts/preprocess_salad.py
→ produces data/processed/processed_data.csv

Push the artifact to GCS:

```bash
dvc push
```

📥 4. Pull Data on Any Machine

```bash
git pull
pip install -r requirements.txt
gcloud auth application-default login
dvc pull
```

🔄 5. When You Modify the Pipeline or Data

```bash
dvc repro
git add dvc.yaml dvc.lock
git commit -m "Update DVC pipeline or data sources"
dvc push
git push
```

⚠️ Notes

data/processed/processed_data.csv is tracked by DVC, not Git.

If you see:

output 'data/processed/processed_data.csv' is already tracked by SCM

fix with:

```bash
git rm --cached data/processed/processed_data.csv
git commit -m "Untrack processed_data.csv (DVC-managed)"
```

✅ Verify

```bash
dvc status   # should show: Data and pipelines are up to date.
```
### 🧩 Data Validation Notes

- The DAG performs **in-place validation** on `data/processed/processed_data.csv`.
- Validation artifacts are versioned under `data/metrics/` and can be tracked via DVC if desired.
- You can optionally run `python scripts/ge_runner.py baseline` manually to regenerate a new baseline schema.
