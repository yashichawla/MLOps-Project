# dags/salad_preprocess_dag.py
"""
SALAD preprocessing DAG (with Test Mode)

Normal mode:
  - loads multi-source config
  - preprocesses to data/processed/processed_data.csv
  - validates and writes reports

Test Mode (Airflow Variable TEST_MODE=true):
  - skips preprocessing
  - validates a separate test CSV at TEST_CSV_PATH
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import logging
from airflow.decorators import dag, task, setup
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.email import EmailOperator
from scripts.preprocess_salad import run_preprocessing
from scripts.validate_salad import validate_output_csv

logger = logging.getLogger(__name__)

# repo layout
REPO_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = DATA_DIR / "processed"
CFG_DIR   = REPO_ROOT / "config"

# allow override via env/Variables if you want
CONFIG_PATH = Path(os.environ.get("SALAD_CONFIG_PATH", CFG_DIR / "data_sources.json"))
OUTPUT_PATH = Path(os.environ.get("SALAD_OUTPUT_PATH", OUT_DIR / "processed_data.csv"))

# Airflow Variables (with defaults)
TEST_MODE      = (Variable.get("TEST_MODE", default_var="false").lower() == "true")
TEST_CSV_PATH  = Variable.get("TEST_CSV_PATH", default_var=str(DATA_DIR / "test_validation" / "test.csv"))

DEFAULT_CONFIG = {
    "data_sources": [
        {
            "type": "hf",
            "name": "OpenSafetyLab/Salad-Data",
            "config": "attack_enhanced_set",
            "split": "train"
        }
        # add more sources here (csv/json) via config
    ]
}

@dag(
    dag_id="salad_preprocess_v1",
    description="Unified preprocessing for Salad-Data + other sources via config (with Test Mode).",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "owner": "airflow",
    },
    tags=["salad", "preprocessing", "v1", "mlops", "testmode"],
)
def salad_preprocess_v1():

    @setup
    @task
    def ensure_dirs() -> dict:
        """Make sure data/, data/processed/, data/test_validation/, config/ exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "validation_reports").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "test_validation").mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CFG_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Dirs ensured: DATA_DIR=%s OUT_DIR=%s CFG_DIR=%s TEST_DIR=%s",
            DATA_DIR, OUT_DIR, CFG_DIR, DATA_DIR / "test_validation"
        )
        return {
            "config_path": str(CONFIG_PATH),
            "output_path": str(OUTPUT_PATH),
            "test_csv_path": str(TEST_CSV_PATH),
        }

    @task
    def ensure_config(paths: dict) -> str:
        """
        Create a minimal config if none exists.
        Returns absolute path to the config file to use.
        """
        cfg_path = Path(paths["config_path"])
        if not cfg_path.exists():
            with open(cfg_path, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            logger.info("Created default config at %s", cfg_path)
        else:
            logger.info("Using existing config at %s", cfg_path)

        # quick sanity check
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            if "data_sources" not in cfg or not isinstance(cfg["data_sources"], list) or len(cfg["data_sources"]) == 0:
                raise AirflowFailException("Config invalid: 'data_sources' is missing or empty.")
        except Exception as e:
            raise AirflowFailException(f"Failed to read/validate config: {e}")

        return str(cfg_path)

    @task
    def select_input_csv(paths_and_cfg: tuple[str, str]) -> str:
        """
        If TEST_MODE is true, return TEST_CSV_PATH (skip preprocessing).
        Otherwise run preprocessing and return OUTPUT_PATH.
        """
        cfg_path, out_path = paths_and_cfg
        if TEST_MODE:
            logger.warning("TEST_MODE is ON — skipping preprocessing. Using test CSV: %s", TEST_CSV_PATH)
            # Ensure file exists (warn if not)
            test_p = Path(TEST_CSV_PATH)
            if not test_p.exists():
                logger.error("Test CSV does not exist: %s", test_p)
                raise AirflowFailException(f"Missing TEST_CSV_PATH: {test_p}")
            return str(test_p)

        # Normal run: preprocess
        logger.info("TEST_MODE is OFF — running preprocessing with config=%s -> %s", cfg_path, out_path)
        run_preprocessing(config_path=cfg_path, save_path=out_path)
        if not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
            raise AirflowFailException(f"Expected output not found or empty: {out_path}")
        logger.info("Preprocessing complete. Output at %s", out_path)
        return out_path

    @task
    def validate_output(out_csv_path: str) -> dict:
        report_dirs = [
            REPO_ROOT / "airflow_artifacts" / "reports",
            REPO_ROOT / "data" / "validation_reports",
        ]
        metrics = validate_output_csv(out_csv_path, report_dirs)
        # DO NOT raise here; always return metrics so downstream can report/alert
        return metrics


    @task(trigger_rule=TriggerRule.ALL_DONE)
    def report_validation_status(metrics: dict | None) -> None:
        if not metrics:
            logger.error("Validation metrics missing (task likely errored). Check logs.")
            return
        if metrics.get("soft_warn"):
            logger.warning("Validation passed with warnings: %s", metrics)
        else:
            logger.info("Validation passed without issues: %s", metrics)

    @task
    def enforce_validation_policy(metrics: dict | None) -> None:
        if not metrics:
            # if validation crashed before producing metrics, fail cleanly
            raise AirflowFailException("Validation metrics missing; validation may have crashed.")
        if metrics.get("hard_fail"):
            raise AirflowFailException(
                f"Validation hard-failed. See reports: {metrics.get('report_paths')}"
            )
        
    # 1) Always email the validation report (runs even if other tasks fail)
    email_validation_report = EmailOperator(
        task_id="email_validation_report",
        to=["yashi.chawla1@gmail.com", "chawla.y@northeastern.edu"],
        subject="[Airflow][{{ dag.dag_id }}][{{ ds }}] Validation Report",
        html_content="""
            <h3>Validation Report for {{ dag.dag_id }}</h3>
            <p><b>Run:</b> {{ run_id }} | <b>Execution date:</b> {{ ds }}</p>
            <p><b>Selected CSV:</b> {{ ti.xcom_pull(task_ids='select_input_csv') }}</p>
            <p><b>Hard Fail:</b> {{ ti.xcom_pull(task_ids='validate_output')['hard_fail'] if ti.xcom_pull(task_ids='validate_output') else 'n/a' }}</p>
            <p><b>Soft Warn:</b> {{ ti.xcom_pull(task_ids='validate_output')['soft_warn'] if ti.xcom_pull(task_ids='validate_output') else 'n/a' }}</p>
            <p>Reports are attached (if available). Locations recorded in XCom:
            <br><code>{{ ti.xcom_pull(task_ids='validate_output')['report_paths'] if ti.xcom_pull(task_ids='validate_output') else [] }}</code></p>
        """,
        files=[
            # Attach the first report (if exists)
            "{{ (ti.xcom_pull(task_ids='validate_output')['report_paths'] or [])[0] if ti.xcom_pull(task_ids='validate_output') else '' }}"
        ],
        trigger_rule=TriggerRule.ALL_DONE,  # send even if upstream failed
    )

    # 2) Email success summary (only if everything succeeded)
    email_success = EmailOperator(
        task_id="email_success",
        to=["yashi.chawla1@gmail.com", "chawla.y@northeastern.edu"],
        subject="[Airflow][{{ dag.dag_id }}][{{ ds }}] ✅ DAG Succeeded",
        html_content="""
            <h3>DAG Succeeded: {{ dag.dag_id }}</h3>
            <p><b>Run:</b> {{ run_id }}</p>
            <p><b>Selected CSV:</b> {{ ti.xcom_pull(task_ids='select_input_csv') }}</p>
            {% set m = ti.xcom_pull(task_ids='validate_output') %}
            {% if m %}
            <p><b>Rows:</b> {{ m['row_count'] }},
                <b>Null Prompts:</b> {{ m['null_prompt_count'] }},
                <b>Dups:</b> {{ m['dup_prompt_count'] }},
                <b>Unknown rate:</b> {{ '%.3f'|format(m['unknown_category_rate']) }},
                <b>text_length min/max:</b> {{ m['text_len_min'] }}/{{ m['text_len_max'] }}</p>
            {% endif %}
            <p>Great job! ✔</p>
        """,
        trigger_rule=TriggerRule.ALL_SUCCESS,  # only when everything upstream passed
    )

    # 3) Email failure summary (if anything failed)
    email_failure = EmailOperator(
        task_id="email_failure",
        to=["yashi.chawla1@gmail.com", "chawla.y@northeastern.edu"],
        subject="[Airflow][{{ dag.dag_id }}][{{ ds }}] ❌ DAG Failed",
        html_content="""
            <h3>DAG Failed: {{ dag.dag_id }}</h3>
            <p><b>Run:</b> {{ run_id }}</p>
            <p>One or more tasks failed. Check Airflow logs/UI.</p>
            {% set m = ti.xcom_pull(task_ids='validate_output') %}
            {% if m %}
            <p><b>Validation summary:</b>
                rows={{ m['row_count'] }},
                nulls={{ m['null_prompt_count'] }},
                dups={{ m['dup_prompt_count'] }},
                unknown_rate={{ '%.3f'|format(m['unknown_category_rate']) }},
                text_len=[{{ m['text_len_min'] }},{{ m['text_len_max'] }}],
                mismatches={{ m['size_label_mismatch_count'] }},
                hard_fail={{ m['hard_fail'] }},
                soft_warn={{ m['soft_warn'] }}</p>
            <p><b>Report:</b> {{ m['report_paths'] }}</p>
            {% else %}
            <p>No metrics available (validation may have crashed before reporting).</p>
            {% endif %}
            <p><b>Tip:</b> In the Airflow UI, open the failed task's “Log” for details.</p>
        """,
        trigger_rule=TriggerRule.ONE_FAILED,  # fire if any upstream failed
    )

    paths = ensure_dirs()
    cfg   = ensure_config(paths)

    # Task instances (XComArg objects)
    selected_csv      = select_input_csv((cfg, str(OUTPUT_PATH)))
    validate_task     = validate_output(selected_csv)               # returns metrics dict
    report_task       = report_validation_status(validate_task)     # logs
    enforce_task      = enforce_validation_policy(validate_task)    # raises on hard_fail

    # Email report should run regardless of success/failure
    report_task >> email_validation_report
    enforce_task >> email_validation_report

    # Success email only if everything passed
    enforce_task >> email_success

    # Failure email if anything failed in the core path
    [selected_csv, validate_task, report_task, enforce_task] >> email_failure


dag = salad_preprocess_v1()