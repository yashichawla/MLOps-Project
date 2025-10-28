# dags/salad_preprocess_dag.py
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import os, subprocess
import json
import logging

from airflow.decorators import dag, task, setup
from airflow.operators.python import get_current_context
from airflow.exceptions import AirflowFailException
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.email import EmailOperator
 
# ✅ NEW: we'll call DVC via BashOperator
from airflow.operators.bash import BashOperator
 
from scripts.preprocess_salad import run_preprocessing
from scripts.validator import run_validation

logger = logging.getLogger(__name__)
 
# repo layout
REPO_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = DATA_DIR / "processed"
CFG_DIR = REPO_ROOT / "config"
 
# allow override via env/Variables if you want
CONFIG_PATH = Path(os.environ.get("SALAD_CONFIG_PATH", CFG_DIR / "data_sources.json"))
OUTPUT_PATH = Path(os.environ.get("SALAD_OUTPUT_PATH", OUT_DIR / "processed_data.csv"))
 
# Airflow Variables (with defaults)
TEST_MODE = Variable.get("TEST_MODE", default_var="false").lower() == "true"
TEST_CSV_PATH = Variable.get(
    "TEST_CSV_PATH", default_var=str(DATA_DIR / "test_validation" / "test.csv")
)
 
DEFAULT_CONFIG = {
    "data_sources": [
        {
            "type": "hf",
            "name": "OpenSafetyLab/Salad-Data",
            "config": "attack_enhanced_set",
            "split": "train",
        }
    ]
}
 
# ✅ NEW: tell DVC which paths to track/push after a successful run.
# Adjust to include any directories/files your pipeline updates and you want versioned.
# DVC_TRACK_PATHS = [
#     str(OUT_DIR)               # data/processed/
#     # str(AIRFLOW_REPORTS_DIR),   # airflow_artifacts/reports/
# ]
DVC_TRACK_PATHS = [
    "data/processed",
    "airflow_artifacts/reports",
]
 
 
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
 
    # ✅ NEW: Pull the latest versioned inputs from remote at the very start.
    dvc_pull = BashOperator(
        task_id="dvc_pull",
        env={
            "GOOGLE_APPLICATION_CREDENTIALS": "/opt/airflow/secrets/gcp-key.json",
            "DVC_NO_ANALYTICS": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        },
        bash_command=f'''
            set -euo pipefail
            cd "{REPO_ROOT}"

            # Ensure DVC is available
            python -m pip install --no-cache-dir -q "dvc[gs]" gcsfs google-cloud-storage

            # 0) Backup any conflicting local file that would block checkout (keeps your work!)
            if [ -f "data/processed/processed_data.csv" ] && ! python -m dvc ls . >/dev/null 2>&1; then
                : # (ls above is just to warm up dvc)
            fi

            if [ -f "data/processed/processed_data.csv" ]; then
                TS=$(date +%Y%m%d_%H%M%S)
                mkdir -p "data/processed/.backup"
                cp -f "data/processed/processed_data.csv" "data/processed/.backup/processed_data.csv.$TS" || true
                echo "[info] Backed up local data/processed/processed_data.csv -> data/processed/.backup/processed_data.csv.$TS"
            fi

            # 1) Optional: clear any hardcoded credentialpath so GOOGLE_APPLICATION_CREDENTIALS is used
            if python -m dvc config --name remote.gcsremote.credentialpath >/dev/null 2>&1; then
                python -m dvc config --local --unset remote.gcsremote.credentialpath || true
            fi

            # 2) Force checkout and pull so DVC can overwrite local files with tracked versions
            python -m dvc checkout -f || true
            python -m dvc pull -v -f
        ''',
    )

  
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
            DATA_DIR,
            OUT_DIR,
            CFG_DIR,
            DATA_DIR / "test_validation",
        )
        return {
            "config_path": str(CONFIG_PATH),
            "output_path": str(OUTPUT_PATH),
            "test_csv_path": str(TEST_CSV_PATH),
        }
 
 
    @task
    def ensure_config(paths: dict) -> str:
        cfg_path = Path(paths["config_path"])
        if not cfg_path.exists():
            with open(cfg_path, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            logger.info("Created default config at %s", cfg_path)
        else:
            logger.info("Using existing config at %s", cfg_path)
        try:
            with open(cfg_path, "r") as f:
                cfg = json.load(f)
            if (
                "data_sources" not in cfg
                or not isinstance(cfg["data_sources"], list)
                or len(cfg["data_sources"]) == 0
            ):
                raise AirflowFailException(
                    "Config invalid: 'data_sources' is missing or empty."
                )
        except Exception as e:
            raise AirflowFailException(f"Failed to read/validate config: {e}")
        return str(cfg_path)
 
    @task
    def select_input_csv(paths_and_cfg: tuple[str, str]) -> str:
        cfg_path, out_path = paths_and_cfg
        if TEST_MODE:
            logger.warning(
                "TEST_MODE is ON — skipping preprocessing. Using test CSV: %s",
                TEST_CSV_PATH,
            )
            test_p = Path(TEST_CSV_PATH)
            if not test_p.exists():
                logger.error("Test CSV does not exist: %s", test_p)
                raise AirflowFailException(f"Missing TEST_CSV_PATH: {test_p}")
            return str(test_p)
 
        logger.info(
            "TEST_MODE is OFF — running preprocessing with config=%s -> %s",
            cfg_path, out_path,
        )
        run_preprocessing(config_path=cfg_path, save_path=out_path)
        if not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
            raise AirflowFailException(f"Expected output not found or empty: {out_path}")
        logger.info("Preprocessing complete. Output at %s", out_path)
        return out_path
 
    @task
    def validate_output(out_csv_path: str) -> dict:
        """
        Single source of truth:
        1) Run pandas validator -> returns metrics dict for emails/gating.
        2) Sidecar: run GE baseline (if missing) and GE validate to emit extra artifacts.
            GE outputs never gate the DAG; failures are logged but ignored here.
        """

        # 1) Run pandas validation (this drives emails + gating)
        allowed_env = os.getenv("ALLOWED_CATEGORIES")
        allowed_categories = [c.strip() for c in allowed_env.split(",")] if allowed_env else None
        ds_nodash = get_current_context()["ds_nodash"]
        metrics = run_validation(
            input_csv=out_csv_path,
            metrics_root=str(METRICS_DIR),   # keep your existing METRICS_DIR
            date_str=ds_nodash,
            allowed_categories=allowed_categories,
        )

        # 2) Sidecar GE artifacts (optional, non-blocking)
        try:
            if not BASELINE_SCHEMA.exists():
                subprocess.run(
                    ["python", str(SCRIPT_GE), "baseline", "--input", out_csv_path, "--date", ds_nodash],
                    check=False,
                )
            subprocess.run(
                ["python", str(SCRIPT_GE), "validate", "--input", out_csv_path,
                "--baseline_schema", str(BASELINE_SCHEMA), "--date", ds_nodash],
                check=False,
            )
        except Exception as e:
            logger.warning("GE sidecar failed (non-blocking): %s", e)

        # Always return metrics for downstream report/enforce/email
        return metrics
 
    @task(trigger_rule=TriggerRule.ALL_DONE)
    def report_validation_status(metrics: dict | None) -> None:
        if not metrics:
            logger.error("Validation metrics missing (task likely errored). Check logs.")
            return
        hard = metrics.get("hard_fail") or []
        soft = metrics.get("soft_warn") or []
        if hard:
            logger.error("Validation HARD FAIL: %s", hard)
        elif soft:
            logger.warning("Validation passed with warnings: %s", soft)
        else:
            logger.info("Validation passed without issues: %s", metrics)
 
    @task
    def enforce_validation_policy(metrics: dict | None) -> None:
        """
        Gate the DAG on HARD failures only; keep soft warnings informational.
        """
        if not metrics:
            raise AirflowFailException(
                "Validation metrics missing; validation may have crashed."
            )
        hard = metrics.get("hard_fail") or []
        if hard:
            raise AirflowFailException(
                f"Validation hard-failed. See reports: {metrics.get('report_paths')}"
            )
 
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
            "{{ (ti.xcom_pull(task_ids='validate_output')['report_paths'] or [])[0] if ti.xcom_pull(task_ids='validate_output') else '' }}"
        ],
        trigger_rule=TriggerRule.ALL_DONE,
    )
 
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
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )
 
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
        trigger_rule=TriggerRule.ONE_FAILED,
    )
 
    paths = ensure_dirs()
    cfg = ensure_config(paths)
    selected_csv = select_input_csv((cfg, str(OUTPUT_PATH)))
    validate_task = validate_output(selected_csv)
    report_task = report_validation_status(validate_task)
    enforce_task = enforce_validation_policy(validate_task)
 
    SCRIPT_GE = REPO_ROOT / "scripts" / "ge_runner.py"
    METRICS_DIR = DATA_DIR / "metrics"
    BASELINE_SCHEMA = METRICS_DIR / "schema" / "baseline" / "schema.json"

    # assumes DVC_TRACK_PATHS is defined (list of strings) and REPO_ROOT points to repo root
    dvc_push = BashOperator(
        task_id="dvc_push",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        env={
            "REPO_ROOT": Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1])),  # root of your repo
            "TEST_MODE": str(TEST_MODE).lower(),
            "GOOGLE_APPLICATION_CREDENTIALS": "/opt/airflow/secrets/gcp-key.json",
            "DVC_NO_ANALYTICS": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        },
        bash_command="""{% raw %}
    set -euo pipefail
    cd "$REPO_ROOT"

    echo "──────────────────────────────────────────"
    echo "🔎 STEP 1: dvc status (cache/remote delta)"
    python -m dvc status -c -v || true

    echo "──────────────────────────────────────────"
    echo "🚀 STEP 2: dvc push (sync to remote)"
    python -m dvc push -v

    echo "──────────────────────────────────────────"
    echo "✅ STEP 3: Quick listing of processed outputs"
    ls -la data/processed || true

    echo "✅ DVC Push complete"
    {% endraw %}""",
    )

    # ── Orchestration ───────────────────────────────────────────────────────────
    dvc_pull >> paths >> cfg >> selected_csv >> validate_task >> [report_task, enforce_task]
    # validate_task >> report_task
    # enforce_task.set_upstream(validate_task)
 
    # Email report always
    validate_task >> email_validation_report
 
    # On success: push artifacts then success email
    enforce_task >> dvc_push >> email_success
 
    # On any failure in the core path: failure email
    [selected_csv, validate_task, enforce_task] >> email_failure
 
 
dag = salad_preprocess_v1()
