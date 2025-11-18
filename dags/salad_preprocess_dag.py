# dags/salad_preprocess_dag.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os, subprocess
import json
import logging

from airflow.decorators import dag, task, setup
from airflow.operators.python import get_current_context, PythonOperator
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.models import Variable
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.email import EmailOperator
from airflow.utils.email import send_email_smtp

from airflow.operators.bash import BashOperator

from scripts.preprocess_salad import run_preprocessing

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

# Identify which DVC paths to track/push after a successful run.
# Adjust to include any directories/files your pipeline updates and you want versioned.
# DVC_TRACK_PATHS = [
#     str(OUT_DIR)               # data/processed/
#     # str(AIRFLOW_REPORTS_DIR),   # airflow_artifacts/reports/
# ]
DVC_TRACK_PATHS = [
    "data/processed",
    "airflow_artifacts/reports",
]


def send_email_with_conditional_files(
    to: list[str],
    subject: str,
    html_content: str,
    file_paths: list[str] | None = None,
) -> None:
    """
    Helper function to send emails with conditional file attachments.
    Only attaches files that actually exist.
    """
    try:
        # Filter out non-existent files
        existing_files = []
        if file_paths:
            for file_path in file_paths:
                if file_path and file_path.strip():
                    file_path_obj = Path(file_path)
                    # Convert relative paths to absolute using REPO_ROOT
                    if not file_path_obj.is_absolute():
                        file_path_obj = REPO_ROOT / file_path_obj
                    if file_path_obj.exists() and file_path_obj.is_file():
                        existing_files.append(str(file_path_obj))
                        logger.info(f"Email attachment file found: {file_path_obj}")
                    else:
                        logger.warning(f"Email attachment file does not exist, skipping: {file_path_obj}")
        
        # Send email with only existing files
        send_email_smtp(
            to=to,
            subject=subject,
            html_content=html_content,
            files=existing_files if existing_files else None,
        )
        logger.info(f"Email sent successfully to {to} with {len(existing_files) if existing_files else 0} attachment(s)")
    except Exception as e:
        logger.error(f"Failed to send email: {e}", exc_info=True)
        raise  # Re-raise so the task can handle it


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

    # Pull the latest versioned inputs from remote at the very start.
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

            # DVC packages should be pre-installed via requirements-docker.txt
            # If not available, this will fail and indicate a Docker image build issue

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
        (DATA_DIR / "responses").mkdir(parents=True, exist_ok=True)
        (DATA_DIR / "judge").mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CFG_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Dirs ensured: DATA_DIR=%s OUT_DIR=%s CFG_DIR=%s TEST_DIR=%s RESPONSES_DIR=%s JUDGE_DIR=%s",
            DATA_DIR,
            OUT_DIR,
            CFG_DIR,
            DATA_DIR / "test_validation",
            DATA_DIR / "responses",
            DATA_DIR / "judge",
        )
        return {
            "config_path": str(CONFIG_PATH),
            "output_path": str(OUTPUT_PATH),
            "test_csv_path": str(TEST_CSV_PATH),
        }

    @task
    def ensure_config(paths: dict) -> str:
        """Ensure config file exists and contains a valid data_sources section."""
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
    def preprocess_input_csv(paths_and_cfg: tuple[str, str]) -> str:
        """Run preprocessing if TEST_MODE is off; otherwise return test CSV."""
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

    # Define GE validation paths before validate_output task uses them
    SCRIPT_GE = REPO_ROOT / "scripts" / "ge_runner.py"
    METRICS_DIR = DATA_DIR / "metrics"
    BASELINE_SCHEMA = METRICS_DIR / "schema" / "baseline" / "schema.json"

    @task
    def validate_output(out_csv_path: str) -> dict:
        """
        Run Great Expectations validation (compulsory).
        GE baseline and validation are required; failures will cause DAG to fail.
        Returns metrics dict for emails/gating.
        """

        # 1) Run GE baseline (if missing) and GE validate (compulsory)
        ds_nodash = get_current_context()["ds_nodash"]
        anomalies_path = METRICS_DIR / "validation" / ds_nodash / "anomalies.json"
        stats_path = METRICS_DIR / "stats" / ds_nodash / "stats.json"
        
        def _create_fallback_anomalies(error_type: str, error_message: str, exit_code: int | None = None) -> None:
            """Create a fallback anomalies.json when validation fails before GE script creates it."""
            if not anomalies_path.exists():
                fallback_anomalies = {
                    "hard_fail": [f"{error_type}: {error_message}"],
                    "soft_warn": [],
                    "metadata": {
                        "validation_timestamp": datetime.now(timezone.utc).isoformat(),
                        "validation_source": "airflow_fallback",
                        "error_type": error_type,
                        "exit_code": exit_code,
                    },
                    "summary": {
                        "validation_status": "hard_fail",
                        "total_rows": 0,
                        "hard_fail_count": 1,
                        "soft_warn_count": 0,
                    }
                }
                anomalies_path.parent.mkdir(parents=True, exist_ok=True)
                with open(anomalies_path, "w") as f:
                    json.dump(fallback_anomalies, f, indent=2)
                logger.warning("Created fallback anomalies.json due to validation failure: %s", error_message)
        
        try:
            if not BASELINE_SCHEMA.exists():
                logger.info("Creating baseline schema at %s", BASELINE_SCHEMA)
                subprocess.run(
                    ["python", str(SCRIPT_GE), "baseline", "--input", out_csv_path, "--date", ds_nodash],
                    check=True,
                    timeout=300,  # 5 minute timeout
                )
            
            logger.info("Running GE validation for %s", out_csv_path)
            res = subprocess.run(
                [
                    "python", str(SCRIPT_GE), "validate", "--input", out_csv_path,
                    "--baseline_schema", str(BASELINE_SCHEMA), "--date", ds_nodash,
                ],
                check=False,  # We check returncode manually to handle validation failures
                timeout=300,  # 5 minute timeout
            )
            
            # GE validation returns:
            # - 0: validation passed
            # - 1: validation hard failed (hard_fail reasons exist) - GE script creates anomalies.json
            # - 2: baseline schema missing or other error - GE script may not create anomalies.json
            if res.returncode == 1:
                # Exit code 1 means validation failed but GE script should have created anomalies.json
                # Check if it exists, if not create fallback
                if not anomalies_path.exists():
                    _create_fallback_anomalies(
                        "ValidationHardFail",
                        "Great Expectations validation failed (hard fail). GE script exited with code 1 but did not create anomalies.json.",
                        exit_code=1
                    )
                raise AirflowFailException(
                    f"Great Expectations validation failed (hard fail). "
                    f"Check validation report at {anomalies_path}"
                )
            elif res.returncode == 2:
                # Exit code 2 means baseline schema missing or other error
                _create_fallback_anomalies(
                    "BaselineSchemaError",
                    f"Baseline schema missing or validation script error. Exit code: {res.returncode}",
                    exit_code=2
                )
                raise AirflowFailException(
                    f"Great Expectations validation returned exit code {res.returncode} (baseline schema missing or other error). "
                    f"Check logs for details."
                )
            elif res.returncode != 0:
                # Unexpected exit code
                _create_fallback_anomalies(
                    "UnexpectedExitCode",
                    f"Great Expectations validation returned unexpected exit code {res.returncode}",
                    exit_code=res.returncode
                )
                raise AirflowFailException(
                    f"Great Expectations validation returned unexpected exit code {res.returncode}. "
                    f"Check logs for details."
                )
        except subprocess.CalledProcessError as e:
            # Subprocess failed (e.g., baseline creation failed)
            _create_fallback_anomalies(
                "SubprocessError",
                f"Great Expectations subprocess failed: {str(e)}",
                exit_code=e.returncode if hasattr(e, 'returncode') else None
            )
            raise AirflowFailException(
                f"Great Expectations validation subprocess failed: {e}. "
                f"Ensure great_expectations==0.18.21 is installed."
            ) from e
        except FileNotFoundError as e:
            # Script or input file not found
            _create_fallback_anomalies(
                "FileNotFoundError",
                f"Required file or script not found: {str(e)}",
                exit_code=None
            )
            raise AirflowFailException(
                f"Great Expectations validation failed: required file or script not found: {e}"
            ) from e
        except Exception as e:
            # Any other exception
            _create_fallback_anomalies(
                "ValidationInvocationError",
                f"Validation invocation failed: {str(e)}",
                exit_code=None
            )
            raise AirflowFailException(
                f"Great Expectations validation invocation failed: {e}"
            ) from e

        # 2) Read GE-produced artifacts as the single source of metrics
        # GE validation must produce these artifacts - fail if they don't exist
        if not stats_path.exists():
            # Create fallback anomalies if stats.json is missing
            if not anomalies_path.exists():
                _create_fallback_anomalies(
                    "MissingArtifacts",
                    f"Great Expectations validation artifacts missing: {stats_path} not found. GE validation must produce stats.json.",
                    exit_code=None
                )
            raise AirflowFailException(
                f"Great Expectations validation artifacts missing: {stats_path} not found. "
                f"GE validation must produce stats.json."
            )
        if not anomalies_path.exists():
            # This should not happen if we've handled all failure cases above, but just in case
            _create_fallback_anomalies(
                "MissingAnomalies",
                f"Great Expectations validation artifacts missing: {anomalies_path} not found. GE validation must produce anomalies.json.",
                exit_code=None
            )
            raise AirflowFailException(
                f"Great Expectations validation artifacts missing: {anomalies_path} not found. "
                f"GE validation must produce anomalies.json."
            )
        
        # Read artifacts - fail if reading fails
        try:
            with open(stats_path) as f:
                s = json.load(f)
            with open(anomalies_path) as f:
                a = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            # If reading fails, create fallback with error info
            if not anomalies_path.exists():
                _create_fallback_anomalies(
                    "ArtifactReadError",
                    f"Failed to read validation artifacts: {str(e)}",
                    exit_code=None
                )
            raise AirflowFailException(
                f"Failed to read validation artifacts: {e}"
            ) from e
        
        metrics = {
            "row_count": s.get("row_count", 0),
            "null_prompt_count": s.get("null_prompt_count"),
            "dup_prompt_count": s.get("dup_prompt_count"),
            "unknown_category_rate": s.get("unknown_category_rate"),
            "text_len_min": s.get("text_len_min"),
            "text_len_max": s.get("text_len_max"),
            "size_label_mismatch_count": s.get("size_label_mismatch_count"),
            "hard_fail": a.get("hard_fail", []),
            "soft_warn": a.get("soft_warn", []),
            "report_paths": [str(anomalies_path), str(stats_path)],
        }
        
        return metrics

    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def report_validation_status(metrics: dict | None) -> None:
        """Log validation outcome (pass, warnings, or hard fail)."""
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
        """Fail the DAG if there are hard validation failures."""
        if not metrics:
            raise AirflowFailException(
                "Validation metrics missing; validation may have crashed."
            )
        hard = metrics.get("hard_fail") or []
        if hard:
            raise AirflowFailException(
                f"Validation hard-failed. See reports: {metrics.get('report_paths')}"
            )

    def send_validation_report_email(**context):
        """Send validation report email with conditional file attachments."""
        try:
            ti = context['ti']
            dag = context['dag']
            ds = context['ds']
            ds_nodash = context.get('ds_nodash')
            run_id = context['run_id']
            
            # Handle missing ds_nodash gracefully
            if not ds_nodash and ds:
                ds_nodash = ds.replace('-', '')
            
            # Get data from XCom
            csv_path = ti.xcom_pull(task_ids='preprocess_input_csv', default=None)
            metrics = ti.xcom_pull(task_ids='validate_output', default=None)
            
            # Determine file path - always check both XCom and fallback
            file_paths = []
            
            # First, try to get path from XCom metrics
            if metrics and metrics.get('report_paths'):
                anomalies_path_from_xcom = metrics['report_paths'][0] if metrics['report_paths'] else None
                if anomalies_path_from_xcom:
                    path_obj = Path(anomalies_path_from_xcom)
                    # Convert to absolute if relative
                    if not path_obj.is_absolute():
                        path_obj = REPO_ROOT / path_obj
                    if path_obj.exists() and path_obj.is_file():
                        file_paths.append(str(path_obj))
                        logger.info(f"Using anomalies.json from XCom: {path_obj}")
            
            # Always check fallback path regardless of whether metrics exist
            if ds_nodash:
                fallback_path = REPO_ROOT / "data" / "metrics" / "validation" / ds_nodash / "anomalies.json"
                if fallback_path.exists() and fallback_path.is_file():
                    fallback_str = str(fallback_path)
                    if fallback_str not in file_paths:  # Avoid duplicates
                        file_paths.append(fallback_str)
                        logger.info(f"Using anomalies.json from fallback path: {fallback_path}")
                else:
                    logger.warning(f"Fallback anomalies.json not found at: {fallback_path}")
            
            # Build HTML content
            html_content = f"""
                <h3>Validation Report for {dag.dag_id}</h3>
                <p><b>Run:</b> {run_id} | <b>Execution date:</b> {ds}</p>
                <p><b>Selected CSV:</b> {csv_path if csv_path else 'N/A'}</p>
            """
            if metrics:
                html_content += f"""
                <p><b>Hard Fail:</b> {metrics.get('hard_fail', [])}</p>
                <p><b>Soft Warn:</b> {metrics.get('soft_warn', [])}</p>
                <p>Reports are attached (if available). Locations recorded in XCom:
                <br><code>{metrics.get('report_paths', [])}</code></p>
                """
            else:
                html_content += """
                <p><b>Warning:</b> Validation metrics not available. Validation task may have failed before producing metrics.</p>
                <p>Check the validation task logs for details.</p>
                <p><b>Note:</b> If anomalies.json was created, it will be attached to this email.</p>
                """
            
            # Log file attachment status
            if file_paths:
                logger.info(f"Attaching {len(file_paths)} file(s) to email: {file_paths}")
            else:
                logger.warning("No anomalies.json file found to attach to email")
            
            # Send email with conditional files
            send_email_with_conditional_files(
                to=["athatalnikar@gmail.com"],
                subject=f"[Airflow][{dag.dag_id}][{ds}] Validation Report",
                html_content=html_content,
                file_paths=file_paths,
            )
            logger.info("Validation report email sent successfully")
        except Exception as e:
            logger.error(f"Failed to send validation report email: {e}", exc_info=True)
            # Don't re-raise - we don't want email failures to fail the DAG
    
    email_validation_report = PythonOperator(
        task_id="email_validation_report",
        python_callable=send_validation_report_email,
        trigger_rule=TriggerRule.ALL_DONE,
        retries=0,  # Disable retries for email tasks
    )

    email_success = EmailOperator(
        task_id="email_success",
        to=["athatalnikar@gmail.com"],
        subject="[Airflow][{{ dag.dag_id }}][{{ ds }}] ✅ DAG Succeeded",
        html_content="""
            <h3>DAG Succeeded: {{ dag.dag_id }}</h3>
            <p><b>Run:</b> {{ run_id }}</p>
            <p><b>Selected CSV:</b> {{ ti.xcom_pull(task_ids='preprocess_input_csv', default_var='N/A') }}</p>
            {% set m = ti.xcom_pull(task_ids='validate_output') %}
            {% if m %}
            <p><b>Data Validation:</b></p>
            <ul>
                <li><b>Rows:</b> {{ m['row_count'] }}</li>
                <li><b>Null Prompts:</b> {{ m['null_prompt_count'] }}</li>
                <li><b>Duplicates:</b> {{ m['dup_prompt_count'] }}</li>
                <li><b>Unknown rate:</b> {{ '%.3f'|format(m['unknown_category_rate']) }}</li>
                <li><b>Text length range:</b> {{ m['text_len_min'] }}/{{ m['text_len_max'] }}</li>
            </ul>
            {% endif %}
            {% set model_gen = ti.xcom_pull(task_ids='generate_model_responses', default_var=None) %}
            {% set model_judge = ti.xcom_pull(task_ids='judge_responses', default_var=None) %}
            {% set model_metrics = ti.xcom_pull(task_ids='compute_additional_metrics', default_var=None) %}
            {% set bias_detection = ti.xcom_pull(task_ids='compute_bias_detection', default_var=None) %}
            <p><b>Model Pipeline Status:</b></p>
            <ul>
                {% if model_gen %}
                <li><b>Model Generation:</b> ✅ Executed successfully</li>
                {% else %}
                <li><b>Model Generation:</b> ⏭️ Skipped (data unchanged)</li>
                {% endif %}
                {% if model_judge %}
                <li><b>Response Judging:</b> ✅ Executed successfully</li>
                {% else %}
                <li><b>Response Judging:</b> ⏭️ Skipped (data unchanged)</li>
                {% endif %}
                {% if model_metrics %}
                <li><b>Additional Metrics:</b> ✅ Executed successfully</li>
                {% else %}
                <li><b>Additional Metrics:</b> ⏭️ Skipped (data unchanged)</li>
                {% endif %}
                {% if bias_detection %}
                <li><b>Bias Detection:</b> ✅ Executed successfully</li>
                {% else %}
                <li><b>Bias Detection:</b> ⏭️ Skipped (data unchanged)</li>
                {% endif %}
            </ul>
            <p>Great job! ✔</p>
        """,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    # ── Context-Specific Failure Email Operators ────────────────────────────────
    
    # Email 1: DVC Pull Failure (Infrastructure/Data Source Issue)
    email_failure_dvc_pull = EmailOperator(
        task_id="email_failure_dvc_pull",
        to=["athatalnikar@gmail.com"],
        subject="[Airflow][{{ dag.dag_id }}][{{ ds }}] ❌ DVC Pull Failed",
        html_content="""
            <h3>DVC Pull Failed: {{ dag.dag_id }}</h3>
            <p><b>Run:</b> {{ run_id }} | <b>Execution date:</b> {{ ds }}</p>
            <p><b>Failed Task:</b> dvc_pull</p>
            <p><b>Issue:</b> Failed to pull data from DVC remote storage.</p>
            <p><b>Possible Causes:</b></p>
            <ul>
                <li>GCP credentials issue (check GOOGLE_APPLICATION_CREDENTIALS)</li>
                <li>DVC remote configuration problem</li>
                <li>Network connectivity issue</li>
                <li>Remote storage bucket access denied</li>
            </ul>
            <p><b>Action Required:</b> Check DVC remote configuration and GCP credentials.</p>
            <p><b>Tip:</b> In the Airflow UI, open the "dvc_pull" task's "Log" for details.
            {% if ti.log_url %}
            <br><code>{{ ti.log_url }}</code>
            {% else %}
            <br>Navigate to the failed task in Airflow UI to view logs.
            {% endif %}
            </p>
        """,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # Email 2: Setup/Config Failure (Configuration Issue)
    email_failure_setup = EmailOperator(
        task_id="email_failure_setup",
        to=["athatalnikar@gmail.com"],
        subject="[Airflow][{{ dag.dag_id }}][{{ ds }}] ❌ Setup/Config Failed",
        html_content="""
            <h3>Setup/Config Failed: {{ dag.dag_id }}</h3>
            <p><b>Run:</b> {{ run_id }} | <b>Execution date:</b> {{ ds }}</p>
            <p><b>Failed Task:</b> ensure_dirs or ensure_config</p>
            <p><b>Issue:</b> Failed during setup or configuration validation.</p>
            {% set cfg_path = ti.xcom_pull(task_ids='ensure_config', default_var=None) %}
            {% if cfg_path %}
            <p><b>Config Path:</b> {{ cfg_path }}</p>
            {% else %}
            <p><b>Config Path:</b> N/A (config task may have failed before producing path)</p>
            {% endif %}
            <p><b>Possible Causes:</b></p>
            <ul>
                <li>Invalid or missing data_sources.json configuration</li>
                <li>Directory creation permission issues</li>
                <li>Config file format errors (invalid JSON)</li>
            </ul>
            <p><b>Action Required:</b> Verify config file and check directory permissions.</p>
            <p><b>Tip:</b> In the Airflow UI, check the failed task's "Log" for details.
            {% if ti.log_url %}
            <br><code>{{ ti.log_url }}</code>
            {% else %}
            <br>Navigate to the failed task in Airflow UI to view logs.
            {% endif %}
            </p>
        """,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # Email 3: Preprocessing Failure (Data Processing Issue)
    email_failure_preprocessing = EmailOperator(
        task_id="email_failure_preprocessing",
        to=["athatalnikar@gmail.com"],
        subject="[Airflow][{{ dag.dag_id }}][{{ ds }}] ❌ Preprocessing Failed",
        html_content="""
            <h3>Preprocessing Failed: {{ dag.dag_id }}</h3>
            <p><b>Run:</b> {{ run_id }} | <b>Execution date:</b> {{ ds }}</p>
            <p><b>Failed Task:</b> preprocess_input_csv</p>
            <p><b>Issue:</b> Failed to preprocess input data.</p>
            {% set cfg_path = ti.xcom_pull(task_ids='ensure_config', default_var=None) %}
            {% if cfg_path %}
            <p><b>Config Used:</b> {{ cfg_path }}</p>
            {% else %}
            <p><b>Config Used:</b> N/A (config may not be available)</p>
            {% endif %}
            <p><b>Possible Causes:</b></p>
            <ul>
                <li>Data source unavailable (HuggingFace dataset download failed)</li>
                <li>Preprocessing script error</li>
                <li>Output file write permission issue</li>
                <li>Empty or invalid output generated</li>
            </ul>
            <p><b>Action Required:</b> Check data source availability and preprocessing script logs.</p>
            <p><b>Tip:</b> In the Airflow UI, open the "preprocess_input_csv" task's "Log" for details.
            {% if ti.log_url %}
            <br><code>{{ ti.log_url }}</code>
            {% else %}
            <br>Navigate to the failed task in Airflow UI to view logs.
            {% endif %}
            </p>
        """,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # Email 4: Validation Failure (Data Quality Issue)
    def send_validation_failure_email(**context):
        """Send validation failure email with conditional file attachments."""
        try:
            ti = context['ti']
            dag = context['dag']
            ds = context['ds']
            ds_nodash = context.get('ds_nodash')
            run_id = context['run_id']
            
            # Handle missing ds_nodash gracefully
            if not ds_nodash and ds:
                ds_nodash = ds.replace('-', '')
            
            # Get data from XCom
            csv_path = ti.xcom_pull(task_ids='preprocess_input_csv', default=None)
            validate_metrics = ti.xcom_pull(task_ids='validate_output', default=None)
            enforce_metrics = ti.xcom_pull(task_ids='enforce_validation_policy', default=None)
            
            # Determine failed task
            if validate_metrics is None:
                failed_task = "validate_output (task failed before producing metrics)"
            elif enforce_metrics is None:
                failed_task = "enforce_validation_policy (validation policy enforcement failed)"
            else:
                failed_task = "validate_output or enforce_validation_policy"
            
            # Determine file path - always check both XCom and fallback
            file_paths = []
            
            # First, try to get path from XCom metrics
            if validate_metrics and validate_metrics.get('report_paths'):
                anomalies_path_from_xcom = validate_metrics['report_paths'][0] if validate_metrics['report_paths'] else None
                if anomalies_path_from_xcom:
                    path_obj = Path(anomalies_path_from_xcom)
                    # Convert to absolute if relative
                    if not path_obj.is_absolute():
                        path_obj = REPO_ROOT / path_obj
                    if path_obj.exists() and path_obj.is_file():
                        file_paths.append(str(path_obj))
                        logger.info(f"Using anomalies.json from XCom: {path_obj}")
            
            # Always check fallback path regardless of whether metrics exist
            if ds_nodash:
                fallback_path = REPO_ROOT / "data" / "metrics" / "validation" / ds_nodash / "anomalies.json"
                if fallback_path.exists() and fallback_path.is_file():
                    fallback_str = str(fallback_path)
                    if fallback_str not in file_paths:  # Avoid duplicates
                        file_paths.append(fallback_str)
                        logger.info(f"Using anomalies.json from fallback path: {fallback_path}")
                else:
                    logger.warning(f"Fallback anomalies.json not found at: {fallback_path}")
            
            # Build HTML content
            html_content = f"""
                <h3>Validation Failed: {dag.dag_id}</h3>
                <p><b>Run:</b> {run_id} | <b>Execution date:</b> {ds}</p>
                <p><b>Failed Task:</b> {failed_task}</p>
                <p><b>Issue:</b> Data validation failed or validation policy enforcement blocked the pipeline.</p>
            """
            if csv_path:
                html_content += f"<p><b>Validated CSV:</b> {csv_path}</p>"
            
            if validate_metrics:
                html_content += f"""
                <p><b>Validation Metrics (before failure):</b></p>
                <ul>
                    <li><b>Rows:</b> {validate_metrics.get('row_count', 'N/A')}</li>
                    <li><b>Null Prompts:</b> {validate_metrics.get('null_prompt_count', 'N/A')}</li>
                    <li><b>Duplicate Prompts:</b> {validate_metrics.get('dup_prompt_count', 'N/A')}</li>
                    <li><b>Unknown Category Rate:</b> {validate_metrics.get('unknown_category_rate', 0):.3f}</li>
                    <li><b>Text Length Range:</b> [{validate_metrics.get('text_len_min', 'N/A')}, {validate_metrics.get('text_len_max', 'N/A')}]</li>
                    <li><b>Size Label Mismatches:</b> {validate_metrics.get('size_label_mismatch_count', 'N/A')}</li>
                </ul>
                <p><b>Hard Failures:</b> {validate_metrics.get('hard_fail', [])}</p>
                <p><b>Soft Warnings:</b> {validate_metrics.get('soft_warn', [])}</p>
                <p><b>Report Paths:</b> {validate_metrics.get('report_paths', [])}</p>
                """
            else:
                html_content += """
                <p><b>Warning:</b> Validation metrics not available. Validation may have crashed before completion.</p>
                <p><b>Action:</b> Check the validate_output task logs in Airflow UI for detailed error information.</p>
                """
            
            html_content += """
                <p><b>Possible Causes:</b></p>
                <ul>
                    <li>Data quality issues (hard_fail conditions met)</li>
                    <li>Great Expectations validation script error</li>
                    <li>Baseline schema missing or corrupted</li>
                    <li>Validation artifacts not generated</li>
                    <li>Subprocess execution failure</li>
                </ul>
                <p><b>Action Required:</b> Review validation reports and fix data quality issues.</p>
                <p><b>Tip:</b> Check validation reports at the paths above, or view the failed task's "Log" in Airflow UI.</p>
            """
            
            # Log file attachment status
            if file_paths:
                logger.info(f"Attaching {len(file_paths)} file(s) to email: {file_paths}")
            else:
                logger.warning("No anomalies.json file found to attach to email")
            
            # Send email with conditional files
            send_email_with_conditional_files(
                to=["athatalnikar@gmail.com"],
                subject=f"[Airflow][{dag.dag_id}][{ds}] ❌ Validation Failed",
                html_content=html_content,
                file_paths=file_paths,
            )
            logger.info("Validation failure email sent successfully")
        except Exception as e:
            logger.error(f"Failed to send validation failure email: {e}", exc_info=True)
            # Don't re-raise - we don't want email failures to fail the DAG
    
    email_failure_validation = PythonOperator(
        task_id="email_failure_validation",
        python_callable=send_validation_failure_email,
        trigger_rule=TriggerRule.ONE_FAILED,
        retries=0,  # Disable retries for email tasks
    )

    # Email 5: Enforce Validation Policy Failure (Policy Enforcement Issue)
    def send_enforce_policy_failure_email(**context):
        """Send enforce policy failure email only if enforce_task actually failed (not skipped)."""
        try:
            ti = context['ti']
            dag = context['dag']
            ds = context['ds']
            ds_nodash = context.get('ds_nodash')
            run_id = context['run_id']
            dag_run = context['dag_run']
            
            # Handle missing ds_nodash gracefully
            if not ds_nodash and ds:
                ds_nodash = ds.replace('-', '')
            
            # Check if enforce_validation_policy actually failed (not skipped)
            enforce_task_instance = dag_run.get_task_instance('enforce_validation_policy')
            
            # Only send email if task actually failed (not skipped or upstream_failed)
            if not enforce_task_instance or enforce_task_instance.state not in ['failed']:
                state = enforce_task_instance.state if enforce_task_instance else 'not found'
                logger.info(
                    f"enforce_validation_policy is in state '{state}', not 'failed'. "
                    f"Skipping task. This is expected when validate_output fails."
                )
                raise AirflowSkipException(f"Upstream task enforce_validation_policy is in state '{state}', not 'failed'. Skipping email.")
            
            # Get data from XCom
            csv_path = ti.xcom_pull(task_ids='preprocess_input_csv', default=None)
            metrics = ti.xcom_pull(task_ids='validate_output', default=None)
            
            # Determine file path - always check both XCom and fallback
            file_paths = []
            
            # First, try to get path from XCom metrics
            if metrics and metrics.get('report_paths'):
                anomalies_path_from_xcom = metrics['report_paths'][0] if metrics['report_paths'] else None
                if anomalies_path_from_xcom:
                    path_obj = Path(anomalies_path_from_xcom)
                    # Convert to absolute if relative
                    if not path_obj.is_absolute():
                        path_obj = REPO_ROOT / path_obj
                    if path_obj.exists() and path_obj.is_file():
                        file_paths.append(str(path_obj))
                        logger.info(f"Using anomalies.json from XCom: {path_obj}")
            
            # Always check fallback path regardless of whether metrics exist
            if ds_nodash:
                fallback_path = REPO_ROOT / "data" / "metrics" / "validation" / ds_nodash / "anomalies.json"
                if fallback_path.exists() and fallback_path.is_file():
                    fallback_str = str(fallback_path)
                    if fallback_str not in file_paths:  # Avoid duplicates
                        file_paths.append(fallback_str)
                        logger.info(f"Using anomalies.json from fallback path: {fallback_path}")
            
            # Build HTML content
            html_content = f"""
                <h3>Validation Policy Enforcement Failed: {dag.dag_id}</h3>
                <p><b>Run:</b> {run_id} | <b>Execution date:</b> {ds}</p>
                <p><b>Failed Task:</b> enforce_validation_policy</p>
                <p><b>Issue:</b> Validation completed but policy enforcement blocked the pipeline due to hard validation failures.</p>
            """
            if csv_path:
                html_content += f"<p><b>Validated CSV:</b> {csv_path}</p>"
            
            if metrics:
                html_content += f"""
                <p><b>Validation Metrics:</b></p>
                <ul>
                    <li><b>Rows:</b> {metrics.get('row_count', 'N/A')}</li>
                    <li><b>Null Prompts:</b> {metrics.get('null_prompt_count', 'N/A')}</li>
                    <li><b>Duplicate Prompts:</b> {metrics.get('dup_prompt_count', 'N/A')}</li>
                    <li><b>Unknown Category Rate:</b> {metrics.get('unknown_category_rate', 0):.3f}</li>
                    <li><b>Text Length Range:</b> [{metrics.get('text_len_min', 'N/A')}, {metrics.get('text_len_max', 'N/A')}]</li>
                    <li><b>Size Label Mismatches:</b> {metrics.get('size_label_mismatch_count', 'N/A')}</li>
                </ul>
                <p><b>Hard Failures (blocking pipeline):</b> {metrics.get('hard_fail', [])}</p>
                <p><b>Soft Warnings:</b> {metrics.get('soft_warn', [])}</p>
                <p><b>Report Paths:</b> {metrics.get('report_paths', [])}</p>
                """
            else:
                html_content += "<p><b>Warning:</b> Validation metrics not available.</p>"
            
            html_content += """
                <p><b>Possible Causes:</b></p>
                <ul>
                    <li>Hard validation failures detected (data quality issues that block pipeline)</li>
                    <li>Validation metrics missing or corrupted</li>
                    <li>Policy enforcement logic error</li>
                </ul>
                <p><b>Action Required:</b> Review validation reports and fix data quality issues. The pipeline was blocked to prevent processing invalid data.</p>
                <p><b>Tip:</b> Check validation reports at the paths above, or view the failed task's "Log" in Airflow UI.</p>
            """
            
            # Log file attachment status
            if file_paths:
                logger.info(f"Attaching {len(file_paths)} file(s) to email: {file_paths}")
            
            # Send email with conditional files
            send_email_with_conditional_files(
                to=["athatalnikar@gmail.com"],
                subject=f"[Airflow][{dag.dag_id}][{ds}] ❌ Validation Policy Enforcement Failed",
                html_content=html_content,
                file_paths=file_paths,
            )
            logger.info("Enforce policy failure email sent successfully")
        except AirflowSkipException:
            # Re-raise skip exceptions so task is properly skipped
            raise
        except Exception as e:
            logger.error(f"Failed to send enforce policy failure email: {e}", exc_info=True)
            # Don't re-raise - we don't want email failures to fail the DAG
    
    email_failure_enforce_policy = PythonOperator(
        task_id="email_failure_enforce_policy",
        python_callable=send_enforce_policy_failure_email,
        trigger_rule=TriggerRule.ALL_DONE,  # Run regardless, but function will skip if upstream wasn't actually failed
        retries=0,  # Disable retries for email tasks
    )

    # Email: DVC Push (validation) Failure
    def send_dvc_push_validation_failure_email(**context):
        """Send DVC push (validation) failure email only if dvc_push_validation actually failed."""
        try:
            ti = context['ti']
            dag = context['dag']
            ds = context['ds']
            run_id = context['run_id']
            dag_run = context['dag_run']
            
            # Check if dvc_push_validation actually failed
            dvc_push_task_instance = dag_run.get_task_instance('dvc_push_validation')
            
            if not dvc_push_task_instance or dvc_push_task_instance.state not in ['failed']:
                state = dvc_push_task_instance.state if dvc_push_task_instance else 'not found'
                logger.info(f"dvc_push_validation is in state '{state}', not 'failed'. Skipping email.")
                raise AirflowSkipException(f"Upstream task dvc_push_validation is in state '{state}', not 'failed'. Skipping email.")
            
            csv_path = ti.xcom_pull(task_ids='preprocess_input_csv', default=None)
            metrics = ti.xcom_pull(task_ids='validate_output', default=None)
            
            html_content = f"""
                <h3>DVC Push (Validation) Failed: {dag.dag_id}</h3>
                <p><b>Run:</b> {run_id} | <b>Execution date:</b> {ds}</p>
                <p><b>Failed Task:</b> dvc_push_validation</p>
                <p><b>Issue:</b> Failed to push validated data to DVC remote storage.</p>
            """
            if csv_path:
                html_content += f"<p><b>Validated CSV:</b> {csv_path}</p>"
            
            if metrics:
                html_content += f"""
                <p><b>Validation Summary:</b></p>
                <ul>
                    <li><b>Rows:</b> {metrics.get('row_count', 'N/A')}</li>
                    <li><b>Validation Status:</b> Passed</li>
                </ul>
                """
            
            html_content += """
                <p><b>Possible Causes:</b></p>
                <ul>
                    <li>GCP credentials issue</li>
                    <li>DVC remote storage quota exceeded</li>
                    <li>Network connectivity issue during push</li>
                    <li>Permission denied on remote bucket</li>
                </ul>
                <p><b>Action Required:</b> Data is validated locally but not versioned remotely. Check DVC remote configuration and retry push manually if needed.</p>
                <p><b>Tip:</b> In the Airflow UI, open the "dvc_push_validation" task's "Log" for details.</p>
            """
            
            send_email_with_conditional_files(
                to=["athatalnikar@gmail.com"],
                subject=f"[Airflow][{dag.dag_id}][{ds}] ❌ DVC Push (Validation) Failed",
                html_content=html_content,
                file_paths=None,
            )
            logger.info("DVC push (validation) failure email sent successfully")
        except AirflowSkipException:
            raise
        except Exception as e:
            logger.error(f"Failed to send DVC push (validation) failure email: {e}", exc_info=True)
    
    email_failure_dvc_push_validation = PythonOperator(
        task_id="email_failure_dvc_push_validation",
        python_callable=send_dvc_push_validation_failure_email,
        trigger_rule=TriggerRule.ALL_DONE,
        retries=0,
    )

    # Email: Model Generation Failure
    def send_model_generation_failure_email(**context):
        """Send model generation failure email only if generate_model_responses actually failed (not skipped)."""
        try:
            ti = context['ti']
            dag = context['dag']
            ds = context['ds']
            run_id = context['run_id']
            dag_run = context['dag_run']
            
            # Check if generate_model_responses actually failed (not skipped)
            model_gen_task_instance = dag_run.get_task_instance('generate_model_responses')
            
            # Only send email if task actually failed (not skipped or upstream_failed)
            if not model_gen_task_instance or model_gen_task_instance.state not in ['failed']:
                state = model_gen_task_instance.state if model_gen_task_instance else 'not found'
                logger.info(
                    f"generate_model_responses is in state '{state}', not 'failed'. "
                    f"Skipping email. This is expected when upstream tasks fail or are skipped."
                )
                raise AirflowSkipException(f"Upstream task generate_model_responses is in state '{state}', not 'failed'. Skipping email.")
            
            # Get data from XCom
            csv_path = ti.xcom_pull(task_ids='preprocess_input_csv', default=None)
            
            # Build HTML content
            html_content = f"""
                <h3>Model Generation Failed: {dag.dag_id}</h3>
                <p><b>Run:</b> {run_id} | <b>Execution date:</b> {ds}</p>
                <p><b>Failed Task:</b> generate_model_responses</p>
                <p><b>Issue:</b> Failed to generate model responses for prompts.</p>
            """
            if csv_path:
                html_content += f"<p><b>Input CSV:</b> {csv_path}</p>"
            
            html_content += """
                <p><b>Possible Causes:</b></p>
                <ul>
                    <li>HuggingFace API token missing or invalid (check HF_TOKEN environment variable)</li>
                    <li>Model API rate limit exceeded</li>
                    <li>Network connectivity issue</li>
                    <li>Script execution error</li>
                </ul>
                <p><b>Action Required:</b> Check model generation script logs and verify API credentials.</p>
                <p><b>Tip:</b> In the Airflow UI, open the "generate_model_responses" task's "Log" for details.</p>
            """
            
            # Send email
            send_email_with_conditional_files(
                to=["athatalnikar@gmail.com"],
                subject=f"[Airflow][{dag.dag_id}][{ds}] ❌ Model Generation Failed",
                html_content=html_content,
                file_paths=None,
            )
            logger.info("Model generation failure email sent successfully")
        except AirflowSkipException:
            # Re-raise skip exceptions so task is properly skipped
            raise
        except Exception as e:
            logger.error(f"Failed to send model generation failure email: {e}", exc_info=True)
            # Don't re-raise - we don't want email failures to fail the DAG
    
    email_failure_model_generation = PythonOperator(
        task_id="email_failure_model_generation",
        python_callable=send_model_generation_failure_email,
        trigger_rule=TriggerRule.ALL_DONE,  # Run regardless, but function will skip if upstream wasn't actually failed
        retries=0,  # Disable retries for email tasks
    )

    # Email: Model Judging Failure
    email_failure_model_judging = EmailOperator(
        task_id="email_failure_model_judging",
        to=["athatalnikar@gmail.com"],
        subject="[Airflow][{{ dag.dag_id }}][{{ ds }}] ❌ Model Judging Failed",
        html_content="""
            <h3>Model Judging Failed: {{ dag.dag_id }}</h3>
            <p><b>Run:</b> {{ run_id }} | <b>Execution date:</b> {{ ds }}</p>
            <p><b>Failed Task:</b> judge_responses</p>
            <p><b>Issue:</b> Failed to judge model responses using judge LLM.</p>
            <p><b>Possible Causes:</b></p>
            <ul>
                <li>Groq API key missing or invalid (check GROQ_API_KEY environment variable)</li>
                <li>Judge LLM API rate limit exceeded</li>
                <li>Network connectivity issue</li>
                <li>Script execution error</li>
            </ul>
            <p><b>Action Required:</b> Check judge responses script logs and verify API credentials.</p>
            <p><b>Tip:</b> In the Airflow UI, open the "judge_responses" task's "Log" for details.</p>
        """,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # Email: Additional Metrics Failure
    def send_model_metrics_failure_email(**context):
        """Send additional metrics failure email only if compute_additional_metrics actually failed (not skipped)."""
        try:
            ti = context['ti']
            dag = context['dag']
            ds = context['ds']
            run_id = context['run_id']
            dag_run = context['dag_run']
            
            # Check if compute_additional_metrics actually failed (not skipped)
            metrics_task_instance = dag_run.get_task_instance('compute_additional_metrics')
            
            # Only send email if task actually failed (not skipped or upstream_failed)
            if not metrics_task_instance or metrics_task_instance.state not in ['failed']:
                state = metrics_task_instance.state if metrics_task_instance else 'not found'
                logger.info(
                    f"compute_additional_metrics is in state '{state}', not 'failed'. "
                    f"Skipping email. This is expected when upstream tasks fail or are skipped."
                )
                raise AirflowSkipException(f"Upstream task compute_additional_metrics is in state '{state}', not 'failed'. Skipping email.")
            
            # Get data from XCom
            csv_path = ti.xcom_pull(task_ids='preprocess_input_csv', default=None)
            model_gen_result = ti.xcom_pull(task_ids='generate_model_responses', default=None)
            model_judge_result = ti.xcom_pull(task_ids='judge_responses', default=None)
            
            # Build HTML content
            html_content = f"""
                <h3>Additional Metrics Computation Failed: {dag.dag_id}</h3>
                <p><b>Run:</b> {run_id} | <b>Execution date:</b> {ds}</p>
                <p><b>Failed Task:</b> compute_additional_metrics</p>
                <p><b>Issue:</b> Failed to compute additional metrics from judged responses.</p>
            """
            if csv_path:
                html_content += f"<p><b>Input CSV:</b> {csv_path}</p>"
            
            if model_gen_result:
                html_content += "<p><b>Model Generation:</b> Completed successfully</p>"
            if model_judge_result:
                html_content += "<p><b>Response Judging:</b> Completed successfully</p>"
            
            html_content += """
                <p><b>Possible Causes:</b></p>
                <ul>
                    <li>Missing judged responses CSV files</li>
                    <li>Data format error in judged responses</li>
                    <li>Script execution error</li>
                    <li>File permission issues</li>
                </ul>
                <p><b>Action Required:</b> Check additional metrics script logs and verify input files exist.</p>
                <p><b>Tip:</b> In the Airflow UI, open the "compute_additional_metrics" task's "Log" for details.</p>
            """
            
            # Send email
            send_email_with_conditional_files(
                to=["athatalnikar@gmail.com"],
                subject=f"[Airflow][{dag.dag_id}][{ds}] ❌ Additional Metrics Computation Failed",
                html_content=html_content,
                file_paths=None,
            )
            logger.info("Additional metrics failure email sent successfully")
        except AirflowSkipException:
            # Re-raise skip exceptions so task is properly skipped
            raise
        except Exception as e:
            logger.error(f"Failed to send additional metrics failure email: {e}", exc_info=True)
            # Don't re-raise - we don't want email failures to fail the DAG
    
    email_failure_model_metrics = PythonOperator(
        task_id="email_failure_model_metrics",
        python_callable=send_model_metrics_failure_email,
        trigger_rule=TriggerRule.ALL_DONE,  # Run regardless, but function will skip if upstream wasn't actually failed
        retries=0,  # Disable retries for email tasks
    )

    # Email: Bias Detection Failure
    def send_bias_detection_failure_email(**context):
        """Send bias detection failure email only if compute_bias_detection actually failed (not skipped)."""
        try:
            ti = context['ti']
            dag = context['dag']
            ds = context['ds']
            run_id = context['run_id']
            dag_run = context['dag_run']
            
            # Check if compute_bias_detection actually failed (not skipped)
            bias_task_instance = dag_run.get_task_instance('compute_bias_detection')
            
            # Only send email if task actually failed (not skipped or upstream_failed)
            if not bias_task_instance or bias_task_instance.state not in ['failed']:
                state = bias_task_instance.state if bias_task_instance else 'not found'
                logger.info(
                    f"compute_bias_detection is in state '{state}', not 'failed'. "
                    f"Skipping email. This is expected when upstream tasks fail or are skipped."
                )
                raise AirflowSkipException(f"Upstream task compute_bias_detection is in state '{state}', not 'failed'. Skipping email.")
            
            # Get data from XCom
            csv_path = ti.xcom_pull(task_ids='preprocess_input_csv', default=None)
            model_gen_result = ti.xcom_pull(task_ids='generate_model_responses', default=None)
            model_judge_result = ti.xcom_pull(task_ids='judge_responses', default=None)
            
            # Build HTML content
            html_content = f"""
                <h3>Bias Detection Failed: {dag.dag_id}</h3>
                <p><b>Run:</b> {run_id} | <b>Execution date:</b> {ds}</p>
                <p><b>Failed Task:</b> compute_bias_detection</p>
                <p><b>Issue:</b> Failed to compute bias detection metrics from judged responses.</p>
            """
            if csv_path:
                html_content += f"<p><b>Input CSV:</b> {csv_path}</p>"
            
            if model_gen_result:
                html_content += "<p><b>Model Generation:</b> Completed successfully</p>"
            if model_judge_result:
                html_content += "<p><b>Response Judging:</b> Completed successfully</p>"
            
            html_content += """
                <p><b>Possible Causes:</b></p>
                <ul>
                    <li>Missing judged responses CSV files</li>
                    <li>Data format error in judged responses</li>
                    <li>Missing required columns in judgements (prompt_id, prompt, response, safe, category, size_label, refusal_score)</li>
                    <li>Script execution error</li>
                    <li>File permission issues</li>
                </ul>
                <p><b>Action Required:</b> Check bias detection script logs and verify input files exist with correct format.</p>
                <p><b>Tip:</b> In the Airflow UI, open the "compute_bias_detection" task's "Log" for details.</p>
            """
            
            # Send email
            send_email_with_conditional_files(
                to=["athatalnikar@gmail.com"],
                subject=f"[Airflow][{dag.dag_id}][{ds}] ❌ Bias Detection Failed",
                html_content=html_content,
                file_paths=None,
            )
            logger.info("Bias detection failure email sent successfully")
        except AirflowSkipException:
            # Re-raise skip exceptions so task is properly skipped
            raise
        except Exception as e:
            logger.error(f"Failed to send bias detection failure email: {e}", exc_info=True)
            # Don't re-raise - we don't want email failures to fail the DAG
    
    email_failure_bias_detection = PythonOperator(
        task_id="email_failure_bias_detection",
        python_callable=send_bias_detection_failure_email,
        trigger_rule=TriggerRule.ALL_DONE,  # Run regardless, but function will skip if upstream wasn't actually failed
        retries=0,  # Disable retries for email tasks
    )

    # Email: DVC Push (final) Failure
    def send_dvc_push_final_failure_email(**context):
        """Send DVC push (final) failure email only if dvc_push_final actually failed."""
        try:
            ti = context['ti']
            dag = context['dag']
            ds = context['ds']
            run_id = context['run_id']
            dag_run = context['dag_run']
            
            # Check if dvc_push_final actually failed
            dvc_push_task_instance = dag_run.get_task_instance('dvc_push_final')
            
            if not dvc_push_task_instance or dvc_push_task_instance.state not in ['failed']:
                state = dvc_push_task_instance.state if dvc_push_task_instance else 'not found'
                logger.info(f"dvc_push_final is in state '{state}', not 'failed'. Skipping email.")
                raise AirflowSkipException(f"Upstream task dvc_push_final is in state '{state}', not 'failed'. Skipping email.")
            
            csv_path = ti.xcom_pull(task_ids='preprocess_input_csv', default=None)
            model_gen_result = ti.xcom_pull(task_ids='generate_model_responses', default=None)
            
            html_content = f"""
                <h3>DVC Push (Final) Failed (But Pipeline Succeeded): {dag.dag_id}</h3>
                <p><b>Run:</b> {run_id} | <b>Execution date:</b> {ds}</p>
                <p><b>Failed Task:</b> dvc_push_final</p>
                <p><b>Status:</b> ⚠️ <b>Pipeline completed successfully, but final artifact push to remote failed.</b></p>
            """
            if csv_path:
                html_content += f"<p><b>Processed CSV:</b> {csv_path}</p>"
            
            if model_gen_result:
                html_content += "<p><b>Model Pipeline:</b> Completed successfully</p>"
            
            html_content += """
                <p><b>Issue:</b> Failed to push all artifacts (including model outputs) to DVC remote storage.</p>
                <p><b>Possible Causes:</b></p>
                <ul>
                    <li>GCP credentials issue</li>
                    <li>DVC remote storage quota exceeded</li>
                    <li>Network connectivity issue during push</li>
                    <li>Permission denied on remote bucket</li>
                </ul>
                <p><b>Action Required:</b> Data and model outputs are processed locally, but not versioned remotely. Check DVC remote configuration and retry push manually if needed.</p>
                <p><b>Tip:</b> In the Airflow UI, open the "dvc_push_final" task's "Log" for details.</p>
            """
            
            send_email_with_conditional_files(
                to=["athatalnikar@gmail.com"],
                subject=f"[Airflow][{dag.dag_id}][{ds}] ⚠️ DVC Push (Final) Failed (Pipeline Succeeded)",
                html_content=html_content,
                file_paths=None,
            )
            logger.info("DVC push (final) failure email sent successfully")
        except AirflowSkipException:
            raise
        except Exception as e:
            logger.error(f"Failed to send DVC push (final) failure email: {e}", exc_info=True)
    
    email_failure_dvc_push_final = PythonOperator(
        task_id="email_failure_dvc_push_final",
        python_callable=send_dvc_push_final_failure_email,
        trigger_rule=TriggerRule.ALL_DONE,
        retries=0,
    )

    paths = ensure_dirs()
    cfg = ensure_config(paths)
    preprocessed_csv = preprocess_input_csv((cfg, str(OUTPUT_PATH)))
    
    validate_task = validate_output(preprocessed_csv)
    report_task = report_validation_status(validate_task)
    enforce_task = enforce_validation_policy(validate_task)

    # First DVC push: after validation succeeds (safe checkpoint for validated data)
    dvc_push_validation = BashOperator(
        task_id="dvc_push_validation",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        env={
            "REPO_ROOT": str(REPO_ROOT),
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
    echo "🚀 STEP 2: dvc push (sync validated data to remote)"
    python -m dvc push -v

    echo "──────────────────────────────────────────"
    echo "✅ STEP 3: Quick listing of processed outputs"
    ls -la data/processed || true

    echo "✅ DVC Push (validation) complete"
    {% endraw %}""",
    )

    @task
    def generate_model_responses() -> dict:
        """Generate model responses for all models in config/attack_llm_config.json."""
        script_path = REPO_ROOT / "scripts" / "generate_model_responses.py"
        if not script_path.exists():
            raise AirflowFailException(f"Model response generation script not found: {script_path}")
        
        config_path = REPO_ROOT / "config" / "attack_llm_config.json"
        if not config_path.exists():
            raise AirflowFailException(f"Model config file not found: {config_path}")
        
        logger.info("Running model response generation using config: %s", config_path)
        result = subprocess.run(
            ["python", str(script_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,  # 1 hour timeout for model generation
        )
        
        if result.returncode != 0:
            logger.error("Model response generation failed:\nSTDOUT: %s\nSTDERR: %s", result.stdout, result.stderr)
            raise AirflowFailException(
                f"Model response generation failed with exit code {result.returncode}. "
                f"Check logs for details."
            )
        
        # Try to parse summary from output (generate_model_responses.py prints summaries)
        logger.info("Model response generation completed successfully")
        logger.info("STDOUT: %s", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        
        # Return summary info if available
        return {
            "status": "success",
            "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        }

    @task
    def judge_responses() -> dict:
        """Judge model responses using judge LLM."""
        # Note: If generate_model_responses fails, this task will automatically skip
        
        script_path = REPO_ROOT / "scripts" / "judge_responses.py"
        if not script_path.exists():
            raise AirflowFailException(f"Judge responses script not found: {script_path}")
        
        logger.info("Running response judging")
        result = subprocess.run(
            ["python", str(script_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,  # 1 hour timeout for response judging
        )
        
        if result.returncode != 0:
            logger.error("Response judging failed:\nSTDOUT: %s\nSTDERR: %s", result.stdout, result.stderr)
            raise AirflowFailException(
                f"Response judging failed with exit code {result.returncode}. "
                f"Check logs for details."
            )
        
        logger.info("Response judging completed successfully")
        return {
            "status": "success",
            "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        }

    @task
    def compute_additional_metrics() -> dict:
        """Compute additional metrics from judged responses."""
        # Note: If generate_model_responses fails, this task will automatically skip
        
        script_path = REPO_ROOT / "scripts" / "additional_metrics.py"
        if not script_path.exists():
            raise AirflowFailException(f"Additional metrics script not found: {script_path}")
        
        logger.info("Running additional metrics computation")
        result = subprocess.run(
            ["python", str(script_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=300,  # 5 minute timeout for metrics computation
        )
        
        if result.returncode != 0:
            logger.error("Additional metrics computation failed:\nSTDOUT: %s\nSTDERR: %s", result.stdout, result.stderr)
            raise AirflowFailException(
                f"Additional metrics computation failed with exit code {result.returncode}. "
                f"Check logs for details."
            )
        
        logger.info("Additional metrics computation completed successfully")
        return {
            "status": "success",
            "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        }

    @task
    def compute_bias_detection() -> dict:
        """Compute bias detection metrics from judged responses."""
        # Note: If generate_model_responses fails, this task will automatically skip
        
        script_path = REPO_ROOT / "scripts" / "bias_detection.py"
        if not script_path.exists():
            raise AirflowFailException(f"Bias detection script not found: {script_path}")
        
        logger.info("Running bias detection")
        result = subprocess.run(
            ["python", str(script_path)],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
            timeout=300,  # 5 minute timeout for bias detection
        )
        
        if result.returncode != 0:
            logger.error("Bias detection failed:\nSTDOUT: %s\nSTDERR: %s", result.stdout, result.stderr)
            raise AirflowFailException(
                f"Bias detection failed with exit code {result.returncode}. "
                f"Check logs for details."
            )
        
        logger.info("Bias detection completed successfully")
        return {
            "status": "success",
            "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        }

    # Second DVC push: after model pipeline completes (will include model outputs when added to DVC)
    dvc_push_final = BashOperator(
        task_id="dvc_push_final",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        env={
            "REPO_ROOT": str(REPO_ROOT),
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
    echo "🚀 STEP 2: dvc push (sync all artifacts to remote)"
    echo "Note: This task is configured to always succeed (model outputs may not be implemented yet)"
    if python -m dvc push -v; then
        echo "✅ dvc push completed successfully"
    else
        echo "⚠️  WARNING: dvc push encountered issues, but task will still succeed"
        echo "This is expected if model outputs are not yet tracked in DVC"
    fi

    echo "──────────────────────────────────────────"
    echo "✅ STEP 3: Quick listing of outputs"
    ls -la data/processed || true
    ls -la data/responses || true

    echo "✅ DVC Push (final) complete - task succeeded"
    {% endraw %}""",
    )

    # ── Orchestration ───────────────────────────────────────────────────────────
    # If validate_task fails, both report_task and enforce_task are skipped
    # (report_task uses ALL_SUCCESS, enforce_task uses default ALL_SUCCESS)
    dvc_pull >> paths >> cfg >> preprocessed_csv >> validate_task
    validate_task >> [report_task, enforce_task]
 
    # Email validation report - runs regardless of validation outcome
    # Only depend on validate_task - report_task is optional and may be skipped when validate_task fails
    # This prevents email_validation_report from waiting indefinitely for report_task
    validate_task >> email_validation_report

    # Model pipeline - uses config/attack_llm_config.json as input
    # Model pipeline tasks run sequentially: model_gen >> model_judge >> [model_metrics, bias_detection]
    # model_metrics and bias_detection run in parallel after model_judge completes
    # dvc_push_validation and model_gen can run in parallel after validation
    # dvc_push_final waits for both model_metrics and bias_detection to complete
    model_gen = generate_model_responses()
    model_judge = judge_responses()  # Auto-skips if model_gen fails
    model_metrics = compute_additional_metrics()  # Auto-skips if model_gen fails
    bias_detection = compute_bias_detection()  # Auto-skips if model_gen fails
    
    enforce_task >> [dvc_push_validation, model_gen]
    model_gen >> model_judge >> [model_metrics, bias_detection]
    [model_metrics, bias_detection] >> dvc_push_final >> email_success

    # Context-specific failure emails based on where failure occurs
    # Stage 1: DVC Pull failures
    dvc_pull >> email_failure_dvc_pull
    
    # Stage 2: Setup/Config failures (only if dvc_pull succeeded)
    # Note: Only connect cfg, not paths (setup task), to avoid violating setup task trigger rule requirement
    # If paths (setup task) fails, DAG will fail and cfg won't run, so we only need to catch cfg failures
    cfg >> email_failure_setup
    
    # Stage 3: Preprocessing failures (only if setup succeeded)
    preprocessed_csv >> email_failure_preprocessing
    
    # Stage 4: Validation failures (only if preprocessing succeeded)
    # Only connect validate_task - if it fails, enforce_task is skipped so we don't need to wait for it
    validate_task >> email_failure_validation
    
    # Stage 4b: Enforce validation policy failures (only if validate_task succeeded)
    # This handles the case where validate_task succeeded but enforce_task failed
    enforce_task >> email_failure_enforce_policy
    
    # Stage 5: DVC Push (validation) failures (only if validation succeeded)
    dvc_push_validation >> email_failure_dvc_push_validation
    
    # Stage 6: Model pipeline failures
    model_gen >> email_failure_model_generation
    model_judge >> email_failure_model_judging
    model_metrics >> email_failure_model_metrics
    bias_detection >> email_failure_bias_detection
    
    # Stage 7: DVC Push (final) failures (only if model pipeline succeeded)
    dvc_push_final >> email_failure_dvc_push_final


dag = salad_preprocess_v1()