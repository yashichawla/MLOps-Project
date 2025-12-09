# dags/salad_preprocess_dag.py
from __future__ import annotations
from datetime import datetime, timedelta, timezone
from pathlib import Path
import os, subprocess, sys
import json
import logging

from airflow.decorators import dag, task, setup
from airflow.operators.python import get_current_context, PythonOperator
from airflow.exceptions import AirflowFailException, AirflowSkipException
from airflow.models import Variable
from airflow.hooks.base import BaseHook
from airflow.utils.trigger_rule import TriggerRule
from airflow.operators.email import EmailOperator
from airflow.utils.email import send_email_smtp

from airflow.operators.bash import BashOperator

# Initialize logger early for path detection logging
logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Environment Detection and Path Resolution
# ──────────────────────────────────────────────────────────────────────────────
# This section handles both local development and Google Cloud Composer:
# - Single source of truth: scripts/ are at <repo>/dags/scripts/ (works for both local and Composer)
# - Composer: only dags/ folder is synced, so scripts must be in dags/scripts/
# - Local: also uses dags/scripts/ for consistency
# 
# Strategy:
#   1. Always use dags/scripts/ as the single source of truth
#   2. This ensures consistency between local and Composer environments
# ──────────────────────────────────────────────────────────────────────────────

# 1. Compute DAGS_DIR (directory containing this DAG file)
DAGS_DIR = Path(__file__).resolve().parent

# 2. Detect REPO_ROOT and SCRIPTS_DIR
# Always use dags/scripts/ as single source of truth

if "PROJECT_ROOT" in os.environ:
    # Manual override via environment variable
    REPO_ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()
    # Always use dags/scripts/ as single source of truth
    if (REPO_ROOT / "dags" / "scripts").exists():
        SCRIPTS_DIR = REPO_ROOT / "dags" / "scripts"
        logger.info("Using PROJECT_ROOT override: found dags/scripts/")
    else:
        # Last resort: assume scripts is at dags/scripts/
        SCRIPTS_DIR = REPO_ROOT / "dags" / "scripts"
        logger.warning("Using PROJECT_ROOT override: assuming dags/scripts/ (may not exist)")
else:
    # Auto-detect based on file structure
    # Always use dags/scripts/ as single source of truth (works for both local and Composer)
    dags_scripts_dir = DAGS_DIR / "scripts"
    if dags_scripts_dir.exists() and dags_scripts_dir.is_dir():
        # dags/scripts/ exists - use it (works for both local and Composer)
        REPO_ROOT = DAGS_DIR.parent
        SCRIPTS_DIR = dags_scripts_dir
        logger.info("Auto-detected: Using dags/scripts/ (single source of truth)")
    else:
        # Fallback: assume DAGS_DIR.parent is REPO_ROOT and look for dags/scripts/
        REPO_ROOT = DAGS_DIR.parent
        if (REPO_ROOT / "dags" / "scripts").exists():
            SCRIPTS_DIR = REPO_ROOT / "dags" / "scripts"
            logger.info("Fallback: Found dags/scripts/")
        else:
            # Last resort: assume scripts is at dags/scripts/ (may not exist)
            SCRIPTS_DIR = REPO_ROOT / "dags" / "scripts"
            logger.warning("Fallback: Assuming dags/scripts/ (may not exist)")

# 3. Insert the correct parent directory into sys.path to allow `from scripts.X import Y`
# We need the parent of SCRIPTS_DIR in sys.path so that `from scripts.X import Y` works
# - If SCRIPTS_DIR is dags/scripts/, then scripts_parent is dags/, so scripts/ is accessible
# - SCRIPTS_DIR is always dags/scripts/, so scripts_parent is dags/, making scripts/ accessible
scripts_parent = SCRIPTS_DIR.parent
scripts_parent_str = str(scripts_parent)
if scripts_parent_str not in sys.path:
    sys.path.insert(0, scripts_parent_str)

# Log detected paths for debugging
logger.info(
    f"Path detection: DAGS_DIR={DAGS_DIR}, REPO_ROOT={REPO_ROOT}, "
    f"SCRIPTS_DIR={SCRIPTS_DIR}, scripts_parent={scripts_parent}"
)

# DVC project directory: Single source of truth for both local and Composer
# This directory (dags/dvc_project/) is synced to Composer via:
#   gsutil -m rsync -r -d dags/ gs://$COMPOSER_BUCKET/dags/
# The rsync command automatically syncs:
#   - dags/dvc_project/.dvc/ (DVC configuration directory)
#   - dags/dvc_project/dvc.yaml (DVC pipeline definition)
#   - dags/dvc_project/dvc.lock (DVC lock file, if present)
# Both local and Composer use dags/dvc_project/ as the DVC repository
# Note: Data is stored inside DVC repo at dvc_project/data/ folder
DVC_PROJECT_DIR = DAGS_DIR / "dvc_project"

# repo layout - use data directory inside dvc_project
DATA_DIR = DVC_PROJECT_DIR / "data"
OUT_DIR = DATA_DIR / "processed"
# Config is at dags/config/ (synced to Composer), use DAGS_DIR for consistency
CFG_DIR = DAGS_DIR / "config"

# Helper function to get environment dict with PROJECT_ROOT and DVC_PROJECT_DIR if set
def get_subprocess_env():
    """Get environment dict for subprocess calls, including PROJECT_ROOT, DVC_PROJECT_DIR, DVC_DATA_DIR, and API keys if set."""
    env = os.environ.copy()
    # Always set PROJECT_ROOT so scripts know where to find config files
    env["PROJECT_ROOT"] = str(REPO_ROOT)
    # Set DVC_PROJECT_DIR for DVC commands (DVC repo is at dags/dvc_project/)
    env["DVC_PROJECT_DIR"] = str(DVC_PROJECT_DIR)
    # Set DVC_DATA_DIR so scripts know where to find data (inside DVC repo)
    env["DVC_DATA_DIR"] = str(DATA_DIR)
    # Set HF_TOKEN from Airflow Variables if available (for Hugging Face API calls)
    try:
        hf_token = Variable.get("HF_TOKEN", default_var=None)
        if hf_token:
            env["HF_TOKEN"] = hf_token
    except Exception:
        # Variable might not exist, that's okay - script will handle it
        pass
    # Set GROQ_API_KEY from Airflow Variables if available
    try:
        groq_key = Variable.get("GROQ_API_KEY", default_var=None)
        if groq_key:
            env["GROQ_API_KEY"] = groq_key
    except Exception:
        # Variable might not exist, that's okay
        pass
    return env

# Import after sys.path is configured
from scripts.preprocess_salad import run_preprocessing

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
# Note: These paths are relative to DVC_PROJECT_DIR (dags/dvc_project/).
# Data is now inside dvc_project/data/, so use data/ for data paths.
DVC_TRACK_PATHS = [
    "data/processed",  # Data inside dvc_project/data/processed/ (relative from dags/dvc_project/)
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
    Uses Airflow SMTP Connection 'gmail_smtp' for authentication.
    """
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        
        # Retrieve the SMTP connection
        conn = BaseHook.get_connection("gmail_smtp")
        logger.info(f"Retrieved SMTP connection: {conn.conn_id} (host: {conn.host}, port: {conn.port})")
        
        # Get connection details
        smtp_host = conn.host or "smtp.gmail.com"
        smtp_port = conn.port or 587
        smtp_user = conn.login
        smtp_password = conn.password
        smtp_extra = conn.extra_dejson if conn.extra else {}
        use_starttls = smtp_extra.get("starttls", True)
        use_ssl = smtp_extra.get("ssl", False)
        
        if not smtp_user or not smtp_password:
            raise ValueError("SMTP connection missing login or password")
        
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
        
        # Create message
        msg = MIMEMultipart()
        msg["From"] = smtp_user
        msg["To"] = ", ".join(to)
        msg["Subject"] = subject
        msg.attach(MIMEText(html_content, "html"))
        
        # Attach files
        for file_path in existing_files:
            with open(file_path, "rb") as f:
                part = MIMEBase("application", "octet-stream")
                part.set_payload(f.read())
                encoders.encode_base64(part)
                part.add_header(
                    "Content-Disposition",
                    f'attachment; filename= "{Path(file_path).name}"'
                )
                msg.attach(part)
        
        # Send email
        if use_ssl:
            server = smtplib.SMTP_SSL(smtp_host, smtp_port)
        else:
            server = smtplib.SMTP(smtp_host, smtp_port)
            if use_starttls:
                server.starttls()
        
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email sent successfully to {to} with {len(existing_files) if existing_files else 0} attachment(s)")
    except Exception as e:
        # Don't re-raise - email failures should be non-fatal
        logger.error(f"Failed to send email: {e}", exc_info=True)
        logger.warning("Email task will succeed but email was not sent.")


@dag(
    dag_id="salad_ml_evaluation_pipeline_v1",
    description="Full ML evaluation pipeline: preprocessing, validation, model response generation, judging, metrics, and bias detection.",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "owner": "airflow",
    },
    tags=["salad", "ml-evaluation", "pipeline", "v1", "mlops", "testmode"],
)
def salad_ml_evaluation_pipeline_v1():

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
            cd "{DVC_PROJECT_DIR}"

            # DVC packages should be pre-installed via requirements-docker.txt
            # If not available, this will fail and indicate a Docker image build issue

            # 0) Backup any conflicting local file that would block checkout (keeps your work!)
            # Data is now inside dvc_project/data/ (relative from dags/dvc_project/)
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

            # 2) Create directories for tracked outputs that may not exist in remote yet
            # This prevents DVC pull from failing on missing outputs
            # Data is now inside dvc_project/data/ (relative from dags/dvc_project/)
            # Using || true to make this idempotent (won't fail if directories already exist)
            mkdir -p data/metrics/additional \
                     data/metrics/schema/baseline \
                     data/responses \
                     data/judge \
                     data/bias \
                     data/processed || true
            echo "[info] Created directories for tracked outputs"

            # 3) Pull latest from DVC remote, but don't force overwrite newer local files
            # DVC remote should have latest after dvc_push_final, but we need to handle edge cases:
            # - If dvc_push_final hasn't run yet (current run in progress)
            # - If dvc_push_final failed in previous run
            # - If files are newer locally (just generated), don't overwrite them
            
            echo "[info] Running dvc pull (will update dvc.lock and pull missing files)..."
            # Pull without -f flag first - this won't overwrite newer local files
            # This is safer: it pulls missing files but preserves newer local changes
            set +e  # Temporarily disable exit on error to handle missing outputs
            PULL_OUTPUT=$(python -m dvc pull -v 2>&1)
            PULL_EXIT=$?
            set -e  # Re-enable exit on error
            
            # Only do force checkout for preprocessing stage (inputs), not model outputs
            # Model outputs (responses, judge, bias, metrics) are generated by DAG and should not be overwritten
            echo "[info] Running dvc checkout for preprocessing stage only..."
            python -m dvc checkout salad_preprocess -f 2>&1 || echo "[warn] dvc checkout had some errors (may be missing outputs), continuing..."
            
            echo "$PULL_OUTPUT"
            
            # Check if error is due to missing outputs (common for new stages)
            if [ $PULL_EXIT -ne 0 ]; then
                if echo "$PULL_OUTPUT" | grep -qi "failed to restore\|missing\|not found"; then
                    echo "[info] Some tracked outputs may not exist in remote yet (this is OK for new stages)"
                    echo "[info] Continuing with pipeline - outputs will be created by DAG tasks"
                else
                    echo "[error] DVC pull failed with unexpected error (exit code: $PULL_EXIT)"
                    exit $PULL_EXIT
                fi
            else
                echo "[info] DVC pull completed successfully"
            fi
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
        # Ensure parent directory exists before trying to write
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
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
    SCRIPT_GE = SCRIPTS_DIR / "ge_runner.py"
    METRICS_DIR = DVC_PROJECT_DIR / "data" / "metrics"
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
                    ["python", str(SCRIPT_GE), "baseline", "--input", out_csv_path, 
                     "--baseline_schema", str(BASELINE_SCHEMA), 
                     "--metrics_dir", str(METRICS_DIR), "--date", ds_nodash],
                    env=get_subprocess_env(),
                    check=True,
                    timeout=300,  # 5 minute timeout
                )
            
            logger.info("Running GE validation for %s", out_csv_path)
            res = subprocess.run(
                [
                    "python", str(SCRIPT_GE), "validate", "--input", out_csv_path,
                    "--baseline_schema", str(BASELINE_SCHEMA), 
                    "--metrics_dir", str(METRICS_DIR), "--date", ds_nodash,
                ],
                env=get_subprocess_env(),
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
                fallback_path = DATA_DIR / "metrics" / "validation" / ds_nodash / "anomalies.json"
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

    def send_success_email(**context):
        """Send success email with comprehensive logging and error handling."""
        try:
            ti = context['ti']
            dag = context['dag']
            ds = context['ds']
            run_id = context['run_id']
            execution_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            
            logger.info("=" * 60)
            logger.info("Starting email_success task")
            logger.info(f"DAG: {dag.dag_id}, Run: {run_id}, Date: {ds}")
            logger.info("=" * 60)
            
            # Get data from XCom with logging
            logger.info("Fetching XCom data from upstream tasks...")
            
            csv_path = ti.xcom_pull(task_ids='preprocess_input_csv', default=None)
            logger.info(f"CSV path from XCom: {csv_path}")
            
            metrics = ti.xcom_pull(task_ids='validate_output', default=None)
            if metrics:
                logger.info(f"Validation metrics found: {list(metrics.keys())}")
                logger.info(f"Row count: {metrics.get('row_count', 'N/A')}")
                logger.info(f"Null prompts: {metrics.get('null_prompt_count', 'N/A')}")
                logger.info(f"Duplicates: {metrics.get('dup_prompt_count', 'N/A')}")
            else:
                logger.warning("Validation metrics not found in XCom")
            
            model_gen = ti.xcom_pull(task_ids='generate_model_responses', default=None)
            logger.info(f"Model generation result: {'Found' if model_gen else 'Not found'}")
            
            model_judge = ti.xcom_pull(task_ids='judge_responses', default=None)
            logger.info(f"Model judging result: {'Found' if model_judge else 'Not found'}")
            
            model_metrics = ti.xcom_pull(task_ids='compute_additional_metrics', default=None)
            logger.info(f"Additional metrics result: {'Found' if model_metrics else 'Not found'}")
            
            bias_detection = ti.xcom_pull(task_ids='compute_bias_detection', default=None)
            logger.info(f"Bias detection result: {'Found' if bias_detection else 'Not found'}")
            
            # Load model configuration
            logger.info("Loading model configuration...")
            config_path = CFG_DIR / "attack_llm_config.json"
            models_config = []
            if config_path.exists():
                try:
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)
                        models_config = config_data.get('models', [])
                    logger.info(f"Loaded {len(models_config)} models from config")
                except Exception as e:
                    logger.warning(f"Failed to load model config: {e}")
            else:
                logger.warning(f"Model config file not found: {config_path}")
            
            # Build HTML content
            logger.info("Building email HTML content...")
            html_content = f"""
                <h3>DAG Succeeded: {dag.dag_id}</h3>
                <p><b>Run:</b> {run_id}</p>
                <p><b>Execution Date:</b> {ds}</p>
                <p><b>Execution Time:</b> {execution_time}</p>
                <p><b>Selected CSV:</b> {csv_path if csv_path else 'N/A'}</p>
            """
            
            # Data Validation Section
            if metrics:
                try:
                    # Safe dictionary access with .get() and proper None checks
                    row_count = metrics.get('row_count', 'N/A')
                    null_prompt_count = metrics.get('null_prompt_count', 'N/A')
                    dup_prompt_count = metrics.get('dup_prompt_count', 'N/A')
                    unknown_rate = metrics.get('unknown_category_rate')
                    text_len_min = metrics.get('text_len_min', 'N/A')
                    text_len_max = metrics.get('text_len_max', 'N/A')
                    soft_warn = metrics.get('soft_warn', [])
                    
                    # Format unknown_rate safely
                    if unknown_rate is not None and isinstance(unknown_rate, (int, float)):
                        unknown_rate_str = f"{unknown_rate:.3f}"
                    else:
                        unknown_rate_str = 'N/A'
                    
                    html_content += f"""
                    <h4>Data Validation</h4>
                    <ul>
                        <li><b>Rows:</b> {row_count}</li>
                        <li><b>Null Prompts:</b> {null_prompt_count}</li>
                        <li><b>Duplicates:</b> {dup_prompt_count}</li>
                        <li><b>Unknown rate:</b> {unknown_rate_str}</li>
                        <li><b>Text length range:</b> {text_len_min}/{text_len_max}</li>
                    </ul>
                    """
                    
                    # Add soft warnings if any
                    if soft_warn:
                        html_content += f"<p><b>⚠️ Soft Warnings:</b></p><ul>"
                        for warn in soft_warn:
                            html_content += f"<li>{warn}</li>"
                        html_content += "</ul>"
                    
                    # Add validation report paths
                    report_paths = metrics.get('report_paths', [])
                    if report_paths:
                        html_content += "<p><b>Validation Reports:</b></p><ul>"
                        for path in report_paths:
                            html_content += f"<li><code>{path}</code></li>"
                        html_content += "</ul>"
                    
                    logger.info("Added validation metrics to email")
                except Exception as e:
                    logger.warning(f"Error formatting validation metrics: {e}", exc_info=True)
                    html_content += "<p><b>Data Validation:</b> Metrics available but formatting error occurred</p>"
            else:
                html_content += "<p><b>Data Validation:</b> Metrics not available</p>"
                logger.warning("Validation metrics not available for email")
            
            # Model Pipeline Status
            html_content += "<h4>Model Pipeline Status</h4><ul>"
            
            if model_gen:
                html_content += "<li><b>Model Generation:</b> ✅ Executed successfully</li>"
                logger.info("Model generation: Executed")
            else:
                html_content += "<li><b>Model Generation:</b> ⏭️ Skipped (data unchanged)</li>"
                logger.info("Model generation: Skipped")
            
            if model_judge:
                html_content += "<li><b>Response Judging:</b> ✅ Executed successfully</li>"
                logger.info("Response judging: Executed")
            else:
                html_content += "<li><b>Response Judging:</b> ⏭️ Skipped (data unchanged)</li>"
                logger.info("Response judging: Skipped")
            
            if model_metrics:
                html_content += "<li><b>Additional Metrics:</b> ✅ Executed successfully</li>"
                logger.info("Additional metrics: Executed")
            else:
                html_content += "<li><b>Additional Metrics:</b> ⏭️ Skipped (data unchanged)</li>"
                logger.info("Additional metrics: Skipped")
            
            if bias_detection:
                html_content += "<li><b>Bias Detection:</b> ✅ Executed successfully</li>"
                logger.info("Bias detection: Executed")
            else:
                html_content += "<li><b>Bias Detection:</b> ⏭️ Skipped (data unchanged)</li>"
                logger.info("Bias detection: Skipped")
            
            html_content += "</ul>"
            
            # Model Metrics Section
            if models_config and model_metrics:
                logger.info("Loading model metrics files...")
                html_content += "<h4>Model Metrics Summary</h4>"
                html_content += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
                html_content += "<tr><th>Model</th><th>Total Prompts</th><th>Categories</th><th>Over-Refusal Rate</th></tr>"
                
                models_with_metrics = 0
                for model in models_config:
                    model_name = model.get('name', 'Unknown')
                    metrics_file = DATA_DIR / "metrics" / "additional" / f"additional_metrics_{model_name}.json"
                    
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                model_metrics_data = json.load(f)
                            
                            coverage = model_metrics_data.get('coverage_metrics', {})
                            over_refusal = model_metrics_data.get('over_refusal_metrics', {})
                            
                            total_prompts = coverage.get('total_prompts', 'N/A')
                            num_categories = coverage.get('num_categories', 'N/A')
                            over_refusal_rate = over_refusal.get('over_refusal_rate', 'N/A')
                            
                            # Format over_refusal_rate
                            if isinstance(over_refusal_rate, (int, float)):
                                over_refusal_str = f"{over_refusal_rate:.3f}"
                            else:
                                over_refusal_str = str(over_refusal_rate)
                            
                            html_content += f"<tr><td>{model_name}</td><td>{total_prompts}</td><td>{num_categories}</td><td>{over_refusal_str}</td></tr>"
                            models_with_metrics += 1
                            logger.info(f"Loaded metrics for {model_name}")
                        except Exception as e:
                            logger.warning(f"Failed to load metrics for {model_name}: {e}")
                            html_content += f"<tr><td>{model_name}</td><td colspan='3'>Error loading metrics</td></tr>"
                    else:
                        logger.warning(f"Metrics file not found for {model_name}: {metrics_file}")
                        html_content += f"<tr><td>{model_name}</td><td colspan='3'>Metrics not available</td></tr>"
                
                html_content += "</table>"
                logger.info(f"Added metrics for {models_with_metrics} models")
            
            # Bias Detection Summary Section
            if models_config and bias_detection:
                logger.info("Loading bias detection reports...")
                html_content += "<h4>Bias Detection Summary</h4>"
                html_content += "<table border='1' cellpadding='5' cellspacing='0' style='border-collapse: collapse;'>"
                html_content += "<tr><th>Model</th><th>Global ASR</th><th>Samples</th><th>Biased Categories</th><th>Biased Sizes</th></tr>"
                
                models_with_bias = 0
                for model in models_config:
                    model_name = model.get('name', 'Unknown')
                    bias_file = DATA_DIR / "bias" / model_name / "bias_report.json"
                    
                    if bias_file.exists():
                        try:
                            with open(bias_file, 'r') as f:
                                bias_data = json.load(f)
                            
                            global_metrics = bias_data.get('global', {})
                            biased_slices = bias_data.get('biased_slices', {})
                            
                            global_asr = global_metrics.get('asr', 'N/A')
                            sample_count = global_metrics.get('count', 'N/A')
                            biased_categories = len(biased_slices.get('category', []))
                            biased_sizes = len(biased_slices.get('size_label', []))
                            
                            # Format ASR
                            if isinstance(global_asr, (int, float)):
                                asr_str = f"{global_asr:.3f}"
                            else:
                                asr_str = str(global_asr)
                            
                            html_content += f"<tr><td>{model_name}</td><td>{asr_str}</td><td>{sample_count}</td><td>{biased_categories}</td><td>{biased_sizes}</td></tr>"
                            models_with_bias += 1
                            logger.info(f"Loaded bias report for {model_name}")
                        except Exception as e:
                            logger.warning(f"Failed to load bias report for {model_name}: {e}")
                            html_content += f"<tr><td>{model_name}</td><td colspan='4'>Error loading bias report</td></tr>"
                    else:
                        logger.warning(f"Bias report not found for {model_name}: {bias_file}")
                        html_content += f"<tr><td>{model_name}</td><td colspan='4'>Bias report not available</td></tr>"
                
                html_content += "</table>"
                logger.info(f"Added bias summaries for {models_with_bias} models")
            
            html_content += "<p>Great job! ✔</p>"
            
            logger.info("Email HTML content built successfully")
            logger.info(f"Email content length: {len(html_content)} characters")
            
            # Send email
            logger.info("Sending success email...")
            logger.info(f"Recipient: athatalnikar@gmail.com")
            logger.info(f"Subject: [Airflow][{dag.dag_id}][{ds}] ✅ DAG Succeeded")
            
            send_email_with_conditional_files(
                to=["athatalnikar@gmail.com"],
                subject=f"[Airflow][{dag.dag_id}][{ds}] ✅ DAG Succeeded",
                html_content=html_content,
                file_paths=None,
            )
            
            logger.info("=" * 60)
            logger.info("✅ Success email sent successfully!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error("=" * 60)
            logger.error(f"❌ Failed to send success email: {e}", exc_info=True)
            logger.error("=" * 60)
            # Re-raise so the task fails and you can see the error in Airflow
            raise

    email_success = PythonOperator(
        task_id="email_success",
        python_callable=send_success_email,
        trigger_rule=TriggerRule.ALL_SUCCESS,
        retries=0,  # Disable retries for email tasks
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
                fallback_path = DATA_DIR / "metrics" / "validation" / ds_nodash / "anomalies.json"
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
                fallback_path = DATA_DIR / "metrics" / "validation" / ds_nodash / "anomalies.json"
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
    def send_model_judging_failure_email(**context):
        """Send model judging failure email only if judge_responses actually failed (not skipped)."""
        try:
            ti = context['ti']
            dag = context['dag']
            ds = context['ds']
            run_id = context['run_id']
            dag_run = context['dag_run']
            
            # Check if judge_responses actually failed (not skipped)
            judge_task_instance = dag_run.get_task_instance('judge_responses')
            
            # Only send email if task actually failed (not skipped or upstream_failed)
            if not judge_task_instance or judge_task_instance.state not in ['failed']:
                state = judge_task_instance.state if judge_task_instance else 'not found'
                logger.info(
                    f"judge_responses is in state '{state}', not 'failed'. "
                    f"Skipping email. This is expected when upstream tasks fail or are skipped."
                )
                raise AirflowSkipException(f"Upstream task judge_responses is in state '{state}', not 'failed'. Skipping email.")
            
            # Get data from XCom
            csv_path = ti.xcom_pull(task_ids='preprocess_input_csv', default=None)
            model_gen_result = ti.xcom_pull(task_ids='generate_model_responses', default=None)
            
            # Build HTML content
            html_content = f"""
                <h3>Model Judging Failed: {dag.dag_id}</h3>
                <p><b>Run:</b> {run_id} | <b>Execution date:</b> {ds}</p>
                <p><b>Failed Task:</b> judge_responses</p>
                <p><b>Issue:</b> Failed to judge model responses using judge LLM.</p>
            """
            if csv_path:
                html_content += f"<p><b>Input CSV:</b> {csv_path}</p>"
            
            if model_gen_result:
                html_content += "<p><b>Model Generation:</b> Completed successfully</p>"
            
            html_content += """
                <p><b>Possible Causes:</b></p>
                <ul>
                    <li>Groq API key missing or invalid (check GROQ_API_KEY environment variable)</li>
                    <li>Judge LLM API rate limit exceeded</li>
                    <li>Network connectivity issue</li>
                    <li>Script execution error</li>
                </ul>
                <p><b>Action Required:</b> Check judge responses script logs and verify API credentials.</p>
                <p><b>Tip:</b> In the Airflow UI, open the "judge_responses" task's "Log" for details.</p>
            """
            
            # Send email
            send_email_with_conditional_files(
                to=["athatalnikar@gmail.com"],
                subject=f"[Airflow][{dag.dag_id}][{ds}] ❌ Model Judging Failed",
                html_content=html_content,
                file_paths=None,
            )
            logger.info("Model judging failure email sent successfully")
        except AirflowSkipException:
            # Re-raise skip exceptions so task is properly skipped
            raise
        except Exception as e:
            logger.error(f"Failed to send model judging failure email: {e}", exc_info=True)
            # Don't re-raise - we don't want email failures to fail the DAG
    
    email_failure_model_judging = PythonOperator(
        task_id="email_failure_model_judging",
        python_callable=send_model_judging_failure_email,
        trigger_rule=TriggerRule.ALL_DONE,  # Run regardless, but function will skip if upstream wasn't actually failed
        retries=0,  # Disable retries for email tasks
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
            "DVC_PROJECT_DIR": str(DVC_PROJECT_DIR),
            "TEST_MODE": str(TEST_MODE).lower(),
            "GOOGLE_APPLICATION_CREDENTIALS": "/opt/airflow/secrets/gcp-key.json",
            "DVC_NO_ANALYTICS": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        },
        bash_command="""{% raw %}
    set -euo pipefail
    cd "$DVC_PROJECT_DIR"

    echo "──────────────────────────────────────────"
    echo "🔎 STEP 1: dvc status (cache/remote delta)"
    python -m dvc status -c -v || true

    echo "──────────────────────────────────────────"
    echo "📝 STEP 2: Commit preprocessing outputs to DVC (if any)"
    echo "This ensures newly generated preprocessing outputs are tracked before pushing"
    set +e  # Temporarily disable exit on error to handle commit failures gracefully
    COMMIT_OUTPUT=$(python -m dvc commit -f 2>&1)
    COMMIT_EXIT=$?
    set -e  # Re-enable exit on error
    echo "$COMMIT_OUTPUT"
    if [ "$COMMIT_EXIT" -ne 0 ]; then
        echo "❌ ERROR: DVC commit failed (exit code: $COMMIT_EXIT)"
        # Check for common errors
        if echo "$COMMIT_OUTPUT" | grep -qi "permission\|access\|credential"; then
            echo "   Error: Authentication/credential issue"
            echo "   Action: Check GOOGLE_APPLICATION_CREDENTIALS and GCP service account permissions"
        elif echo "$COMMIT_OUTPUT" | grep -qi "network\|connection\|timeout"; then
            echo "   Error: Network connectivity issue"
            echo "   Action: Check network connection"
        fi
        echo "❌ Task failed: DVC commit is required before push"
        exit $COMMIT_EXIT
    fi
    echo "✅ DVC commit completed successfully"

    echo "──────────────────────────────────────────"
    echo "🚀 STEP 3: dvc push (sync validated data to remote)"
    PUSH_OUTPUT=$(python -m dvc push -v 2>&1)
    PUSH_EXIT=$?
    echo "$PUSH_OUTPUT"
    
    # Parse push results and errors
    if [ "$PUSH_EXIT" -eq 0 ]; then
        PUSHED_COUNT=$(echo "$PUSH_OUTPUT" | grep -c "Pushed" 2>/dev/null || echo "0")
        # Handle case where grep returns empty (no matches)
        if [ -z "$PUSHED_COUNT" ]; then
            PUSHED_COUNT=0
        fi
        if [ "$PUSHED_COUNT" -gt 0 ]; then
            echo "✅ dvc push completed successfully"
            echo "📊 Summary: Pushed $PUSHED_COUNT item(s) to remote"
        else
            echo "✅ dvc push completed (no new items to push)"
        fi
    else
        echo "❌ ERROR: dvc push failed (exit code: $PUSH_EXIT)"
        # Parse common error patterns
        if echo "$PUSH_OUTPUT" | grep -qi "permission\|access\|credential\|unauthorized"; then
            echo "   Error: Authentication/credential issue"
            echo "   Action: Check GOOGLE_APPLICATION_CREDENTIALS and GCP service account permissions"
        elif echo "$PUSH_OUTPUT" | grep -qi "network\|connection\|timeout\|unreachable"; then
            echo "   Error: Network connectivity issue"
            echo "   Action: Check network connection and GCS bucket accessibility"
        elif echo "$PUSH_OUTPUT" | grep -qi "bucket\|not found\|404"; then
            echo "   Error: GCS bucket not found or inaccessible"
            echo "   Action: Verify bucket name and service account has storage.objects.* permissions"
        else
            echo "   Error details:"
            echo "$PUSH_OUTPUT" | grep -i "error\|failed\|exception" | head -5 || echo "   (See full output above)"
        fi
        echo "❌ Task failed: DVC push must complete successfully"
        exit $PUSH_EXIT
    fi

    echo "──────────────────────────────────────────"
    echo "✅ STEP 4: Quick listing of processed outputs"
    ls -la data/processed || true

    echo "✅ DVC Push (validation) complete"
    {% endraw %}""",
    )

    @task(execution_timeout=timedelta(hours=2))
    def generate_model_responses() -> dict:
        """Generate model responses for all models in config/attack_llm_config.json."""
        script_path = SCRIPTS_DIR / "generate_model_responses.py"
        if not script_path.exists():
            raise AirflowFailException(f"Model response generation script not found: {script_path}")
        
        config_path = CFG_DIR / "attack_llm_config.json"
        if not config_path.exists():
            raise AirflowFailException(f"Model config file not found: {config_path}")
        
        # PRE-EXECUTION DIAGNOSTICS
        logger.info("=" * 80)
        logger.info("PRE-EXECUTION DIAGNOSTICS: generate_model_responses")
        logger.info("=" * 80)
        env = get_subprocess_env()
        logger.info("Environment variables:")
        logger.info("  DVC_DATA_DIR: %s", env.get("DVC_DATA_DIR", "NOT SET"))
        logger.info("  PROJECT_ROOT: %s", env.get("PROJECT_ROOT", "NOT SET"))
        logger.info("  DVC_PROJECT_DIR: %s", env.get("DVC_PROJECT_DIR", "NOT SET"))
        logger.info("Data directory (DATA_DIR): %s", DATA_DIR)
        logger.info("Data directory exists: %s", DATA_DIR.exists())
        logger.info("Data directory is writable: %s", os.access(DATA_DIR, os.W_OK) if DATA_DIR.exists() else False)
        
        # Load config and log expected output paths
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            models = config_data.get("models", [])
            logger.info("Found %d models in config", len(models))
            
            for model in models:
                model_name = model.get("name", "unknown")
                out_path_str = model.get("out_path", "")
                # Resolve output path
                if out_path_str.startswith("data/"):
                    expected_out_path = DATA_DIR / out_path_str[5:]
                else:
                    expected_out_path = DATA_DIR / "responses" / Path(out_path_str).name
                
                logger.info("Model: %s", model_name)
                logger.info("  Expected output path: %s", expected_out_path)
                logger.info("  Output path exists: %s", expected_out_path.exists())
                if expected_out_path.exists():
                    stat = expected_out_path.stat()
                    logger.info("  File size: %d bytes", stat.st_size)
                    logger.info("  Last modified: %s", datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc))
                    # Count rows if CSV
                    try:
                        import pandas as pd
                        df = pd.read_csv(expected_out_path)
                        logger.info("  Current row count: %d", len(df))
                    except Exception as e:
                        logger.warning("  Could not read CSV to count rows: %s", e)
                else:
                    logger.info("  Will create new file")
        except Exception as e:
            logger.warning("Could not load config for diagnostics: %s", e)
        
        logger.info("=" * 80)
        
        logger.info("Running model response generation using config: %s", config_path)
        result = subprocess.run(
            ["python", str(script_path)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,  # 1 hour timeout for model generation
        )
        
        # POST-EXECUTION VERIFICATION
        logger.info("=" * 80)
        logger.info("POST-EXECUTION VERIFICATION: generate_model_responses")
        logger.info("=" * 80)
        logger.info("Script exit code: %d", result.returncode)
        
        if result.returncode != 0:
            logger.error("Model response generation failed:\nSTDOUT: %s\nSTDERR: %s", result.stdout, result.stderr)
            raise AirflowFailException(
                f"Model response generation failed with exit code {result.returncode}. "
                f"Check logs for details."
            )
        
        # Verify expected output files
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            models = config_data.get("models", [])
            
            files_created = 0
            files_updated = 0
            files_unchanged = 0
            files_missing = 0
            
            # Store pre-execution file sizes for comparison (from PRE-EXECUTION diagnostics)
            # We logged them earlier, but we need to recalculate here since we don't store them
            pre_execution_sizes = {}
            for model in models:
                model_name = model.get("name", "unknown")
                out_path_str = model.get("out_path", "")
                if out_path_str.startswith("data/"):
                    expected_out_path = DATA_DIR / out_path_str[5:]
                else:
                    expected_out_path = DATA_DIR / "responses" / Path(out_path_str).name
                # Note: We can't get pre-execution sizes here since file was already modified
                # Instead, we'll use file modification time and size change as indicators
                pre_execution_sizes[model_name] = 0  # Will be set if we can determine from logs
            
            # Use FULL stdout for status detection (not truncated)
            full_stdout = result.stdout
            
            for model in models:
                model_name = model.get("name", "unknown")
                model_id = model.get("model_id", model_name)
                out_path_str = model.get("out_path", "")
                if out_path_str.startswith("data/"):
                    expected_out_path = DATA_DIR / out_path_str[5:]
                else:
                    expected_out_path = DATA_DIR / "responses" / Path(out_path_str).name
                
                if expected_out_path.exists():
                    stat = expected_out_path.stat()
                    current_size = stat.st_size
                    file_mtime = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
                    # Check if file was modified during task execution (within last 10 minutes)
                    task_start_time = datetime.now(timezone.utc) - timedelta(minutes=10)
                    file_modified_during_task = file_mtime >= task_start_time
                    
                    logger.info("Model %s: File exists at %s", model_name, expected_out_path)
                    logger.info("  File size: %d bytes", current_size)
                    logger.info("  Last modified: %s", file_mtime)
                    logger.info("  Modified during task: %s", file_modified_during_task)
                    
                    # Check model-specific output in FULL stdout (look for model name or output file path)
                    # Each model's output is separated by "Running model:" headers
                    # Format in stdout: "Running model: {name} ({model_id})"
                    model_section_start = full_stdout.find(f"Running model: {model_name}")
                    if model_section_start == -1:
                        # Try with model_id as fallback
                        model_section_start = full_stdout.find(f"Running model: {model_id}")
                    if model_section_start == -1:
                        # Try finding by output file path
                        model_section_start = full_stdout.find(str(expected_out_path))
                    if model_section_start == -1:
                        # Try finding by just the filename
                        model_section_start = full_stdout.find(Path(out_path_str).name)
                    
                    model_section_end = full_stdout.find("\n" + "="*60, model_section_start + 1)
                    if model_section_end == -1:
                        # Look for next "Running model:" or end of string
                        next_model = full_stdout.find("\nRunning model:", model_section_start + 1)
                        model_section_end = next_model if next_model != -1 else len(full_stdout)
                    
                    model_section = full_stdout[model_section_start:model_section_end] if model_section_start != -1 else ""
                    
                    # Also check for output file path in the section to confirm it's the right model
                    out_path_in_section = (str(expected_out_path) in model_section or 
                                          out_path_str in model_section or 
                                          model_name.lower() in model_section.lower() or
                                          model_id.lower() in model_section.lower() or
                                          Path(out_path_str).name in model_section)
                    
                    # Determine status based on model-specific output AND file modification time
                    if "Created new file" in model_section and (out_path_in_section or file_modified_during_task):
                        files_created += 1
                        logger.info("  Status: FILE CREATED")
                    elif ("Appended" in model_section and (out_path_in_section or file_modified_during_task)) or file_modified_during_task:
                        # File was modified during task = file was updated (fallback if stdout parsing fails)
                        files_updated += 1
                        if "Appended" in model_section:
                            logger.info("  Status: FILE UPDATED")
                        else:
                            logger.info("  Status: FILE UPDATED (file modified during task, but stdout section not found)")
                    elif "All prompts already processed" in model_section or "Skipping API calls" in model_section:
                        files_unchanged += 1
                        logger.info("  Status: FILE UNCHANGED (all prompts already processed)")
                    elif not file_modified_during_task:
                        files_unchanged += 1
                        logger.info("  Status: FILE UNCHANGED (no new prompts or skipped)")
                    else:
                        # File was modified but couldn't find evidence in stdout - assume updated
                        files_updated += 1
                        logger.info("  Status: FILE UPDATED (file modified during task)")
                else:
                    files_missing += 1
                    logger.warning("Model %s: Expected file missing: %s", model_name, expected_out_path)
            
            logger.info("Summary: Created=%d, Updated=%d, Unchanged=%d, Missing=%d", 
                       files_created, files_updated, files_unchanged, files_missing)
        except Exception as e:
            logger.warning("Could not verify output files: %s", e)
        
        logger.info("=" * 80)
        
        # Try to parse summary from output (generate_model_responses.py prints summaries)
        logger.info("Model response generation completed successfully")
        logger.info("STDOUT: %s", result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        
        # Return summary info if available
        return {
            "status": "success",
            "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        }

    @task(execution_timeout=timedelta(hours=2))
    def judge_responses() -> dict:
        """Judge model responses using judge LLM."""
        # Note: If generate_model_responses fails, this task will automatically skip
        
        script_path = SCRIPTS_DIR / "judge_responses.py"
        if not script_path.exists():
            raise AirflowFailException(f"Judge responses script not found: {script_path}")
        
        # PRE-EXECUTION DIAGNOSTICS
        logger.info("=" * 80)
        logger.info("PRE-EXECUTION DIAGNOSTICS: judge_responses")
        logger.info("=" * 80)
        env = get_subprocess_env()
        logger.info("Environment variables:")
        logger.info("  DVC_DATA_DIR: %s", env.get("DVC_DATA_DIR", "NOT SET"))
        logger.info("  PROJECT_ROOT: %s", env.get("PROJECT_ROOT", "NOT SET"))
        judge_output_dir = DATA_DIR / "judge"
        logger.info("Judge output directory: %s", judge_output_dir)
        logger.info("Judge output directory exists: %s", judge_output_dir.exists())
        
        # Check for expected input files (response CSVs) and output files (judgement CSVs)
        try:
            config_path = CFG_DIR / "attack_llm_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                models = config_data.get("models", [])
                logger.info("Found %d models in config", len(models))
                
                for model in models:
                    model_name = model.get("name", "unknown")
                    expected_judgement_path = judge_output_dir / f"judgements_{model_name}.csv"
                    logger.info("Model: %s", model_name)
                    logger.info("  Expected judgement file: %s", expected_judgement_path)
                    logger.info("  Judgement file exists: %s", expected_judgement_path.exists())
                    if expected_judgement_path.exists():
                        stat = expected_judgement_path.stat()
                        logger.info("  File size: %d bytes", stat.st_size)
                        logger.info("  Last modified: %s", datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc))
        except Exception as e:
            logger.warning("Could not check expected files: %s", e)
        
        logger.info("=" * 80)
        
        logger.info("Running response judging")
        result = subprocess.run(
            ["python", str(script_path)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=3600,  # 1 hour timeout for response judging
        )
        
        # POST-EXECUTION VERIFICATION
        logger.info("=" * 80)
        logger.info("POST-EXECUTION VERIFICATION: judge_responses")
        logger.info("=" * 80)
        logger.info("Script exit code: %d", result.returncode)
        
        if result.returncode != 0:
            logger.error("Response judging failed:\nSTDOUT: %s\nSTDERR: %s", result.stdout, result.stderr)
            raise AirflowFailException(
                f"Response judging failed with exit code {result.returncode}. "
                f"Check logs for details."
            )
        
        # Verify expected output files
        try:
            config_path = CFG_DIR / "attack_llm_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                models = config_data.get("models", [])
                
                files_found = 0
                for model in models:
                    model_name = model.get("name", "unknown")
                    expected_judgement_path = judge_output_dir / f"judgements_{model_name}.csv"
                    if expected_judgement_path.exists():
                        stat = expected_judgement_path.stat()
                        files_found += 1
                        logger.info("Model %s: Judgement file exists at %s", model_name, expected_judgement_path)
                        logger.info("  File size: %d bytes", stat.st_size)
                        logger.info("  Last modified: %s", datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc))
                    else:
                        logger.warning("Model %s: Expected judgement file missing: %s", model_name, expected_judgement_path)
                
                logger.info("Summary: %d/%d judgement files found", files_found, len(models))
        except Exception as e:
            logger.warning("Could not verify output files: %s", e)
        
        logger.info("=" * 80)
        
        logger.info("Response judging completed successfully")
        return {
            "status": "success",
            "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        }

    @task
    def compute_additional_metrics() -> dict:
        """Compute additional metrics from judged responses."""
        # Note: If generate_model_responses fails, this task will automatically skip
        
        script_path = SCRIPTS_DIR / "additional_metrics.py"
        if not script_path.exists():
            raise AirflowFailException(f"Additional metrics script not found: {script_path}")
        
        # PRE-EXECUTION DIAGNOSTICS
        logger.info("=" * 80)
        logger.info("PRE-EXECUTION DIAGNOSTICS: compute_additional_metrics")
        logger.info("=" * 80)
        env = get_subprocess_env()
        logger.info("Environment variables:")
        logger.info("  DVC_DATA_DIR: %s", env.get("DVC_DATA_DIR", "NOT SET"))
        metrics_output_dir = DATA_DIR / "metrics" / "additional"
        logger.info("Metrics output directory: %s", metrics_output_dir)
        logger.info("Metrics output directory exists: %s", metrics_output_dir.exists())
        
        # Check for expected output files
        try:
            config_path = CFG_DIR / "attack_llm_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                models = config_data.get("models", [])
                logger.info("Found %d models in config", len(models))
                
                for model in models:
                    model_name = model.get("name", "unknown")
                    expected_metrics_path = metrics_output_dir / f"additional_metrics_{model_name}.json"
                    logger.info("Model: %s", model_name)
                    logger.info("  Expected metrics file: %s", expected_metrics_path)
                    logger.info("  Metrics file exists: %s", expected_metrics_path.exists())
                    if expected_metrics_path.exists():
                        stat = expected_metrics_path.stat()
                        logger.info("  File size: %d bytes", stat.st_size)
                        logger.info("  Last modified: %s", datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc))
        except Exception as e:
            logger.warning("Could not check expected files: %s", e)
        
        logger.info("=" * 80)
        
        logger.info("Running additional metrics computation")
        result = subprocess.run(
            ["python", str(script_path)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,  # 5 minute timeout for metrics computation
        )
        
        # POST-EXECUTION VERIFICATION
        logger.info("=" * 80)
        logger.info("POST-EXECUTION VERIFICATION: compute_additional_metrics")
        logger.info("=" * 80)
        logger.info("Script exit code: %d", result.returncode)
        
        if result.returncode != 0:
            logger.error("Additional metrics computation failed:\nSTDOUT: %s\nSTDERR: %s", result.stdout, result.stderr)
            raise AirflowFailException(
                f"Additional metrics computation failed with exit code {result.returncode}. "
                f"Check logs for details."
            )
        
        # Verify expected output files
        try:
            config_path = CFG_DIR / "attack_llm_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                models = config_data.get("models", [])
                
                files_found = 0
                for model in models:
                    model_name = model.get("name", "unknown")
                    expected_metrics_path = metrics_output_dir / f"additional_metrics_{model_name}.json"
                    if expected_metrics_path.exists():
                        stat = expected_metrics_path.stat()
                        files_found += 1
                        logger.info("Model %s: Metrics file exists at %s", model_name, expected_metrics_path)
                        logger.info("  File size: %d bytes", stat.st_size)
                        logger.info("  Last modified: %s", datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc))
                    else:
                        logger.warning("Model %s: Expected metrics file missing: %s", model_name, expected_metrics_path)
                
                logger.info("Summary: %d/%d metrics files found", files_found, len(models))
        except Exception as e:
            logger.warning("Could not verify output files: %s", e)
        
        logger.info("=" * 80)
        
        logger.info("Additional metrics computation completed successfully")
        return {
            "status": "success",
            "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        }

    @task(execution_timeout=timedelta(minutes=10))
    def compute_bias_detection() -> dict:
        """Compute bias detection metrics from judged responses."""
        # Note: If generate_model_responses fails, this task will automatically skip
        
        script_path = SCRIPTS_DIR / "bias_detection.py"
        if not script_path.exists():
            raise AirflowFailException(f"Bias detection script not found: {script_path}")
        
        # PRE-EXECUTION DIAGNOSTICS
        logger.info("=" * 80)
        logger.info("PRE-EXECUTION DIAGNOSTICS: compute_bias_detection")
        logger.info("=" * 80)
        env = get_subprocess_env()
        logger.info("Environment variables:")
        logger.info("  DVC_DATA_DIR: %s", env.get("DVC_DATA_DIR", "NOT SET"))
        bias_output_dir = DATA_DIR / "bias"
        logger.info("Bias output directory: %s", bias_output_dir)
        logger.info("Bias output directory exists: %s", bias_output_dir.exists())
        
        # Check for expected output files
        try:
            config_path = CFG_DIR / "attack_llm_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                models = config_data.get("models", [])
                logger.info("Found %d models in config", len(models))
                
                for model in models:
                    model_name = model.get("name", "unknown")
                    expected_bias_path = bias_output_dir / model_name / "bias_report.json"
                    logger.info("Model: %s", model_name)
                    logger.info("  Expected bias report: %s", expected_bias_path)
                    logger.info("  Bias report exists: %s", expected_bias_path.exists())
                    if expected_bias_path.exists():
                        stat = expected_bias_path.stat()
                        logger.info("  File size: %d bytes", stat.st_size)
                        logger.info("  Last modified: %s", datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc))
        except Exception as e:
            logger.warning("Could not check expected files: %s", e)
        
        logger.info("=" * 80)
        
        logger.info("Running bias detection")
        result = subprocess.run(
            ["python", str(script_path)],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=300,  # 5 minute timeout for bias detection
        )
        
        # POST-EXECUTION VERIFICATION
        logger.info("=" * 80)
        logger.info("POST-EXECUTION VERIFICATION: compute_bias_detection")
        logger.info("=" * 80)
        logger.info("Script exit code: %d", result.returncode)
        
        if result.returncode != 0:
            logger.error("Bias detection failed:\nSTDOUT: %s\nSTDERR: %s", result.stdout, result.stderr)
            raise AirflowFailException(
                f"Bias detection failed with exit code {result.returncode}. "
                f"Check logs for details."
            )
        
        # Verify expected output files
        try:
            config_path = CFG_DIR / "attack_llm_config.json"
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                models = config_data.get("models", [])
                
                files_found = 0
                for model in models:
                    model_name = model.get("name", "unknown")
                    expected_bias_path = bias_output_dir / model_name / "bias_report.json"
                    if expected_bias_path.exists():
                        stat = expected_bias_path.stat()
                        files_found += 1
                        logger.info("Model %s: Bias report exists at %s", model_name, expected_bias_path)
                        logger.info("  File size: %d bytes", stat.st_size)
                        logger.info("  Last modified: %s", datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc))
                    else:
                        logger.warning("Model %s: Expected bias report missing: %s", model_name, expected_bias_path)
                
                logger.info("Summary: %d/%d bias reports found", files_found, len(models))
        except Exception as e:
            logger.warning("Could not verify output files: %s", e)
        
        logger.info("=" * 80)
        
        logger.info("Bias detection completed successfully")
        return {
            "status": "success",
            "stdout": result.stdout[-500:] if len(result.stdout) > 500 else result.stdout,
        }

    @task
    def sync_data_to_gcs() -> dict:
        """Force sync of data files to GCS bucket to ensure visibility.
        
        In Composer, /home/airflow/gcs/ is mounted via gcsfuse to the bucket.
        gcsfuse can have caching delays, so we explicitly copy files to ensure
        they're immediately visible in the bucket.
        
        Strategy: Copy response files directly using gcloud storage cp (more reliable
        than rsync from gcsfuse mount).
        """
        logger.info("=" * 80)
        logger.info("EXPLICIT GCS SYNC: Forcing sync of data files to bucket")
        logger.info("=" * 80)
        
        bucket_name = "us-central1-mlops-airflow-c-47afa20c-bucket"
        data_dir = DATA_DIR
        responses_dir = data_dir / "responses"
        
        logger.info("Data directory path: %s", data_dir)
        logger.info("Responses directory: %s", responses_dir)
        logger.info("Target bucket: gs://%s/dags/dvc_project/data/", bucket_name)
        logger.info("Note: In Composer, this path is mounted via gcsfuse")
        logger.info("Using direct copy to bypass gcsfuse caching delays")
        
        env = os.environ.copy()
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            env["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        
        files_synced = 0
        files_failed = 0
        
        try:
            # First, try to sync the entire data directory using rsync (faster for many files)
            logger.info("Attempting full directory sync with gsutil rsync...")
            result = subprocess.run(
                ["gsutil", "-m", "rsync", "-r", "-d", str(data_dir), 
                 f"gs://{bucket_name}/dags/dvc_project/data/"],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=False,
                env=env,
            )
            
            logger.info("gsutil rsync exit code: %d", result.returncode)
            if result.stdout:
                logger.info("STDOUT: %s", result.stdout[-1000:] if len(result.stdout) > 1000 else result.stdout)
            if result.stderr:
                logger.warning("STDERR: %s", result.stderr[-1000:] if len(result.stderr) > 1000 else result.stderr)
            
            # If rsync failed or we want to be extra sure, copy response files individually
            if result.returncode != 0 or responses_dir.exists():
                logger.info("Copying response files individually to ensure visibility...")
                if responses_dir.exists():
                    response_files = list(responses_dir.glob("*.csv"))
                    logger.info("Found %d response CSV files to sync", len(response_files))
                    
                    for response_file in response_files:
                        try:
                            # Use gcloud storage cp (newer, more reliable than gsutil)
                            target_path = f"gs://{bucket_name}/dags/dvc_project/data/responses/{response_file.name}"
                            logger.info("Copying %s → %s", response_file, target_path)
                            
                            cp_result = subprocess.run(
                                ["gcloud", "storage", "cp", "--content-type", "text/csv", str(response_file), target_path],
                                capture_output=True,
                                text=True,
                                timeout=60,
                                check=False,
                                env=env,
                            )
                            
                            if cp_result.returncode == 0:
                                files_synced += 1
                                logger.info("  ✅ Synced: %s", response_file.name)
                            else:
                                files_failed += 1
                                logger.warning("  ❌ Failed to sync %s: %s", response_file.name, cp_result.stderr[-200:])
                        except Exception as e:
                            files_failed += 1
                            logger.error("  ❌ Error syncing %s: %s", response_file.name, e)
                else:
                    logger.warning("Responses directory not found: %s", responses_dir)
            
            if result.returncode == 0 and files_failed == 0:
                logger.info("✅ Data directory synced to GCS bucket (rsync succeeded)")
                return {"status": "success", "synced": True, "files_synced": files_synced}
            elif files_synced > 0:
                logger.info("✅ Response files synced individually (%d files, %d failed)", files_synced, files_failed)
                return {"status": "partial_success", "files_synced": files_synced, "files_failed": files_failed}
            else:
                logger.warning("⚠️  Sync completed with issues (rsync exit code: %d, files synced: %d, failed: %d)", 
                             result.returncode, files_synced, files_failed)
                return {"status": "warning", "exit_code": result.returncode, "files_synced": files_synced, "files_failed": files_failed}
            
        except subprocess.TimeoutExpired:
            logger.error("Sync timed out after 5 minutes")
            return {"status": "error", "error": "Sync timed out"}
        except Exception as e:
            logger.error("Failed to sync data directory: %s", e, exc_info=True)
            return {"status": "error", "error": str(e)}
    
    @task
    def verify_files_in_gcs() -> dict:
        """Verify that generated files are visible in GCS bucket."""
        try:
            from google.cloud import storage
        except ImportError:
            logger.warning("google-cloud-storage not available - skipping GCS verification")
            return {"status": "skipped", "reason": "google-cloud-storage not available"}
        
        logger.info("=" * 80)
        logger.info("GCS BUCKET VERIFICATION")
        logger.info("=" * 80)
        
        # Get bucket name from environment or use default
        # In Composer, files are written to /home/airflow/gcs which maps to the Composer bucket
        bucket_name = "us-central1-mlops-airflow-c-47afa20c-bucket"
        expected_paths = [
            "dags/dvc_project/data/responses/",
            "dags/dvc_project/data/judge/",
            "dags/dvc_project/data/metrics/additional/",
            "dags/dvc_project/data/bias/",
        ]
        
        try:
            client = storage.Client()
            bucket = client.bucket(bucket_name)
            
            logger.info("Checking bucket: gs://%s", bucket_name)
            
            total_files = 0
            recent_files = 0
            current_time = datetime.now(timezone.utc)
            
            for path_prefix in expected_paths:
                blobs = list(bucket.list_blobs(prefix=path_prefix))
                logger.info("Path: gs://%s/%s", bucket_name, path_prefix)
                logger.info("  Found %d file(s)", len(blobs))
                
                # Log first 5 files with their timestamps
                for blob in blobs[:5]:
                    age_minutes = (current_time - blob.updated.replace(tzinfo=timezone.utc)).total_seconds() / 60
                    is_recent = age_minutes < 30  # Files updated in last 30 minutes
                    if is_recent:
                        recent_files += 1
                    logger.info("    - %s (size: %d bytes, updated: %s, age: %.1f min)", 
                              blob.name, blob.size, blob.updated, age_minutes)
                    total_files += 1
                
                if len(blobs) > 5:
                    logger.info("    ... and %d more file(s)", len(blobs) - 5)
                    total_files += len(blobs) - 5
            
            logger.info("=" * 80)
            logger.info("GCS Verification Summary: %d total files found, %d updated in last 30 min", 
                       total_files, recent_files)
            
            if recent_files == 0 and total_files > 0:
                logger.warning("⚠️  WARNING: Files exist in bucket but none were updated in the last 30 minutes")
                logger.warning("   This suggests files from the latest run are not syncing to GCS")
            elif recent_files > 0:
                logger.info("✅ Found %d recently updated file(s) - sync appears to be working", recent_files)
            
            logger.info("=" * 80)
            
            return {
                "status": "success",
                "bucket": bucket_name,
                "total_files": total_files,
                "recent_files": recent_files,
            }
        except Exception as e:
            logger.error("GCS verification failed: %s", e, exc_info=True)
            return {
                "status": "error",
                "error": str(e),
            }

    # Second DVC push: after model pipeline completes (includes all model outputs: responses, judge, bias, metrics)
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
        bash_command=f'''
    set -euo pipefail
    cd "{DVC_PROJECT_DIR}"

    echo "──────────────────────────────────────────"
    echo "🔎 STEP 1: dvc status (cache/remote delta)"
    python -m dvc status -c -v || true

    echo "──────────────────────────────────────────"
    echo "📊 STEP 1.5: Parsing DVC status (JSON)"
    STATUS_JSON=$(python -m dvc status -c --json 2>/dev/null || echo "{{}}")
    if [ "$STATUS_JSON" != "{{}}" ] && [ -n "$STATUS_JSON" ]; then
        echo "Status summary:"
        echo "$STATUS_JSON" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    total_stages = len(data)
    total_files = sum(len(items) for items in data.values() if isinstance(items, dict))
    print(f'  Stages with changes: {{total_stages}}')
    print(f'  Total files to push: {{total_files}}')
    for stage, items in data.items():
        if isinstance(items, dict) and items:
            file_count = len(items)
            print(f'  - {{stage}}: {{file_count}} file(s)')
except:
    pass
" 2>/dev/null || echo "  (Could not parse status JSON)"
    else
        echo "  No pending changes to push"
    fi

    echo "──────────────────────────────────────────"
    echo "📝 STEP 2: Commit new outputs to DVC (if any)"
    echo "This ensures newly generated model outputs are tracked before pushing"
    set +e  # Temporarily disable exit on error to handle commit failures gracefully
    COMMIT_OUTPUT=$(python -m dvc commit -f 2>&1)
    COMMIT_EXIT=$?
    set -e  # Re-enable exit on error
    echo "$COMMIT_OUTPUT"
    if [ "$COMMIT_EXIT" -ne 0 ]; then
        echo "❌ ERROR: DVC commit failed (exit code: $COMMIT_EXIT)"
        # Check for common errors
        if echo "$COMMIT_OUTPUT" | grep -qi "permission\|access\|credential"; then
            echo "   Error: Authentication/credential issue"
            echo "   Action: Check GOOGLE_APPLICATION_CREDENTIALS and GCP service account permissions"
        elif echo "$COMMIT_OUTPUT" | grep -qi "network\|connection\|timeout"; then
            echo "   Error: Network connectivity issue"
            echo "   Action: Check network connection"
        fi
        echo "❌ Task failed: DVC commit is required before push"
        exit $COMMIT_EXIT
    fi
    echo "✅ DVC commit completed successfully"

    echo "──────────────────────────────────────────"
    echo "🚀 STEP 3: dvc push (sync all artifacts to remote)"
    echo "Pushing all stages including: salad_preprocess, model_responses, judge_outputs, bias_detection, additional_metrics"
    PUSH_OUTPUT=$(python -m dvc push -v 2>&1)
    PUSH_EXIT=$?
    echo "$PUSH_OUTPUT"
    
    # Parse push results and errors
    PUSHED_COUNT=$(echo "$PUSH_OUTPUT" | grep -c "Pushed" 2>/dev/null || echo "0")
    # Handle case where grep returns empty (no matches)
    if [ -z "$PUSHED_COUNT" ]; then
        PUSHED_COUNT=0
    fi
    ERROR_COUNT=$(echo "$PUSH_OUTPUT" | grep -ci "error" 2>/dev/null || echo "0")
    if [ -z "$ERROR_COUNT" ]; then
        ERROR_COUNT=0
    fi
    
    if [ "$PUSH_EXIT" -eq 0 ] && [ "$PUSHED_COUNT" -gt 0 ]; then
        echo "✅ dvc push completed successfully"
        echo "📊 Summary: Pushed $PUSHED_COUNT item(s) to remote"
    elif [ "$PUSH_EXIT" -eq 0 ]; then
        echo "✅ dvc push completed (no new items to push)"
    else
        echo "❌ ERROR: dvc push failed (exit code: $PUSH_EXIT)"
        # Parse common error patterns
        if echo "$PUSH_OUTPUT" | grep -qi "permission\|access\|credential\|unauthorized"; then
            echo "   Error: Authentication/credential issue"
            echo "   Action: Check GOOGLE_APPLICATION_CREDENTIALS and GCP service account permissions"
        elif echo "$PUSH_OUTPUT" | grep -qi "network\|connection\|timeout\|unreachable"; then
            echo "   Error: Network connectivity issue"
            echo "   Action: Check network connection and GCS bucket accessibility"
        elif echo "$PUSH_OUTPUT" | grep -qi "bucket\|not found\|404"; then
            echo "   Error: GCS bucket not found or inaccessible"
            echo "   Action: Verify bucket name and service account has storage.objects.* permissions"
        elif [ "$ERROR_COUNT" -gt 0 ]; then
            echo "   Found $ERROR_COUNT error(s) in output:"
            echo "$PUSH_OUTPUT" | grep -i "error\|failed\|exception" | head -5
        fi
        echo "❌ Task failed: DVC push must complete successfully"
        exit $PUSH_EXIT
    fi

    echo "──────────────────────────────────────────"
    echo "✅ STEP 4: Quick listing of all output directories"
    echo "Processed data:"
    ls -la data/processed || true
    echo "Model responses:"
    ls -la data/responses || true
    echo "Judge outputs:"
    ls -la data/judge || true
    echo "Bias detection:"
    ls -la data/bias || true
    echo "Metrics:"
    ls -la data/metrics || true

    echo "✅ DVC Push (final) complete - all model pipeline outputs pushed to remote"
    ''',
    )

    @task
    def verify_dvc_sync() -> dict:
        """Verify DVC sync was successful by checking for pending changes."""
        import subprocess
        import json
        
        logger.info("Verifying DVC sync completion...")
        
        try:
            # Check DVC status after push
            result = subprocess.run(
                ["python", "-m", "dvc", "status", "-c", "--json"],
                cwd=str(DVC_PROJECT_DIR),
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode != 0:
                logger.warning("DVC status check failed: %s", result.stderr)
                return {
                    "status": "error",
                    "message": f"DVC status check failed: {result.stderr}",
                    "pending": -1,
                }
            
            # Parse JSON output
            status_json = {}
            if result.stdout.strip():
                try:
                    status_json = json.loads(result.stdout)
                except json.JSONDecodeError:
                    logger.warning("Could not parse DVC status JSON: %s", result.stdout)
                    return {
                        "status": "unknown",
                        "message": "Could not parse DVC status",
                        "pending": -1,
                    }
            
            # Check if there are any pending changes
            if not status_json or status_json == {}:
                logger.info("✅ DVC fully synced - no pending changes")
                return {
                    "status": "synced",
                    "message": "DVC fully synced - no pending changes",
                    "pending": 0,
                    "stages": [],
                }
            else:
                # Count pending changes
                total_pending = sum(
                    len(items) if isinstance(items, dict) else 1
                    for items in status_json.values()
                )
                stages_with_changes = list(status_json.keys())
                
                logger.warning(
                    "⚠️  DVC sync incomplete - %d pending change(s) in %d stage(s): %s",
                    total_pending,
                    len(stages_with_changes),
                    ", ".join(stages_with_changes),
                )
                
                return {
                    "status": "pending",
                    "message": f"DVC has {total_pending} pending change(s)",
                    "pending": total_pending,
                    "stages": stages_with_changes,
                    "details": status_json,
                }
                
        except subprocess.TimeoutExpired:
            logger.error("DVC status check timed out")
            return {
                "status": "timeout",
                "message": "DVC status check timed out",
                "pending": -1,
            }
        except Exception as e:
            logger.error("Error verifying DVC sync: %s", str(e), exc_info=True)
            return {
                "status": "error",
                "message": f"Error verifying DVC sync: {str(e)}",
                "pending": -1,
            }

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
    # Sync response files immediately after generation to ensure visibility (gcsfuse may have delays)
    sync_responses = sync_data_to_gcs()
    model_gen >> sync_responses
    # model_judge needs the response files to be synced first
    sync_responses >> model_judge >> [model_metrics, bias_detection]
    # Also sync after metrics/bias for completeness
    sync_gcs = sync_data_to_gcs()
    [model_metrics, bias_detection] >> sync_gcs
    # Verify files are visible in GCS after sync
    verify_gcs = verify_files_in_gcs()
    sync_gcs >> verify_gcs
    verify_gcs >> dvc_push_final >> email_success

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
    
    # Stage 8: DVC Sync Verification (runs after successful push)
    verify_sync = verify_dvc_sync()
    dvc_push_final >> verify_sync
    
    # Stage 9: Sync files to structured paths for API access
    # DVC stores files by hash, but API needs structured paths at bucket root
    # Using gcloud storage cp (natively available in Composer, more reliable than gsutil)
    @task(trigger_rule=TriggerRule.ALL_SUCCESS)
    def copy_to_api_paths() -> dict:
        """Copy files to structured paths for API access using gcloud storage cp."""
        logger.info("=" * 80)
        logger.info("Copying files to structured paths for API access")
        logger.info("Source: %s", DATA_DIR)
        logger.info("Destination: gs://mlops-project-dvc-480422/data/")
        logger.info("=" * 80)
        
        # Setup environment
        env = os.environ.copy()
        if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
            env["GOOGLE_APPLICATION_CREDENTIALS"] = os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
        
        # Track overall success
        files_copied = 0
        files_failed = 0
        metrics_synced = False
        bias_synced = False
        overall_success = True
        
        data_dir = Path(DATA_DIR)
        api_bucket = "gs://mlops-project-dvc-480422/data"
        
        if not data_dir.exists():
            logger.error("Data directory not found at %s", DATA_DIR)
            raise AirflowFailException(f"Data directory not found at {DATA_DIR}")
        
        # Copy metrics/additional/ files
        logger.info("")
        logger.info("Copying data/metrics/additional/ (additional metrics for API)...")
        metrics_dir = data_dir / "metrics" / "additional"
        
        if metrics_dir.exists():
            json_files = list(metrics_dir.glob("*.json"))
            if json_files:
                for json_file in json_files:
                    if json_file.is_file():
                        filename = json_file.name
                        target_path = f"{api_bucket}/metrics/additional/{filename}"
                        logger.info("  Copying: %s", filename)
                        try:
                            cp_result = subprocess.run(
                                ["gcloud", "storage", "cp", "--content-type", "application/json", str(json_file), target_path],
                                capture_output=True,
                                text=True,
                                timeout=60,
                                check=False,
                                env=env,
                            )
                            
                            if cp_result.returncode == 0:
                                files_copied += 1
                                logger.info("    [OK] Copied: %s", filename)
                            else:
                                files_failed += 1
                                logger.error("    [ERROR] Failed: %s - %s", filename, cp_result.stderr[-200:] if cp_result.stderr else "Unknown error")
                                overall_success = False
                        except Exception as e:
                            files_failed += 1
                            logger.error("    [ERROR] Failed: %s - %s", filename, str(e))
                            overall_success = False
                
                if files_failed == 0 and files_copied > 0:
                    logger.info("[SUCCESS] Additional metrics synced (copied %d files)", files_copied)
                    metrics_synced = True
                elif files_copied > 0:
                    logger.warning("[WARNING] %d file(s) failed to sync, %d succeeded", files_failed, files_copied)
                    metrics_synced = True  # Partial success
                else:
                    logger.error("[ERROR] No files were successfully synced")
            else:
                logger.warning("[WARNING] No JSON files found in data/metrics/additional/")
        else:
            logger.warning("[WARNING] data/metrics/additional/ directory not found (may be first run)")
        
        # Reset counters for bias files
        bias_files_copied = 0
        bias_files_failed = 0
        
        # Copy bias/ files
        logger.info("")
        logger.info("Copying data/bias/ (bias reports for API)...")
        bias_dir = data_dir / "bias"
        
        if bias_dir.exists():
            model_dirs = [d for d in bias_dir.iterdir() if d.is_dir()]
            if model_dirs:
                for model_dir in model_dirs:
                    model_name = model_dir.name
                    logger.info("  Copying files for model: %s", model_name)
                    
                    # Copy all files in this model directory
                    for file in model_dir.iterdir():
                        if file.is_file():
                            filename = file.name
                            target_path = f"{api_bucket}/bias/{model_name}/{filename}"
                            logger.info("    Copying: %s/%s", model_name, filename)
                            try:
                                # Determine content-type based on file extension
                                content_type = "application/json" if file.suffix == ".json" else "text/csv" if file.suffix == ".csv" else None
                                cp_cmd = ["gcloud", "storage", "cp", str(file), target_path]
                                if content_type:
                                    cp_cmd.insert(-1, "--content-type")
                                    cp_cmd.insert(-1, content_type)
                                
                                cp_result = subprocess.run(
                                    cp_cmd,
                                    capture_output=True,
                                    text=True,
                                    timeout=60,
                                    check=False,
                                    env=env,
                                )
                                
                                if cp_result.returncode == 0:
                                    bias_files_copied += 1
                                    logger.info("      [OK] Copied: %s/%s", model_name, filename)
                                else:
                                    bias_files_failed += 1
                                    logger.error("      [ERROR] Failed: %s/%s - %s", model_name, filename, cp_result.stderr[-200:] if cp_result.stderr else "Unknown error")
                                    overall_success = False
                            except Exception as e:
                                bias_files_failed += 1
                                logger.error("      [ERROR] Failed: %s/%s - %s", model_name, filename, str(e))
                                overall_success = False
                
                if bias_files_failed == 0 and bias_files_copied > 0:
                    logger.info("[SUCCESS] Bias reports synced (copied %d files)", bias_files_copied)
                    bias_synced = True
                elif bias_files_copied > 0:
                    logger.warning("[WARNING] %d file(s) failed to sync, %d succeeded", bias_files_failed, bias_files_copied)
                    bias_synced = True  # Partial success
                else:
                    logger.error("[ERROR] No bias files were successfully synced")
            else:
                logger.warning("[WARNING] No model subdirectories found in data/bias/")
        else:
            logger.warning("[WARNING] data/bias/ directory not found (may be first run)")
        
        # Summary
        logger.info("")
        if metrics_synced or bias_synced:
            logger.info("[SUCCESS] API-required files synced to structured paths")
            if metrics_synced:
                logger.info("   ✓ Additional metrics: %s/metrics/additional/", api_bucket)
            if bias_synced:
                logger.info("   ✓ Bias reports: %s/bias/", api_bucket)
            return {
                "status": "success",
                "metrics_synced": metrics_synced,
                "bias_synced": bias_synced,
                "files_copied": files_copied + bias_files_copied,
                "files_failed": files_failed + bias_files_failed,
            }
        elif not overall_success:
            logger.error("[ERROR] Some files failed to sync")
            raise AirflowFailException("Some files failed to sync to API paths")
        else:
            logger.warning("[WARNING] No files were found to sync (may be first run or directories empty)")
            return {
                "status": "no_files",
                "metrics_synced": False,
                "bias_synced": False,
                "files_copied": 0,
                "files_failed": 0,
            }
    
    copy_to_api_paths_task = copy_to_api_paths()
    dvc_push_final >> copy_to_api_paths_task


dag = salad_ml_evaluation_pipeline_v1()