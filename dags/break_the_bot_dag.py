# dags/salad_pipeline_v2.py
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import os, json, hashlib, logging, subprocess

from airflow.decorators import dag, task, task_group
from airflow.operators.python import get_current_context, ShortCircuitOperator
from airflow.operators.email import EmailOperator
from airflow.utils.trigger_rule import TriggerRule
from airflow.exceptions import AirflowFailException
from airflow.operators.bash import BashOperator

# === External scripts (unchanged calls) ===
from scripts.preprocess_salad import run_preprocessing
from scripts.generate_model_responses import run_model_response_generation

logger = logging.getLogger(__name__)

# === Repo layout ===
REPO_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = REPO_ROOT / "data"
OUT_DIR = DATA_DIR / "processed"
CFG_DIR = REPO_ROOT / "config"
ARTIFACTS_DIR = REPO_ROOT / "airflow_artifacts"
REPORTS_DIR = ARTIFACTS_DIR / "reports"
STATE_DIR = ARTIFACTS_DIR / "state"    # for tiny persistent state like hashes

# === Inputs/Outputs ===
CONFIG_PATH = Path(os.environ.get("SALAD_CONFIG_PATH", CFG_DIR / "data_sources.json"))
PROCESSED_CSV = Path(os.environ.get("SALAD_OUTPUT_PATH", OUT_DIR / "processed_data.csv"))
MODEL_RESPONSES = OUT_DIR / "model_responses.csv"

# === GE runner (same as before) ===
SCRIPT_GE = REPO_ROOT / "scripts" / "ge_runner.py"
METRICS_DIR = DATA_DIR / "metrics"
BASELINE_SCHEMA = METRICS_DIR / "schema" / "baseline" / "schema.json"

# === Airflow Variables / toggles ===
# You can still flip SAMPLE/N_SAMPLES in the script; these are just for future use or logging
TEST_MODE = (os.environ.get("TEST_MODE", "false").lower() == "true")

EMAIL_TO = ["yashi.chawla1@gmail.com", "chawla.y@northeastern.edu"]

# Track/push these paths via DVC
DVC_TRACK_PATHS = [
    "data/processed",
    "airflow_artifacts/reports",
]

# ========================= Helpers =========================

def _sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except Exception:
        return None

def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)

# ========================= DAG =========================

@dag(
    dag_id="salad_pipeline_v2",
    description="Preprocess, validate, (conditionally) generate model responses, DVC push, and notify.",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "owner": "airflow",
    },
    tags=["salad", "v2", "preprocess", "validate", "generate", "mlops"],
)
def salad_pipeline_v2():

    # ---------- Setup / DVC pull ----------
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

            python -m pip install --no-cache-dir -q "dvc[gs]" gcsfs google-cloud-storage

            # clear any forced credentialpath so env is used
            python -m dvc config --local --unset remote.gcsremote.credentialpath || true

            python -m dvc checkout -f || true
            python -m dvc pull -v -f
        ''',
    )

    @task_group(group_id="setup")
    def setup_group():
        @task
        def ensure_dirs() -> dict:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            CFG_DIR.mkdir(parents=True, exist_ok=True)
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            METRICS_DIR.mkdir(parents=True, exist_ok=True)
            logger.info("Dirs ensured: %s %s %s %s %s",
                        DATA_DIR, OUT_DIR, CFG_DIR, REPORTS_DIR, STATE_DIR)
            return {
                "config_path": str(CONFIG_PATH),
                "processed_path": str(PROCESSED_CSV),
                "reports_dir": str(REPORTS_DIR),
                "state_dir": str(STATE_DIR),
            }

        @task
        def ensure_config(paths: dict) -> str:
            cfg_path = Path(paths["config_path"])
            if not cfg_path.exists():
                default_cfg = {
                    "data_sources": [
                        {
                            "type": "hf",
                            "name": "OpenSafetyLab/Salad-Data",
                            "config": "attack_enhanced_set",
                            "split": "train",
                        }
                    ]
                }
                cfg_path.write_text(json.dumps(default_cfg, indent=2))
                logger.info("Created default config at %s", cfg_path)
            else:
                logger.info("Using existing config at %s", cfg_path)

            try:
                cfg = json.loads(cfg_path.read_text())
                if not isinstance(cfg.get("data_sources"), list) or not cfg["data_sources"]:
                    raise AirflowFailException("Config invalid: 'data_sources' missing/empty.")
            except Exception as e:
                raise AirflowFailException(f"Failed to parse/validate config: {e}")
            return str(cfg_path)

        return ensure_dirs(), ensure_config(ensure_dirs())

    setup_out, cfg_out = setup_group()

    # ---------- Preprocess ----------
    @task
    def preprocess_input_csv(cfg_path: str) -> str:
        if TEST_MODE:
            # In test mode, expect a prepared test CSV (optional behavior)
            test_csv = DATA_DIR / "test_validation" / "test.csv"
            if not test_csv.exists():
                raise AirflowFailException(f"TEST_MODE ON but missing {test_csv}")
            logger.warning("TEST_MODE ON — using %s", test_csv)
            return str(test_csv)

        run_preprocessing(config_path=cfg_path, save_path=str(PROCESSED_CSV))
        if not PROCESSED_CSV.exists() or PROCESSED_CSV.stat().st_size == 0:
            raise AirflowFailException(f"Expected preprocessed CSV missing/empty: {PROCESSED_CSV}")
        logger.info("Preprocessing OK: %s", PROCESSED_CSV)
        return str(PROCESSED_CSV)

    preprocessed_csv = preprocess_input_csv(cfg_out)

    # ---------- Validate (GE once, then read metrics) ----------
    @task_group(group_id="validate")
    def validate_group():
        @task
        def run_ge_and_collect(csv_path: str) -> dict:
            """Run GE baseline if missing, then validate; collect metrics from artifacts."""
            ds_nodash = get_current_context()["ds_nodash"]

            # run baseline once if missing
            try:
                if not BASELINE_SCHEMA.exists():
                    subprocess.run(
                        ["python", str(SCRIPT_GE), "baseline", "--input", csv_path, "--date", ds_nodash],
                        check=False,
                    )
                # validate
                res = subprocess.run(
                    ["python", str(SCRIPT_GE), "validate",
                     "--input", csv_path,
                     "--baseline_schema", str(BASELINE_SCHEMA),
                     "--date", ds_nodash],
                    check=False,
                )
                if res.returncode not in (0, 1):
                    logger.warning("GE validate returned code %s", res.returncode)
            except Exception as e:
                logger.warning("GE validation invocation failed (non-blocking): %s", e)

            anomalies_path = METRICS_DIR / "validation" / ds_nodash / "anomalies.json"
            stats_path = METRICS_DIR / "stats" / ds_nodash / "stats.json"

            metrics = {
                "row_count": 0,
                "null_prompt_count": None,
                "dup_prompt_count": None,
                "unknown_category_rate": None,
                "text_len_min": None,
                "text_len_max": None,
                "size_label_mismatch_count": None,
                "hard_fail": [],
                "soft_warn": [],
                "report_paths": [str(anomalies_path), str(stats_path)],
            }
            try:
                if stats_path.exists():
                    s = json.loads(stats_path.read_text())
                    metrics.update({
                        "row_count": s.get("row_count", 0),
                        "null_prompt_count": s.get("null_prompt_count"),
                        "dup_prompt_count": s.get("dup_prompt_count"),
                        "unknown_category_rate": s.get("unknown_category_rate"),
                        "text_len_min": s.get("text_len_min"),
                        "text_len_max": s.get("text_len_max"),
                        "size_label_mismatch_count": s.get("size_label_mismatch_count"),
                    })
                if anomalies_path.exists():
                    a = json.loads(anomalies_path.read_text())
                    metrics.update({
                        "hard_fail": a.get("hard_fail", []),
                        "soft_warn": a.get("soft_warn", []),
                    })
            except Exception as e:
                logger.warning("Failed reading GE artifacts: %s", e)
            return metrics

        @task
        def enforce(metrics: dict) -> None:
            if not metrics:
                raise AirflowFailException("Validation metrics missing.")
            hard = metrics.get("hard_fail") or []
            if hard:
                raise AirflowFailException(f"Validation hard-failed. See reports: {metrics.get('report_paths')}")

        return run_ge_and_collect, enforce

    run_ge_and_collect, enforce = validate_group()
    metrics = run_ge_and_collect(preprocessed_csv)
    enforce_task = enforce(metrics)

    # ---------- Change detection (skip generation if no new data) ----------
    @task
    def detect_change(csv_path: str) -> bool:
        """Returns True if processed CSV content is new (different hash) since last successful run."""
        current_hash = _sha1_file(Path(csv_path))
        hash_file = STATE_DIR / "processed_data.sha1"
        previous_hash = _read_text(hash_file)
        if previous_hash == current_hash:
            logger.info("No change in processed_data.csv (hash=%s) — skipping generation.", current_hash[:12])
            return False
        _write_text(hash_file, current_hash)
        logger.info("Change detected (old=%s, new=%s) — will generate.",
                    (previous_hash or "None")[:12], current_hash[:12])
        return True

    should_generate = detect_change(preprocessed_csv)

    gate_generate = ShortCircuitOperator(
        task_id="gate_generate_if_changed",
        python_callable=lambda x: x,
        op_args=[should_generate],
    )

    # ---------- Generate model responses (unchanged function call) ----------
    @task(task_id="generate_model_responses", retries=1, retry_delay=timedelta(minutes=2))
    def generate_model_responses_task() -> dict:
        summary = run_model_response_generation()
        logger.info("Generation summary: %s", summary)
        return summary

    gen_summary = generate_model_responses_task()

    # ---------- Versioning (DVC push) ----------
    dvc_push = BashOperator(
        task_id="dvc_push",
        trigger_rule=TriggerRule.ALL_SUCCESS,
        env={
            "REPO_ROOT": REPO_ROOT,
            "GOOGLE_APPLICATION_CREDENTIALS": "/opt/airflow/secrets/gcp-key.json",
            "DVC_NO_ANALYTICS": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        },
        bash_command="""{% raw %}
set -euo pipefail
cd "$REPO_ROOT"

echo "──────────────────────────────────────────"
echo "🔎 dvc status"
python -m dvc status -c -v || true

echo "──────────────────────────────────────────"
echo "🚀 dvc push"
python -m dvc push -v

echo "──────────────────────────────────────────"
echo "✅ outputs"
ls -la data/processed || true
{% endraw %}""",
    )

    # ---------- Emails ----------
    email_validation_report = EmailOperator(
        task_id="email_validation_report",
        to=EMAIL_TO,
        subject="[Airflow][{{ dag.dag_id }}][{{ ds }}] Validation Report",
        html_content="""
            <h3>Validation Report for {{ dag.dag_id }}</h3>
            <p><b>Run:</b> {{ run_id }} | <b>Execution date:</b> {{ ds }}</p>
            <p><b>Processed CSV:</b> {{ ti.xcom_pull(task_ids='preprocess_input_csv') }}</p>
            {% set m = ti.xcom_pull(task_ids='validate.run_ge_and_collect') %}
            {% if m %}
            <ul>
              <li>Rows: {{ m['row_count'] }}</li>
              <li>Null Prompts: {{ m['null_prompt_count'] }}</li>
              <li>Duplicates: {{ m['dup_prompt_count'] }}</li>
              <li>Unknown Category Rate: {{ '%.3f'|format(m['unknown_category_rate'] or 0) }}</li>
              <li>Text Length min/max: {{ m['text_len_min'] }}/{{ m['text_len_max'] }}</li>
              <li>Hard Fail: {{ m['hard_fail'] }}</li>
              <li>Soft Warn: {{ m['soft_warn'] }}</li>
              <li>Reports: {{ m['report_paths'] }}</li>
            </ul>
            {% else %}
              <p>No metrics available.</p>
            {% endif %}
        """,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    email_success = EmailOperator(
        task_id="email_success",
        to=EMAIL_TO,
        subject="[Airflow][{{ dag.dag_id }}][{{ ds }}] ✅ Succeeded",
        html_content="""
            <h3>DAG Succeeded: {{ dag.dag_id }}</h3>
            <p><b>Run:</b> {{ run_id }}</p>
            <p><b>Processed CSV:</b> {{ ti.xcom_pull(task_ids='preprocess_input_csv') }}</p>
            {% set g = ti.xcom_pull(task_ids='generate_model_responses') %}
            {% if g %}
              <p><b>Model Generation:</b> rows={{ g['rows_written'] }}, ok={{ g['ok'] }}, errors={{ g['errors'] }}, avg_latency_ms={{ g['avg_latency_ms'] }}</p>
              <p><b>Output file:</b> {{ g['output_file'] }}</p>
            {% else %}
              <p><b>Model Generation:</b> skipped (no data change)</p>
            {% endif %}
        """,
        trigger_rule=TriggerRule.ALL_SUCCESS,
    )

    email_failure = EmailOperator(
        task_id="email_failure",
        to=EMAIL_TO,
        subject="[Airflow][{{ dag.dag_id }}][{{ ds }}] ❌ Failed",
        html_content="""
            <h3>DAG Failed: {{ dag.dag_id }}</h3>
            <p><b>Run:</b> {{ run_id }}</p>
            <p>One or more tasks failed. Check Airflow logs/UI.</p>
            {% set m = ti.xcom_pull(task_ids='validate.run_ge_and_collect') %}
            {% if m %}
              <p><b>Validation summary:</b>
              rows={{ m['row_count'] }},
              nulls={{ m['null_prompt_count'] }},
              dups={{ m['dup_prompt_count'] }},
              unknown_rate={{ '%.3f'|format(m['unknown_category_rate'] or 0) }},
              text_len=[{{ m['text_len_min'] }},{{ m['text_len_max'] }}],
              hard_fail={{ m['hard_fail'] }},
              soft_warn={{ m['soft_warn'] }},
              reports={{ m['report_paths'] }}</p>
            {% endif %}
        """,
        trigger_rule=TriggerRule.ONE_FAILED,
    )

    # ---------- Orchestration ----------
    dvc_pull >> setup_out >> cfg_out >> preprocessed_csv
    # validate
    metrics.set_upstream(preprocessed_csv)
    enforce_task.set_upstream(metrics)
    email_validation_report.set_upstream(metrics)

    # change detection gate → generate → push
    should_gen = detect_change(preprocessed_csv)
    gate_generate.set_upstream(should_gen)
    gen_summary.set_upstream(gate_generate)

    # success path
    gen_summary >> dvc_push >> email_success

    # failure path
    [preprocessed_csv, metrics, enforce_task] >> email_failure


dag = salad_pipeline_v2()
