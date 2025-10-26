# dags/salad_preprocess_dag.py
"""
SALAD preprocessing DAG

New pipeline uses scripts/preprocess_salad_data.run_preprocessing() which:
  - loads multiple sources via config/config.json
  - cleans, maps categories, adds metadata
  - writes: data/processed/processed_data.csv
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import os
import json
import logging
from airflow.decorators import dag, task, setup
from airflow.exceptions import AirflowFailException
from scripts.preprocess_salad import run_preprocessing

logger = logging.getLogger(__name__)

# repo layout
REPO_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR  = REPO_ROOT / "data"
OUT_DIR   = DATA_DIR / "processed"
CFG_DIR   = REPO_ROOT / "config"

# allow override via env/Variables if you want
CONFIG_PATH = Path(os.environ.get("SALAD_CONFIG_PATH", CFG_DIR / "data_sources.json"))
OUTPUT_PATH = Path(os.environ.get("SALAD_OUTPUT_PATH", OUT_DIR / "processed_data.csv"))

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
    description="Unified preprocessing for Salad-Data + other sources via config.",
    start_date=datetime(2025, 1, 1),
    schedule=None,          # set cron later if needed
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "owner": "airflow",
    },
    tags=["salad", "preprocessing", "v1", "mlops"],
)
def salad_preprocess_v1():

    @setup
    @task
    def ensure_dirs() -> dict:
        """Make sure data/, data/processed/, config/ exist."""
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        CFG_DIR.mkdir(parents=True, exist_ok=True)
        logger.info("Dirs ensured: DATA_DIR=%s OUT_DIR=%s CFG_DIR=%s", DATA_DIR, OUT_DIR, CFG_DIR)
        return {"config_path": str(CONFIG_PATH), "output_path": str(OUTPUT_PATH)}

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
    def run_preprocess_task(cfg_path: str) -> str:
        """
        Call the new preprocessing entrypoint. Writes a single CSV.
        Returns the output file path.
        """
        out_path = str(OUTPUT_PATH)
        logger.info("Running preprocessing with config=%s -> %s", cfg_path, out_path)
        run_preprocessing(config_path=cfg_path, save_path=out_path)

        if not Path(out_path).exists() or Path(out_path).stat().st_size == 0:
            raise AirflowFailException(f"Expected output not found or empty: {out_path}")

        logger.info("Preprocessing complete. Output at %s", out_path)
        return out_path

    paths = ensure_dirs()
    cfg   = ensure_config(paths)
    final = run_preprocess_task(cfg)

dag = salad_preprocess_v1()