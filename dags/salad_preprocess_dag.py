# dags/salad_preprocess_dag.py
"""
SALAD preprocessing DAG

Runs a multi-step, file-based preprocessing pipeline:
  1) ensure_dirs  – create tmp/output folders
  2) t_load       – load HF SALAD split -> write 00_raw.parquet
  3) t_remove_duplicates -> 01_dedup.parquet
  4) t_map_categories   -> 02_mapped.parquet
  5) t_drop_cols        -> 03_dropcols.parquet
  6) t_clean_nulls      -> 04_clean.parquet + data/processed/salad_cleaned.csv

Outputs:
  - TMP parquet steps under data/tmp/salad/
  - Final CSV under data/processed/salad_cleaned.csv

Notes:
  - Airflow captures Python logging; check each task’s “Log” in the UI.
  - Requires: datasets, pandas, pyarrow (for parquet).
"""

from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import os
import logging
import pandas as pd
from airflow.decorators import dag, task, setup

from scripts.preprocess_salad import (
    load_salad_dataset,
    remove_duplicates,
    map_categories,
    drop_unnecessary_columns,
    clean_null_values,
)

# Use a module-level logger; Airflow will route this to task logs.
logger = logging.getLogger(__name__)


REPO_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = REPO_ROOT / "data"
TMP_DIR = DATA_DIR / "tmp" / "salad"
OUT_DIR = DATA_DIR / "processed"


@dag(
    dag_id="salad_preprocess_multistep",
    description="Multi-step SALAD preprocessing pipeline (file-based, parquet between steps).",
    start_date=datetime(2025, 1, 1),
    schedule=None,  # set a cron later if you want (e.g., "0 3 * * *")
    catchup=False,
    default_args={
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
        "owner": "airflow",
    },
    tags=["salad", "preprocessing", "mlops"],
)
def salad_preprocess_multistep():
    """Define the SALAD multi-step preprocessing DAG."""

    @setup
    @task
    def ensure_dirs() -> None:
        """Create the tmp/output directories if they don't exist."""
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(
            "Ensured directories:\n  TMP_DIR=%s\n  OUT_DIR=%s", TMP_DIR, OUT_DIR
        )

    @task
    def t_load() -> str:
        """
        Load the HF SALAD split and persist the raw dataframe to parquet.

        Returns
        -------
        str
            Path to 00_raw.parquet
        """
        logger.info("Loading SALAD dataset from Hugging Face…")
        df = load_salad_dataset()
        logger.info("Loaded dataframe shape: %s", df.shape)

        out_path = TMP_DIR / "00_raw.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Wrote raw parquet: %s", out_path)
        return str(out_path)

    @task
    def t_remove_duplicates(in_path: str) -> str:
        """
        Drop duplicate (baseq, augq) rows.

        Parameters
        ----------
        in_path : str
            Path to input parquet from previous step.

        Returns
        -------
        str
            Path to 01_dedup.parquet
        """
        logger.info("Reading raw parquet: %s", in_path)
        df = pd.read_parquet(in_path)
        before = df.shape
        df = remove_duplicates(df)
        after = df.shape
        logger.info("Deduplicated: %s -> %s", before, after)

        out_path = TMP_DIR / "01_dedup.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Wrote dedup parquet: %s", out_path)
        return str(out_path)

    @task
    def t_map_categories(in_path: str) -> str:
        """
        Map raw category labels to standardized categories.

        Returns
        -------
        str
            Path to 02_mapped.parquet
        """
        logger.info("Reading dedup parquet: %s", in_path)
        df = pd.read_parquet(in_path)
        df = map_categories(df)
        logger.info(
            "Category mapping complete. Unique std_category: %s",
            df["std_category"].nunique(),
        )

        out_path = TMP_DIR / "02_mapped.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Wrote mapped parquet: %s", out_path)
        return str(out_path)

    @task
    def t_drop_cols(in_path: str) -> str:
        """
        Drop unnecessary columns.

        Returns
        -------
        str
            Path to 03_dropcols.parquet
        """
        logger.info("Reading mapped parquet: %s", in_path)
        df = pd.read_parquet(in_path)
        before_cols = list(df.columns)
        df = drop_unnecessary_columns(df)
        after_cols = list(df.columns)
        logger.info(
            "Dropped columns. Before: %d, After: %d", len(before_cols), len(after_cols)
        )
        logger.debug("Remaining columns: %s", after_cols)

        out_path = TMP_DIR / "03_dropcols.parquet"
        df.to_parquet(out_path, index=False)
        logger.info("Wrote dropcols parquet: %s", out_path)
        return str(out_path)

    @task
    def t_clean_nulls(in_path: str) -> str:
        """
        Remove null/empty question rows, write final parquet and CSV.

        Returns
        -------
        str
            Path to final CSV at data/processed/salad_cleaned.csv
        """
        logger.info("Reading dropcols parquet: %s", in_path)
        df = pd.read_parquet(in_path)
        before = len(df)
        df = clean_null_values(df)
        after = len(df)
        logger.info("Cleaned nulls/empties: %d -> %d rows", before, after)

        final_parquet = TMP_DIR / "04_clean.parquet"
        final_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(final_parquet, index=False)
        logger.info("Wrote final parquet: %s", final_parquet)

        final_csv = OUT_DIR / "salad_cleaned.csv"
        final_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(final_csv, index=False)
        logger.info("Wrote final CSV: %s (rows=%d, cols=%d)", final_csv, *df.shape)
        return str(final_csv)

    # Dependencies: setup → linear chain of tasks
    s = ensure_dirs()
    p0 = t_load()
    p1 = t_remove_duplicates(p0)
    p2 = t_map_categories(p1)
    p3 = t_drop_cols(p2)
    s >> p0  # ensure directories before first step
    _ = t_clean_nulls(p3)


# DAG object for the scheduler
dag = salad_preprocess_multistep()
