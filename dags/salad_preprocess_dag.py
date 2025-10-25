# dags/salad_preprocess_dag.py
from __future__ import annotations
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd
from airflow.decorators import dag, task, setup
from scripts.preprocess_salad import (
    load_salad_dataset,
    remove_duplicates,
    map_categories,
    drop_unnecessary_columns,
    clean_null_values,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
TMP_DIR = DATA_DIR / "tmp" / "salad"
OUT_DIR = DATA_DIR / "processed"


@dag(
    dag_id="salad_preprocess_multistep",
    start_date=datetime(2025, 1, 1),
    schedule=None,
    catchup=False,
    default_args={"retries": 1, "retry_delay": timedelta(minutes=5)},
    tags=["salad", "preprocessing"],
)
def salad_preprocess_multistep():

    @setup
    @task
    def ensure_dirs() -> None:
        TMP_DIR.mkdir(parents=True, exist_ok=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)

    @task
    def t_load() -> str:
        df = load_salad_dataset()
        out_path = TMP_DIR / "00_raw.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        return str(out_path)

    @task
    def t_remove_duplicates(in_path: str) -> str:
        df = pd.read_parquet(in_path)
        df = remove_duplicates(df)
        out_path = TMP_DIR / "01_dedup.parquet"
        df.to_parquet(out_path, index=False)
        return str(out_path)

    @task
    def t_map_categories(in_path: str) -> str:
        df = pd.read_parquet(in_path)
        df = map_categories(df)
        out_path = TMP_DIR / "02_mapped.parquet"
        df.to_parquet(out_path, index=False)
        return str(out_path)

    @task
    def t_drop_cols(in_path: str) -> str:
        df = pd.read_parquet(in_path)
        df = drop_unnecessary_columns(df)
        out_path = TMP_DIR / "03_dropcols.parquet"
        df.to_parquet(out_path, index=False)
        return str(out_path)

    @task
    def t_clean_nulls(in_path: str) -> str:
        df = pd.read_parquet(in_path)
        df = clean_null_values(df)
        final_parquet = TMP_DIR / "04_clean.parquet"
        final_parquet.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(final_parquet, index=False)
        final_csv = OUT_DIR / "salad_cleaned.csv"
        final_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(final_csv, index=False)
        return str(final_csv)

    # dependencies: setup -> linear chain
    s = ensure_dirs()
    p0 = t_load()
    p1 = t_remove_duplicates(p0)
    p2 = t_map_categories(p1)
    p3 = t_drop_cols(p2)
    s >> p0  # only ensure_dirs -> first task
    _ = t_clean_nulls(p3)


dag = salad_preprocess_multistep()
