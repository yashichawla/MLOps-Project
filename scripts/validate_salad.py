# scripts/validate_salad.py
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import logging
import pandas as pd
from typing import Iterable, List, Union

REQUIRED_COLS = ["prompt", "category", "prompt_id", "text_length", "size_label"]
ALLOWED_SIZE = {"S", "M", "L"}
ALLOWED_CATEGORIES = None  # None = skip strict allowlist for v1

DQ_MIN_ROWS = 50
DQ_MAX_TEXT_LEN = 8000
DQ_MAX_UNKNOWN_RATE = 0.30  # soft warn only

# Airflow logs will capture this logger automatically inside tasks
logger = logging.getLogger(__name__)


def _size_ok(n: int) -> str:
    if n < 50:
        return "S"
    if n <= 200:
        return "M"
    return "L"


def validate_output_csv(
    csv_path: str,
    report_dirs: Union[str, Path, Iterable[Union[str, Path]]],
) -> dict:
    """
    Validate the preprocessed output CSV and write a JSON report
    into one or more report directories.

    Parameters
    ----------
    csv_path : str
        Path to processed_data.csv
    report_dirs : str | Path | Iterable[str|Path]
        One or more directories where validation_report.json will be written.
        Each directory will receive the same report JSON under a timestamped subfolder.

    Returns
    -------
    dict
        Metrics dict. Caller can decide whether to fail on hard_fail.
    """
    # Normalize report_dirs to a list
    if isinstance(report_dirs, (str, Path)):
        report_dirs = [report_dirs]  
    report_dirs = [Path(d) for d in report_dirs]

    p = Path(csv_path)
    if not p.exists() or p.stat().st_size == 0:
        msg = f"Output not found or empty: {p}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    logger.info("Reading CSV for validation: %s", p)
    df = pd.read_csv(p)

    metrics = {
        "row_count": int(len(df)),
        "hard_fail": False,
        "soft_warn": False,
        "null_prompt_count": 0,
        "invalid_category_count": 0,
        "unknown_category_rate": 0.0,
        "dup_prompt_count": 0,
        "text_len_min": None,
        "text_len_max": None,
        "size_label_mismatch_count": 0,
        "csv_path": str(p),
        "report_paths": [], 
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    }

    # --- HARD checks ---
    # Schema
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        metrics["hard_fail"] = True
        metrics["missing_columns"] = missing
        logger.error("Missing required columns: %s", missing)
        _write_report_to_all(report_dirs, metrics)
        return metrics

    # Minimum rows
    if len(df) < DQ_MIN_ROWS:
        metrics["hard_fail"] = True
        logger.error("Row count too low: %d (min required: %d)", len(df), DQ_MIN_ROWS)

    # Null/empty prompt
    null_prompts = df["prompt"].isna().sum() + (df["prompt"].astype(str).str.strip() == "").sum()
    metrics["null_prompt_count"] = int(null_prompts)
    if null_prompts > 0:
        metrics["hard_fail"] = True
        logger.error("Found null/empty prompts: %d", null_prompts)

    # Duplicate prompts (not failing on v1, but track)
    dup_prompts = int(df.duplicated(subset=["prompt"]).sum())
    metrics["dup_prompt_count"] = dup_prompts
    if dup_prompts > 0:
        logger.warning("Duplicate prompts detected (non-fatal in v1): %d", dup_prompts)

    # Category validity (optional strict mode)
    if ALLOWED_CATEGORIES is not None:
        invalid_mask = ~df["category"].isin(ALLOWED_CATEGORIES)
        invalid = int(invalid_mask.sum())
        metrics["invalid_category_count"] = invalid
        if invalid > 0:
            metrics["hard_fail"] = True
            logger.error("Invalid categories encountered: %d", invalid)

    # Unknown rate (soft warn only)
    if "Unknown" in df["category"].unique():
        unknown_rate = float((df["category"] == "Unknown").mean())
        metrics["unknown_category_rate"] = unknown_rate
        if unknown_rate > DQ_MAX_UNKNOWN_RATE:
            metrics["soft_warn"] = True
            logger.warning(
                "Unknown category rate high: %.2f (threshold=%.2f)",
                unknown_rate, DQ_MAX_UNKNOWN_RATE
            )

    # text_length sanity + size_label check
    try:
        tl = df["text_length"].astype(int)
        metrics["text_len_min"] = int(tl.min())
        metrics["text_len_max"] = int(tl.max())
        if metrics["text_len_max"] > DQ_MAX_TEXT_LEN:
            metrics["soft_warn"] = True
            logger.warning(
                "Max text_length high: %d (threshold=%d)",
                metrics["text_len_max"], DQ_MAX_TEXT_LEN
            )
        sizes = df["size_label"].astype(str)
        predicted = tl.apply(_size_ok)
        mismatch = int((sizes != predicted).sum())
        metrics["size_label_mismatch_count"] = mismatch
        if mismatch > 0:
            metrics["soft_warn"] = True
            logger.warning("Size label mismatches detected: %d", mismatch)
    except Exception as e:
        metrics["hard_fail"] = True
        logger.exception("Failed size/length checks: %s", e)

    # Write report(s)
    report_paths = _write_report_to_all(report_dirs, metrics)
    metrics["report_paths"] = [str(r) for r in report_paths]

    # Final summary line for quick scanning in Airflow logs
    level = logging.ERROR if metrics["hard_fail"] else (logging.WARNING if metrics["soft_warn"] else logging.INFO)
    logger.log(
        level,
        "Validation summary | rows=%d nulls=%d dups=%d unknown_rate=%.3f text_len=[%s,%s] mismatches=%d hard_fail=%s soft_warn=%s",
        metrics["row_count"], metrics["null_prompt_count"], metrics["dup_prompt_count"],
        metrics["unknown_category_rate"], metrics["text_len_min"], metrics["text_len_max"],
        metrics["size_label_mismatch_count"], metrics["hard_fail"], metrics["soft_warn"]
    )

    return metrics


def _write_report_to_all(report_dirs: List[Path], metrics: dict) -> List[Path]:
    """
    Write the same JSON report with timestamped names.
    Returns the concrete file paths written.
    """
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    paths = []
    for base in report_dirs:
        base.mkdir(parents=True, exist_ok=True)
        out_path = base / f"validation_{ts}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Wrote validation report: %s", out_path)
        paths.append(out_path)
    return paths