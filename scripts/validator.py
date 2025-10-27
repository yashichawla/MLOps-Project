#!/usr/bin/env python3
import os, json, datetime as dt
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

# ----- constants (tweak as needed) -----
REQUIRED_COLS = ["prompt", "category", "prompt_id", "text_length", "size_label"]
MIN_ROWS_HARD = 50
TEXT_LEN_MAX_SOFT_WARN = 8000
UNKNOWN_FRACTION_SOFT_WARN = 0.30

def _ensure_parent(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def _nonempty(s: pd.Series) -> pd.Series:
    return (~s.isna()) & (s.astype(str).str.strip() != "")

def _size_from_len(n: int) -> str:
    if n < 50: return "S"
    if n <= 200: return "M"
    return "L"

def run_validation(
    input_csv: str,
    metrics_root: str,
    date_str: Optional[str] = None,
    allowed_categories: Optional[List[str]] = None,
) -> Dict:
    """
    Returns a dict that your EmailOperator templates already expect.
    Writes:
      - {metrics_root}/stats/{date}/stats.json
      - {metrics_root}/validation/{date}/anomalies.json
    """
    date_str = date_str or dt.datetime.utcnow().strftime("%Y%m%d")
    input_path = Path(input_csv)
    metrics_root = Path(metrics_root)

    hard_fail: List[str] = []
    soft_warn: List[str] = []
    info: List[str] = []

    # ---------- File-level ----------
    if not input_path.exists():
        hard_fail.append(f"File not found: {input_path}")
        row_count = 0
        # Write minimal artifacts and return early
        anomalies_path = metrics_root / "validation" / date_str / "anomalies.json"
        stats_path = metrics_root / "stats" / date_str / "stats.json"
        _ensure_parent(anomalies_path); _ensure_parent(stats_path)
        with open(anomalies_path, "w") as f: json.dump({"hard_fail": hard_fail, "soft_warn": [], "info": []}, f, indent=2)
        with open(stats_path, "w") as f: json.dump({"row_count": 0, "columns": []}, f, indent=2)
        return {
            "row_count": 0,
            "null_prompt_count": 0,
            "dup_prompt_count": 0,
            "unknown_category_rate": 0.0,
            "text_len_min": None,
            "text_len_max": None,
            "size_label_mismatch_count": 0,
            "hard_fail": hard_fail,
            "soft_warn": soft_warn,
            "report_paths": [str(anomalies_path), str(stats_path)],
        }

    df = pd.read_csv(input_path)
    row_count = int(len(df))
    if row_count == 0:
        hard_fail.append("File is empty.")
    if row_count < MIN_ROWS_HARD:
        hard_fail.append(f"Row count {row_count} < {MIN_ROWS_HARD}.")

    # ---------- Schema ----------
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        hard_fail.append(f"Missing required columns: {missing}")

    # Prepare stat outputs early
    stats = {
        "row_count": row_count,
        "columns": list(df.columns),
    }

    # If schema/file already failed, write artifacts and return
    stats_path = metrics_root / "stats" / date_str / "stats.json"
    anomalies_path = metrics_root / "validation" / date_str / "anomalies.json"
    _ensure_parent(stats_path); _ensure_parent(anomalies_path)
    if hard_fail:
        with open(stats_path, "w") as f: json.dump(stats, f, indent=2)
        with open(anomalies_path, "w") as f: json.dump({"hard_fail": hard_fail, "soft_warn": [], "info": []}, f, indent=2)
        return {
            "row_count": row_count,
            "null_prompt_count": None,
            "dup_prompt_count": None,
            "unknown_category_rate": None,
            "text_len_min": None,
            "text_len_max": None,
            "size_label_mismatch_count": None,
            "hard_fail": hard_fail,
            "soft_warn": soft_warn,
            "report_paths": [str(anomalies_path), str(stats_path)],
        }

    # ---------- Prompt-level ----------
    null_prompt_count = int((~_nonempty(df["prompt"])).sum())
    if null_prompt_count > 0:
        hard_fail.append(f"Null/empty prompt values: {null_prompt_count}")

    dup_prompt_count = int(df.duplicated(subset=["prompt"]).sum())
    if dup_prompt_count > 0:
        soft_warn.append(f"Duplicate prompts: {dup_prompt_count}")

    # ---------- Category ----------
    unknown_rate = float((df["category"].astype(str) == "Unknown").mean())
    if unknown_rate > UNKNOWN_FRACTION_SOFT_WARN:
        soft_warn.append(f'Unknown category fraction {unknown_rate:.3f} > {UNKNOWN_FRACTION_SOFT_WARN}')

    if allowed_categories:
        invalid_vals = sorted(set(df.loc[~df["category"].isin(allowed_categories), "category"].astype(str)))
        if invalid_vals:
            hard_fail.append(f"category outside ALLOWED_CATEGORIES: {invalid_vals}")

    # ---------- Text length / size ----------
    tl = pd.to_numeric(df["text_length"], errors="coerce")
    text_len_min = int(tl.min()) if tl.notna().any() else None
    text_len_max = int(tl.max()) if tl.notna().any() else None
    info.append(f"text_length min={text_len_min}, max={text_len_max}")

    if text_len_max is not None and text_len_max > TEXT_LEN_MAX_SOFT_WARN:
        soft_warn.append(f"max(text_length)={text_len_max} > {TEXT_LEN_MAX_SOFT_WARN}")

    expected_size = tl.fillna(-1).astype(int).apply(lambda n: _size_from_len(n))
    size_label_mismatch_count = int((expected_size.astype(str) != df["size_label"].astype(str)).sum())
    if size_label_mismatch_count > 0:
        soft_warn.append(f"size_label mismatches: {size_label_mismatch_count}")

    # ---------- Write artifacts ----------
    stats.update({
        "null_prompt_count": null_prompt_count,
        "dup_prompt_count": dup_prompt_count,
        "unknown_category_rate": unknown_rate,
        "text_len_min": text_len_min,
        "text_len_max": text_len_max,
        "size_label_mismatch_count": size_label_mismatch_count,
    })
    with open(stats_path, "w") as f: json.dump(stats, f, indent=2)
    with open(anomalies_path, "w") as f: json.dump({"hard_fail": hard_fail, "soft_warn": soft_warn, "info": info}, f, indent=2)

    # ---------- XCom payload (matches your emails) ----------
    return {
        "row_count": row_count,
        "null_prompt_count": null_prompt_count,
        "dup_prompt_count": dup_prompt_count,
        "unknown_category_rate": unknown_rate,
        "text_len_min": text_len_min,
        "text_len_max": text_len_max,
        "size_label_mismatch_count": size_label_mismatch_count,
        "hard_fail": hard_fail,
        "soft_warn": soft_warn,
        "report_paths": [str(anomalies_path), str(stats_path)],
    }
