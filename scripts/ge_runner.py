#!/usr/bin/env python3
# Python 3.12-friendly, no external validation libs required.
import argparse, os, sys, json, datetime as dt
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# --------- Paths ----------
PIPELINE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PIPELINE_ROOT / "data" / "processed" / "processed_data.csv"

METRICS_DIR = PIPELINE_ROOT / "data" / "metrics"
SCHEMA_BASELINE = METRICS_DIR / "schema" / "baseline" / "schema.json"
STATS_BASELINE  = METRICS_DIR / "stats"  / "baseline" / "stats.json"

# --------- Contract (your schema) ----------
REQUIRED_COLS = ["prompt", "category", "prompt_id", "text_length", "size_label"]

# Size bins (you can tweak here if needed)
def size_from_len(n: int) -> str:
    if n < 50:
        return "S"
    elif n <= 200:
        return "M"
    else:
        return "L"

# Soft/hard thresholds
MIN_ROWS_HARD = 50
TEXT_LEN_MAX_SOFT_WARN = 8000
UNKNOWN_FRACTION_SOFT_WARN = 0.30

def _ensure_parent(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)

def _load_df(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def _nonempty(s: pd.Series) -> pd.Series:
    return (~s.isna()) & (s.astype(str).str.strip() != "")

def _basic_stats(df: pd.DataFrame) -> Dict[str, Any]:
    stats: Dict[str, Any] = {
        "row_count": int(len(df)),
        "columns": list(df.columns),
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns if c in df},
        "numerics": {},
        "categoricals": {}
    }
    if "text_length" in df.columns:
        s = pd.to_numeric(df["text_length"], errors="coerce").dropna()
        stats["numerics"]["text_length"] = {
            "min": int(s.min()) if len(s) else None,
            "max": int(s.max()) if len(s) else None,
            "mean": float(s.mean()) if len(s) else None,
        }
    if "size_label" in df.columns:
        vc = df["size_label"].astype(str).value_counts().to_dict()
        stats["categoricals"]["size_label"] = {str(k): int(v) for k, v in vc.items()}
    if "category" in df.columns:
        vc = df["category"].astype(str).value_counts().to_dict()
        stats["categoricals"]["category"] = {str(k): int(v) for k, v in vc.items()}
    return stats

def _save_json(obj: Dict[str, Any], path: Path):
    _ensure_parent(path)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _load_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)

# ---- BASELINE ----
def do_baseline(input_csv: Path, date_str: str, allowed_categories_env: Optional[str]):
    df = _load_df(input_csv)
    stats = _basic_stats(df)

    # Build baseline "schema" JSON (simple, explicit)
    baseline: Dict[str, Any] = {
        "required_columns": REQUIRED_COLS,
        # If env var provided, store in baseline for future runs; else leave None = open set
        "allowed_categories": None,
        "rules": {
            "file_min_rows": MIN_ROWS_HARD,
            "text_length_soft_max": TEXT_LEN_MAX_SOFT_WARN,
            "unknown_fraction_soft_warn": UNKNOWN_FRACTION_SOFT_WARN,
            "size_bins": {"S": "<50", "M": "50-200", "L": ">200"}
        }
    }
    if allowed_categories_env:
        baseline["allowed_categories"] = [c.strip() for c in allowed_categories_env.split(",") if c.strip()]

    _save_json(baseline, SCHEMA_BASELINE)
    _save_json(stats, STATS_BASELINE)

    print(f"[BASELINE] rows={stats['row_count']} -> {SCHEMA_BASELINE}, {STATS_BASELINE}")

# ---- VALIDATE ----
def do_validate(input_csv: Path, baseline_path: Path, date_str: str, allowed_categories_env: Optional[str]):
    if not baseline_path.exists():
        print(f"Baseline schema missing: {baseline_path}", file=sys.stderr)
        sys.exit(2)

    df = _load_df(input_csv)

    # File-level validations
    hard_fail_reasons: List[str] = []
    soft_warn_reasons: List[str] = []
    info_notes: List[str] = []

    if df.shape[0] == 0:
        hard_fail_reasons.append("File is empty.")
    if df.shape[0] < MIN_ROWS_HARD:
        hard_fail_reasons.append(f"Row count {df.shape[0]} < {MIN_ROWS_HARD}.")

    # Schema: required columns exist
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        hard_fail_reasons.append(f"Missing required columns: {missing_cols}")

    # If schema is already broken, write artifacts & exit hard fail
    run_stats = _basic_stats(df) if df.shape[0] > 0 else {"row_count": 0}
    out_stats = METRICS_DIR / "stats" / date_str / "stats.json"
    _save_json(run_stats, out_stats)

    if hard_fail_reasons:
        anomalies = {"hard_fail": hard_fail_reasons, "soft_warn": soft_warn_reasons, "info": info_notes}
        _save_json(anomalies, METRICS_DIR / "validation" / date_str / "anomalies.json")
        print(f"[VALIDATE:FAIL] {hard_fail_reasons}", file=sys.stderr)
        sys.exit(1)

    # ---- Prompt-level ----
    # Null/empty prompt -> hard fail
    if not _nonempty(df["prompt"]).all():
        hard_fail_reasons.append("Null or empty values found in 'prompt'.")

    # Duplicate prompt -> soft warn (tracked, not failing)
    dup_count = int(df.duplicated(subset=["prompt"]).sum())
    if dup_count > 0:
        soft_warn_reasons.append(f"Duplicate 'prompt' values: {dup_count}")

    # ---- Category validations ----
    # Determine allowed categories (env overrides baseline; baseline may be None = open set)
    baseline = _load_json(baseline_path)
    allowed = None
    if allowed_categories_env:
        allowed = [c.strip() for c in allowed_categories_env.split(",") if c.strip()]
    elif isinstance(baseline.get("allowed_categories"), list):
        allowed = baseline["allowed_categories"]

    if allowed is not None:
        invalid = df[~df["category"].isin(allowed)]
        if not invalid.empty:
            invalid_vals = sorted(set(invalid["category"].astype(str)))
            hard_fail_reasons.append(f"category values not in ALLOWED_CATEGORIES: {invalid_vals}")

    # "Unknown" fraction soft warn
    if "Unknown" in df["category"].astype(str).unique():
        unknown_frac = float((df["category"].astype(str) == "Unknown").mean())
        if unknown_frac > UNKNOWN_FRACTION_SOFT_WARN:
            soft_warn_reasons.append(f'Fraction of "Unknown" categories = {unknown_frac:.2f} (> {UNKNOWN_FRACTION_SOFT_WARN})')

    # ---- Text length / size validations ----
    # Record min/max (info)
    tl = pd.to_numeric(df["text_length"], errors="coerce")
    min_len = int(tl.min()) if tl.notna().any() else None
    max_len = int(tl.max()) if tl.notna().any() else None
    info_notes.append(f"text_length min={min_len}, max={max_len}")

    # Soft warn if max > threshold
    if max_len is not None and max_len > TEXT_LEN_MAX_SOFT_WARN:
        soft_warn_reasons.append(f"max(text_length)={max_len} > {TEXT_LEN_MAX_SOFT_WARN} (soft warn)")

    # size_label consistency with bins
    # Build expected label from text_length
    expected = tl.fillna(-1).astype(int).apply(size_from_len)
    mismatches = (expected.astype(str) != df["size_label"].astype(str)).sum()
    if mismatches > 0:
        soft_warn_reasons.append(f"size_label mismatches with text_length bins: {mismatches}")

    # ---- Decide pass/fail ----
    anomalies = {"hard_fail": hard_fail_reasons, "soft_warn": soft_warn_reasons, "info": info_notes}
    _save_json(anomalies, METRICS_DIR / "validation" / date_str / "anomalies.json")

    if hard_fail_reasons:
        print(f"[VALIDATE:FAIL] {hard_fail_reasons}", file=sys.stderr)
        sys.exit(1)
    else:
        # Soft warns do not fail
        if soft_warn_reasons:
            print(f"[VALIDATE:WARN] {soft_warn_reasons}")
        else:
            print("[VALIDATE:OK]")

def main():
    ap = argparse.ArgumentParser("Schema & data validator (baseline/validate)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    env_allowed = os.getenv("ALLOWED_CATEGORIES")  # e.g., "toxicity,pii,injection,illicit,benign"

    p1 = sub.add_parser("baseline")
    p1.add_argument("--input", default=str(DEFAULT_INPUT))
    p1.add_argument("--date", default=dt.datetime.utcnow().strftime("%Y%m%d"))

    p2 = sub.add_parser("validate")
    p2.add_argument("--input", default=str(DEFAULT_INPUT))
    p2.add_argument("--baseline_schema", default=str(SCHEMA_BASELINE))
    p2.add_argument("--date", default=dt.datetime.utcnow().strftime("%Y%m%d"))

    args = ap.parse_args()
    if args.cmd == "baseline":
        do_baseline(Path(args.input), args.date, env_allowed)
    else:
        do_validate(Path(args.input), Path(args.baseline_schema), args.date, env_allowed)

if __name__ == "__main__":
    sys.exit(main())
