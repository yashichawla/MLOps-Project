#!/usr/bin/env python3
# Uses Great Expectations when available; falls back to built-in checks.
import argparse, os, sys, json, datetime as dt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

# Try to import Great Expectations in a backward-compatible way.
_GE_AVAILABLE = False
try:
    import great_expectations as gx  # type: ignore
    try:
        # Older-style dataset API is simpler for inline use.
        from great_expectations.dataset import PandasDataset  # type: ignore
        _GE_AVAILABLE = True
    except Exception:
        _GE_AVAILABLE = False
except Exception:
    _GE_AVAILABLE = False

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


# ---- GE helpers (optional) ----
def _ge_build_and_validate(
    df: pd.DataFrame,
    allowed_categories: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    """Run a small suite of Great Expectations checks.

    Returns (hard_fail_messages, soft_warn_messages).

    If GE is not available, returns two empty lists and lets callers use fallback checks.
    """
    hard: List[str] = []
    soft: List[str] = []

    if not _GE_AVAILABLE:
        return hard, soft

    # Wrap DataFrame in a GE dataset (dataset API kept for simplicity/compatibility)
    gdf = PandasDataset(df.copy())

    # Hard checks
    gdf.expect_table_row_count_to_be_between(min_value=MIN_ROWS_HARD, max_value=None)
    for col in REQUIRED_COLS:
        gdf.expect_column_to_exist(col)

    gdf.expect_column_values_to_not_be_null("prompt")
    # Also ensure non-empty strings (at least one non-space char)
    gdf.expect_column_values_to_match_regex("prompt", r"\S", mostly=1.0)

    # Allowed categories (hard when provided)
    if allowed_categories is not None:
        gdf.expect_column_values_to_be_in_set("category", allowed_categories)

    # Soft checks
    # Duplicate prompts -> soft warn (use GE but we will treat as soft)
    gdf.expect_column_values_to_be_unique("prompt", mostly=1.0)

    # Unknown fraction soft warn: require non-Unknown proportion >= (1 - threshold)
    try:
        gdf.expect_column_values_to_not_be_in_set(
            "category",
            ["Unknown"],
            mostly=max(0.0, 1.0 - UNKNOWN_FRACTION_SOFT_WARN),
        )
    except Exception:
        # In case column types are odd, ignore GE error here; fallback checks will still run.
        pass

    # Text length soft cap
    gdf.expect_column_values_to_be_between(
        "text_length", min_value=None, max_value=TEXT_LEN_MAX_SOFT_WARN, mostly=1.0
    )

    # Execute expectations and partition messages.
    results = gdf.validate(result_format="SUMMARY", catch_exceptions=False)
    for r in results.get("results", []):
        etype = r.get("expectation_config", {}).get("expectation_type", "")
        success = bool(r.get("success", False))

        # Map expectation types to severity (hard vs soft) aligned with our policy.
        if etype in {
            "expect_table_row_count_to_be_between",
            "expect_column_to_exist",
            "expect_column_values_to_not_be_null",
            "expect_column_values_to_be_in_set",
        }:
            if not success:
                hard.append(f"GE failed: {etype}")
        elif etype in {
            "expect_column_values_to_be_unique",
            "expect_column_values_to_not_be_in_set",
            "expect_column_values_to_be_between",
            "expect_column_values_to_match_regex",
        }:
            if not success:
                soft.append(f"GE warn: {etype}")
        else:
            # Unknown expectations: keep non-blocking
            if not success:
                soft.append(f"GE warn (other): {etype}")

    return hard, soft

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

    # File-level validations (early)
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

    # ---- Category/Prompt/Text validations (GE + fallback) ----
    # Determine allowed categories (env overrides baseline; baseline may be None = open set)
    baseline = _load_json(baseline_path)
    allowed = None
    if allowed_categories_env:
        allowed = [c.strip() for c in allowed_categories_env.split(",") if c.strip()]
    elif isinstance(baseline.get("allowed_categories"), list):
        allowed = baseline["allowed_categories"]

    # Run Great Expectations if available; collect messages.
    ge_hard, ge_soft = _ge_build_and_validate(df, allowed)
    hard_fail_reasons.extend(ge_hard)
    soft_warn_reasons.extend(ge_soft)

    # Fallback/manual checks to preserve metrics and messages that GE may not cover fully.
    # Prompt null/empty -> hard fail (redundant if GE ran, but keeps behavior consistent)
    if not _nonempty(df["prompt"]).all():
        hard_fail_reasons.append("Null or empty values found in 'prompt'.")

    # Duplicate prompt -> soft warn with explicit count
    dup_count = int(df.duplicated(subset=["prompt"]).sum())
    if dup_count > 0:
        soft_warn_reasons.append(f"Duplicate 'prompt' values: {dup_count}")

    # Allowed category hard check if provided (explicit listing of invalids)
    if allowed is not None:
        invalid = df[~df["category"].isin(allowed)]
        if not invalid.empty:
            invalid_vals = sorted(set(invalid["category"].astype(str)))
            hard_fail_reasons.append(f"category values not in ALLOWED_CATEGORIES: {invalid_vals}")

    # Unknown fraction soft warn (explicit value)
    if "Unknown" in df["category"].astype(str).unique():
        unknown_frac = float((df["category"].astype(str) == "Unknown").mean())
        if unknown_frac > UNKNOWN_FRACTION_SOFT_WARN:
            soft_warn_reasons.append(
                f'Fraction of "Unknown" categories = {unknown_frac:.2f} (> {UNKNOWN_FRACTION_SOFT_WARN})'
            )

    # Text length / size validations — record and warn
    tl = pd.to_numeric(df["text_length"], errors="coerce")
    min_len = int(tl.min()) if tl.notna().any() else None
    max_len = int(tl.max()) if tl.notna().any() else None
    info_notes.append(f"text_length min={min_len}, max={max_len}")
    if max_len is not None and max_len > TEXT_LEN_MAX_SOFT_WARN:
        soft_warn_reasons.append(
            f"max(text_length)={max_len} > {TEXT_LEN_MAX_SOFT_WARN} (soft warn)"
        )

    expected = tl.fillna(-1).astype(int).apply(size_from_len)
    mismatches = (expected.astype(str) != df["size_label"].astype(str)).sum()
    if mismatches > 0:
        soft_warn_reasons.append(
            f"size_label mismatches with text_length bins: {mismatches}"
        )

    # ---- Decide pass/fail ----
    anomalies = {"hard_fail": hard_fail_reasons, "soft_warn": soft_warn_reasons, "info": info_notes}
    _save_json(anomalies, METRICS_DIR / "validation" / date_str / "anomalies.json")

    # Enrich stats.json with derived metrics expected by downstream tasks
    try:
        out_stats = METRICS_DIR / "stats" / date_str / "stats.json"
        metrics_stats = _basic_stats(df)
        # Derive numeric fields (keep None when not applicable)
        null_prompt_count = int((~_nonempty(df["prompt"]).astype(bool)).sum()) if "prompt" in df.columns else 0
        unknown_rate = None
        if "category" in df.columns:
            unknown_rate = float((df["category"].astype(str) == "Unknown").mean())
        # 'dup_count', 'min_len', 'max_len', 'mismatches' were computed above when columns exist
        try:
            dup_count_val = int(df.duplicated(subset=["prompt"]).sum()) if "prompt" in df.columns else 0
        except Exception:
            dup_count_val = None
        try:
            tl_loc = pd.to_numeric(df["text_length"], errors="coerce") if "text_length" in df.columns else pd.Series([], dtype=float)
            min_len_val = int(tl_loc.min()) if tl_loc.notna().any() else None
            max_len_val = int(tl_loc.max()) if tl_loc.notna().any() else None
        except Exception:
            min_len_val = None
            max_len_val = None
        try:
            expected_loc = tl_loc.fillna(-1).astype(int).apply(size_from_len) if "text_length" in df.columns else pd.Series([], dtype=object)
            mismatch_val = int((expected_loc.astype(str) != df["size_label"].astype(str)).sum()) if "size_label" in df.columns and len(expected_loc) else 0
        except Exception:
            mismatch_val = None

        metrics_stats.update({
            "null_prompt_count": null_prompt_count,
            "dup_prompt_count": dup_count_val,
            "unknown_category_rate": unknown_rate,
            "text_len_min": min_len_val,
            "text_len_max": max_len_val,
            "size_label_mismatch_count": mismatch_val,
        })
        _save_json(metrics_stats, out_stats)
    except Exception:
        pass

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