#!/usr/bin/env python3
# Uses Great Expectations for data validation.
import argparse, os, sys, json, datetime as dt
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

logger = logging.getLogger(__name__)

# Great Expectations is required for validation.
# Note: This code uses the PandasDataset API which is available in GE v0.18.x
# For GE v1.x+, you may need to use the new Validator API or pin to v0.18.x
try:
    import great_expectations as gx  # type: ignore
    from great_expectations.dataset import PandasDataset  # type: ignore
except ImportError as e:
    raise ImportError(
        "Great Expectations is required but not installed. "
        "Please install it with: pip install 'great_expectations==0.18.21' "
        "(pinned version for PandasDataset API compatibility)"
    ) from e

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
DRIFT_THRESHOLD = 0.20  # 20% change from baseline triggers warning

# Expected values for validation
VALID_SIZE_LABELS = {"S", "M", "L"}
VALID_CATEGORIES = {
    "Hate Speech", "Unknown", "Political Lobbying", "Physical Harm",
    "Pornography", "Fraud", "Illegal Activity", "Economic Harm",
    "Legal Opinion", "Malware Generation", "Health Consultation",
    "Child Sexual Abuse", "Privacy Violence", "Government Decision",
    "Financial Advice"
}

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
        "null_counts": {c: int(df[c].isna().sum()) for c in df.columns},
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
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    try:
        with open(path) as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in {path}: {e}")


def _enhance_anomalies_dict(
    hard_fail_reasons: List[str],
    soft_warn_reasons: List[str],
    total_rows: int,
) -> Dict[str, Any]:
    """
    Enhance anomalies dict with metadata and summary while keeping backward compatibility.
    """
    # Determine validation status
    if hard_fail_reasons:
        validation_status = "hard_fail"
    elif soft_warn_reasons:
        validation_status = "soft_warn"
    else:
        validation_status = "passed"
    
    # Determine validation source
    validation_source = "great_expectations"
    
    # Build enhanced structure (keep existing fields for backward compatibility)
    return {
        "metadata": {
            "validation_timestamp": datetime.now(timezone.utc).isoformat(),
            "validation_source": validation_source,
        },
        "summary": {
            "validation_status": validation_status,
            "total_rows": total_rows,
            "hard_fail_count": len(hard_fail_reasons),
            "soft_warn_count": len(soft_warn_reasons),
        },
        # Keep existing structure for backward compatibility
        "hard_fail": hard_fail_reasons,
        "soft_warn": soft_warn_reasons,
    }


def _validate_data_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Validate data types and formats for required columns.
    
    Returns (hard_fail_reasons, soft_warn_reasons).
    """
    hard: List[str] = []
    soft: List[str] = []
    
    # Check prompt_id uniqueness
    if "prompt_id" in df.columns:
        if df["prompt_id"].isna().any():
            hard.append("prompt_id contains null values.")
        dup_ids = df["prompt_id"].duplicated().sum()
        if dup_ids > 0:
            hard.append(f"prompt_id has {dup_ids} duplicate values (must be unique).")
    
    # Check text_length is numeric
    if "text_length" in df.columns:
        non_numeric = pd.to_numeric(df["text_length"], errors="coerce").isna().sum()
        if non_numeric > 0:
            hard.append(f"text_length has {non_numeric} non-numeric values.")
        # Check for negative values
        tl_numeric = pd.to_numeric(df["text_length"], errors="coerce")
        negative_count = (tl_numeric < 0).sum()
        if negative_count > 0:
            hard.append(f"text_length has {negative_count} negative values.")
    
    # Check size_label values
    if "size_label" in df.columns:
        invalid_sizes = df[~df["size_label"].astype(str).isin(VALID_SIZE_LABELS)]
        if not invalid_sizes.empty:
            invalid_vals = sorted(set(invalid_sizes["size_label"].astype(str)))
            hard.append(f"size_label contains invalid values: {invalid_vals} (must be S, M, or L).")
    
    return hard, soft

def _compare_to_baseline(current_stats: Dict[str, Any], baseline_stats: Dict[str, Any]) -> List[str]:
    """Compare current stats to baseline and detect significant drift.
    
    Returns list of drift warnings.
    """
    warnings: List[str] = []
    
    if not baseline_stats or "row_count" not in baseline_stats:
        return warnings
    
    baseline_count = baseline_stats.get("row_count", 0)
    current_count = current_stats.get("row_count", 0)
    
    if baseline_count > 0:
        row_change = abs(current_count - baseline_count) / baseline_count
        if row_change > DRIFT_THRESHOLD:
            warnings.append(f"Row count changed by {row_change:.1%} from baseline ({baseline_count} -> {current_count}).")
    
    # Compare category distributions
    baseline_cats = baseline_stats.get("categoricals", {}).get("category", {})
    current_cats = current_stats.get("categoricals", {}).get("category", {})
    
    if baseline_cats and current_cats:
        baseline_total = sum(baseline_cats.values())
        current_total = sum(current_cats.values())
        
        if baseline_total > 0 and current_total > 0:
            for cat in set(list(baseline_cats.keys()) + list(current_cats.keys())):
                baseline_pct = baseline_cats.get(cat, 0) / baseline_total
                current_pct = current_cats.get(cat, 0) / current_total
                change = abs(current_pct - baseline_pct)
                
                if change > DRIFT_THRESHOLD:
                    warnings.append(f"Category '{cat}' distribution changed by {change:.1%} from baseline ({baseline_pct:.1%} -> {current_pct:.1%}).")
    
    # Compare text_length distribution
    baseline_tl = baseline_stats.get("numerics", {}).get("text_length", {})
    current_tl = current_stats.get("numerics", {}).get("text_length", {})
    
    if baseline_tl and current_tl and baseline_tl.get("mean") and current_tl.get("mean"):
        baseline_mean = baseline_tl["mean"]
        current_mean = current_tl["mean"]
        if baseline_mean > 0:
            mean_change = abs(current_mean - baseline_mean) / baseline_mean
            if mean_change > DRIFT_THRESHOLD:
                warnings.append(f"Average text_length changed by {mean_change:.1%} from baseline ({baseline_mean:.1f} -> {current_mean:.1f} words).")
    
    # Compare size_label distribution
    baseline_sizes = baseline_stats.get("categoricals", {}).get("size_label", {})
    current_sizes = current_stats.get("categoricals", {}).get("size_label", {})
    
    if baseline_sizes and current_sizes:
        baseline_total = sum(baseline_sizes.values())
        current_total = sum(current_sizes.values())
        
        if baseline_total > 0 and current_total > 0:
            for size in set(list(baseline_sizes.keys()) + list(current_sizes.keys())):
                baseline_pct = baseline_sizes.get(size, 0) / baseline_total
                current_pct = current_sizes.get(size, 0) / current_total
                change = abs(current_pct - baseline_pct)
                
                if change > DRIFT_THRESHOLD:
                    warnings.append(f"Size label '{size}' distribution changed by {change:.1%} from baseline ({baseline_pct:.1%} -> {current_pct:.1%}).")
    
    return warnings


# ---- GE helpers ----
def _ge_build_and_validate(
    df: pd.DataFrame,
    allowed_categories: Optional[List[str]],
) -> Tuple[List[str], List[str]]:
    """Run a small suite of Great Expectations checks.

    Returns (hard_fail_messages, soft_warn_messages).
    """
    hard: List[str] = []
    soft: List[str] = []

    # Wrap DataFrame in a GE dataset/validator (compatibility wrapper handles both APIs)
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
    gdf.expect_column_values_to_not_be_in_set(
        "category",
        ["Unknown"],
        mostly=max(0.0, 1.0 - UNKNOWN_FRACTION_SOFT_WARN),
    )

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
        # Default to VALID_CATEGORIES if env var not provided
        "allowed_categories": list(VALID_CATEGORIES),
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

    if df.shape[0] == 0:
        hard_fail_reasons.append("File is empty.")
    if df.shape[0] < MIN_ROWS_HARD:
        hard_fail_reasons.append(f"Row count {df.shape[0]} < {MIN_ROWS_HARD}.")

    # Schema: required columns exist
    missing_cols = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing_cols:
        hard_fail_reasons.append(f"Missing required columns: {missing_cols}")

    # Data type and format validation (early, before stats computation)
    if df.shape[0] > 0:
        dt_hard, dt_soft = _validate_data_types(df)
        hard_fail_reasons.extend(dt_hard)
        soft_warn_reasons.extend(dt_soft)

    # If schema is already broken, write artifacts & exit hard fail
    run_stats = _basic_stats(df) if df.shape[0] > 0 else {"row_count": 0}
    out_stats = METRICS_DIR / "stats" / date_str / "stats.json"
    _save_json(run_stats, out_stats)

    if hard_fail_reasons:
        anomalies = _enhance_anomalies_dict(
            hard_fail_reasons, soft_warn_reasons, df.shape[0]
        )
        _save_json(anomalies, METRICS_DIR / "validation" / date_str / "anomalies.json")
        print(f"[VALIDATE:FAIL] {hard_fail_reasons}", file=sys.stderr)
        sys.exit(1)

    # ---- Category/Prompt/Text validations using Great Expectations ----
    # Determine allowed categories (env overrides baseline; default to VALID_CATEGORIES)
    baseline = _load_json(baseline_path)
    allowed = None
    if allowed_categories_env:
        allowed = [c.strip() for c in allowed_categories_env.split(",") if c.strip()]
    elif isinstance(baseline.get("allowed_categories"), list):
        allowed = baseline["allowed_categories"]
    else:
        # Default to VALID_CATEGORIES if neither env var nor baseline specifies
        allowed = list(VALID_CATEGORIES)

    # Run Great Expectations validation; collect messages.
    ge_hard, ge_soft = _ge_build_and_validate(df, allowed)
    hard_fail_reasons.extend(ge_hard)
    soft_warn_reasons.extend(ge_soft)

    # Baseline comparison and drift detection
    # Failures in baseline comparison are logged as soft warnings, not hard failures
    baseline_stats_path = STATS_BASELINE
    if baseline_stats_path.exists():
        try:
            baseline_stats = _load_json(baseline_stats_path)
            drift_warnings = _compare_to_baseline(run_stats, baseline_stats)
            soft_warn_reasons.extend(drift_warnings)
        except Exception as e:
            # Log baseline comparison failures as soft warnings, don't fail validation
            logger.warning(f"Failed to compare to baseline: {e}")
            soft_warn_reasons.append(f"Baseline comparison failed: {e}")

    # ---- Decide pass/fail ----
    anomalies = _enhance_anomalies_dict(
        hard_fail_reasons, soft_warn_reasons, df.shape[0]
    )
    _save_json(anomalies, METRICS_DIR / "validation" / date_str / "anomalies.json")

    # Enrich stats.json with derived metrics expected by downstream tasks
    # No fallbacks - validation must complete successfully
    out_stats = METRICS_DIR / "stats" / date_str / "stats.json"
    metrics_stats = _basic_stats(df)
    # Derive numeric fields (keep None when not applicable)
    null_prompt_count = int((~_nonempty(df["prompt"])).sum()) if "prompt" in df.columns else 0
    unknown_rate = None
    if "category" in df.columns:
        unknown_rate = float((df["category"].astype(str) == "Unknown").mean())
    # Compute additional metrics for stats.json
    dup_count_val = int(df.duplicated(subset=["prompt"]).sum()) if "prompt" in df.columns else 0
    tl_loc = pd.to_numeric(df["text_length"], errors="coerce") if "text_length" in df.columns else pd.Series([], dtype=float)
    min_len_val = int(tl_loc.min()) if tl_loc.notna().any() else None
    max_len_val = int(tl_loc.max()) if tl_loc.notna().any() else None
    expected_loc = tl_loc.fillna(-1).astype(int).apply(size_from_len) if "text_length" in df.columns else pd.Series([], dtype=object)
    mismatch_val = int((expected_loc.astype(str) != df["size_label"].astype(str)).sum()) if "size_label" in df.columns and len(expected_loc) else 0

    metrics_stats.update({
        "null_prompt_count": null_prompt_count,
        "dup_prompt_count": dup_count_val,
        "unknown_category_rate": unknown_rate,
        "text_len_min": min_len_val,
        "text_len_max": max_len_val,
        "size_label_mismatch_count": mismatch_val,
    })
    _save_json(metrics_stats, out_stats)

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