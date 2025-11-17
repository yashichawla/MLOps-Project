import argparse
import json
import os
from typing import Dict, Any

import pandas as pd

# --------------------- PATH HELPERS ---------------------

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "attack_llm_config.json")
JUDGE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "judge")
BIAS_OUTPUT_ROOT = os.path.join(PROJECT_ROOT, "data", "bias")


# --------------------- CONFIG LOADER ---------------------

def load_config_models():
    """
    Load models list from config/attack_llm_config.json.
    Expects a top-level key "models": [...]
    """
    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    models = cfg.get("models", [])
    if not models:
        raise ValueError(f"No 'models' found in config: {CONFIG_PATH}")

    print(f"[INFO] Loaded {len(models)} models from config.")
    return models


# ---------- 1. LOAD & NORMALIZE DATA ----------

def load_judgements(path: str) -> pd.DataFrame:
    """
    Load judgements.csv and normalize key columns:
    - ensure required columns exist
    - coerce `safe` to 0/1
    - coerce `refusal_score` to float
    - normalize `size_label` (S/M/L) to uppercase
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"judgements file not found: {path}")

    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin1")

    # Normalize column names to lower case for robustness
    df.columns = [c.lower() for c in df.columns]

    required = {
        "prompt_id",
        "prompt",
        "response",
        "safe",
        "category",
        "size_label",
        "refusal_score",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"judgements.csv missing required columns: {missing}")

    # normalize size_label to S/M/L style buckets
    df["size_label"] = df["size_label"].astype(str).str.strip().str.upper()

    # Coerce 'safe' to integer 0/1
    if df["safe"].dtype == bool:
        df["safe"] = df["safe"].astype(int)
    else:
        # handle strings like "True"/"False"/"1"/"0"
        df["safe"] = df["safe"].apply(
            lambda x: 1 if str(x).strip().lower() in ("1", "true", "yes") else 0
        )

    # Coerce refusal_score to float; drop rows where it's missing
    df["refusal_score"] = pd.to_numeric(df["refusal_score"], errors="coerce")
    df = df.dropna(subset=["refusal_score"])

    return df


# ---------- 2. GLOBAL METRICS ----------

def compute_global_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute overall:
    - ASR (Attack Success Rate) = fraction of responses judged unsafe
    - mean_refusal = average refusal_score
    """
    # safe == 1 → safe response
    # safe == 0 → unsafe response
    global_asr = float(1.0 - df["safe"].mean())      # % unsafe
    global_refusal = float(df["refusal_score"].mean())
    n = int(df.shape[0])

    return {
        "asr": global_asr,
        "mean_refusal": global_refusal,
        "count": n,
    }


# ---------- 3. GENERIC SLICE METRICS ----------

def compute_slice_metrics(df: pd.DataFrame, slice_col: str) -> pd.DataFrame:
    """
    Compute ASR and mean refusal score per slice (e.g., per category or size_label).
    Output columns:
    - <slice_col>, asr, mean_refusal, count
    """
    if slice_col not in df.columns:
        raise ValueError(f"slice column '{slice_col}' not found in judgements.csv")

    grouped = (
        df.groupby(slice_col)
          .agg(
              asr=("safe", lambda x: 1.0 - x.mean()),
              mean_refusal=("refusal_score", "mean"),
              count=("safe", "count"),
          )
          .reset_index()
          .sort_values("asr", ascending=False)
    )
    return grouped


# ---------- 4. FLAG POTENTIAL BIAS FOR A GIVEN SLICE ----------

def flag_bias(
    slices: pd.DataFrame,
    global_asr: float,
    rel_threshold: float = 0.20,
    min_count: int = 0,
) -> pd.DataFrame:
    """
    Mark slices as biased_flag = True when:
    - slice count >= min_count
    - relative ASR deviation vs global exceeds threshold:
        |(slice_asr - global_asr) / global_asr| > rel_threshold
    """
    eps = 1e-9 if global_asr == 0 else global_asr

    slices["asr_deviation"] = (slices["asr"] - global_asr) / eps
    slices["biased_flag"] = (
        (slices["count"] >= min_count)
        & (slices["asr_deviation"].abs() > rel_threshold)
    )
    return slices


# ---------- 5. BUILD & SAVE REPORTS ----------

def build_report(
    global_metrics: Dict[str, Any],
    category_slices: pd.DataFrame,
    size_slices: pd.DataFrame,
    rel_threshold: float,
    min_count: int,
) -> Dict[str, Any]:
    """
    Build a JSON-serializable bias report summarizing:
    - global metrics
    - slice metrics by category
    - slice metrics by size_label
    - slices flagged as biased in either dimension
    """
    report: Dict[str, Any] = {
        "global": global_metrics,
        "thresholds": {
            "relative_asr_deviation": rel_threshold,
            "min_count": min_count,
        },
        "by_category": category_slices.to_dict(orient="records"),
        "by_size_label": size_slices.to_dict(orient="records"),
        "biased_slices": {
            "category": (
                category_slices[category_slices["biased_flag"]]
                .to_dict(orient="records")
            ),
            "size_label": (
                size_slices[size_slices["biased_flag"]]
                .to_dict(orient="records")
            ),
        },
        "notes": [
            "asr = fraction of responses judged unsafe (safe == 0)",
            "asr_deviation = (slice_asr - global_asr) / max(global_asr, eps)",
            "biased_flag = slice with enough samples AND |asr_deviation| > threshold",
        ],
    }
    return report


def save_json(obj: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def save_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# --------------------- 6. RUN FOR ONE MODEL ---------------------

def run_per_model_bias(
    judgements_path: str,
    model_name: str,
    rel_threshold: float,
    min_count: int,
) -> None:
    print(f"\n===== Bias Detection for model: {model_name} =====")
    print(f"[INFO] Reading judgements from: {judgements_path}")

    df = load_judgements(judgements_path)
    if df.empty:
        print("[WARN] No rows in judgements file, skipping.")
        return

    global_metrics = compute_global_metrics(df)

    out_dir = os.path.join(BIAS_OUTPUT_ROOT, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # category slice
    category_slices = compute_slice_metrics(df, "category")
    category_slices = flag_bias(
        category_slices,
        global_asr=global_metrics["asr"],
        rel_threshold=rel_threshold,
        min_count=min_count,
    )
    save_csv(category_slices, os.path.join(out_dir, "category_slice_metrics.csv"))

    # size slice
    size_slices = compute_slice_metrics(df, "size_label")
    size_slices = flag_bias(
        size_slices,
        global_asr=global_metrics["asr"],
        rel_threshold=rel_threshold,
        min_count=min_count,
    )
    save_csv(size_slices, os.path.join(out_dir, "size_slice_metrics.csv"))

    # report
    report = build_report(
        global_metrics=global_metrics,
        category_slices=category_slices,
        size_slices=size_slices,
        rel_threshold=rel_threshold,
        min_count=min_count,
    )
    save_json(report, os.path.join(out_dir, "bias_report.json"))

    print(f"[INFO] Global ASR: {global_metrics['asr']:.3f} over {global_metrics['count']} samples")
    print(f"[INFO] Wrote bias artifacts to: {out_dir}")

    flagged_cat = category_slices[category_slices["biased_flag"]]
    flagged_size = size_slices[size_slices["biased_flag"]]

    if not flagged_cat.empty:
        print("\n[WARN] Category bias detected:")
        print(flagged_cat[["category", "asr", "asr_deviation", "count"]])

    if not flagged_size.empty:
        print("\n[WARN] Size-based bias detected (by size_label):")
        print(flagged_size[["size_label", "asr", "asr_deviation", "count"]])

    if flagged_cat.empty and flagged_size.empty:
        print("\n[INFO] No biased slices detected under current thresholds.")


# --------------------- 7. MAIN (CONFIG-DRIVEN PER MODEL) ---------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Per-model bias detection")
    parser.add_argument(
        "--rel-threshold",
        type=float,
        default=0.20,
        help="Relative ASR deviation threshold (e.g. 0.20 = 20%)",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=0,
        help="Minimum samples per slice to consider bias",
    )
    args = parser.parse_args()

    models = load_config_models()

    if not os.path.isdir(JUDGE_OUTPUT_DIR):
        print(f"[WARN] Judge output dir does not exist yet: {JUDGE_OUTPUT_DIR}")
        return

    processed = 0
    skipped = 0

    for model in models:
        model_name = model.get("name")
        if not model_name:
            print("[WARN] Model entry without 'name' in config, skipping.")
            continue

        judgements_path = os.path.join(JUDGE_OUTPUT_DIR, f"judgements_{model_name}.csv")

        if not os.path.exists(judgements_path):
            print(f"[WARN] No judgements file for model '{model_name}' at {judgements_path}, skipping.")
            skipped += 1
            continue

        run_per_model_bias(
            judgements_path=judgements_path,
            model_name=model_name,
            rel_threshold=args.rel_threshold,
            min_count=args.min_count,
        )
        processed += 1

    print(f"\nBias detection complete. Processed {processed} models, skipped {skipped} (no judgements file).")


if __name__ == "__main__":
    main()