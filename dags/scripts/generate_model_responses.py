# scripts/generate_model_responses.py
"""
Generate model responses for one or more Hugging Face models.

- Reads per-model config from: config/attack_llm_config.json
- For each model: loads prompts from CSV, calls the model (chat-first by default), and writes a per-model CSV with responses, latency, and status.
"""

from __future__ import annotations

import json, os, time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dotenv import find_dotenv, load_dotenv
from huggingface_hub import InferenceClient
import shutil

# Use PROJECT_ROOT env var if set (Docker), otherwise calculate from script location (Composer)
SCRIPT_DIR = Path(__file__).resolve().parent
DAGS_DIR = SCRIPT_DIR.parent  # This is /home/airflow/gcs/dags/ in Composer, /opt/airflow/dags/ in local
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(DAGS_DIR.parent))).resolve()

# Config path: try PROJECT_ROOT/config/ first (for local), then DAGS_DIR/config/ (for Composer)
# In local Docker: PROJECT_ROOT=/opt/airflow/app, config is at /opt/airflow/app/config/
# In Composer: PROJECT_ROOT=/home/airflow/gcs, config is at /home/airflow/gcs/dags/config/
if (PROJECT_ROOT / "config" / "attack_llm_config.json").exists():
    CONFIG_PATH = PROJECT_ROOT / "config" / "attack_llm_config.json"
else:
    CONFIG_PATH = DAGS_DIR / "config" / "attack_llm_config.json"

# Determine data directory: use DVC_DATA_DIR if set (data inside DVC repo), otherwise fall back to PROJECT_ROOT/data/
# Data is now stored inside DVC repo at dvc_project/data/ when DVC_DATA_DIR is set
if "DVC_DATA_DIR" in os.environ:
    DATA_DIR = Path(os.environ["DVC_DATA_DIR"]).resolve()
elif "PROJECT_ROOT" in os.environ:
    DATA_DIR = Path(os.environ["PROJECT_ROOT"]) / "data"
else:
    # Fallback: calculate from script location (dags/scripts/ -> root-level data/)
    DATA_DIR = SCRIPT_DIR.parent.parent.parent / "data"

PROMPT_COL = "prompt"
DEFAULT_META_COLS = ["category", "prompt_id", "text_length", "size_label"]


def _get_hf_token() -> str:
    """Load HF token from .env or environment."""
    env_path = find_dotenv(usecwd=True)
    load_dotenv(env_path if env_path else None)
    tok = os.getenv("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN missing. Add HF_TOKEN=hf_xxxxx to your .env")
    return tok


def _load_config() -> List[Dict[str, Any]]:
    """Load and validate model configurations from CONFIG_PATH."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing {CONFIG_PATH}. Please define model configs.")
    cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    models = cfg.get("models")
    if not isinstance(models, list):
        raise ValueError("Config must contain a 'models' list.")
    required = ["model_id", "csv_path", "out_path", "max_new_tokens", "temperature", "top_p", "seed", "sample", "n_samples"]
    for m in models:
        for k in required:
            if k not in m:
                raise ValueError(f"Missing required key '{k}' in model config: {m}")
    return models


def _load_prompts(csv_path: str, prompt_col: str = PROMPT_COL, meta_cols: Optional[List[str]] = None, sample: bool = False, n_samples: int = 5) -> pd.DataFrame:
    """Load prompts (and optional metadata) from CSV."""
    df = pd.read_csv(csv_path)
    meta_cols = meta_cols or DEFAULT_META_COLS
    keep = [prompt_col] + [c for c in meta_cols if c in df.columns]
    df = df[keep].dropna(subset=[prompt_col])
    # Don't sample here - we'll sample AFTER filtering out processed prompts
    return df


def _normalize_prompt(prompt: str) -> str:
    """Normalize prompt text for consistent comparison (strip whitespace, normalize newlines)."""
    return str(prompt).strip().replace('\r\n', '\n').replace('\r', '\n')


def _get_processed_prompts(out_path: Path, model_id: str) -> set:
    """
    Load existing responses CSV and return set of normalized prompts that have already been processed for this model.
    Uses (prompt_text, model_id) as the unique key to prevent duplicate API calls.
    """
    if not out_path.exists():
        return set()
    
    try:
        existing_df = pd.read_csv(out_path)
        
        # Filter to only this model's responses
        if "model" in existing_df.columns:
            existing_df = existing_df[existing_df["model"] == model_id]
        
        # Extract and normalize prompts
        if "prompt" in existing_df.columns:
            processed_prompts = {
                _normalize_prompt(p) 
                for p in existing_df["prompt"].dropna().astype(str)
            }
            print(f"[INFO] Found {len(processed_prompts)} already-processed prompts for model {model_id}")
            return processed_prompts
        else:
            print(f"[WARN] Existing CSV {out_path} missing 'prompt' column. Will reprocess all prompts.")
            return set()
    except Exception as e:
        print(f"[WARN] Failed to load existing responses from {out_path}: {e}. Will reprocess all prompts.")
        return set()


def _filter_new_prompts(df: pd.DataFrame, processed_prompts: set, prompt_col: str = PROMPT_COL, sample: bool = False, n_samples: int = 5) -> pd.DataFrame:
    """
    Filter DataFrame to only include prompts that haven't been processed yet for this model.
    If sample=True, takes the first n_samples from the unprocessed prompts.
    Uses normalized prompt text for comparison.
    """
    if not processed_prompts:
        print(f"[INFO] No existing responses found.")
        if sample:
            print(f"[INFO] Sampling mode: taking first {n_samples} prompts from {len(df)} total.")
            return df.head(n_samples)
        else:
            print(f"[INFO] Processing all {len(df)} prompts.")
            return df
    
    # Normalize prompts in the input DataFrame
    df["_normalized_prompt"] = df[prompt_col].astype(str).apply(_normalize_prompt)
    
    # Filter out already-processed prompts
    df_new = df[~df["_normalized_prompt"].isin(processed_prompts)].copy()
    df_new = df_new.drop(columns=["_normalized_prompt"])
    
    skipped = len(df) - len(df_new)
    
    # If sample mode, take first n_samples from the unprocessed prompts
    if sample and len(df_new) > 0:
        original_count = len(df_new)
        df_new = df_new.head(n_samples)
        sampled_count = len(df_new)
        print(f"[INFO] Filtered prompts: {len(df)} total, {skipped} already processed, {original_count} unprocessed available.")
        print(f"[INFO] Sampling mode: taking first {sampled_count} unprocessed prompts.")
    else:
        print(f"[INFO] Filtered prompts: {len(df)} total, {skipped} already processed, {len(df_new)} new prompts to process.")
    
    return df_new


def _make_client(model_id: str, provider: Optional[str] = None, timeout_s: int = 120) -> InferenceClient:
    """Create an InferenceClient. If provider is None, the router decides."""
    kwargs: Dict[str, Any] = {"model": model_id, "token": _get_hf_token(), "timeout": timeout_s}
    if provider: kwargs["provider"] = provider
    return InferenceClient(**kwargs)


def _infer_one(client: InferenceClient, prompt: str, params: Dict[str, Any], system_prompt: str, prefer_chat: bool) -> Tuple[str, int]:
    """Run inference using chat or text-generation, with a single fallback."""
    t0 = time.time()
    effective_prompt = prompt if not system_prompt.strip() else f"{system_prompt}\n\n{prompt}"

    try:
        if prefer_chat:
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + [{"role": "user", "content": prompt}]
            out = client.chat.completions.create(messages=messages, max_tokens=params["max_new_tokens"], temperature=params["temperature"], top_p=params["top_p"], seed=params["seed"])
            text = out.choices[0].message.content
        else:
            out = client.text_generation(prompt=effective_prompt, max_new_tokens=params["max_new_tokens"], temperature=params["temperature"], top_p=params["top_p"], return_full_text=False)
            text = out if isinstance(out, str) else out[0].get("generated_text", str(out))
    except Exception:
        # Fallback once via text-generation (keeps it simple and broadly supported)
        out = client.text_generation(prompt=effective_prompt, max_new_tokens=min(64, int(params["max_new_tokens"])), temperature=params["temperature"], top_p=params["top_p"], return_full_text=False)
        text = out if isinstance(out, str) else (out[0].get("generated_text", str(out)) if isinstance(out, list) and out else "")

    return text, int((time.time() - t0) * 1000)


def run_model_response_generation() -> List[Dict[str, Any]]:
    """Run the pipeline across all models from the config and return per-model summaries."""
    models_cfg = _load_config()
    summaries: List[Dict[str, Any]] = []

    for m in models_cfg:
        name = m.get("name", m["model_id"])
        model_id = m["model_id"]
        print(f"\n{'='*60}")
        print(f"Running model: {name} ({model_id})")
        print(f"{'='*60}")

        # Resolve paths relative to DATA_DIR (handles both DVC_DATA_DIR and PROJECT_ROOT/data)
        # Config paths are relative like "data/processed/processed_data.csv"
        csv_path = m["csv_path"]
        if not Path(csv_path).is_absolute():
            # Remove "data/" prefix if present, then prepend DATA_DIR
            if csv_path.startswith("data/"):
                csv_path = str(DATA_DIR / csv_path[5:])  # Remove "data/" prefix
            else:
                csv_path = str(DATA_DIR / csv_path)
        
        # Load ALL prompts from processed CSV (don't sample yet)
        df_all = _load_prompts(
            csv_path=csv_path, 
            sample=False,  # Don't sample here - we'll do it after filtering
            n_samples=int(m["n_samples"])
        )
        
        # Resolve out_path relative to DATA_DIR
        out_path_str = m["out_path"]
        if not Path(out_path_str).is_absolute():
            # Remove "data/" prefix if present, then prepend DATA_DIR
            if out_path_str.startswith("data/"):
                out_path = DATA_DIR / out_path_str[5:]  # Remove "data/" prefix
            else:
                out_path = DATA_DIR / out_path_str
        else:
            out_path = Path(out_path_str)
        processed_prompts = _get_processed_prompts(out_path, model_id)
        print(f"[INFO] Loaded {len(df_all)} total prompts from {csv_path}")
        print(f"[INFO] Found {len(processed_prompts)} already-processed prompts for model {model_id}")
        
        # Filter out processed prompts, then sample if needed
        df = _filter_new_prompts(
            df_all, 
            processed_prompts,
            sample=bool(m["sample"]),  # Pass sample flag here
            n_samples=int(m["n_samples"])
        )
        
        if len(df) == 0:
            print(f"[INFO] All prompts already processed for {name} ({model_id}). Skipping API calls.")
            print(f"[INFO]   - Total prompts in CSV: {len(df_all)}")
            print(f"[INFO]   - Already processed: {len(processed_prompts)}")
            print(f"[INFO]   - Unprocessed available: {len(df_all) - len(processed_prompts)}")
            # Load existing file to get summary stats
            if out_path.exists():
                try:
                    existing_df = pd.read_csv(out_path)
                    model_df = existing_df[existing_df["model"] == model_id] if "model" in existing_df.columns else existing_df
                    summaries.append({
                        "model": model_id,
                        "provider": m.get("provider", "router-default"),
                        "rows_written": 0,
                        "ok": len(model_df[model_df.get("status", "").astype(str).str.contains("ok", na=False)]),
                        "errors": len(model_df[model_df.get("status", "").astype(str).str.contains("error", na=False)]),
                        "avg_latency_ms": int(model_df["latency_ms"].mean()) if "latency_ms" in model_df.columns and len(model_df) > 0 else 0,
                        "output_file": str(out_path),
                        "skipped": True,
                    })
                except Exception as e:
                    print(f"[WARN] Could not load existing stats: {e}")
            continue
        
        client = _make_client(model_id=model_id, provider=m.get("provider"), timeout_s=m.get("timeout_s", 120))

        prefer_chat = bool(m.get("prefer_chat", True))
        retries, backoff = int(m.get("retries", 2)), float(m.get("retry_backoff_s", 1.5))
        ts_iso = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
        gen_params = {
            "max_new_tokens": m["max_new_tokens"], 
            "temperature": m["temperature"], 
            "top_p": m["top_p"], 
            "seed": m["seed"]
        }
        system_prompt = m.get("system_prompt", "")

        rows: List[Dict[str, Any]] = []
        ok = err = total_latency = 0

        print(f"[INFO] Processing {len(df)} new prompts...")
        for idx, (_, r) in enumerate(df.iterrows(), 1):
            prompt = str(r[PROMPT_COL])
            meta = {k: r.get(k, "") for k in DEFAULT_META_COLS}

            if idx % 10 == 0 or idx == len(df):
                print(f"  Progress: {idx}/{len(df)} prompts processed...")

            attempt = 0
            while True:
                try:
                    text, latency = _infer_one(client, prompt, gen_params, system_prompt, prefer_chat)
                    status = "ok"
                    ok += 1
                    total_latency += latency
                    break
                except Exception as e:
                    attempt += 1
                    error_msg = str(e)
                    error_type = type(e).__name__
                    print(f"[ERROR] Attempt {attempt}/{retries + 1} failed for prompt {idx}: {error_type}: {error_msg}")
                    if attempt > retries:
                        # Store full error message (up to 200 chars) in status for debugging
                        text, latency, status = "", -1, f"error:{error_type}:{error_msg[:200]}"
                        err += 1
                        print(f"[ERROR] All retries exhausted for prompt {idx}. Final error: {error_type}: {error_msg}")
                        # Print traceback for first failure to help debug
                        if idx == 1:
                            import traceback
                            print(f"[ERROR] Full traceback for first failure:")
                            traceback.print_exc()
                        break
                    print(f"[INFO] Retrying in {backoff * attempt}s...")
                    time.sleep(backoff * attempt)

            rows.append({
                "ts_iso": ts_iso,
                "prompt_id": meta.get("prompt_id", ""),
                "category": meta.get("category", ""),
                "text_length": meta.get("text_length", ""),
                "size_label": meta.get("size_label", ""), 
                "model": model_id, 
                "provider": m.get("provider", ""), 
                "prompt": prompt, 
                "response": text, 
                "latency_ms": latency, 
                "status": status, 
                "meta_json": json.dumps(gen_params)
            })

        # Filter rows to only include successful responses (status == "ok")
        successful_rows = [row for row in rows if row.get("status") == "ok"]
        failed_count = len(rows) - len(successful_rows)
        
        if failed_count > 0:
            print(f"[WARNING] Filtering out {failed_count} failed responses (status != 'ok') - will retry on next run")
            # Print details of failed responses for debugging
            failed_rows = [row for row in rows if row.get("status") != "ok"]
            for failed_row in failed_rows[:5]:  # Show first 5 failures
                print(f"  Failed response status: {failed_row.get('status', 'unknown')}")
            if len(failed_rows) > 5:
                print(f"  ... and {len(failed_rows) - 5} more failures")

        # Append to existing file or create new one (only successful responses)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(successful_rows) > 0:
            if out_path.exists():
                # Append new successful rows to existing CSV, avoiding duplicates
                try:
                    existing_df = pd.read_csv(out_path)
                    new_df = pd.DataFrame(successful_rows)
                    
                    # Filter out duplicates based on (prompt, model) combination
                    # Normalize prompts for comparison
                    if "prompt" in existing_df.columns and "model" in existing_df.columns:
                        existing_df["_normalized_prompt"] = existing_df["prompt"].astype(str).apply(_normalize_prompt)
                        new_df["_normalized_prompt"] = new_df["prompt"].astype(str).apply(_normalize_prompt)
                        
                        # Create composite key for duplicate detection
                        existing_keys = set(zip(existing_df["_normalized_prompt"], existing_df["model"]))
                        new_keys = set(zip(new_df["_normalized_prompt"], new_df["model"]))
                        
                        # Filter out rows that already exist
                        duplicates = new_keys & existing_keys
                        if duplicates:
                            print(f"[WARN] Found {len(duplicates)} duplicate (prompt, model) combinations, filtering them out")
                            new_df = new_df[~new_df.apply(lambda row: (_normalize_prompt(str(row["prompt"])), row["model"]) in existing_keys, axis=1)]
                            new_df = new_df.drop(columns=["_normalized_prompt"])
                            existing_df = existing_df.drop(columns=["_normalized_prompt"])
                        
                        if len(new_df) == 0:
                            print(f"[INFO] All {len(successful_rows)} new responses were duplicates. No rows to append.")
                        else:
                            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                            combined_df.to_csv(out_path, index=False)
                            # Explicitly flush and sync to ensure file is written to disk
                            with open(out_path, 'r+b') as f:
                                f.flush()
                                os.fsync(f.fileno())
                            print(f"[INFO]  Appended {len(new_df)} new responses (filtered {len(successful_rows) - len(new_df)} duplicates) to existing file: {out_path}")
                    else:
                        # Fallback: simple append if columns are missing
                        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                        combined_df.to_csv(out_path, index=False)
                        with open(out_path, 'r+b') as f:
                            f.flush()
                            os.fsync(f.fileno())
                        print(f"[INFO]  Appended {len(successful_rows)} successful responses to existing file: {out_path}")
                except Exception as e:
                    # If append fails, create backup and write new file
                    backup_path = out_path.with_suffix(f'.backup_{int(time.time())}.csv')
                    print(f"[WARN] Failed to append, backing up existing file to {backup_path}")
                    if out_path.exists():
                        
                        shutil.copy2(out_path, backup_path)
                    pd.DataFrame(successful_rows).to_csv(out_path, index=False)
                    # Explicitly flush and sync
                    with open(out_path, 'r+b') as f:
                        f.flush()
                        os.fsync(f.fileno())
                    print(f"[INFO]  Created new file with {len(successful_rows)} responses: {out_path}")
            else:
                # Create new file
                pd.DataFrame(successful_rows).to_csv(out_path, index=False)
                # Explicitly flush and sync to ensure file is written to disk
                with open(out_path, 'r+b') as f:
                    f.flush()
                    os.fsync(f.fileno())
                print(f"[INFO]  Created new file with {len(successful_rows)} successful responses: {out_path}")
        else:
            print(f"[INFO] No successful responses to write for {name} (all {len(rows)} failed)")

        avg_latency = int(total_latency / ok) if ok else 0
        summaries.append({
            "model": model_id,
            "provider": m.get("provider", "router-default"),
            "rows_written": len(successful_rows),  # Only count successful ones written to CSV
            "ok": ok,
            "errors": err,
            "avg_latency_ms": avg_latency,
            "output_file": str(out_path)
        })

        print(f" Completed {model_id} ({m.get('provider', 'router-default')}) → {out_path}")
        print(f" Summary: {ok} ok, {err} errors, {len(successful_rows)} successful rows written to CSV")

    return summaries


if __name__ == "__main__":
    run_model_response_generation()