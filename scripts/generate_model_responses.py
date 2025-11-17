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


CONFIG_PATH = Path("config/attack_llm_config.json")
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
    """Load prompts (and optional metadata) from CSV; optionally take head(n_samples)."""
    df = pd.read_csv(csv_path)
    meta_cols = meta_cols or DEFAULT_META_COLS
    keep = [prompt_col] + [c for c in meta_cols if c in df.columns]
    df = df[keep].dropna(subset=[prompt_col])
    return df.head(n_samples) if sample else df


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


def _filter_new_prompts(df: pd.DataFrame, processed_prompts: set, prompt_col: str = PROMPT_COL) -> pd.DataFrame:
    """
    Filter DataFrame to only include prompts that haven't been processed yet for this model.
    Uses normalized prompt text for comparison.
    """
    if not processed_prompts:
        print(f"[INFO] No existing responses found. Processing all {len(df)} prompts.")
        return df
    
    # Normalize prompts in the input DataFrame
    df["_normalized_prompt"] = df[prompt_col].astype(str).apply(_normalize_prompt)
    
    # Filter out already-processed prompts
    df_new = df[~df["_normalized_prompt"].isin(processed_prompts)].copy()
    df_new = df_new.drop(columns=["_normalized_prompt"])
    
    skipped = len(df) - len(df_new)
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

        # Load all prompts from processed CSV
        df_all = _load_prompts(
            csv_path=m["csv_path"], 
            sample=bool(m["sample"]), 
            n_samples=int(m["n_samples"])
        )
        
        # Check for existing responses and filter out already-processed prompts
        out_path = Path(m["out_path"])
        processed_prompts = _get_processed_prompts(out_path, model_id)
        df = _filter_new_prompts(df_all, processed_prompts)
        
        if len(df) == 0:
            print(f"[INFO] All prompts already processed for {name}. Skipping API calls.")
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
                    if attempt > retries:
                        text, latency, status = "", -1, f"error:{type(e).__name__}:{str(e)[:50]}"
                        err += 1
                        break
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

        # Append to existing file or create new one
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if len(rows) > 0:
            if out_path.exists():
                # Append new rows to existing CSV
                try:
                    existing_df = pd.read_csv(out_path)
                    new_df = pd.DataFrame(rows)
                    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
                    combined_df.to_csv(out_path, index=False)
                    print(f"[INFO] ✅ Appended {len(rows)} new responses to existing file: {out_path}")
                except Exception as e:
                    # If append fails, create backup and write new file
                    backup_path = out_path.with_suffix(f'.backup_{int(time.time())}.csv')
                    print(f"[WARN] Failed to append, backing up existing file to {backup_path}")
                    if out_path.exists():
                        import shutil
                        shutil.copy2(out_path, backup_path)
                    pd.DataFrame(rows).to_csv(out_path, index=False)
                    print(f"[INFO] ✅ Created new file with {len(rows)} responses: {out_path}")
            else:
                # Create new file
                pd.DataFrame(rows).to_csv(out_path, index=False)
                print(f"[INFO] ✅ Created new file with {len(rows)} responses: {out_path}")
        else:
            print(f"[INFO] No new responses to write for {name}")

        avg_latency = int(total_latency / ok) if ok else 0
        summaries.append({
            "model": model_id,
            "provider": m.get("provider", "router-default"),
            "rows_written": len(rows),
            "ok": ok,
            "errors": err,
            "avg_latency_ms": avg_latency,
            "output_file": str(out_path)
        })

        print(f"✅ Completed {model_id} ({m.get('provider', 'router-default')}) → {out_path}")
        print(f"   Summary: {ok} ok, {err} errors, {len(rows)} new rows written")

    return summaries


if __name__ == "__main__":
    run_model_response_generation()