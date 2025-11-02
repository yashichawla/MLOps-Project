# scripts/generate_model_responses.py
"""
Generate model responses over preprocessed prompts.

- Reads prompts from processed_data.csv
- Optionally samples (for dev/testing)
- Calls a Hugging Face model
- Writes model_responses.csv (overwrite each run)

Only secret needed: HF_TOKEN in .env
"""

from __future__ import annotations
import os, time, json
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Dict, Any, Tuple

import pandas as pd
from dotenv import load_dotenv

# ================= CONFIG  =================
CSV_PATH = Path("data/processed/processed_data.csv")
OUT_PATH = Path("data/processed/model_responses.csv")

PROMPT_COL = "prompt"
META_INPUT_COLS = ["category", "prompt_id", "text_length", "size_label"]

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_NEW_TOKENS = 256
TEMPERATURE = 0.2
TOP_P = 0.9
SEED = 42

# **Dev/Test Mode TOGGLE**
SAMPLE = True        # True → use first N_SAMPLES, False → use full CSV
N_SAMPLES = 5        # Only applies when SAMPLE=True

SYSTEM_PROMPT = ""   # no system constraints — raw jailbreak behavior
# ======================================================


def _get_hf_token() -> str:
    load_dotenv(override=False)
    tok = os.getenv("HF_TOKEN")
    if not tok:
        raise RuntimeError("HF_TOKEN missing. Add to .env: HF_TOKEN=hf_xxxxx")
    return tok


def _load_prompts() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    keep = [PROMPT_COL] + [c for c in META_INPUT_COLS if c in df.columns]
    df = df[keep].dropna(subset=[PROMPT_COL])
    return df.head(N_SAMPLES) if SAMPLE else df


def _make_client():
    from huggingface_hub import InferenceClient
    return InferenceClient(model=MODEL_ID, token=_get_hf_token())


def _infer_one(client, prompt: str) -> Tuple[str, int]:
    t0 = time.time()
    try:
        messages = (
            [{"role": "user", "content": prompt}]
            if not SYSTEM_PROMPT.strip()
            else [{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": prompt}]
        )
        out = client.chat.completions.create(
            messages=messages,
            max_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            seed=SEED,
        )
        text = out.choices[0].message.content
    except Exception:
        out = client.text_generation(
            prompt=prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            return_full_text=False,
        )
        if isinstance(out, str):
            text = out
        elif isinstance(out, list) and out and isinstance(out[0], dict) and "generated_text" in out[0]:
            text = out[0]["generated_text"]
        else:
            text = str(out)
    return text, int((time.time() - t0) * 1000)


def run_model_response_generation() -> Dict[str, Any]:
    df = _load_prompts()
    client = _make_client()
    ts = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    rows = []
    total_latency = 0
    ok_count = err_count = 0

    for _, row in df.iterrows():
        prompt = str(row[PROMPT_COL])
        meta = {k: row[k] for k in META_INPUT_COLS if k in row}
        try:
            resp, latency = _infer_one(client, prompt)
            status = "ok"
            ok_count += 1
            total_latency += latency
        except Exception as e:
            resp, latency = "", -1
            status = f"error:{type(e).__name__}:{str(e)}"
            err_count += 1

        rows.append({
            "ts_iso": ts,
            "prompt_id": meta.get("prompt_id", ""),
            "category": meta.get("category", ""),
            "text_length": meta.get("text_length", ""),
            "size_label": meta.get("size_label", ""),
            "model": MODEL_ID,
            "prompt": prompt,
            "response": resp,
            "latency_ms": latency,
            "status": status,
            "meta_json": json.dumps({
                "max_new_tokens": MAX_NEW_TOKENS,
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "seed": SEED,
                "sample_mode": SAMPLE,
            }),
        })

    out_cols = [
        "ts_iso","prompt_id","category","text_length","size_label",
        "model","prompt","response","latency_ms","status","meta_json"
    ]
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=out_cols).to_csv(OUT_PATH, index=False)

    avg_latency = int(total_latency / ok_count) if ok_count else 0
    result = {
        "rows_written": len(rows),
        "ok": ok_count,
        "errors": err_count,
        "avg_latency_ms": avg_latency,
        "output_file": str(OUT_PATH),
    }
    print(f"\n Generated: {result}\n")
    return result


if __name__ == "__main__":
    run_model_response_generation()
