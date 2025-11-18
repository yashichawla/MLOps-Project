import os
import json
import pandas as pd

from judge import judge_llm, append_judgement_to_csv

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "attack_llm_config.json")
JUDGE_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data", "judge")

os.makedirs(JUDGE_OUTPUT_DIR, exist_ok=True)

def load_config():
    print("Loading config:", CONFIG_PATH)

    if not os.path.exists(CONFIG_PATH):
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    print(f"[INFO] Loaded {len(cfg['models'])} model entries from config.")
    return cfg["models"]

def process_single_csv(csv_path: str, out_path: str):
    print(f"\n=========== Processing {csv_path} ===========")

    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    if "prompt" not in df.columns or "response" not in df.columns:
        print(f"[ERROR] Required columns missing in {csv_path}. Skipping.")
        return

    print(f"[INFO] Loaded {len(df)} rows.")
    
    # Filter out rows with missing/null responses before processing
    df = df.dropna(subset=["response"])
    df = df[df["response"].astype(str).str.strip() != ""]  # Also filter empty strings
    print(f"[INFO] After filtering invalid responses: {len(df)} rows remaining.")

    existing_responses = set()
    if os.path.exists(out_path):
        try:
            judged_df = pd.read_csv(out_path)
            if "response" in judged_df.columns:
                existing_responses = set(judged_df["response"].astype(str).tolist())
                print(f"[INFO] Found {len(existing_responses)} previously judged responses. Skipping duplicates.")
        except:
            pass

    df_new = df[~df["response"].astype(str).isin(existing_responses)]

    print(f"[INFO] {len(df_new)} new responses found to judge.")

    if len(df_new) == 0:
        print("[INFO] Nothing new to judge.")
        return

    if not os.path.exists(out_path):
        print("[INFO] Creating fresh judged CSV file.")
    else:
        print("[INFO] Appending to existing judged CSV file.")

    for i, row in df_new.iterrows():
        prompt = str(row["prompt"]) if pd.notna(row["prompt"]) else ""
        response = str(row["response"]) if pd.notna(row["response"]) else ""

        print(f"\n---- ({i+1}/{len(df_new)}) Evaluating ----")
        print(f"PROMPT: {prompt[:100] if len(prompt) > 100 else prompt}...")
        print(f"RESPONSE: {response[:100] if len(response) > 100 else response}...")

        result = judge_llm(prompt, response)

        append_judgement_to_csv(
            original_row=row.to_dict(),
            judgment=result,
            out_path=out_path
        )

    print(f" Saved judged file to {out_path}")

def judge_all_models():
    models = load_config()

    for model in models:
        csv_path = os.path.join(PROJECT_ROOT, model["out_path"])
        
        model_name = model["name"]
        judge_csv_path = os.path.join(
            JUDGE_OUTPUT_DIR, f"judgements_{model_name}.csv"
        )

        process_single_csv(csv_path, judge_csv_path)

    print("\n🎉 ALL MODELS JUDGED SUCCESSFULLY!")


if __name__ == "__main__":
    judge_all_models()
