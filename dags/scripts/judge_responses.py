import os
import sys
import json
import pandas as pd

# Add scripts directory to Python path so we can import judge
# This is needed both when the script is run directly and when Airflow parses it
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from judge import judge_llm, append_judgement_to_csv
# Use PROJECT_ROOT env var if set (Docker), otherwise calculate from script location (Composer)
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(CURRENT_DIR, "..", "..")))
PROJECT_ROOT = os.path.abspath(PROJECT_ROOT)  # Ensure absolute path

# DAGS_DIR is the parent of scripts directory (dags/scripts/ -> dags/)
DAGS_DIR = os.path.dirname(CURRENT_DIR)

# Config path: try PROJECT_ROOT/config/ first (for local), then DAGS_DIR/config/ (for Composer)
# In local Docker: PROJECT_ROOT=/opt/airflow/app, config is at /opt/airflow/app/config/
# In Composer: PROJECT_ROOT=/home/airflow/gcs, config is at /home/airflow/gcs/dags/config/
config_path_project = os.path.join(PROJECT_ROOT, "config", "attack_llm_config.json")
config_path_dags = os.path.join(DAGS_DIR, "config", "attack_llm_config.json")
if os.path.exists(config_path_project):
    CONFIG_PATH = config_path_project
else:
    CONFIG_PATH = config_path_dags

# Determine data directory: use DVC_DATA_DIR if set (data inside DVC repo), otherwise fall back to PROJECT_ROOT/data/
# Data is now stored inside DVC repo at dvc_project/data/ when DVC_DATA_DIR is set
if "DVC_DATA_DIR" in os.environ:
    DATA_DIR = os.path.abspath(os.environ["DVC_DATA_DIR"])
elif "PROJECT_ROOT" in os.environ:
    DATA_DIR = os.path.join(os.environ["PROJECT_ROOT"], "data")
else:
    # Fallback: calculate from script location (dags/scripts/ -> root-level data/)
    # Go up from dags/scripts/ to dags/ to repo root, then to data/
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_DIR))), "data")

JUDGE_OUTPUT_DIR = os.path.join(DATA_DIR, "judge")
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
