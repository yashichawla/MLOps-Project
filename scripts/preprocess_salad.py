# preprocess_salad_data.py

"""
Preprocessing pipeline for Salad-Data and other safety datasets.

This script:
1. Loads datasets from one or more sources (defined in config).
2. Validates and standardizes data format.
3. Cleans missing or empty prompts.
4. Normalizes scenario categories.
5. Combines all processed data into one unified DataFrame.

Final output columns:
    prompt | category | prompt_id | text_length | size_label

To be integrated later with Airflow for automated execution.
"""


from datasets import load_dataset
import pandas as pd
import json
import os
import uuid

# --------------------------
# CONFIG LOADING
# --------------------------


def load_config(config_path: str):
    """Load data source configuration file (JSON/YAML)."""
    print(f"Loading config from {config_path}...")
    if config_path.endswith(".json"):
        with open(config_path, "r") as f:
            return json.load(f)
    else:
        raise ValueError("Only JSON config supported for now.")


# --------------------------
# DATA LOADING
# --------------------------


def load_dataset_from_hf(source):
    """Load Hugging Face dataset (e.g., Salad)."""
    print(f" Loading HF dataset: {source['name']} ({source['config']})...")
    ds = load_dataset(source["name"], source["config"], split=source["split"])
    df = ds.to_pandas()
    return df


def load_dataset_from_csv(source):
    """Load CSV dataset."""
    print(f" Loading CSV: {source['path']}...")
    return pd.read_csv(source["path"])


def load_dataset_from_json(source):
    """Load JSON dataset."""
    print(f" Loading JSON: {source['path']}...")
    return pd.read_json(source["path"])


def load_datasets(config):
    """Load and combine all datasets listed in config."""
    combined_data = []

    for source in config.get("data_sources", []):
        src_type = source.get("type", "hf")
        try:
            if src_type == "hf":
                df = load_dataset_from_hf(source)
            elif src_type == "csv":
                df = load_dataset_from_csv(source)
            elif src_type == "json":
                df = load_dataset_from_json(source)
            else:
                print(f" Unsupported data type '{src_type}', skipping.")
                continue
        except Exception as e:
            print(f"Failed to load {source}: {e}")
            continue

        # Handle Salad Data separately
        if source.get("name") == "OpenSafetyLab/Salad-Data":
            if "augq" in df.columns and "3-category" in df.columns:
                df = df.rename(columns={"augq": "prompt"})
                df = df.rename(columns={"3-category": "category"})
                df = df[["prompt", "category"]]
            else:
                print(
                    "Salad dataset missing expected columns 'augq' or 'category'. Skipping."
                )
                continue

        # Handle other datasets — keep only 'prompts'
        else:
            if "prompts" in df.columns:
                df = df.rename(columns={"prompts": "prompt"})
                df["category"] = "Unknown"
            else:
                print(
                    f"'prompts' column not found in {source.get('path', source.get('name'))}. Skipping."
                )
                continue

        combined_data.append(df)

    if not combined_data:
        raise ValueError("No valid datasets loaded.")

    df_combined = pd.concat(combined_data, ignore_index=True)
    print(f" Combined {len(combined_data)} datasets, total {len(df_combined)} rows.")
    return df_combined


def clean_null_values(df):
    """Remove rows with null/empty prompts."""
    before = len(df)
    df = df.dropna(subset=["prompt"])
    df = df[df["prompt"].str.strip() != ""]
    after = len(df)
    print(f"Removed {before - after} empty prompts.")
    return df


def remove_duplicates(df):
    """Remove duplicate baseq-augq pairs."""
    print("Removing duplicates...")
    df = df.drop_duplicates(subset=["prompt"], keep="first").reset_index(drop=True)
    print(f"Dataset after deduplication: {len(df)} rows")
    return df


def map_categories(df):
    """Map raw 3-category labels to 13 standardized categories."""
    print("Mapping categories...")

    category_map = {
        # Illegal Activity
        "O19: Illegal Drugs and Regulated/Controlled Substances": "Illegal Activity",
        "O54: Drug-related Crimes": "Illegal Activity",
        "O42: Scams": "Illegal Activity",
        "O37: Malware Generation": "Malware Generation",
        "O35: Weapon Generation and Transportation": "Illegal Activity",
        "O56: Violent Crimes": "Physical Harm",
        "O60: Traffic and Driving Offenses": "Illegal Activity",
        "O59: Environmental Crimes": "Illegal Activity",
        "O43: Financial Fraud": "Fraud",
        "O65: Plagiarism": "Fraud",
        "O45: Fake Review Generation": "Fraud",
        "O50: Manipulation of Public Opinion": "Political Lobbying",
        "O51: Campaign Materials": "Political Lobbying",
        "O31: Government Secrets": "Government Decision",
        "O33: Inferring Personal Information": "Privacy Violence",
        "O34: Leaking Private Information": "Privacy Violence",
        "O48: Spam": "Economic Harm",
        "O25: Paypal Lending": "Economic Harm",
        # Hate Speech
        "O1: Hate Speech": "Hate Speech",
        "O7: Insult": "Hate Speech",
        "O8: Racial and Ethnic Stereotyping": "Hate Speech",
        "O10: Disability Stereotyping": "Hate Speech",
        "O9: Age Stereotyping": "Hate Speech",
        "O11: Gender Stereotyping": "Hate Speech",
        "O12: Religious Stereotyping": "Hate Speech",
        # Sexual / Child Abuse
        "O6: Child Abuse": "Child Sexual Abuse",
        "O15: Sexual Activity": "Pornography",
        "O16: Pornography": "Pornography",
        "O14: Erotic Chat": "Pornography",
        "O55: Sexual Offenses": "Pornography",
        # Health / Legal / Financial Advice
        "O17: Law Advice": "Legal Opinion",
        "O58: Illegal Law Advice": "Legal Opinion",
        "O20: Medical Advice": "Health Consultation",
        "O53: Financial Crimes": "Financial Advice",
        # Physical / Psychological Harm
        "O3: Enjoying someone else's Pain": "Physical Harm",
        "O2: Harass, Threaten, or Bully An Individual": "Physical Harm",
        "O39: Biological and Chemical Harms": "Physical Harm",
        "O63: Psychological Manipulation": "Physical Harm",
        "O38: Cyber Attack": "Physical Harm",
        # Other
        "O21: Historical Revisionism": "Political Lobbying",
        "O22: Science Denialism": "Political Lobbying",
        "O23: Radicalization": "Political Lobbying",
        "O30: Social Media Rumors": "Political Lobbying",
        "O24: Multi-level Marketing": "Economic Harm",
        "O26: Gambling": "Economic Harm",
        "O40: Group Defamation": "Hate Speech",
        "O41: Defamation Someone": "Hate Speech",
        "O46: Fake Online Content": "Political Lobbying",
        "O29: Fake News": "Political Lobbying",
        "O47: Disinformation": "Political Lobbying",
        "O32: Financial Data Leaks": "Financial Advice",
        "O62: Self-Harm": "Physical Harm",
        "O64: Mimicking Writing Style": "Fraud",
        "O28: Trade Affairs": "Economic Harm",
        "O27: Labor Issues": "Economic Harm",
    }

    if "category" in df.columns:
        df["category"] = df["category"].map(category_map).fillna("Unknown")
    else:
        df["category"] = "Unknown"
    return df


def add_metadata(df):
    """Add prompt_id, text_length, and size_label columns."""
    print(" Adding metadata columns...")

    df["prompt_id"] = [str(uuid.uuid4())[:8] for _ in range(len(df))]
    df["text_length"] = df["prompt"].apply(lambda x: len(str(x).split()))

    def size_label(n):
        if n < 50:
            return "S"
        elif n <= 200:
            return "M"
        else:
            return "L"

    df["size_label"] = df["text_length"].apply(size_label)
    return df


# --------------------------
# MAIN PIPELINE
# --------------------------


def run_preprocessing(
    config_path: str, save_path: str = "data/processed/processed_data.csv"
):
    """Run full preprocessing flow."""
    os.makedirs("data", exist_ok=True)

    config = load_config(config_path)
    df = load_datasets(config)
    df = clean_null_values(df)
    df = map_categories(df)
    df = add_metadata(df)

    df = df[["prompt", "category", "prompt_id", "text_length", "size_label"]]
    df.to_csv(save_path, index=False)

    print(f"Preprocessing complete — saved {len(df)} records to {save_path}.")


# --------------------------
# ENTRY POINT
# --------------------------

if __name__ == "__main__":
    CONFIG_PATH = "config/data_sources.json"
    os.makedirs("config", exist_ok=True)

    if not os.path.exists(CONFIG_PATH):
        example_config = {
            "data_sources": [
                {
                    "type": "hf",
                    "name": "OpenSafetyLab/Salad-Data",
                    "config": "attack_enhanced_set",
                    "split": "train",
                },
                {"type": "csv", "path": "data/local_dataset.csv"},
            ]
        }
        with open(CONFIG_PATH, "w") as f:
            json.dump(example_config, f, indent=2)
        print(" Example config file created at config/data_sources.json")

    run_preprocessing(CONFIG_PATH)
