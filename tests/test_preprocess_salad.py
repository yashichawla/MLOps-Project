import pytest
import pandas as pd
import json
import os
from pathlib import Path
from scripts.preprocess_salad import (
    load_config,
    clean_null_values,
    remove_duplicates,
    map_categories,
    add_metadata,
    run_preprocessing,
)

# ---------------- CONFIG LOADING ----------------

def test_load_config_json_valid(tmp_path):
    cfg = {"data_sources": [{"type": "csv", "path": "data.csv"}]}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    result = load_config(str(path))
    assert result["data_sources"][0]["path"] == "data.csv"


# ---------------- DATA CLEANING ----------------

def test_clean_null_values_removes_empty():
    df = pd.DataFrame({"prompt": ["hi", None, " ", "ok"]})
    cleaned = clean_null_values(df)
    assert len(cleaned) == 2
    assert all(cleaned["prompt"].isin(["hi", "ok"]))

def test_clean_null_values_empty_dataframe():
    df = pd.DataFrame({"prompt": [None, None]})
    result = clean_null_values(df)
    assert result.empty


# ---------------- DUPLICATES ----------------

def test_remove_duplicates_drops_repeated_prompts():
    df = pd.DataFrame({"prompt": ["hi", "hi", "bye"]})
    result = remove_duplicates(df)
    assert len(result) == 2

def test_remove_duplicates_no_duplicates():
    df = pd.DataFrame({"prompt": ["a", "b", "c"]})
    result = remove_duplicates(df)
    assert len(result) == 3


# ---------------- CATEGORY MAPPING ----------------

def test_map_categories_known_label():
    df = pd.DataFrame({"prompt": ["x"], "category": ["O19: Illegal Drugs and Regulated/Controlled Substances"]})
    mapped = map_categories(df)
    assert mapped["category"].iloc[0] == "Illegal Activity"

def test_map_categories_handles_unknown_label():
    df = pd.DataFrame({"prompt": ["x"], "category": ["UnknownThing"]})
    mapped = map_categories(df)
    assert mapped["category"].iloc[0] == "Unknown"


# ---------------- METADATA ----------------

def test_add_metadata_generates_fields():
    df = pd.DataFrame({"prompt": ["hello there", "this is longer"]})
    result = add_metadata(df)
    assert all(c in result.columns for c in ["prompt_id", "text_length", "size_label"])
    assert all(result["text_length"] > 0)

def test_add_metadata_size_label_ranges():
    df = pd.DataFrame({"prompt": ["short one", "a " * 80, "a " * 250]})
    result = add_metadata(df)
    assert set(result["size_label"]) == {"S", "M", "L"}


# ---------------- FULL PIPELINE ----------------

def test_full_pipeline_csv(tmp_path):
    csv_path = tmp_path / "input.csv"
    config_path = tmp_path / "config.json"
    output_path = tmp_path / "processed.csv"

    df = pd.DataFrame({"prompts": ["hi", "ok"], "category": ["Unknown", "Unknown"]})
    df.to_csv(csv_path, index=False)
    config_path.write_text(json.dumps({"data_sources": [{"type": "csv", "path": str(csv_path)}]}))

    run_preprocessing(str(config_path), str(output_path))
    assert output_path.exists()

    result = pd.read_csv(output_path)
    assert all(c in result.columns for c in ["prompt", "category", "prompt_id", "text_length", "size_label"])
    assert len(result) == 2
