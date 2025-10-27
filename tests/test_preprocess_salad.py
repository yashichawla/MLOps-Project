import pandas as pd
import pytest
from scripts.preprocess_salad import (
    clean_null_values,
    remove_duplicates,
    map_categories,
    add_metadata,
    run_preprocessing,
)

# --- BASIC FUNCTIONAL TESTS ---

def test_clean_null_values_removes_empty():
    df = pd.DataFrame({"prompt": ["hi", None, " ", "ok"]})
    cleaned = clean_null_values(df)
    assert len(cleaned) == 2
    assert all(cleaned["prompt"].isin(["hi", "ok"]))

def test_remove_duplicates_drops_repeated_prompts():
    df = pd.DataFrame({"prompt": ["hi", "hi", "bye"]})
    deduped = remove_duplicates(df)
    assert len(deduped) == 2

def test_map_categories_known_label():
    df = pd.DataFrame({
        "category": ["O19: Illegal Drugs and Regulated/Controlled Substances"]
    })
    mapped = map_categories(df)
    assert mapped["category"].iloc[0] == "Illegal Activity"

def test_add_metadata_generates_fields():
    df = pd.DataFrame({"prompt": ["short text", "this is a bit longer"]})
    df = add_metadata(df)
    assert set(["prompt_id", "text_length", "size_label"]).issubset(df.columns)
    assert df["text_length"].iloc[0] < df["text_length"].iloc[1]

# --- EDGE CASE TESTS ---

def test_map_categories_handles_unknown_label():
    df = pd.DataFrame({"category": ["O99: Nonexistent Category"]})
    mapped = map_categories(df)
    assert mapped["category"].iloc[0] == "Unknown"

def test_add_metadata_size_label_ranges():
    # Create prompts with different word counts, not character counts
    # S: <50 words, M: 50-200 words, L: >200 words
    df = pd.DataFrame({"prompt": [
        " ".join(["word"] * 30),      # 30 words -> S
        " ".join(["word"] * 100),      # 100 words -> M  
        " ".join(["word"] * 250),      # 250 words -> L
    ]})
    df = add_metadata(df)
    assert set(df["size_label"]).issubset({"S", "M", "L"})
    assert df.loc[0, "size_label"] == "S"
    assert df.loc[1, "size_label"] == "M"
    assert df.loc[2, "size_label"] == "L"

def test_clean_null_values_empty_dataframe():
    df = pd.DataFrame(columns=["prompt"])
    cleaned = clean_null_values(df)
    assert cleaned.empty

def test_remove_duplicates_no_duplicates():
    df = pd.DataFrame({"prompt": ["unique1", "unique2"]})
    deduped = remove_duplicates(df)
    assert len(deduped) == 2

def test_run_preprocessing_with_invalid_config(tmp_path):
    cfg = tmp_path / "config.json"
    cfg.write_text('{"data_sources":[{"type":"unknown","path":"none"}]}')
    out = tmp_path / "processed.csv"
    with pytest.raises(Exception):
        run_preprocessing(str(cfg), str(out))
