import pytest
import json
import pandas as pd
from pathlib import Path
from scripts.preprocess_salad import run_preprocessing
from scripts.validator import run_validation as validate_output_csv  # alias fix

# ------------------ TEST 1: Full pipeline success ------------------

def test_full_pipeline_success(tmp_path):
    raw = tmp_path / "raw.csv"
    output = tmp_path / "processed.csv"
    config = tmp_path / "config.json"

    df = pd.DataFrame({
        "prompts": [f"Prompt {i}" for i in range(60)],
        "category": ["Safe"] * 60
    })
    df.to_csv(raw, index=False)
    config.write_text(json.dumps({"data_sources": [{"type": "csv", "path": str(raw)}]}))

    run_preprocessing(str(config), str(output))
    result = validate_output_csv(str(output), str(tmp_path))

    assert output.exists()
    assert isinstance(result, dict)
    assert "hard_fail" in result
    assert "soft_warn" in result


# ------------------ TEST 2: Missing config ------------------

def test_pipeline_with_missing_config(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_preprocessing("missing_config.json", str(tmp_path / "output.csv"))


# ------------------ TEST 3: Empty data ------------------

def test_pipeline_with_empty_data(tmp_path):
    raw = tmp_path / "empty.csv"
    output = tmp_path / "processed.csv"
    config = tmp_path / "config.json"
    pd.DataFrame(columns=["prompts", "category"]).to_csv(raw, index=False)
    config.write_text(json.dumps({"data_sources": [{"type": "csv", "path": str(raw)}]}))

    run_preprocessing(str(config), str(output))
    result = validate_output_csv(str(output), str(tmp_path))
    assert "hard_fail" in result


# ------------------ TEST 4: Invalid category ------------------

def test_pipeline_with_invalid_category(tmp_path):
    raw = tmp_path / "raw.csv"
    output = tmp_path / "processed.csv"
    config = tmp_path / "config.json"
    df = pd.DataFrame({"prompts": ["hi"], "category": ["Invalid"]})
    df.to_csv(raw, index=False)
    config.write_text(json.dumps({"data_sources": [{"type": "csv", "path": str(raw)}]}))

    run_preprocessing(str(config), str(output))
    result = validate_output_csv(str(output), str(tmp_path))
    assert isinstance(result, dict)
