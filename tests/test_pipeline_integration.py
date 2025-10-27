import pandas as pd
import pytest
import json
from pathlib import Path
from scripts.preprocess_salad import run_preprocessing
from scripts.validate_salad import validate_output_csv

def test_full_pipeline_success(tmp_path):
    # Create enough data to pass validation (DQ_MIN_ROWS=50)
    # Note: preprocess_salad.py expects "prompts" (plural) column for non-Salad datasets
    df = pd.DataFrame({
        "prompts": [f"prompt_{i}" for i in range(60)],
        "category": [
            "O19: Illegal Drugs and Regulated/Controlled Substances",
            "O8: Racial and Ethnic Stereotyping"
        ] * 30
    })
    raw = tmp_path / "raw.csv"
    df.to_csv(raw, index=False)

    cfg = tmp_path / "config.json"
    config_data = {
        "data_sources": [{"type": "csv", "path": str(raw).replace("\\", "/")}]
    }
    with open(cfg, 'w') as f:
        json.dump(config_data, f)

    output = tmp_path / "processed.csv"
    run_preprocessing(str(cfg), str(output))
    assert output.exists()

    result = validate_output_csv(str(output), tmp_path)
    assert result["row_count"] >= 60
    assert not result["hard_fail"]

def test_pipeline_with_missing_config(tmp_path):
    missing_cfg = tmp_path / "missing.json"
    output = tmp_path / "processed.csv"
    with pytest.raises(FileNotFoundError):
        run_preprocessing(str(missing_cfg), str(output))

def test_pipeline_with_empty_data(tmp_path):
    csv = tmp_path / "empty.csv"
    pd.DataFrame(columns=["prompts", "category"]).to_csv(csv, index=False)
    cfg = tmp_path / "config.json"
    config_data = {
        "data_sources": [{"type": "csv", "path": str(csv).replace("\\", "/")}]
    }
    with open(cfg, 'w') as f:
        json.dump(config_data, f)
    output = tmp_path / "processed.csv"
    run_preprocessing(str(cfg), str(output))
    result = validate_output_csv(str(output), tmp_path)
    assert result["hard_fail"]

def test_pipeline_with_invalid_category(tmp_path):
    df = pd.DataFrame({
        "prompts": ["bad prompt"],
        "category": ["O999: Made-up category"]
    })
    raw = tmp_path / "raw.csv"
    df.to_csv(raw, index=False)
    cfg = tmp_path / "config.json"
    config_data = {
        "data_sources": [{"type": "csv", "path": str(raw).replace("\\", "/")}]
    }
    with open(cfg, 'w') as f:
        json.dump(config_data, f)
    output = tmp_path / "processed.csv"
    run_preprocessing(str(cfg), str(output))
    result = validate_output_csv(str(output), tmp_path)
    assert "Unknown" in pd.read_csv(output)["category"].iloc[0]
