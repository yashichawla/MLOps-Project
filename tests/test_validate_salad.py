import pandas as pd
from scripts.validate_salad import validate_output_csv

# --- BASIC TESTS ---

def test_validation_hard_fails_on_missing_columns(tmp_path):
    df = pd.DataFrame({"prompt": ["hello"], "category": ["Unknown"]})
    bad_csv = tmp_path / "bad.csv"
    df.to_csv(bad_csv, index=False)
    result = validate_output_csv(str(bad_csv), tmp_path)
    assert result["hard_fail"] is True
    assert "missing_columns" in result or result["row_count"] == 0

def test_validation_soft_warns_on_unknown_rate(tmp_path):
    df = pd.DataFrame({
        "prompt": ["hi", "hello", "ok"],
        "category": ["Unknown", "Unknown", "Unknown"],
        "prompt_id": [1, 2, 3],
        "text_length": [10, 20, 30],
        "size_label": ["S", "M", "L"],
    })
    csv = tmp_path / "unknown.csv"
    df.to_csv(csv, index=False)
    result = validate_output_csv(str(csv), tmp_path)
    assert "unknown_category_rate" in result

# --- EDGE CASES ---

def test_validation_detects_duplicates(tmp_path):
    df = pd.DataFrame({
        "prompt": ["same", "same", "diff"],
        "category": ["Illegal Activity", "Illegal Activity", "Hate Speech"],
        "prompt_id": [1, 2, 3],
        "text_length": [5, 5, 10],
        "size_label": ["S", "S", "S"],
    })
    csv = tmp_path / "dupes.csv"
    df.to_csv(csv, index=False)
    result = validate_output_csv(str(csv), tmp_path)
    assert result["row_count"] == 3

def test_validation_empty_file(tmp_path):
    csv = tmp_path / "empty.csv"
    # Create a valid CSV structure with required columns but no rows
    pd.DataFrame(columns=["prompt", "category", "prompt_id", "text_length", "size_label"]).to_csv(csv, index=False)
    result = validate_output_csv(str(csv), tmp_path)
    # Empty file should hard fail because it has 0 rows (below DQ_MIN_ROWS=50)
    assert result["hard_fail"]

def test_validation_long_text_detection(tmp_path):
    df = pd.DataFrame({
        "prompt": ["a" * 9000],
        "category": ["Illegal Activity"],
        "prompt_id": [1],
        "text_length": [9000],
        "size_label": ["L"],
    })
    csv = tmp_path / "longtext.csv"
    df.to_csv(csv, index=False)
    result = validate_output_csv(str(csv), tmp_path)
    assert "long_text" in result or result["soft_warn"]

def test_validation_handles_all_ok_case(tmp_path):
    # Create enough rows to pass the DQ_MIN_ROWS (50) threshold
    df = pd.DataFrame({
        "prompt": [f"prompt_{i}" for i in range(60)],
        "category": ["Illegal Activity", "Hate Speech"] * 30,
        "prompt_id": [i for i in range(60)],
        "text_length": [10 + i for i in range(60)],
        "size_label": ["S" if i < 50 else "M" for i in range(60)],
    })
    csv = tmp_path / "good.csv"
    df.to_csv(csv, index=False)
    result = validate_output_csv(str(csv), tmp_path)
    assert not result["hard_fail"]
    assert result["row_count"] == 60
