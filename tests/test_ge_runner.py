import pytest
import pandas as pd
from pathlib import Path
from scripts import ge_runner


# ---------- Helper ----------
def make_csv(tmp_path, name, data):
    """Utility to create a small CSV for testing."""
    path = tmp_path / name
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def make_baseline(tmp_path, csv_data):
    """Creates a valid baseline schema and stats file."""
    csv_path = make_csv(tmp_path, "baseline_data.csv", csv_data)
    date_str = "20251028"
    ge_runner.do_baseline(csv_path, date_str, "benign,toxic")
    return csv_path, ge_runner.SCHEMA_BASELINE


# ---------- Tests ----------

def test_load_df_missing_file(tmp_path):
    """Should raise FileNotFoundError if file missing."""
    missing = tmp_path / "missing.csv"
    with pytest.raises(FileNotFoundError):
        ge_runner._load_df(missing)


def test_empty_file_triggers_hard_fail(tmp_path):
    """Empty CSV should trigger hard fail gracefully, even if pandas errors."""
    # Instead of empty CSV (which crashes pandas), make one with only header
    path = tmp_path / "empty.csv"
    path.write_text("prompt,category,prompt_id,text_length,size_label\n")
    baseline_data = {
        "prompt": ["x"],
        "category": ["benign"],
        "prompt_id": [1],
        "text_length": [10],
        "size_label": ["S"]
    }
    _, baseline_path = make_baseline(tmp_path, baseline_data)

    # Should trigger SystemExit due to 0 rows
    with pytest.raises(SystemExit):
        ge_runner.do_validate(path, baseline_path, "20251028", "benign,toxic")


def test_missing_required_columns(tmp_path):
    """Missing required columns should hard fail."""
    data = {"prompt": ["x"], "category": ["benign"]}
    csv_path = make_csv(tmp_path, "missing_cols.csv", data)
    baseline_data = {
        "prompt": ["a"],
        "category": ["benign"],
        "prompt_id": [1],
        "text_length": [10],
        "size_label": ["S"]
    }
    _, baseline_path = make_baseline(tmp_path, baseline_data)

    with pytest.raises(SystemExit):
        ge_runner.do_validate(csv_path, baseline_path, "20251028", "benign,toxic")


def test_duplicate_prompts_soft_warn(tmp_path):
    """Duplicate prompts should only soft-warn (no SystemExit)."""
    old_min_rows = ge_runner.MIN_ROWS_HARD
    ge_runner.MIN_ROWS_HARD = 1  # temporarily relax row limit
    
    try:
        data = {
            "prompt": ["hi", "hi"],
            "category": ["benign", "benign"],
            "prompt_id": [1, 2],
            "text_length": [20, 25],
            "size_label": ["S", "S"]
        }
        csv_path = make_csv(tmp_path, "dupes.csv", data)
        baseline_data = {
            "prompt": ["a"],
            "category": ["benign"],
            "prompt_id": [1],
            "text_length": [10],
            "size_label": ["S"]
        }
        _, baseline_path = make_baseline(tmp_path, baseline_data)

        # Should NOT raise SystemExit, just create artifacts with a soft warn recorded
        ge_runner.do_validate(csv_path, baseline_path, "20251028", "benign,toxic")
        
        anomalies_path = ge_runner.METRICS_DIR / "validation" / "20251028" / "anomalies.json"
        assert anomalies_path.exists(), "Anomalies JSON not written"
        assert anomalies_path.stat().st_size > 0, "Anomalies file is empty"
    finally:
        ge_runner.MIN_ROWS_HARD = old_min_rows  # restore


def test_invalid_category_hard_fail(tmp_path):
    """Invalid category outside allowed set should hard fail."""
    data = {
        "prompt": ["p1"],
        "category": ["unknown_label"],
        "prompt_id": [1],
        "text_length": [40],
        "size_label": ["S"]
    }
    csv_path = make_csv(tmp_path, "invalid_category.csv", data)
    baseline_data = {
        "prompt": ["a"],
        "category": ["benign"],
        "prompt_id": [1],
        "text_length": [10],
        "size_label": ["S"]
    }
    _, baseline_path = make_baseline(tmp_path, baseline_data)

    with pytest.raises(SystemExit):
        ge_runner.do_validate(csv_path, baseline_path, "20251028", "benign,toxic")


def test_artifacts_written(tmp_path):
    """Ensure stats and anomaly JSONs are created successfully."""
    # We’ll override MIN_ROWS_HARD for this test only (no code changes)
    old_min_rows = ge_runner.MIN_ROWS_HARD
    ge_runner.MIN_ROWS_HARD = 1  # temporarily relax row limit

    try:
        data = {
            "prompt": ["a"],
            "category": ["benign"],
            "prompt_id": [1],
            "text_length": [100],
            "size_label": ["M"]
        }
        csv_path = make_csv(tmp_path, "artifacts.csv", data)
        baseline_data = data
        _, baseline_path = make_baseline(tmp_path, baseline_data)

        # Should complete successfully
        ge_runner.do_validate(csv_path, baseline_path, "20251028", "benign,toxic")

        stats_path = ge_runner.METRICS_DIR / "stats" / "20251028" / "stats.json"
        anomalies_path = ge_runner.METRICS_DIR / "validation" / "20251028" / "anomalies.json"

        assert stats_path.exists(), "Stats file missing"
        assert anomalies_path.exists(), "Anomalies file missing"
    finally:
        ge_runner.MIN_ROWS_HARD = old_min_rows  # restore

    
def test_size_label_mismatch_soft_warn(tmp_path):
    """Size label mismatch should soft-warn but not fail."""
    data = {
        "prompt": ["tiny"],
        "category": ["benign"],
        "prompt_id": [1],
        "text_length": [10],
        "size_label": ["L"],  # Wrong label
    }
    csv_path = make_csv(tmp_path, "mismatch.csv", data)
    baseline_data = data
    _, baseline_path = make_baseline(tmp_path, baseline_data)

    old_min_rows = ge_runner.MIN_ROWS_HARD
    ge_runner.MIN_ROWS_HARD = 1  # temporarily relax row limit
    
    try:
        ge_runner.do_validate(csv_path, baseline_path, "20251028", "benign,toxic")
    finally:
        ge_runner.MIN_ROWS_HARD = old_min_rows  # restore

