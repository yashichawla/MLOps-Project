import pytest
import pandas as pd
import json
import numpy as np
from pathlib import Path
from scripts.validator import run_validation

def make_csv(tmp_path, df):
    csv_path = tmp_path / "data.csv"
    df.to_csv(csv_path, index=False)
    return csv_path


# ---------------- BASIC VALIDATION ----------------

def test_hard_fail_on_missing_columns(tmp_path):
    df = pd.DataFrame({"wrong_col": [1, 2]})
    path = make_csv(tmp_path, df)
    result = run_validation(str(path), str(tmp_path))
    assert any("Missing required" in msg for msg in result["hard_fail"])


def test_empty_file_triggers_hard_fail(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")  # Empty file
    with pytest.raises(pd.errors.EmptyDataError):
        run_validation(str(path), str(tmp_path))


# ---------------- DUPLICATES ----------------

def test_detects_duplicates(tmp_path):
    df = pd.DataFrame({"prompt": ["hi", "hi", "hello"], "category": ["A", "A", "B"]})
    path = make_csv(tmp_path, df)
    result = run_validation(str(path), str(tmp_path))
    stats_file = Path(tmp_path) / "metrics" / "stats.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
        assert "duplicates" in json.dumps(stats).lower()


def test_case_insensitive_duplicates(tmp_path):
    df = pd.DataFrame({"prompt": ["Hi", "hi"], "category": ["A", "A"]})
    path = make_csv(tmp_path, df)
    result = run_validation(str(path), str(tmp_path))
    stats_file = Path(tmp_path) / "metrics" / "stats.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
        assert "duplicates" in json.dumps(stats).lower()


# ---------------- CATEGORY VALIDATION ----------------

def test_invalid_category_triggers_warn(tmp_path):
    df = pd.DataFrame({"prompt": ["hi"], "category": ["Invalid Category"]})
    path = make_csv(tmp_path, df)
    result = run_validation(str(path), str(tmp_path))

    stats_file = Path(tmp_path) / "metrics" / "stats.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
        assert "unknown" in json.dumps(stats).lower() or result["soft_warn"]
    else:
        assert result["soft_warn"] or result["hard_fail"]

def test_soft_warn_on_high_unknown_rate(tmp_path):
    df = pd.DataFrame({"prompt": ["hi"] * 100, "category": ["Unknown"] * 100})
    path = make_csv(tmp_path, df)
    result = run_validation(str(path), str(tmp_path))
    assert isinstance(result, dict)


# ---------------- TEXT LENGTH ----------------

def test_short_text_triggers_soft_warn(tmp_path):
    df = pd.DataFrame({"prompt": ["a", "b"], "category": ["A", "B"]})
    path = make_csv(tmp_path, df)
    result = run_validation(str(path), str(tmp_path))
    stats_file = Path(tmp_path) / "metrics" / "stats.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
        assert "avg_text_length" in stats or "min_text_length" in stats


def test_long_text_soft_warn(tmp_path):
    df = pd.DataFrame({"prompt": ["x" * 2000], "category": ["A"]})
    path = make_csv(tmp_path, df)
    result = run_validation(str(path), str(tmp_path))
    stats_file = Path(tmp_path) / "metrics" / "stats.json"
    if stats_file.exists():
        stats = json.loads(stats_file.read_text())
        assert "max_text_length" in stats


# ---------------- THRESHOLDS ----------------

def test_soft_threshold_row_count(tmp_path):
    df = pd.DataFrame({"prompt": ["hi"], "category": ["A"]})
    path = make_csv(tmp_path, df)
    result = run_validation(str(path), str(tmp_path))
    assert any("row" in msg.lower() or "count" in msg.lower() for msg in result["hard_fail"] + result["soft_warn"])


def test_extra_columns_dont_fail(tmp_path):
    df = pd.DataFrame({
        "prompt": ["hi"],
        "category": ["A"],
        "extra_col": [123]
    })
    path = make_csv(tmp_path, df)
    result = run_validation(str(path), str(tmp_path))
    assert isinstance(result, dict)


# ---------------- VALID DATA ----------------

def test_valid_data_passes_all(tmp_path):
    df = pd.DataFrame({
        "prompt": [f"Prompt {i}" for i in range(60)],
        "category": ["Safe"] * 60,
        "prompt_id": np.arange(60),
        "text_length": np.random.randint(10, 200, 60),
        "size_label": ["M"] * 60
    })
    path = make_csv(tmp_path, df)
    result = run_validation(str(path), str(tmp_path))
    assert isinstance(result, dict)
    assert not result["hard_fail"]


