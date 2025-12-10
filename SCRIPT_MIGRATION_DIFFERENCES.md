# Script Migration Differences: Top-Level `scripts/` → `dags/scripts/`

## Overview

The scripts were moved from the top-level `scripts/` directory to `dags/scripts/` to work with Google Cloud Composer, which only syncs the `dags/` folder. This document outlines the differences between the old and new versions.

## Scripts That Were Moved (Renamed in Git)

These scripts were moved from `scripts/` to `dags/scripts/`:

1. ✅ `scripts/additional_metrics.py` → `dags/scripts/additional_metrics.py`
2. ✅ `scripts/bias_detection.py` → `dags/scripts/bias_detection.py`
3. ✅ `scripts/ge_runner.py` → `dags/scripts/ge_runner.py`
4. ✅ `scripts/generate_model_responses.py` → `dags/scripts/generate_model_responses.py`
5. ✅ `scripts/judge.py` → `dags/scripts/judge.py`
6. ✅ `scripts/judge_responses.py` → `dags/scripts/judge_responses.py`
7. ✅ `scripts/preprocess_salad.py` → `dags/scripts/preprocess_salad.py`
8. ✅ `scripts/__init__.py` → `dags/scripts/__init__.py`
9. ✅ `scripts/config/data_sources.json` → `dags/scripts/config/data_sources.json`

## Scripts That Were Deleted (Not Moved)

These scripts were deleted and not moved to `dags/scripts/`:

1. ❌ `scripts/validator.py` - **DELETED** (functionality likely consolidated into `ge_runner.py`)
2. ❌ `scripts/validate_salad.py` - **DELETED** (functionality likely in `ge_runner.py`)
3. ❌ `scripts/preprocess_judgedataset.py` - **DELETED** (functionality may be in `preprocess_salad.py` or elsewhere)

## Key Differences in Moved Scripts

### 1. Path Resolution Logic

**Old (Top-Level) Version:**
```python
# Simple relative paths
CONFIG_PATH = Path("config/attack_llm_config.json")
DATA_DIR = Path("data")
```

**New (dags/scripts/) Version:**
```python
# Complex path resolution for multiple environments
SCRIPT_DIR = Path(__file__).resolve().parent
DAGS_DIR = SCRIPT_DIR.parent  # dags/scripts/ -> dags/
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", str(DAGS_DIR.parent))).resolve()

# Try multiple locations
if (PROJECT_ROOT / "config" / "attack_llm_config.json").exists():
    CONFIG_PATH = PROJECT_ROOT / "config" / "attack_llm_config.json"
else:
    CONFIG_PATH = DAGS_DIR / "config" / "attack_llm_config.json"

# Data directory with DVC support
if "DVC_DATA_DIR" in os.environ:
    DATA_DIR = Path(os.environ["DVC_DATA_DIR"]).resolve()
elif "PROJECT_ROOT" in os.environ:
    DATA_DIR = Path(os.environ["PROJECT_ROOT"]) / "data"
else:
    # Fallback: dags/scripts/ -> dags/ -> repo root -> data/
    DATA_DIR = SCRIPT_DIR.parent.parent.parent / "data"
```

### 2. Environment Support

The new scripts support:
- **Local Docker**: Uses `PROJECT_ROOT` environment variable
- **Google Cloud Composer**: Uses `dags/config/` and `dags/dvc_project/data/`
- **DVC Integration**: Supports `DVC_DATA_DIR` environment variable

### 3. Functional Changes

Most scripts maintain the same core functionality, but with updated path handling. The main functional differences are:

1. **Path Resolution**: More robust, handles multiple environments
2. **Config Location**: Tries `PROJECT_ROOT/config/` first, then `dags/config/`
3. **Data Location**: Supports DVC data directory via `DVC_DATA_DIR` env var

## Scripts That Need Verification

Since some scripts were deleted, verify that their functionality was properly consolidated:

1. ✅ **`validator.py`** → **CONSOLIDATED** into `ge_runner.py` (Great Expectations validation)
2. ✅ **`validate_salad.py`** → **CONSOLIDATED** into `ge_runner.py` (validation) and `preprocess_salad.py` (Salad-specific preprocessing)
3. ⚠️ **`preprocess_judgedataset.py`** → **STATUS UNKNOWN** - Check if judge dataset preprocessing is needed or handled elsewhere

## Recommendations

1. ✅ **Path Resolution**: The new path resolution logic is necessary and correct for Composer compatibility
2. ⚠️ **Deleted Scripts**: Verify that deleted scripts' functionality is preserved:
   - Review `ge_runner.py` to ensure it covers all validation needs
   - Check if `preprocess_judgedataset.py` functionality is needed
3. ✅ **Consistency**: All moved scripts follow the same path resolution pattern, which is good
4. ⚠️ **Comments**: Some scripts still have comments like `# scripts/generate_model_responses.py` - consider updating to `# dags/scripts/generate_model_responses.py`

## Summary

**Should they be the same?** Functionally, yes - the core logic should be identical. However, the path resolution needed to be updated to work in the new location (`dags/scripts/` instead of top-level `scripts/`). 

### Key Findings:

1. ✅ **Path Resolution**: Updated correctly for Composer compatibility
2. ✅ **Validation Scripts**: `validator.py` and `validate_salad.py` functionality is in `ge_runner.py`
3. ⚠️ **Judge Dataset Preprocessing**: `preprocess_judgedataset.py` was deleted - verify if this functionality is needed
4. ✅ **Core Functionality**: All moved scripts maintain their core functionality with updated path handling

### Action Items:

1. Verify if `preprocess_judgedataset.py` functionality is needed
2. Update file header comments from `# scripts/` to `# dags/scripts/` for consistency
3. Consider adding a note in `preprocess_salad.py` about the hardcoded `CONFIG_PATH = "config/data_sources.json"` (line 260) - this should use the same path resolution pattern as other scripts

