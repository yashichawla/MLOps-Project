#!/bin/bash
# Cleanup script to organize GCS buckets according to intended folder structure
# 
# This script:
# 1. Cleans mlops-project-dvc-480422 bucket:
#    - Removes duplicate metrics files from wrong location
#    - Removes orphaned bias files from root bias/ directory
#    - Keeps only the correct structure: data/metrics/additional/ and data/bias/{model}/
#
# 2. Cleans Composer bucket:
#    - Removes __pycache__ directories (shouldn't be synced)
#    - Keeps dags/, config/, dvc_project/ structure

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

DVC_BUCKET="mlops-project-dvc-480422"
COMPOSER_BUCKET="us-central1-mlops-airflow-c-47afa20c-bucket"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}GCS Bucket Cleanup Script${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ============================================
# 1. Clean DVC Bucket (mlops-project-dvc-480422)
# ============================================
echo -e "${YELLOW}[1/2] Cleaning DVC bucket: ${DVC_BUCKET}${NC}"
echo ""

# Remove duplicate metrics files from wrong location (data/metrics/ instead of data/metrics/additional/)
echo -e "${YELLOW}  Removing duplicate metrics files from data/metrics/ (should only be in data/metrics/additional/)...${NC}"
DUPLICATE_METRICS=$(gsutil ls "gs://${DVC_BUCKET}/data/metrics/additional_metrics_*.json" 2>/dev/null || echo "")
if [ -n "$DUPLICATE_METRICS" ]; then
    echo "$DUPLICATE_METRICS" | while read -r file; do
        if [ -n "$file" ]; then
            echo "    Removing: $file"
            gsutil rm "$file" 2>/dev/null || true
        fi
    done
    echo -e "${GREEN}  ✓ Removed duplicate metrics files${NC}"
else
    echo -e "${GREEN}  ✓ No duplicate metrics files found${NC}"
fi
echo ""

# Remove orphaned bias files from root bias/ directory (should only be in model subdirectories)
echo -e "${YELLOW}  Removing orphaned bias files from data/bias/ root (should only be in model subdirectories)...${NC}"
ORPHANED_BIAS=$(gsutil ls "gs://${DVC_BUCKET}/data/bias/*.json" "gs://${DVC_BUCKET}/data/bias/*.csv" 2>/dev/null | grep -v "/" || echo "")
if [ -n "$ORPHANED_BIAS" ]; then
    echo "$ORPHANED_BIAS" | while read -r file; do
        if [ -n "$file" ] && [[ "$file" != *"/"* ]]; then
            echo "    Removing: $file"
            gsutil rm "$file" 2>/dev/null || true
        fi
    done
    echo -e "${GREEN}  ✓ Removed orphaned bias files${NC}"
else
    echo -e "${GREEN}  ✓ No orphaned bias files found${NC}"
fi
echo ""

# Verify correct structure
echo -e "${YELLOW}  Verifying correct structure...${NC}"
echo ""
echo -e "${BLUE}  Expected structure:${NC}"
echo "    ✓ data/metrics/additional/additional_metrics_*.json"
echo "    ✓ data/bias/{model}/bias_report.json"
echo "    ✓ data/bias/{model}/category_slice_metrics.csv"
echo "    ✓ data/bias/{model}/size_slice_metrics.csv"
echo "    ✓ data/judge/judgements_*.csv"
echo ""

METRICS_COUNT=$(gsutil ls "gs://${DVC_BUCKET}/data/metrics/additional/additional_metrics_*.json" 2>/dev/null | wc -l || echo "0")
BIAS_DIRS=$(gsutil ls "gs://${DVC_BUCKET}/data/bias/" 2>/dev/null | grep "/$" | wc -l || echo "0")
JUDGE_FILES=$(gsutil ls "gs://${DVC_BUCKET}/data/judge/judgements_*.csv" 2>/dev/null | wc -l || echo "0")

echo -e "${GREEN}  Current structure:${NC}"
echo "    Metrics files: $METRICS_COUNT"
echo "    Bias model directories: $BIAS_DIRS"
echo "    Judge files: $JUDGE_FILES"
echo ""

echo -e "${GREEN}✓ DVC bucket cleanup complete${NC}"
echo ""

# ============================================
# 2. Clean Composer Bucket
# ============================================
echo -e "${YELLOW}[2/2] Cleaning Composer bucket: ${COMPOSER_BUCKET}${NC}"
echo ""

# Remove __pycache__ directories (shouldn't be synced)
echo -e "${YELLOW}  Removing __pycache__ directories...${NC}"
PYCACHE_DIRS=$(gsutil ls "gs://${COMPOSER_BUCKET}/dags/__pycache__" 2>/dev/null || echo "")
if [ -n "$PYCACHE_DIRS" ]; then
    echo "    Removing: gs://${COMPOSER_BUCKET}/dags/__pycache__/"
    gsutil -m rm -r "gs://${COMPOSER_BUCKET}/dags/__pycache__" 2>/dev/null || true
    echo -e "${GREEN}  ✓ Removed __pycache__ directories${NC}"
else
    echo -e "${GREEN}  ✓ No __pycache__ directories found${NC}"
fi
echo ""

# Verify correct structure
echo -e "${YELLOW}  Verifying correct structure...${NC}"
echo ""
echo -e "${BLUE}  Expected structure:${NC}"
echo "    ✓ dags/*.py (DAG files)"
echo "    ✓ dags/scripts/ (Python scripts)"
echo "    ✓ dags/config/ (Config files)"
echo "    ✓ dags/dvc_project/ (DVC project)"
echo "    ✓ dags/dvc_project/.dvc/ (DVC config and cache)"
echo ""

DAG_FILES=$(gsutil ls "gs://${COMPOSER_BUCKET}/dags/*.py" 2>/dev/null | wc -l || echo "0")
CONFIG_FILES=$(gsutil ls "gs://${COMPOSER_BUCKET}/dags/config/*.json" 2>/dev/null | wc -l || echo "0")
DVC_EXISTS=$(gsutil ls "gs://${COMPOSER_BUCKET}/dags/dvc_project/dvc.yaml" 2>/dev/null | wc -l || echo "0")

echo -e "${GREEN}  Current structure:${NC}"
echo "    DAG files: $DAG_FILES"
echo "    Config files: $CONFIG_FILES"
echo "    DVC project: $([ "$DVC_EXISTS" -gt 0 ] && echo "✓ Present" || echo "✗ Missing")"
echo ""

echo -e "${GREEN}✓ Composer bucket cleanup complete${NC}"
echo ""

# ============================================
# Summary
# ============================================
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ Cleanup Complete!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Summary:${NC}"
echo "  • DVC Bucket (${DVC_BUCKET}):"
echo "    - Removed duplicate/incorrectly placed files"
echo "    - Structure now matches intended layout"
echo ""
echo "  • Composer Bucket (${COMPOSER_BUCKET}):"
echo "    - Removed __pycache__ directories"
echo "    - Structure verified"
echo ""
echo -e "${YELLOW}Note:${NC} The next DAG run will sync fresh data to the correct locations."
echo ""

