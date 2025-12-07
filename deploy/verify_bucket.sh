#!/bin/bash
# Script to verify bucket contents against dvc.yaml expectations

set -e

BUCKET="gs://mlops-project-dvc-480422"
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/.secrets/gcp-key.json"

echo "=========================================="
echo "Verifying Bucket Contents vs dvc.yaml"
echo "=========================================="
echo ""
echo "Bucket: $BUCKET"
echo ""

# Expected from dvc.yaml stages
echo "Expected from dvc.yaml:"
echo "  1. salad_preprocess:"
echo "     - data/processed/processed_data.csv"
echo "     - data/metrics/stats/"
echo "     - data/metrics/validation/"
echo ""
echo "  2. model_responses:"
echo "     - data/responses/"
echo ""
echo "  3. judge_outputs:"
echo "     - data/judge/"
echo ""
echo "  4. bias_detection:"
echo "     - data/bias/"
echo ""
echo "  5. additional_metrics:"
echo "     - data/metrics/ (additional_metrics_*.json)"
echo ""

echo "=========================================="
echo "Current Bucket Contents:"
echo "=========================================="
echo ""

# Check each expected directory
check_path() {
    local path=$1
    local desc=$2
    echo -n "  $desc: "
    if gsutil ls "$BUCKET/$path" > /dev/null 2>&1; then
        count=$(gsutil ls "$BUCKET/$path" 2>/dev/null | wc -l)
        echo "✓ Found ($count items)"
    else
        echo "✗ Missing"
    fi
}

check_file() {
    local path=$1
    local desc=$2
    echo -n "  $desc: "
    if gsutil ls "$BUCKET/$path" > /dev/null 2>&1; then
        echo "✓ Found"
    else
        echo "✗ Missing"
    fi
}

echo "Processed Data:"
check_file "data/processed/processed_data.csv" "processed_data.csv"

echo ""
echo "Metrics:"
check_path "data/metrics/stats" "stats/"
check_path "data/metrics/validation" "validation/"
echo -n "  additional_metrics_*.json: "
count=$(gsutil ls "$BUCKET/data/metrics/additional_metrics_*.json" 2>/dev/null | wc -l)
if [ $count -gt 0 ]; then
    echo "✓ Found ($count files)"
else
    echo "✗ Missing"
fi

echo ""
echo "Model Responses:"
check_path "data/responses" "responses/"

echo ""
echo "Judge Outputs:"
check_path "data/judge" "judge/"

echo ""
echo "Bias Detection:"
check_path "data/bias" "bias/"

echo ""
echo "=========================================="
echo "Summary:"
echo "=========================================="
echo ""

# Count what's actually there
metrics_count=$(gsutil ls "$BUCKET/data/metrics/additional_metrics_*.json" 2>/dev/null | wc -l)
bias_count=$(gsutil ls "$BUCKET/data/bias/*/bias_report.json" 2>/dev/null | wc -l)

echo "✓ Additional Metrics: $metrics_count files"
echo "✓ Bias Reports: $bias_count files"
echo ""

# Check what's missing
missing=0
if ! gsutil ls "$BUCKET/data/processed/processed_data.csv" > /dev/null 2>&1; then
    echo "✗ Missing: data/processed/processed_data.csv"
    missing=$((missing + 1))
fi
if ! gsutil ls "$BUCKET/data/metrics/stats" > /dev/null 2>&1; then
    echo "✗ Missing: data/metrics/stats/"
    missing=$((missing + 1))
fi
if ! gsutil ls "$BUCKET/data/metrics/validation" > /dev/null 2>&1; then
    echo "✗ Missing: data/metrics/validation/"
    missing=$((missing + 1))
fi
if ! gsutil ls "$BUCKET/data/responses" > /dev/null 2>&1; then
    echo "✗ Missing: data/responses/"
    missing=$((missing + 1))
fi
if ! gsutil ls "$BUCKET/data/judge" > /dev/null 2>&1; then
    echo "✗ Missing: data/judge/"
    missing=$((missing + 1))
fi

if [ $missing -eq 0 ]; then
    echo "✅ All expected data is present!"
else
    echo ""
    echo "⚠️  $missing expected paths are missing"
fi


