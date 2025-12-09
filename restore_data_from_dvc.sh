#!/bin/bash
# Script to restore data files from DVC remote storage to Composer bucket
# Use this if upload_to_composer.sh accidentally overwrote data files

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get project root (script location)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get the actual Composer bucket
echo -e "${YELLOW}📦 Getting Composer bucket name...${NC}"
COMPOSER_BUCKET=$(gcloud composer environments describe mlops-airflow-composer \
  --location us-central1 \
  --project break-the-bot-480422 \
  --format="value(config.dagGcsPrefix)" 2>/dev/null | sed 's|/dags$||' | sed 's|^gs://||' || echo "")

if [ -z "$COMPOSER_BUCKET" ]; then
    echo -e "${RED}❌ Error: Could not get Composer bucket. Make sure Composer environment exists.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Composer Bucket: ${COMPOSER_BUCKET}${NC}"
echo ""

echo -e "${YELLOW}⚠️  WARNING: This script will restore data files from DVC remote storage${NC}"
echo -e "${YELLOW}   It requires DVC to be configured and authenticated${NC}"
echo ""
read -p "Continue? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

# Check if DVC is available
if ! command -v dvc &> /dev/null && ! python -m dvc --version &> /dev/null; then
    echo -e "${RED}❌ Error: DVC is not installed${NC}"
    echo "   Install with: pip install dvc[gcs]"
    exit 1
fi

# Navigate to DVC project directory
DVC_PROJECT_DIR="dags/dvc_project"
if [ ! -d "$DVC_PROJECT_DIR" ]; then
    echo -e "${RED}❌ Error: DVC project directory not found: $DVC_PROJECT_DIR${NC}"
    exit 1
fi

cd "$DVC_PROJECT_DIR"

echo -e "${YELLOW}📥 Pulling latest data from DVC remote...${NC}"
echo ""

# Pull data from DVC remote
if python -m dvc pull -f 2>&1; then
    echo -e "${GREEN}✓ DVC pull completed${NC}"
else
    echo -e "${RED}❌ DVC pull failed${NC}"
    echo "   Make sure DVC is configured and you have access to the remote storage"
    exit 1
fi

echo ""
echo -e "${YELLOW}📤 Syncing restored data files to Composer bucket...${NC}"
echo ""

# Sync the restored data files to Composer bucket
if [ -d "data" ]; then
    echo "Syncing data/ directory to Composer bucket..."
    gsutil -m rsync -r data/ "gs://${COMPOSER_BUCKET}/dags/dvc_project/data/"
    echo -e "${GREEN}✓ Data files synced to Composer bucket${NC}"
else
    echo -e "${RED}❌ Error: data/ directory not found after DVC pull${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Data restoration complete!${NC}"
echo ""
echo -e "${YELLOW}📋 Next steps:${NC}"
echo "1. Verify files in Composer bucket:"
echo "   gsutil ls -lh gs://${COMPOSER_BUCKET}/dags/dvc_project/data/responses/"
echo "2. The next DAG run will use the restored data files"
echo "3. Make sure to update upload_to_composer.sh to exclude data/ directory"

