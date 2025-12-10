#!/bin/bash
# Script to pull dvc.lock from Composer and commit to git
# This syncs the DVC lock file that was updated by DAG runs in Composer

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=========================================="
echo "Sync dvc.lock from Composer to Git"
echo "==========================================${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# Configuration
PROJECT_ID="${PROJECT_ID:-break-the-bot-480422}"
COMPOSER_ENV="${COMPOSER_ENV:-mlops-airflow-composer}"
LOCATION="${LOCATION:-us-central1}"

# Get Composer bucket name
echo -e "${BLUE}Getting Composer bucket name...${NC}"
COMPOSER_BUCKET=$(gcloud composer environments describe ${COMPOSER_ENV} \
  --location ${LOCATION} \
  --project ${PROJECT_ID} \
  --format "value(config.dagGcsPrefix)" | sed 's|gs://||' | sed 's|/.*||')

if [ -z "$COMPOSER_BUCKET" ]; then
    echo -e "${RED}❌ Error: Could not get Composer bucket name${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Composer bucket: ${COMPOSER_BUCKET}${NC}"
echo ""

# Paths
DVC_LOCK_LOCAL="${PROJECT_ROOT}/dags/dvc_project/dvc.lock"
DVC_LOCK_GCS="gs://${COMPOSER_BUCKET}/dags/dvc_project/dvc.lock"

# Check if dvc.lock exists in GCS
echo -e "${BLUE}Checking if dvc.lock exists in Composer...${NC}"
if ! gsutil -q stat "${DVC_LOCK_GCS}" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Warning: dvc.lock not found in Composer bucket${NC}"
    echo "   Path: ${DVC_LOCK_GCS}"
    echo "   The DAG may not have run yet, or dvc.lock hasn't been generated."
    exit 1
fi

echo -e "${GREEN}✓ dvc.lock found in Composer${NC}"
echo ""

# Create local directory if it doesn't exist
mkdir -p "$(dirname "${DVC_LOCK_LOCAL}")"

# Backup existing dvc.lock if it exists
if [ -f "${DVC_LOCK_LOCAL}" ]; then
    BACKUP_FILE="${DVC_LOCK_LOCAL}.backup.$(date +%Y%m%d_%H%M%S)"
    cp "${DVC_LOCK_LOCAL}" "${BACKUP_FILE}"
    echo -e "${YELLOW}📋 Backed up existing dvc.lock to: $(basename ${BACKUP_FILE})${NC}"
fi

# Download dvc.lock from GCS
echo -e "${BLUE}Downloading dvc.lock from Composer...${NC}"
if gsutil cp "${DVC_LOCK_GCS}" "${DVC_LOCK_LOCAL}"; then
    echo -e "${GREEN}✓ Downloaded dvc.lock${NC}"
else
    echo -e "${RED}❌ Error: Failed to download dvc.lock${NC}"
    exit 1
fi

echo ""

# Check if file changed
echo -e "${BLUE}Checking if dvc.lock changed...${NC}"
cd "${PROJECT_ROOT}"

# Check git status
if ! git diff --quiet "${DVC_LOCK_LOCAL}" 2>/dev/null; then
    echo -e "${GREEN}✓ dvc.lock has changes${NC}"
    HAS_CHANGES=true
elif ! git diff --cached --quiet "${DVC_LOCK_LOCAL}" 2>/dev/null; then
    echo -e "${GREEN}✓ dvc.lock is staged${NC}"
    HAS_CHANGES=true
elif [ -n "$(git status --porcelain "${DVC_LOCK_LOCAL}" 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ dvc.lock is untracked or modified${NC}"
    HAS_CHANGES=true
else
    echo -e "${YELLOW}⚠️  No changes detected in dvc.lock${NC}"
    HAS_CHANGES=false
fi

echo ""

# Show diff if there are changes
if [ "$HAS_CHANGES" = true ]; then
    echo -e "${BLUE}Changes in dvc.lock:${NC}"
    git diff "${DVC_LOCK_LOCAL}" || echo "  (New file or untracked)"
    echo ""
    
    # Ask for confirmation
    read -p "Do you want to commit and push these changes? (y/N): " -n 1 -r
    echo ""
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "${YELLOW}⚠️  Cancelled. Changes are saved locally but not committed.${NC}"
        exit 0
    fi
    
    # Stage the file
    echo -e "${BLUE}Staging dvc.lock...${NC}"
    git add "${DVC_LOCK_LOCAL}"
    echo -e "${GREEN}✓ Staged dvc.lock${NC}"
    echo ""
    
    # Commit
    COMMIT_MESSAGE="Update dvc.lock from Composer DAG run
    
    Synced from: ${DVC_LOCK_GCS}
    Composer environment: ${COMPOSER_ENV}
    Timestamp: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
    
    echo -e "${BLUE}Committing changes...${NC}"
    if git commit -m "${COMMIT_MESSAGE}"; then
        echo -e "${GREEN}✓ Committed dvc.lock${NC}"
    else
        echo -e "${RED}❌ Error: Failed to commit${NC}"
        exit 1
    fi
    echo ""
    
    # Push
    echo -e "${BLUE}Pushing to git...${NC}"
    read -p "Push to remote? (y/N): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if git push; then
            echo -e "${GREEN}✓ Pushed to remote${NC}"
        else
            echo -e "${RED}❌ Error: Failed to push${NC}"
            echo -e "${YELLOW}   You can push manually later with: git push${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}⚠️  Skipped push. Commit is local only.${NC}"
        echo -e "${YELLOW}   Push manually with: git push${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}=========================================="
    echo "✅ Successfully synced dvc.lock from Composer"
    echo "==========================================${NC}"
else
    echo -e "${GREEN}✓ dvc.lock is already up to date${NC}"
    echo ""
    echo -e "${BLUE}No changes to commit.${NC}"
fi

echo ""

