# DAG Log Monitoring Guide

## Key Logs to Focus On

### 1. **generate_model_responses Task**

#### Pre-Execution Diagnostics
Look for:
```
PRE-EXECUTION DIAGNOSTICS: generate_model_responses
```
- Check `DVC_DATA_DIR` is set correctly
- Verify data directory exists and is writable
- Check expected output paths for each model

#### Model Processing (in STDOUT)
For each model, look for:
```
Running model: {model_name} ({model_id})
============================================================
[INFO] Loaded {N} total prompts from {csv_path}
[INFO] Found {N} already-processed prompts for model {model_id}
[INFO] Filtered prompts: {total} total, {skipped} already processed, {new} new prompts to process
```

**What to check:**
- ✅ If `new prompts to process` > 0: Model will process prompts
- ⚠️ If `new prompts to process` = 0: Model will skip (all prompts already processed)
- ✅ Look for: `[INFO] Processing {N} new prompts...`
- ⚠️ Look for: `[INFO] All prompts already processed for {name}. Skipping API calls.`

#### Post-Execution Verification
Look for:
```
POST-EXECUTION VERIFICATION: generate_model_responses
Model {model_name}: File exists at {path}
  File size: {bytes} bytes
  Last modified: {timestamp}
  Status: FILE UPDATED / FILE CREATED / FILE UNCHANGED
```

**What to check:**
- ✅ `FILE UPDATED` or `FILE CREATED`: Model actually processed new prompts
- ⚠️ `FILE UNCHANGED`: Model skipped (no new prompts)
- Check file sizes increased if status says UPDATED
- Check timestamps are recent (within last few minutes)

#### Summary
Look for:
```
Summary: Created={N}, Updated={N}, Unchanged={N}, Missing={N}
```

**What to check:**
- `Updated` count should match number of models that actually processed prompts
- `Unchanged` count shows models that skipped

---

### 2. **sync_data_to_gcs Task (sync_responses)**

Look for:
```
EXPLICIT GCS SYNC: Forcing sync of data files to bucket
Data directory path: /home/airflow/gcs/dags/dvc_project/data
Target bucket: gs://us-central1-mlops-airflow-c-47afa20c-bucket/dags/dvc_project/data/
Running: gsutil -m rsync -r -d {data_dir} gs://{bucket}/dags/dvc_project/data/
gsutil rsync exit code: 0
✅ Data directory synced to GCS bucket
```

**What to check:**
- ✅ Exit code should be `0` (success)
- ✅ Should see "Data directory synced to GCS bucket"
- ⚠️ If exit code != 0: Check STDERR for errors
- After this task completes, files should be visible in bucket

---

### 3. **Script STDOUT (Most Important!)**

In the task logs, scroll to find the STDOUT section. Look for each model's section:

```
============================================================
Running model: llama-3-8b (meta-llama/Meta-Llama-3-8B-Instruct)
============================================================
[INFO] Loaded 10 total prompts from /home/airflow/gcs/dags/dvc_project/data/processed/processed_data.csv
[INFO] Found 8 already-processed prompts for model meta-llama/Meta-Llama-3-8B-Instruct
[INFO] Filtered prompts: 10 total, 8 already processed, 2 unprocessed available.
[INFO] Sampling mode: taking first 2 unprocessed prompts.
[INFO] Processing 2 new prompts...
  Progress: 2/2 prompts processed...
[INFO]  Appended 2 successful responses to existing file: /home/airflow/gcs/dags/dvc_project/data/responses/model_responses_llama3_8b.csv
 Completed meta-llama/Meta-Llama-3-8B-Instruct (novita) → /home/airflow/gcs/dags/dvc_project/data/responses/model_responses_llama3_8b.csv
 Summary: 2 ok, 0 errors, 2 successful rows written to CSV
```

**What to check for each model:**
1. **Prompt counts:**
   - `Loaded {N} total prompts` - Total in CSV
   - `Found {N} already-processed` - Already in output file
   - `{N} unprocessed available` - New prompts to process

2. **Processing status:**
   - ✅ `Processing {N} new prompts...` = Model is processing
   - ⚠️ `All prompts already processed` = Model is skipping

3. **Results:**
   - ✅ `Appended {N} successful responses` = New rows added
   - ✅ `Created new file with {N} responses` = First time processing
   - ⚠️ No append/create message = Model skipped

4. **Summary per model:**
   - `{N} ok, {N} errors, {N} successful rows written` = Actual results

---

## Quick Checklist

### ✅ Success Indicators:
- [ ] At least one model shows `Processing {N} new prompts...`
- [ ] Models show `Appended {N} successful responses` or `Created new file`
- [ ] `sync_data_to_gcs` shows exit code 0
- [ ] Status shows `FILE UPDATED` or `FILE CREATED` for models that processed
- [ ] File sizes increased in POST-EXECUTION verification
- [ ] Files visible in bucket after sync completes

### ⚠️ Warning Signs:
- [ ] All models show `All prompts already processed` (no new prompts)
- [ ] `sync_data_to_gcs` exit code != 0
- [ ] Status shows `FILE UNCHANGED` but file size increased (sync issue)
- [ ] Models show `0 successful rows written` (all failed)

### ❌ Error Indicators:
- [ ] Script exit code != 0
- [ ] `Failed to load existing responses` warnings
- [ ] `gsutil rsync` errors in sync task
- [ ] File missing errors in verification

---

## How to Access Logs

1. **Airflow UI:**
   - Go to DAG → Click on the run → Click on task → Click "Log"
   - Or use the Graph view and click on task instances

2. **Cloud Logging:**
   ```bash
   gcloud logging read "resource.type=cloud_composer_environment AND resource.labels.environment_name=mlops-airflow-composer AND jsonPayload.task_id=generate_model_responses" --limit=100 --format=json
   ```

3. **Focus on these tasks in order:**
   1. `generate_model_responses` - Check STDOUT for model processing
   2. `sync_data_to_gcs` (sync_responses) - Check sync success
   3. `judge_responses` - Should see new responses being judged
   4. `compute_additional_metrics` - Should process new data
   5. `compute_bias_detection` - Should process new judgements

---

## Expected Output Example

**Good run (multiple models processing):**
```
Model llama-3-8b: Status: FILE UPDATED
Model llama-3-70b: Status: FILE UPDATED  
Model llama-3.1-8b: Status: FILE UPDATED
Model moonshot-instruct: Status: FILE UNCHANGED (all prompts already processed)
Model minimax: Status: FILE UPDATED
Summary: Created=0, Updated=4, Unchanged=1, Missing=0
```

**Problem run (only one model):**
```
Model llama-3-8b: Status: FILE UNCHANGED
Model llama-3-70b: Status: FILE UNCHANGED
Model llama-3.1-8b: Status: FILE UNCHANGED
Model moonshot-instruct: Status: FILE UNCHANGED
Model minimax: Status: FILE UPDATED
Summary: Created=0, Updated=1, Unchanged=4, Missing=0
```
→ Check STDOUT to see why other models skipped (likely all prompts already processed)

---

## Known Issue: Status Detection Bug (Fixed)

**Problem:** Status detection was incorrectly showing "FILE UNCHANGED" even when files were updated because:
1. STDOUT was truncated to last 500 characters (only showing last model's output)
2. Model section detection couldn't find evidence in truncated output
3. File sizes and timestamps showed updates, but status didn't reflect it

**Example of the bug:**
```
Model llama-3-8b: File size: 127199 bytes (was 70253 bytes)  ← Size DOUBLED!
  Last modified: 2025-12-09 18:24:49+00:00  ← Recent timestamp!
  Status: FILE UNCHANGED (no new prompts or skipped)  ← WRONG!
```

**Fix Applied:**
- Now uses FULL STDOUT for status detection (not truncated)
- Uses file modification time as fallback indicator
- Improved model section detection with multiple fallback strategies
- Status will now correctly show "FILE UPDATED" when file was modified during task

**After fix, you should see:**
```
Model llama-3-8b: File size: 127199 bytes
  Last modified: 2025-12-09 18:24:49+00:00
  Modified during task: True
  Status: FILE UPDATED (file modified during task)
```

---

## Interpreting Your Current Logs

**What the logs show:**
1. ✅ **File sizes increased** for 4 models (llama-3-8b, llama-3.1-8b, moonshot-instruct, minimax)
2. ✅ **Timestamps are recent** (all within 18:24-18:26 UTC, during task execution)
3. ❌ **Status incorrectly shows "UNCHANGED"** for all models (this is the bug)
4. ⚠️ **STDOUT truncated** - Only shows minimax's output (last 500 chars)

**What actually happened:**
- All 4 models (except llama-3-70b) processed new prompts and updated their files
- The status detection failed because STDOUT was truncated
- After the fix, status will correctly show "FILE UPDATED" for these models

**To verify in next run:**
- Check that status now shows "FILE UPDATED" for models with increased file sizes
- Check that STDOUT shows full output for all models (not just last 500 chars)
- Verify file modification times match task execution window

