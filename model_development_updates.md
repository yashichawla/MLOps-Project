# Model Development Update — LLM Response Generation

A new script has been added to generate model responses for prompts in the processed SALAD dataset.

Script: `scripts/generate_model_responses.py`  
Output file: `data/processed/model_responses_<model>.csv`

Note: The script requires `HF_TOKEN` in `.env`. Generate a **read-only** token at https://huggingface.co/settings/tokens and add `HF_TOKEN=hf_xxx`.

Run locally using

```bash
python scripts/generate_model_responses.py
```

Columns captured:

- prompt_id, category, text_length, size_label
- prompt (input), response (model output)
- model, latency_ms, status
- ts_iso and meta_json (run + model settings)

## Design Notes (DAG & DVC)

- Generation should only run **after** preprocessing + validation succeed.
- To avoid unnecessary inference cost, we will add **change detection**:
  - If `processed_data.csv` hasn’t changed → **skip model generation**.
- `model_responses.csv` will be **versioned via DVC**, but only after DAG integration is finalized.
- The DAG’s success email should reflect whether generation was:
  - Executed and produced new responses, or
  - Skipped due to no new data.

## Next Steps

1. Insert the generation step into the DAG after `enforce_validation_policy`. (Can/should refactor this actually)
2. Add change detection to control whether generation runs.
3. Track and push `model_responses.csv` via DVC once stable in the pipeline.
