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

- Metadata: prompt_id, category, text_length, size_label
- Content: prompt (input), response (model output)
- Runtime: model, provider, latency_ms, status
- Run info: ts_iso, meta_json (run + model settings)

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

## Judge LLM

Added an automated safety scoring via judge.py, which evaluates every (prompt, response) pair executed in judge_responses.py. All judgments—including safety label, refusal score, and violation categories—are written to judgements.csv for analysis.

Note: The judge.py script uses "llama-3.3-70b-versatile" with the help of Groq API key. Below are the steps to get a Groq API key fo free.

## Steps to get a free Groq API key

1. Go to https://console.groq.com/
2. Sign in using Google, GitHub, or email
3. Go to API Keys in the left sidebar
4. Click “Create API Key”
5. Copy the key and store it in your .env file or environment variables

