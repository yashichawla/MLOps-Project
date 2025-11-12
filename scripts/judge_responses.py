import pandas as pd
from judge import judge_llm, append_judgement_to_csv

def judge_responses(
    input_csv="model_responses_minimax.csv",
    output_csv="judgements.csv",
    limit=None,
):
    """
    Reads responses.csv, judges each row using Llama-3-70B,
    and writes results to judgements.csv.
    """

    df = pd.read_csv(input_csv)

    if "prompt" not in df.columns or "response" not in df.columns:
        raise ValueError("CSV must contain 'prompt' and 'response' columns.")

    print(f"[INFO] Loaded {len(df)} rows from {input_csv}")

    if limit:
        df = df.head(limit)
        print(f"[INFO] Limiting to first {limit} rows")

    for i, row in df.iterrows():
        prompt = row["prompt"]
        response = row["response"]

        print(f"\n---- ({i+1}/{len(df)}) Evaluating response ----")
        print(f"PROMPT: {prompt[:120]}...")
        print(f"RESPONSE: {response[:120]}...")

        result = judge_llm(prompt, response)

        append_judgement_to_csv(
            original_row=row.to_dict(),
            judgment=result,
            out_path=output_csv
        )

    print("\n✓ All judgements complete.")
    print(f"Results saved to {output_csv}")


if __name__ == "__main__":
    judge_responses()
