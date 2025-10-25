from datasets import load_dataset
import pandas as pd


def load_judge_dataset():
    """
    Load the JailbreakBench 'judge_comparison' dataset from Hugging Face.
    Returns:
        pd.DataFrame: The dataset as a pandas DataFrame.
    """
    print("Loading the dataset...")
    dataset = load_dataset("JailbreakBench/JBB-Behaviors", name="judge_comparison", split="test")
    df = pd.DataFrame(dataset)
    return df


def drop_unnecessary_columns(df):
    """
    Drop unnecessary columns from the DataFrame.
    Args:
        df (pd.DataFrame): The original dataset.
    Returns:
        pd.DataFrame: Dataset after dropping irrelevant columns.
    """
    print("Dropping unnecessary columns...")
    cols_to_drop = ["human1", "human2", "human3"]
    df = df.drop(columns=cols_to_drop, errors="ignore")
    return df


def add_label_column(df):
    """
    Compute and add a label column based on the number of 1s in vote columns.
    Label is 'jailbreak' if >3 ones, else 'benign'.
    Args:
        df (pd.DataFrame): The dataset after dropping unnecessary columns.
    Returns:
        pd.DataFrame: Dataset with the new 'label' column.
    """
    print("Adding Labels...")
    vote_cols = ["human_majority", "harmbench_cf", "gpt4_cf", "llamaguard2_cf", "llama3_cf"]
    df["ones_count"] = df[vote_cols].sum(axis=1)
    df["label"] = df["ones_count"].apply(lambda x: "jailbreak" if x > 3 else "benign")
    df = df.drop(columns=["ones_count"], errors="ignore")
    return df


def preprocess_judge_dataset():
    """
    Complete preprocessing pipeline:
    1. Load dataset
    2. Drop unnecessary columns
    3. Add computed label column
    Returns:
        pd.DataFrame: Fully preprocessed dataset.
    """
    print("Preprocessing judge now...")
    df = load_judge_dataset()
    df = drop_unnecessary_columns(df)
    df = add_label_column(df)
    return df


if __name__ == "__main__":
    processed_df = preprocess_judge_dataset()
    print("Saving the processed dataset...")
    processed_df.to_csv("data/processed/judge_cleaned.csv", index=False)
