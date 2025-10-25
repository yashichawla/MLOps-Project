# preprocess_salad_data.py

"""
Data Preprocessing Module for SALAD Dataset

This script performs modular preprocessing on the SALAD dataset:
- Loads dataset from Hugging Face
- Converts to DataFrame
- Drops unnecessary columns and duplicates
- Maps raw categories to 13 standardized evaluation categories
- Removes null or invalid entries

Each step is encapsulated in a function for integration into an Airflow pipeline.
"""

from datasets import load_dataset
import pandas as pd


def load_salad_dataset():
    """Load the attack_enhanced_set split from the SALAD dataset."""
    print("Loading SALAD dataset...")
    dataset = load_dataset("OpenSafetyLab/Salad-Data", "attack_enhanced_set", split="train")
    return pd.DataFrame(dataset)


def drop_unnecessary_columns(df):
    """Drop irrelevant columns."""
    print("Dropping unnecessary columns...")
    return df.drop(columns=["1-category", "2-category","3-category","method", "aid", "qid"], errors="ignore")


def remove_duplicates(df):
    """Remove duplicate baseq-augq pairs."""
    print("Removing duplicates...")
    df = df.drop_duplicates(subset=["baseq", "augq"], keep="first").reset_index(drop=True)
    print(f"Dataset after deduplication: {len(df)} rows")
    return df


def map_categories(df):
    """Map raw 3-category labels to 13 standardized categories."""
    print("Mapping categories...")

    category_mapping = {
    
    # Illegal Activity
    'O19: Illegal Drugs and Regulated/Controlled Substances': 'Illegal Activity',
    'O54: Drug-related Crimes': 'Illegal Activity',
    'O42: Scams': 'Illegal Activity',
    'O37: Malware Generation': 'Malware Generation',
    'O35: Weapon Generation and Transportation': 'Illegal Activity',
    'O56: Violent Crimes': 'Physical Harm',
    'O60: Traffic and Driving Offenses': 'Illegal Activity',
    'O59: Environmental Crimes': 'Illegal Activity',
    'O43: Financial Fraud': 'Fraud',
    'O65: Plagiarism': 'Fraud',
    'O45: Fake Review Generation': 'Fraud',
    'O50: Manipulation of Public Opinion': 'Political Lobbying',
    'O51: Campaign Materials': 'Political Lobbying',
    'O31: Government Secrets': 'Government Decision',
    'O33: Inferring Personal Information': 'Privacy Violence',
    'O34: Leaking Private Information': 'Privacy Violence',
    'O48: Spam': 'Economic Harm',
    'O25: Paypal Lending': 'Economic Harm',
    
    # Hate Speech
    'O1: Hate Speech': 'Hate Speech',
    'O7: Insult': 'Hate Speech',
    'O8: Racial and Ethnic Stereotyping': 'Hate Speech',
    'O10: Disability Stereotyping': 'Hate Speech',
    'O9: Age Stereotyping': 'Hate Speech',
    'O11: Gender Stereotyping': 'Hate Speech',
    'O12: Religious Stereotyping': 'Hate Speech',
    
    # Sexual / Child Abuse
    'O6: Child Abuse': 'Child Sexual Abuse',
    'O15: Sexual Activity': 'Pornography',
    'O16: Pornography': 'Pornography',
    'O14: Erotic Chat': 'Pornography',
    'O55: Sexual Offenses': 'Pornography',
    
    # Health / Legal / Financial Advice
    'O17: Law Advice': 'Legal Opinion',
    'O58: Illegal Law Advice': 'Legal Opinion',
    'O20: Medical Advice': 'Health Consultation',
    'O53: Financial Crimes': 'Financial Advice',
    
    # Physical / Psychological Harm
    "O3: Enjoying someone else's Pain": 'Physical Harm',
    'O2: Harass, Threaten, or Bully An Individual': 'Physical Harm',
    'O39: Biological and Chemical Harms': 'Physical Harm',
    'O63: Psychological Manipulation': 'Physical Harm',
    'O38: Cyber Attack': 'Physical Harm',
    
    # Other
    'O21: Historical Revisionism': 'Political Lobbying',
    'O22: Science Denialism': 'Political Lobbying',
    'O23: Radicalization': 'Political Lobbying',
    'O30: Social Media Rumors': 'Political Lobbying',
    'O24: Multi-level Marketing': 'Economic Harm',
    'O26: Gambling': 'Economic Harm',
    'O40: Group Defamation': 'Hate Speech',
    'O41: Defamation Someone': 'Hate Speech',
    'O46: Fake Online Content': 'Political Lobbying',
    'O29: Fake News': 'Political Lobbying',
    'O47: Disinformation': 'Political Lobbying',
    'O32: Financial Data Leaks': 'Financial Advice',
    'O62: Self-Harm': 'Physical Harm',
    'O64: Mimicking Writing Style': 'Fraud',
    'O28: Trade Affairs': 'Economic Harm',
    'O27: Labor Issues': 'Economic Harm',

    }

    df["std_category"] = df["3-category"].map(category_mapping).fillna("Other")
    return df


def clean_null_values(df):
    """Remove rows with null or empty questions."""
    print("Removing null/empty text fields...")
    df = df.dropna(subset=["baseq", "augq"])
    df = df[df["baseq"].str.strip() != ""]
    return df


def preprocess_salad_data():
    """Full preprocessing pipeline — ready for Airflow task."""
    df = load_salad_dataset()
    df = remove_duplicates(df)
    df = map_categories(df)
    df = drop_unnecessary_columns(df)
    df = clean_null_values(df)
    print("Preprocessing completed successfully!")
    print(f"Final dataset shape: {df.shape}")
    return df


if __name__ == "__main__":
    final_df = preprocess_salad_data()
    final_df.to_csv("data/processed/salad_cleaned.csv", index=False)
    print("Processed dataset saved as salad_cleaned.csv")
