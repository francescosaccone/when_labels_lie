import pandas as pd
import os
import random

"""
This script prepares datasets for evaluating an AI "judge" model. 
It generates two sets of inputs:
1. A 'correct' set where the provided label matches the ground truth.
2. A 'miss' set where a deliberately incorrect label is provided to test the model's critical ability.
"""

# File Configuration
CORRECT_TEXT_FILE = "in_csv/subdataset_correct_minimal.csv"
MISS_TEXT_FILE = "in_csv/subdataset_miss_minimal.csv"
GROUND_TRUTH_FILE = "csv/dataset.csv"

OUT_DIR = "out_step_2"
OUT_CORRECT_JUDGE = f"{OUT_DIR}/judge_input_correct.csv"
OUT_MISS_JUDGE = f"{OUT_DIR}/judge_input_miss.csv"

# Column names
URL_COL = "issue_URL"
TITLE_COL = "issue_title"
BODY_COL = "issue_body"
GROUND_URL_COL = "issue_url"
GROUND_LABEL_COL = "issue_label"

ALLOWED_LABELS = ["bug", "question", "enhancement"]
RANDOM_STATE = 42

def setup_environment():
    """Creates output directories and initializes random seed."""
    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(RANDOM_STATE)

def random_wrong_label(correct_label: str) -> str:
    """
    Returns a random label from ALLOWED_LABELS that is different from the correct one.

    Args:
        correct_label: The actual ground truth label.

    Returns:
        A randomly selected incorrect label, or an empty string if no alternatives exist.
    """
    correct_label = str(correct_label).strip().lower()
    candidates = [label for label in ALLOWED_LABELS if label != correct_label]
    if not candidates:
        return ""
    return random.choice(candidates)

def process_datasets():
    """
    Main processing function to load, clean, and generate judge input files.
    """
    setup_environment()

    # Load datasets
    df_correct_text = pd.read_csv(CORRECT_TEXT_FILE)
    df_miss_text = pd.read_csv(MISS_TEXT_FILE)
    df_ground = pd.read_csv(GROUND_TRUTH_FILE)

    # Validation
    required_correct_text = [URL_COL, TITLE_COL, BODY_COL]
    required_ground = [GROUND_URL_COL, GROUND_LABEL_COL]

    for df, req, name in [
        (df_correct_text, required_correct_text, CORRECT_TEXT_FILE),
        (df_miss_text, required_correct_text, MISS_TEXT_FILE),
        (df_ground, required_ground, GROUND_TRUTH_FILE)
    ]:
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name}: {missing}")

    # Data Cleaning
    for df in [df_correct_text, df_miss_text]:
        df[URL_COL] = df[URL_COL].astype(str).str.strip()
        df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str)
        df[BODY_COL] = df[BODY_COL].fillna("").astype(str)

    df_ground[GROUND_URL_COL] = df_ground[GROUND_URL_COL].astype(str).str.strip()
    df_ground[GROUND_LABEL_COL] = (
        df_ground[GROUND_LABEL_COL].fillna("").astype(str).str.strip().str.lower()
    )

    # Remove duplicates from ground truth
    df_ground = df_ground[[GROUND_URL_COL, GROUND_LABEL_COL]].drop_duplicates(
        subset=[GROUND_URL_COL], keep="first"
    )

    # --- Generate Correct Label Dataset ---
    judge_correct = df_correct_text.merge(
        df_ground, how="left", left_on=URL_COL, right_on=GROUND_URL_COL
    )
    judge_correct["label_given_to_model"] = judge_correct[GROUND_LABEL_COL]
    
    judge_correct = judge_correct[
        [URL_COL, TITLE_COL, BODY_COL, "label_given_to_model", GROUND_LABEL_COL]
    ].copy()
    
    judge_correct.columns = [
        "issue_URL", "issue_title", "issue_body", "label_given_to_model", "correct_label"
    ]

    # --- Generate Wrong Label Dataset (Miss) ---
    judge_miss = df_miss_text.merge(
        df_ground, how="left", left_on=URL_COL, right_on=GROUND_URL_COL
    )
    judge_miss["wrong_label"] = judge_miss[GROUND_LABEL_COL].apply(random_wrong_label)
    judge_miss["label_given_to_model"] = judge_miss["wrong_label"]

    judge_miss = judge_miss[
        [URL_COL, TITLE_COL, BODY_COL, "label_given_to_model", GROUND_LABEL_COL, "wrong_label"]
    ].copy()

    judge_miss.columns = [
        "issue_URL", "issue_title", "issue_body", "label_given_to_model", "correct_label", "wrong_label"
    ]

    # Save outputs
    judge_correct.to_csv(OUT_CORRECT_JUDGE, index=False)
    judge_miss.to_csv(OUT_MISS_JUDGE, index=False)

    # Summary Statistics
    print(f"Saved: {OUT_CORRECT_JUDGE}")
    print(f"Total Rows: {len(judge_correct)}")
    print(f"Valid labels found: {judge_correct['correct_label'].astype(str).str.strip().ne('').sum()}\n")

    print(f"Saved: {OUT_MISS_JUDGE}")
    print(f"Total Rows: {len(judge_miss)}")
    print(f"Valid labels found: {judge_miss['correct_label'].astype(str).str.strip().ne('').sum()}")
    print(f"Wrong labels generated: {judge_miss['wrong_label'].astype(str).str.strip().ne('').sum()}\n")
    
    print("Wrong label distribution:")
    print(judge_miss["wrong_label"].value_counts(dropna=False))

if __name__ == "__main__":
    process_datasets()