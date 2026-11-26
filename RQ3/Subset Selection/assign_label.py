"""Prepare label-verification inputs for the LLM-as-a-judge experiment.

The script produces two CSV files with one record per sampled GitHub issue:

* CORRECT: the judge receives the issue's ground-truth label, as in the
  original evaluation pipeline.
* MISS: the judge receives the historical ``removed_label`` associated with
  that sampled issue, rather than a randomly generated incorrect label.

For MISS, the script verifies that the sampled issue URLs and minimal text
subdataset contain the same 376 issues, that each issue has one unambiguous
corresponding ground-truth label, and that ``removed_label`` is different from
``issue_label``. Only bug, enhancement, and question labels are eligible for
MISS. The 384-issue CORRECT branch retains its original label assignment.

Expected directory layout (relative to this script)::

    RQ3/
        assing_label_verified.py
        csv/
            dataset.csv
            sample_miss_combined_perplexity_bodywords.csv
        in_csv/
            subdataset_correct_minimal.csv
            subdataset_miss_minimal.csv
        out_step_2/                # Created automatically

Run with ``python assing_label_verified.py``. Requires pandas.
No machine-specific paths or random-label assignments are used.
"""

from pathlib import Path

import pandas as pd


# Resolve paths relative to this script, not the current working directory.
BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "csv"
IN_CSV_DIR = BASE_DIR / "in_csv"

CORRECT_TEXT_FILE = IN_CSV_DIR / "subdataset_correct_minimal.csv"
MISS_TEXT_FILE = IN_CSV_DIR / "subdataset_miss_minimal.csv"
GROUND_TRUTH_FILE = CSV_DIR / "dataset.csv"
MISS_SAMPLE_FILE = CSV_DIR / "sample_miss_combined_perplexity_bodywords.csv"

OUT_DIR = BASE_DIR / "out_step_2"
OUT_CORRECT_JUDGE = OUT_DIR / "judge_input_correct.csv"
OUT_MISS_JUDGE = OUT_DIR / "judge_input_miss.csv"

# Column configuration.
URL_COL = "issue_URL"
TITLE_COL = "issue_title"
BODY_COL = "issue_body"
GROUND_URL_COL = "issue_url"
GROUND_LABEL_COL = "issue_label"
REMOVED_LABEL_COL = "removed_label"

# Only the MISS labels are restricted to these three classes.
# CORRECT labels are handled exactly as in the original script.
ALLOWED_MISS_LABELS = {"bug", "question", "enhancement"}
EXPECTED_CORRECT_ROWS = 384
EXPECTED_MISS_ROWS = 376


def setup_environment() -> None:
    """Create the output directory, if necessary."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def validate_columns(df: pd.DataFrame, required: list, file_path: Path) -> None:
    """Check required columns and report which ones are missing."""
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}")


def normalize_urls(series: pd.Series) -> pd.Series:
    """Strip URL whitespace/trailing slashes for joins, without changing output URLs."""
    return series.fillna("").astype(str).str.strip().str.rstrip("/")


def normalize_labels(series: pd.Series) -> pd.Series:
    """Normalize label values for case-insensitive matching and validation."""
    return series.fillna("").astype(str).str.strip().str.lower()


def require_unique_urls(df: pd.DataFrame, key_col: str, dataset_name: str) -> None:
    """Reject missing URL keys or duplicate issue keys in a selected dataset."""
    if df[key_col].eq("").any():
        raise ValueError(f"Empty issue URLs found in {dataset_name}.")
    duplicates = df.loc[df[key_col].duplicated(keep=False), key_col].unique()
    if len(duplicates):
        raise ValueError(
            f"Duplicate issue URLs in {dataset_name}. Examples: {duplicates[:5].tolist()}"
        )


def build_miss_dataset(
    df_miss_text: pd.DataFrame,
    df_miss_sample: pd.DataFrame,
    df_ground_all: pd.DataFrame,
) -> pd.DataFrame:
    """
    Assign to each selected MISS issue the historical label that was removed.

    The stratified MISS sample is the source of removed_label; this guarantees
    that the label belongs to the same record selected by the sampling step.
    Reject missing or ambiguous matches and labels that equal the ground truth.

    Returns the six-column judge input, in the original MISS subset order.
    """
    miss_text = df_miss_text.copy()
    miss_sample = df_miss_sample[[URL_COL, REMOVED_LABEL_COL]].copy()
    ground = df_ground_all[[GROUND_URL_COL, GROUND_LABEL_COL]].copy()

    miss_text["__url_key"] = normalize_urls(miss_text[URL_COL])
    miss_sample["__url_key"] = normalize_urls(miss_sample[URL_COL])
    ground["__url_key"] = normalize_urls(ground[GROUND_URL_COL])

    require_unique_urls(miss_text, "__url_key", MISS_TEXT_FILE)
    require_unique_urls(miss_sample, "__url_key", MISS_SAMPLE_FILE)

    if len(miss_text) != EXPECTED_MISS_ROWS or len(miss_sample) != EXPECTED_MISS_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_MISS_ROWS} MISS issues: "
            f"found {len(miss_text)} in the minimal subset and "
            f"{len(miss_sample)} in the stratified sample."
        )

    missing_from_sample = set(miss_text["__url_key"]) - set(miss_sample["__url_key"])
    missing_from_text = set(miss_sample["__url_key"]) - set(miss_text["__url_key"])
    if missing_from_sample or missing_from_text:
        raise ValueError(
            "The MISS minimal subset and stratified sample contain different issues. "
            f"Only in minimal subset: {sorted(missing_from_sample)[:5]}; "
            f"only in stratified sample: {sorted(missing_from_text)[:5]}. "
            "Run extract_title_body.py again after sampling."
        )

    miss_sample[REMOVED_LABEL_COL] = normalize_labels(miss_sample[REMOVED_LABEL_COL])

    # Preserve all ground-truth rows until ambiguous labels are detected.
    # We must not silently choose the first label for an ambiguous MISS issue.
    ground = ground.loc[ground["__url_key"].ne("")].copy()
    ground[GROUND_LABEL_COL] = normalize_labels(ground[GROUND_LABEL_COL])
    ground_lookup = ground.groupby("__url_key", as_index=False).agg(
        correct_label=(GROUND_LABEL_COL, "first"),
        ground_label_count=(GROUND_LABEL_COL, "nunique"),
    )

    # One-to-one matching keeps exactly one row per selected MISS issue.
    judge_miss = miss_text.merge(
        miss_sample[["__url_key", REMOVED_LABEL_COL]],
        how="left",
        on="__url_key",
        validate="one_to_one",
    )
    judge_miss = judge_miss.merge(
        ground_lookup,
        how="left",
        on="__url_key",
        validate="one_to_one",
    )

    invalid = (
        judge_miss["ground_label_count"].isna()
        | judge_miss["ground_label_count"].ne(1)
        | ~judge_miss["correct_label"].isin(ALLOWED_MISS_LABELS)
        | ~judge_miss[REMOVED_LABEL_COL].isin(ALLOWED_MISS_LABELS)
        | judge_miss[REMOVED_LABEL_COL].eq(judge_miss["correct_label"])
    )
    if invalid.any():
        issues = judge_miss.loc[
            invalid, [URL_COL, "correct_label", REMOVED_LABEL_COL, "ground_label_count"]
        ]
        raise ValueError(
            f"Invalid MISS label associations: {int(invalid.sum())} issue(s). "
            "Missing, ambiguous, out-of-scope, or identical labels found.\n"
            + issues.head(10).to_string(index=False)
        )

    judge_miss["wrong_label"] = judge_miss[REMOVED_LABEL_COL]
    judge_miss["label_given_to_model"] = judge_miss["wrong_label"]

    return judge_miss[
        [URL_COL, TITLE_COL, BODY_COL, "label_given_to_model", "correct_label", "wrong_label"]
    ].copy()


def process_datasets() -> None:
    """Build and validate CORRECT/MISS judge inputs, then write both CSV outputs."""
    # Load the same CORRECT, MISS, and ground-truth inputs as before, plus
    # the stratified MISS sample containing the actual removed labels.
    df_correct_text = pd.read_csv(CORRECT_TEXT_FILE)
    df_miss_text = pd.read_csv(MISS_TEXT_FILE)
    df_ground = pd.read_csv(GROUND_TRUTH_FILE)
    df_miss_sample = pd.read_csv(MISS_SAMPLE_FILE)

    # Validate required columns.
    text_columns = [URL_COL, TITLE_COL, BODY_COL]
    for df, required, file_name in [
        (df_correct_text, text_columns, CORRECT_TEXT_FILE),
        (df_miss_text, text_columns, MISS_TEXT_FILE),
        (df_ground, [GROUND_URL_COL, GROUND_LABEL_COL], GROUND_TRUTH_FILE),
        (df_miss_sample, [URL_COL, REMOVED_LABEL_COL], MISS_SAMPLE_FILE),
    ]:
        validate_columns(df, required, file_name)

    # Retain the original text-cleaning behavior, including empty issue bodies.
    for df in [df_correct_text, df_miss_text]:
        df[URL_COL] = df[URL_COL].astype(str).str.strip()
        df[TITLE_COL] = df[TITLE_COL].fillna("").astype(str)
        df[BODY_COL] = df[BODY_COL].fillna("").astype(str)

    df_ground[GROUND_URL_COL] = df_ground[GROUND_URL_COL].astype(str).str.strip()
    df_ground[GROUND_LABEL_COL] = normalize_labels(df_ground[GROUND_LABEL_COL])

    # --- CORRECT dataset: unchanged from the original script. ---
    df_ground_correct = df_ground[[GROUND_URL_COL, GROUND_LABEL_COL]].drop_duplicates(
        subset=[GROUND_URL_COL], keep="first"
    )
    judge_correct = df_correct_text.merge(
        df_ground_correct, how="left", left_on=URL_COL, right_on=GROUND_URL_COL
    )
    judge_correct["label_given_to_model"] = judge_correct[GROUND_LABEL_COL]
    judge_correct = judge_correct[
        [URL_COL, TITLE_COL, BODY_COL, "label_given_to_model", GROUND_LABEL_COL]
    ].copy()
    judge_correct.columns = [
        "issue_URL", "issue_title", "issue_body", "label_given_to_model", "correct_label"
    ]

    # --- MISS dataset: actual removed label, never a randomly generated one. ---
    judge_miss = build_miss_dataset(df_miss_text, df_miss_sample, df_ground)

    if len(judge_correct) != EXPECTED_CORRECT_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_CORRECT_ROWS} CORRECT issues, found {len(judge_correct)}."
        )
    if len(judge_miss) != EXPECTED_MISS_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_MISS_ROWS} MISS issues, found {len(judge_miss)}."
        )

    # Save files only after all validation checks have passed.
    setup_environment()
    judge_correct.to_csv(OUT_CORRECT_JUDGE, index=False)
    judge_miss.to_csv(OUT_MISS_JUDGE, index=False)

    print("=== CORRECT JUDGE INPUT ===")
    print(f"Saved: {OUT_CORRECT_JUDGE}")
    print(f"Total rows: {len(judge_correct)}")
    print(f"Ground-truth labels found: {judge_correct['correct_label'].notna().sum()}")
    print(f"Missing issue bodies: {judge_correct[BODY_COL].eq('').sum()}\n")

    print("=== MISS JUDGE INPUT ===")
    print(f"Saved: {OUT_MISS_JUDGE}")
    print(f"Total rows: {len(judge_miss)}")
    print(f"Validated removed labels matched: {len(judge_miss)}")
    print(f"Different from ground truth: {(judge_miss['wrong_label'] != judge_miss['correct_label']).sum()}")
    print(f"Missing issue bodies: {judge_miss[BODY_COL].eq('').sum()}")
    print("Removed-label distribution:")
    print(judge_miss["wrong_label"].value_counts().to_string())


if __name__ == "__main__":
    process_datasets()
