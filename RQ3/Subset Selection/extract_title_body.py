from typing import List
import pandas as pd

"""
Build minimal issue subdatasets from sampled correct and misclassified datasets.

This script reads the original GitHub issue dataset and two sampled datasets:
one containing correctly classified issues and one containing misclassified
issues. The sampled datasets contain issue URLs, while the original dataset
contains the corresponding issue text fields.

The script matches each sampled issue URL with the original dataset and retrieves
the issue title and issue body. It then creates two minimal subdatasets: one for
the correctly classified sample and one for the misclassified sample.

The input paths can be changed through ORIGINAL_DATASET, CORRECT_SAMPLE_FILE,
and MISS_SAMPLE_FILE. The output CSV files are defined by OUT_CORRECT_SUBDATASET
and OUT_MISS_SUBDATASET.

The generated output files contain only the sampled issue URL, issue title, and
issue body. These minimal files can be used as input for later LLM classification
or LLM-as-a-judge steps.
"""



# CONFIGURATION
ORIGINAL_DATASET = "csv/dataset.csv"

CORRECT_SAMPLE_FILE = "csv/sample_correct_combined_perplexity_bodywords.csv"
MISS_SAMPLE_FILE = "csv/sample_miss_combined_perplexity_bodywords.csv"

OUT_CORRECT_SUBDATASET = "in_csv/subdataset_correct_minimal.csv"
OUT_MISS_SUBDATASET = "in_csv/subdataset_miss_minimal.csv"

ORIGINAL_URL_COL = "issue_url"
SAMPLE_URL_COL = "issue_URL"

ORIGINAL_TEXT_COLUMNS = [
    "issue_url",
    "issue_title",
    "issue_body",
]


def validate_required_column(
    df: pd.DataFrame,
    column_name: str,
    dataset_name: str
) -> None:
    """
    Check whether a required column exists in a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.
    column_name : str
        Name of the required column.
    dataset_name : str
        Name or path of the dataset, used in the error message.

    Raises
    ------
    ValueError
        If the required column is missing.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' was not found in {dataset_name}.")


def validate_required_columns(
    df: pd.DataFrame,
    column_names: List[str],
    dataset_name: str
) -> None:
    """
    Check whether all required columns exist in a dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset to validate.
    column_names : List[str]
        Names of the required columns.
    dataset_name : str
        Name or path of the dataset, used in the error message.

    Raises
    ------
    ValueError
        If one or more required columns are missing.
    """
    missing_columns = [col for col in column_names if col not in df.columns]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {dataset_name}: {missing_columns}"
        )


def build_minimal_subdataset(
    sample_df: pd.DataFrame,
    original_text_df: pd.DataFrame,
    sample_url_col: str,
    original_url_col: str
) -> pd.DataFrame:
    """
    Build a minimal subdataset by matching sampled issue URLs with original text fields.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Sample dataset containing issue URLs.
    original_text_df : pd.DataFrame
        Original dataset reduced to URL, title, and body columns.
    sample_url_col : str
        URL column name in the sample dataset.
    original_url_col : str
        URL column name in the original dataset.

    Returns
    -------
    pd.DataFrame
        Minimal subdataset containing the sampled issue URL, issue title, and issue body.
    """
    sample_urls = sample_df[[sample_url_col]].copy()

    subdataset = sample_urls.merge(
        original_text_df,
        how="left",
        left_on=sample_url_col,
        right_on=original_url_col
    )

    subdataset = subdataset.drop(columns=[original_url_col])

    return subdataset


def print_subdataset_report(
    df: pd.DataFrame,
    name: str,
    title_col: str = "issue_title",
    body_col: str = "issue_body"
) -> None:
    """
    Print a matching report for a generated subdataset.

    Parameters
    ----------
    df : pd.DataFrame
        Generated subdataset.
    name : str
        Name displayed in the report.
    title_col : str
        Column containing issue titles.
    body_col : str
        Column containing issue bodies.
    """
    print(f"=== {name} SUBDATASET REPORT ===")
    print("Rows:", len(df))
    print("Matched title:", df[title_col].notna().sum())
    print("Matched body:", df[body_col].notna().sum())
    print()


def main() -> None:
    """
    Create minimal correct and misclassified subdatasets.

    The script reads the original dataset and the two sampled datasets,
    validates the required columns, retrieves issue titles and bodies from
    the original dataset through an issue URL merge, and writes two minimal
    subdatasets to disk.
    """
    df_original = pd.read_csv(ORIGINAL_DATASET)
    df_correct = pd.read_csv(CORRECT_SAMPLE_FILE)
    df_miss = pd.read_csv(MISS_SAMPLE_FILE)

    validate_required_column(
        df_original,
        ORIGINAL_URL_COL,
        ORIGINAL_DATASET
    )

    validate_required_column(
        df_correct,
        SAMPLE_URL_COL,
        CORRECT_SAMPLE_FILE
    )

    validate_required_column(
        df_miss,
        SAMPLE_URL_COL,
        MISS_SAMPLE_FILE
    )

    validate_required_columns(
        df_original,
        ORIGINAL_TEXT_COLUMNS,
        ORIGINAL_DATASET
    )

    df_original_text = df_original[ORIGINAL_TEXT_COLUMNS].copy()
    df_original_text = df_original_text.drop_duplicates(subset=[ORIGINAL_URL_COL])

    df_correct_sub = build_minimal_subdataset(
        sample_df=df_correct,
        original_text_df=df_original_text,
        sample_url_col=SAMPLE_URL_COL,
        original_url_col=ORIGINAL_URL_COL
    )

    df_miss_sub = build_minimal_subdataset(
        sample_df=df_miss,
        original_text_df=df_original_text,
        sample_url_col=SAMPLE_URL_COL,
        original_url_col=ORIGINAL_URL_COL
    )

    print_subdataset_report(df_correct_sub, "CORRECT")
    print_subdataset_report(df_miss_sub, "MISCLASSIFIED")

    df_correct_sub.to_csv(OUT_CORRECT_SUBDATASET, index=False)
    df_miss_sub.to_csv(OUT_MISS_SUBDATASET, index=False)

    print("Completed.")
    print(f"Saved: {OUT_CORRECT_SUBDATASET}")
    print(f"Saved: {OUT_MISS_SUBDATASET}")


if __name__ == "__main__":
    main()