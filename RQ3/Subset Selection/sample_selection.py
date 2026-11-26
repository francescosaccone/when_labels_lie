"""Create reproducible stratified samples of GitHub issue reports.

This script samples two populations independently:

* CORRECT: correctly classified issue reports, sampled without label filtering.
* MISS: misclassified reports eligible only when their historically removed
  label is one of {bug, enhancement, question}, the ground-truth label is also
  in this set, and the two labels differ.

MISS issue URLs are matched against the ground-truth dataset before sampling.
Ambiguous ground-truth labels and multiple distinct eligible removed labels
for the same issue are excluded; duplicate eligible rows are collapsed.

Each population is divided into up to 16 strata using quartiles of
``perplexity`` and ``body_word_length``, then sampled proportionally with a
fixed random seed. The two samples contain 384 CORRECT and 376 MISS issues.

Input CSV files must be in the ``csv`` directory beside this script.
Output CSV files, the audit report, and the sampling summary are written back
into the same directory. File paths are resolved relative to this script,
independently of the working directory used to launch it.

Run from the repository root or from inside the RQ3 directory:
    python RQ3/sample_selection_no_feature.py
    python sample_selection_no_feature.py

Requirements: pandas, numpy.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# Portable paths: resolve input/output locations relative to this script.
CSV_DIR = Path(__file__).resolve().parent / "csv"
CORRECT_FILE = CSV_DIR / "correctly_class.csv"
MISS_FILE = CSV_DIR / "miss_class_filtered.csv"
GROUND_FILE = CSV_DIR / "dataset.csv"

N_SAMPLE_CORRECT = 384
N_SAMPLE_MISS = 376

METRIC_COL_1 = "perplexity"
METRIC_COL_2 = "body_word_length"

OUT_CORRECT = CSV_DIR / "sample_correct_combined_perplexity_bodywords.csv"
OUT_MISS = CSV_DIR / "sample_miss_combined_perplexity_bodywords.csv"
OUT_SUMMARY = CSV_DIR / "sampling_summary_combined_perplexity_bodywords.txt"

RANDOM_STATE = 42

ALLOWED_REMOVED_LABELS = {"bug", "enhancement", "question"}
# The MISS audit records issues excluded during ground-truth matching.
OUT_MISS_AUDIT = CSV_DIR / "miss_ground_truth_matching_audit.csv"
MISS_URL_COL = "issue_URL"
GROUND_URL_COL = "issue_url"
GROUND_LABEL_COL = "issue_label"


def clean_metrics(df: pd.DataFrame, metric_cols: List[str]) -> pd.DataFrame:
    """
    Convert selected metric columns to numeric values and remove invalid rows.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    metric_cols : List[str]
        Names of the metric columns that must be converted and validated.

    Returns
    -------
    pd.DataFrame
        A cleaned copy of the dataset containing only rows with valid metric values.

    Raises
    ------
    ValueError
        If one or more metric columns are missing from the dataset.
    """
    df = df.copy()

    for col in metric_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' was not found in the dataset.")

        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)

    df = df.dropna(subset=metric_cols)
    return df


def filter_miss_by_removed_label(
    df: pd.DataFrame,
    label_col: str = "removed_label",
    allowed_labels: Optional[Set[str]] = None
) -> pd.DataFrame:
    """
    Filter the misclassified dataset by allowed values in the removed label column.

    Parameters
    ----------
    df : pd.DataFrame
        Input misclassified dataset.
    label_col : str
        Name of the column containing the removed label.
    allowed_labels : Optional[Set[str]]
        Set of allowed label values. If None, the default labels are used.

    Returns
    -------
    pd.DataFrame
        A filtered copy of the dataset containing only rows with allowed labels.

    Raises
    ------
    ValueError
        If the label column is missing from the dataset.
    """
    df = df.copy()

    if allowed_labels is None:
        allowed_labels = {"bug", "enhancement", "question"}

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' was not found in the misclassified dataset.")

    normalized = df[label_col].astype(str).str.strip().str.lower()
    mask = normalized.isin(allowed_labels)

    df = df[mask].copy()
    df[label_col] = normalized[mask]

    return df


def filter_miss_by_ground_truth(
    df_miss: pd.DataFrame,
    df_ground: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Match MISS issues to unambiguous ground truth and select eligible rows.

    This filter is applied only to the MISS population, before metric cleaning
    and stratification. An issue is eligible if its removed label differs from
    a single valid ground-truth label. Duplicate eligible rows are collapsed;
    issues with multiple distinct eligible removed labels are excluded.

    Parameters
    ----------
    df_miss : pd.DataFrame
        MISS rows after filtering the allowed removed labels.
    df_ground : pd.DataFrame
        Reference dataset containing issue URLs and ground-truth labels.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Eligible MISS rows with the original input columns, and an audit table
        identifying the matching status of every checked MISS row.
    """
    for col in [MISS_URL_COL, "removed_label"]:
        if col not in df_miss.columns:
            raise ValueError(f"Missing column in MISS: {col}")
    for col in [GROUND_URL_COL, GROUND_LABEL_COL]:
        if col not in df_ground.columns:
            raise ValueError(f"Missing column in {GROUND_FILE}: {col}")

    original_cols = df_miss.columns.tolist()
    miss = df_miss.copy()
    ground = df_ground[[GROUND_URL_COL, GROUND_LABEL_COL]].copy()

    # Normalize URL keys and ground-truth labels without altering output fields.
    miss["__url_key"] = (
        miss[MISS_URL_COL].fillna("").astype(str).str.strip().str.rstrip("/")
    )
    ground["__url_key"] = (
        ground[GROUND_URL_COL].fillna("").astype(str).str.strip().str.rstrip("/")
    )
    ground[GROUND_LABEL_COL] = (
        ground[GROUND_LABEL_COL].fillna("").astype(str).str.strip().str.lower()
    )
    ground = ground[ground["__url_key"].ne("")]

    # Identify conflicting ground-truth labels before matching MISS rows.
    ground_one_per_url = ground.groupby("__url_key", as_index=False).agg(
        ground_labels=(GROUND_LABEL_COL, lambda s: "|".join(sorted(set(s)))),
        ground_label_count=(GROUND_LABEL_COL, lambda s: len(set(s))),
    )
    audit = miss.merge(
        ground_one_per_url,
        how="left",
        on="__url_key",
        validate="many_to_one",
    )

    # Missing, invalid, or ambiguous ground truth is never eligible.
    audit["match_status"] = np.select(
        [
            audit["__url_key"].eq(""),
            audit["ground_label_count"].isna(),
            audit["ground_label_count"].gt(1),
            ~audit["ground_labels"].isin(ALLOWED_REMOVED_LABELS),
            audit["removed_label"].eq(audit["ground_labels"]),
        ],
        [
            "EMPTY_URL",
            "GROUND_TRUTH_NOT_FOUND",
            "AMBIGUOUS_GROUND_TRUTH",
            "INVALID_GROUND_TRUTH_LABEL",
            "LABELS_EQUAL",
        ],
        default="LABELS_DIFFERENT",
    )

    # Exclude issues with conflicting eligible removed labels.
    eligible_mask = audit["match_status"].eq("LABELS_DIFFERENT")
    candidates = audit.loc[eligible_mask]
    ambiguous_removed_urls = candidates.groupby("__url_key")["removed_label"].nunique()
    ambiguous_removed_urls = ambiguous_removed_urls[ambiguous_removed_urls > 1].index
    audit.loc[
        eligible_mask & audit["__url_key"].isin(ambiguous_removed_urls),
        "match_status",
    ] = "MULTIPLE_ELIGIBLE_REMOVED_LABELS"

    eligible = audit.loc[audit["match_status"].eq("LABELS_DIFFERENT")]
    eligible = eligible.drop_duplicates(subset=["__url_key"], keep="first")

    # Include the ground-truth label in the audit, not the sampled data.
    audit = audit.rename(columns={"ground_labels": "issue_label"})
    audit = audit.drop(columns=["__url_key", "ground_label_count"])
    return eligible[original_cols].copy(), audit


def get_quartiles(df: pd.DataFrame, metric_col: str) -> Dict[str, float]:
    """
    Compute the first, second, and third quartiles for a metric column.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    metric_col : str
        Name of the metric column.

    Returns
    -------
    Dict[str, float]
        Dictionary containing Q1, Q2, and Q3 threshold values.
    """
    return {
        "Q1": df[metric_col].quantile(0.25),
        "Q2": df[metric_col].quantile(0.50),
        "Q3": df[metric_col].quantile(0.75),
    }


def label_quartile(x: float, q1: float, q2: float, q3: float) -> str:
    """
    Assign a value to one of four quartile labels.

    Parameters
    ----------
    x : float
        Value to classify.
    q1 : float
        First quartile threshold.
    q2 : float
        Second quartile threshold.
    q3 : float
        Third quartile threshold.

    Returns
    -------
    str
        Quartile label: Q1, Q2, Q3, or Q4.
    """
    if x <= q1:
        return "Q1"
    if x <= q2:
        return "Q2"
    if x <= q3:
        return "Q3"
    return "Q4"


def assign_combined_quartile_strata(
    df: pd.DataFrame,
    metric_col_1: str,
    metric_col_2: str
) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Assign records to combined strata based on the quartiles of two metric columns.

    Each metric is divided into four quartiles. The final stratum is created by
    combining the quartile label of the first metric with the quartile label of
    the second metric.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset.
    metric_col_1 : str
        Name of the first metric column.
    metric_col_2 : str
        Name of the second metric column.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]
        Dataset with added quartile and combined stratum columns, plus quartile
        threshold dictionaries for both metrics.
    """
    df = df.copy()

    info_1 = get_quartiles(df, metric_col_1)
    info_2 = get_quartiles(df, metric_col_2)

    col1_stratum = f"{metric_col_1}_quartile"
    col2_stratum = f"{metric_col_2}_quartile"

    df[col1_stratum] = df[metric_col_1].apply(
        lambda x: label_quartile(x, info_1["Q1"], info_1["Q2"], info_1["Q3"])
    )

    df[col2_stratum] = df[metric_col_2].apply(
        lambda x: label_quartile(x, info_2["Q1"], info_2["Q2"], info_2["Q3"])
    )

    df["combined_stratum"] = (
        f"{metric_col_1}_" + df[col1_stratum].astype(str) +
        f"__{metric_col_2}_" + df[col2_stratum].astype(str)
    )

    return df, info_1, info_2


def proportional_combined_sample(
    df: pd.DataFrame,
    total_n: int,
    stratum_col: str = "combined_stratum",
    random_state: int = 42
) -> pd.DataFrame:
    """
    Draw a proportional stratified sample from combined strata.

    The function allocates the target sample size proportionally across strata,
    corrects rounding differences, and avoids sampling more rows than available
    in each stratum.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with a stratum column.
    total_n : int
        Desired total sample size.
    stratum_col : str
        Name of the column containing stratum labels.
    random_state : int
        Random seed used for reproducible sampling.

    Returns
    -------
    pd.DataFrame
        Stratified sample.

    Raises
    ------
    ValueError
        If the input dataset is empty or no sample can be extracted.
    """
    df = df.copy()

    if len(df) == 0:
        raise ValueError("The dataset is empty after cleaning and filtering.")

    if stratum_col not in df.columns:
        raise ValueError(f"Column '{stratum_col}' was not found in the dataset.")

    if total_n >= len(df):
        return df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    counts = df[stratum_col].value_counts().sort_index()
    total = len(df)

    exact_alloc = (counts / total) * total_n
    alloc = pd.Series(np.floor(exact_alloc).astype(int), index=counts.index)

    non_empty = counts[counts > 0].index.tolist()
    if total_n >= len(non_empty):
        for stratum in non_empty:
            if alloc[stratum] == 0:
                alloc[stratum] = 1

    alloc = pd.concat([alloc, counts], axis=1)
    alloc.columns = ["alloc", "count"]
    alloc["alloc"] = alloc[["alloc", "count"]].min(axis=1)
    alloc = alloc["alloc"].astype(int)

    current_total = int(alloc.sum())

    if current_total < total_n:
        remainder = total_n - current_total
        fractional = (exact_alloc - np.floor(exact_alloc)).sort_values(ascending=False)

        for stratum in fractional.index:
            if remainder == 0:
                break
            if alloc[stratum] < counts[stratum]:
                alloc[stratum] += 1
                remainder -= 1

        if remainder > 0:
            for stratum in counts.sort_values(ascending=False).index:
                if remainder == 0:
                    break
                while alloc[stratum] < counts[stratum] and remainder > 0:
                    alloc[stratum] += 1
                    remainder -= 1

    elif current_total > total_n:
        surplus = current_total - total_n
        for stratum in alloc.sort_values(ascending=False).index:
            if surplus == 0:
                break
            while alloc[stratum] > 0 and surplus > 0:
                alloc[stratum] -= 1
                surplus -= 1

    sampled_parts = []

    for stratum in counts.index:
        n_take = int(alloc[stratum])
        if n_take == 0:
            continue

        subset = df[df[stratum_col] == stratum]

        if len(subset) < n_take:
            n_take = len(subset)

        sampled_part = subset.sample(n=n_take, random_state=random_state)
        sampled_parts.append(sampled_part)

    if not sampled_parts:
        raise ValueError("No sample was extracted.")

    sampled = pd.concat(sampled_parts, axis=0)

    if len(sampled) < total_n:
        missing = total_n - len(sampled)
        residual = df.drop(sampled.index, errors="ignore")

        if len(residual) > 0:
            extra_n = min(missing, len(residual))
            extra = residual.sample(n=extra_n, random_state=random_state)
            sampled = pd.concat([sampled, extra], axis=0)

    sampled = sampled.sample(frac=1, random_state=random_state).reset_index(drop=True)

    return sampled


def summarize_dataset(
    df: pd.DataFrame,
    name: str,
    metric_col_1: str,
    metric_col_2: str,
    info_1: Dict[str, float],
    info_2: Dict[str, float]
) -> str:
    """
    Create a text summary for a processed dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Processed dataset.
    name : str
        Dataset name to display in the summary.
    metric_col_1 : str
        Name of the first metric column.
    metric_col_2 : str
        Name of the second metric column.
    info_1 : Dict[str, float]
        Quartile thresholds for the first metric.
    info_2 : Dict[str, float]
        Quartile thresholds for the second metric.

    Returns
    -------
    str
        Formatted dataset summary.
    """
    counts = df["combined_stratum"].value_counts().sort_index()

    lines = []
    lines.append(f"=== {name} ===")
    lines.append(f"Valid rows: {len(df)}")
    lines.append("")
    lines.append(f"{metric_col_1} quartiles:")
    lines.append(f"  Q1 threshold: {info_1['Q1']}")
    lines.append(f"  Q2 threshold: {info_1['Q2']}")
    lines.append(f"  Q3 threshold: {info_1['Q3']}")
    lines.append("")
    lines.append(f"{metric_col_2} quartiles:")
    lines.append(f"  Q1 threshold: {info_2['Q1']}")
    lines.append(f"  Q2 threshold: {info_2['Q2']}")
    lines.append(f"  Q3 threshold: {info_2['Q3']}")
    lines.append("")
    lines.append("Distribution by combined_stratum:")
    for stratum, count in counts.items():
        lines.append(f"  {stratum}: {count}")
    lines.append("")

    if "removed_label" in df.columns:
        label_counts = df["removed_label"].value_counts().sort_index()
        lines.append("removed_label distribution:")
        for label, count in label_counts.items():
            lines.append(f"  {label}: {count}")
        lines.append("")

    return "\n".join(lines)


def summarize_sample(sample_df: pd.DataFrame, name: str) -> str:
    """
    Create a text summary for a sampled dataset.

    Parameters
    ----------
    sample_df : pd.DataFrame
        Sampled dataset.
    name : str
        Sample name to display in the summary.

    Returns
    -------
    str
        Formatted sample summary.
    """
    counts = sample_df["combined_stratum"].value_counts().sort_index()

    lines = []
    lines.append(f"=== SAMPLE {name} ===")
    lines.append(f"Sample size: {len(sample_df)}")
    lines.append("Sample distribution by combined_stratum:")
    for stratum, count in counts.items():
        lines.append(f"  {stratum}: {count}")
    lines.append("")

    if "removed_label" in sample_df.columns:
        label_counts = sample_df["removed_label"].value_counts().sort_index()
        lines.append("removed_label distribution in the sample:")
        for label, count in label_counts.items():
            lines.append(f"  {label}: {count}")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    """Sample both populations and write their CSV, summary, and audit files.

    MISS-only label filtering is performed before the two independent quartile
    stratifications. CORRECT records are not subject to MISS label filtering.
    """
    df_correct = pd.read_csv(CORRECT_FILE)
    df_miss = pd.read_csv(MISS_FILE)
    df_ground = pd.read_csv(GROUND_FILE)

    df_miss = filter_miss_by_removed_label(
        df_miss,
        label_col="removed_label",
        allowed_labels=ALLOWED_REMOVED_LABELS
    )

    # Validate MISS labels against ground truth before stratification.
    df_miss, miss_audit = filter_miss_by_ground_truth(df_miss, df_ground)
    miss_audit.to_csv(OUT_MISS_AUDIT, index=False)

    df_correct = clean_metrics(df_correct, [METRIC_COL_1, METRIC_COL_2])
    df_miss = clean_metrics(df_miss, [METRIC_COL_1, METRIC_COL_2])

    if len(df_correct) == 0:
        raise ValueError(
            "The correctly_class.csv dataset does not contain valid values "
            "for the selected metrics."
        )

    if len(df_miss) == 0:
        raise ValueError(
            "The miss_class_filtered.csv dataset does not contain valid values "
            "after applying the removed_label filter and metric cleaning."
        )

    # Fail explicitly if the eligible MISS population is too small.
    if len(df_miss) < N_SAMPLE_MISS:
        raise ValueError(
            f"Insufficient eligible unique MISS issues: {len(df_miss)} available, "
            f"but {N_SAMPLE_MISS} required. Review {OUT_MISS_AUDIT}. "
            "Do not fill the sample with unchanged or invented labels."
        )

    df_correct, info_correct_1, info_correct_2 = assign_combined_quartile_strata(
        df_correct, METRIC_COL_1, METRIC_COL_2
    )

    df_miss, info_miss_1, info_miss_2 = assign_combined_quartile_strata(
        df_miss, METRIC_COL_1, METRIC_COL_2
    )

    sample_correct = proportional_combined_sample(
        df_correct,
        total_n=N_SAMPLE_CORRECT,
        stratum_col="combined_stratum",
        random_state=RANDOM_STATE
    )

    sample_miss = proportional_combined_sample(
        df_miss,
        total_n=N_SAMPLE_MISS,
        stratum_col="combined_stratum",
        random_state=RANDOM_STATE
    )

    if len(sample_miss) != N_SAMPLE_MISS or sample_miss[MISS_URL_COL].nunique() != N_SAMPLE_MISS:
        raise ValueError("MISS sampling did not produce the required number of distinct issue URLs.")

    sample_correct.to_csv(OUT_CORRECT, index=False)
    sample_miss.to_csv(OUT_MISS, index=False)

    summary_lines = []
    summary_lines.append("Combined stratified sampling on perplexity and body_word_length")
    summary_lines.append("")
    summary_lines.append("Method:")
    summary_lines.append("1. Filter the misclassified dataset by removed_label using an OR condition.")
    summary_lines.append(f"   Allowed labels: {sorted(ALLOWED_REMOVED_LABELS)}")
    summary_lines.append("1a. Match MISS issue_URL to dataset.csv issue_url.")
    summary_lines.append("1b. Keep only valid removed_label != issue_label; exclude ambiguous and duplicated issues.")
    summary_lines.append("    MISS matching outcomes (before metric cleaning):")
    for status, count in miss_audit["match_status"].value_counts().items():
        summary_lines.append(f"      {status}: {count}")
    summary_lines.append(f"    Eligible unique MISS before metric cleaning: "
                         f"{miss_audit.loc[miss_audit['match_status'].eq('LABELS_DIFFERENT'), MISS_URL_COL].nunique()}")
    summary_lines.append("2. Convert perplexity and body_word_length to numeric values.")
    summary_lines.append("3. Remove NaN, inf, and -inf values from both metrics.")
    summary_lines.append("4. Compute quartiles separately for each dataset and each metric.")
    summary_lines.append("5. Assign each record to a perplexity quartile and a body_word_length quartile.")
    summary_lines.append("6. Build a combined 4x4 stratum from the two quartile assignments.")
    summary_lines.append("7. Apply proportional stratified sampling on the combined strata.")
    summary_lines.append("")

    summary_lines.append(
        summarize_dataset(
            df_correct,
            "CORRECT DATASET",
            METRIC_COL_1,
            METRIC_COL_2,
            info_correct_1,
            info_correct_2
        )
    )
    summary_lines.append(summarize_sample(sample_correct, "CORRECT"))

    summary_lines.append(
        summarize_dataset(
            df_miss,
            "FILTERED MISCLASSIFIED DATASET",
            METRIC_COL_1,
            METRIC_COL_2,
            info_miss_1,
            info_miss_2
        )
    )
    summary_lines.append(summarize_sample(sample_miss, "FILTERED MISCLASSIFIED"))

    with open(OUT_SUMMARY, "w", encoding="utf-8") as file:
        file.write("\n".join(summary_lines))

    print("Sampling completed.")
    print(f"Correctly classified sample output: {OUT_CORRECT}")
    print(f"Misclassified sample output: {OUT_MISS}")
    print(f"Summary output: {OUT_SUMMARY}")
    print(f"MISS label matching audit: {OUT_MISS_AUDIT}")
    print("MISS label matching outcomes:")
    print(miss_audit["match_status"].value_counts().to_string())
    print(f"Valid unique MISS issues after metric cleaning: {len(df_miss)}")


if __name__ == "__main__":
    main()
