"""Compute and visualize Spearman correlations on the combined RQ2 datasets.

The correctly classified and misclassified issue datasets are concatenated
before computing one pairwise Spearman correlation matrix. Rows are retained
as they appear in the source files, including repeated issue URLs.

Examples
--------
python spearman.py
python spearman.py --input-dir csv --output-dir results --no-show
python spearman.py --matrix results/matrice_spearman.csv --output-dir results --no-show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Metrics extracted from GitHub issue titles and bodies in the original RQ2 study.
METRICS = """
title_word_length title_character_length title_polarity title_subjectivity
title_ambiguity_index title_informality_index title_flesch_reading_ease
title_flesch_kincaid_grade body_word_length body_character_length body_polarity
body_subjectivity num_paragraphs body_ambiguity_index body_informality_index
body_flesch_reading_ease body_flesch_kincaid_grade vagueness_index bullet_points
alpha_ratio mean_word_length stop_words ellipsis_proportion first_order_coherence
entropy perplexity pos_prop_ADJ pos_prop_ADP pos_prop_ADV pos_prop_AUX pos_prop_NOUN
pos_prop_NUM pos_prop_VERB pos_prop_PRON lexical_diversity_less
""".split()

SCRIPT_DIR = Path(__file__).resolve().parent


def load_dataset(csv_path: Path, group: str, metadata_columns: list[str]) -> pd.DataFrame:
    """Load a dataset, supporting both headered and legacy headerless CSV files.

    Legacy files produced by IssueProject.py have an issue URL as their first
    column but no header row. The schema therefore includes the extra metadata
    columns used by the misclassified dataset (label and window).
    """
    columns = ["issue_url", *metadata_columns, *METRICS]
    data = pd.read_csv(csv_path)

    # A URL in the first column name indicates a legacy headerless file.
    if str(data.columns[0]).startswith(("https://", "http://")):
        data = pd.read_csv(csv_path, header=None)
        if data.shape[1] != len(columns):
            raise ValueError(f"Unexpected headerless CSV schema: {csv_path}")
        data.columns = columns

    print(f"{group}: {len(data)} rows")
    return data.assign(group=group)


def plot_correlation_matrix(rho: pd.DataFrame, output_path: Path, show: bool) -> None:
    """Save a heatmap of the Spearman rho matrix and optionally display it."""
    figure_size = max(7, len(rho) * 0.45)
    fig, ax = plt.subplots(figsize=(figure_size, figure_size), layout="constrained")
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad("#cccccc")

    heatmap = ax.imshow(rho.to_numpy(dtype=float), cmap=colormap, vmin=-1, vmax=1)
    positions = np.arange(len(rho))
    ax.set_xticks(positions, labels=rho.columns, rotation=90, fontsize=8)
    ax.set_yticks(positions, labels=rho.index, fontsize=8)
    ax.set_title("Spearman correlations across the combined subsets")
    fig.colorbar(heatmap, ax=ax, shrink=0.7, label="Spearman coefficient (ρ)")
    if rho.isna().any().any():
        ax.set_xlabel("Gray = correlation cannot be computed")

    fig.savefig(output_path, dpi=200)
    print(f"Heatmap saved to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    """Parse command-line options and run the combined correlation analysis."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=SCRIPT_DIR,
        help="Directory containing correctly_class.csv and miss_class_filtered.csv. "
             "Defaults to the script directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR,
        help="Directory for the correlation CSV and heatmap. "
             "Defaults to the script directory.",
    )
    parser.add_argument(
        "--matrix", type=Path,
        help="Plot a previously computed correlation matrix instead of reading the issue datasets.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the plot without opening an interactive window (useful on servers).",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_path = args.output_dir / "matrice_spearman.png"

    if args.matrix is not None:
        rho = pd.read_csv(args.matrix, index_col=0)
        if list(rho.index) != list(rho.columns):
            raise ValueError("The row and column metrics of the matrix must match.")
        plot_correlation_matrix(rho, heatmap_path, show=not args.no_show)
        return

    correct = load_dataset(args.input_dir / "correctly_class.csv", "correctly_classified", [])
    misclassified = load_dataset(
        args.input_dir / "miss_class_filtered.csv", "misclassified", ["label", "window"]
    )

    shared_metrics = [metric for metric in METRICS if metric in correct and metric in misclassified]
    if len(shared_metrics) < 2:
        raise ValueError("At least two common metrics are required. Check the CSV headers.")
    missing_metrics = [metric for metric in METRICS if metric not in shared_metrics]
    if missing_metrics:
        print("Metrics missing from at least one subset:", ", ".join(missing_metrics))

    combined = pd.concat([correct, misclassified], ignore_index=True)
    if "issue_url" in combined:
        duplicates = combined["issue_url"].dropna().duplicated().sum()
        if duplicates:
            print(f"Warning: {duplicates} repeated issue_url occurrences were retained.")

    # Preserve the original analysis: pairwise Spearman rho with a minimum of
    # two non-missing observations, excluding infinities but not duplicate URLs.
    values = combined[shared_metrics].apply(pd.to_numeric, errors="raise")
    values = values.replace([float("inf"), -float("inf")], float("nan"))
    rho = values.corr(method="spearman", min_periods=2)

    matrix_path = args.output_dir / "matrice_spearman.csv"
    rho.to_csv(matrix_path, index_label="metrica", na_rep="NA")
    print(f"Correlation matrix saved to {matrix_path} ({len(shared_metrics)} × {len(shared_metrics)}).")
    plot_correlation_matrix(rho, heatmap_path, show=not args.no_show)


if __name__ == "__main__":
    main()
