"""Compare Spearman correlations between the two RQ2 issue subsets.

This descriptive analysis computes one Spearman correlation matrix for the
correctly classified issues and one for the misclassified issues. It then
reports their pairwise coefficient differences (misclassified minus correctly
classified). 

Example
-------
python spearman_sep.py --input-dir csv --output-dir results --no-show
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Keep the metric ordering consistent with the combined Spearman analysis.
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
    """Load a headered or legacy headerless dataset and annotate its group."""
    columns = ["issue_url", *metadata_columns, *METRICS]
    data = pd.read_csv(csv_path)

    # Legacy IssueProject.py exports begin with an URL rather than a header.
    if str(data.columns[0]).startswith(("https://", "http://")):
        data = pd.read_csv(csv_path, header=None)
        if data.shape[1] != len(columns):
            raise ValueError(f"Unexpected headerless CSV schema: {csv_path}")
        data.columns = columns

    print(f"{group}: {len(data)} rows")
    return data.assign(group=group)


def main() -> None:
    """Compute and export per-group correlation matrices and pairwise changes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, default=SCRIPT_DIR,
        help="Directory containing the two input CSV files. Defaults to the script directory.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=SCRIPT_DIR,
        help="Directory for output CSV and PNG files. Defaults to the script directory.",
    )
    parser.add_argument(
        "--no-show", action="store_true",
        help="Save the plot without opening an interactive window (useful on servers).",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    groups = {
        "correctly": load_dataset(args.input_dir / "correctly_class.csv", "correctly", []),
        "misclassified": load_dataset(
            args.input_dir / "miss_class_filtered.csv", "misclassified", ["label", "window"]
        ),
    }

    shared_metrics = [metric for metric in METRICS if all(metric in data for data in groups.values())]
    if len(shared_metrics) < 2:
        raise ValueError("At least two common metrics are required.")
    missing_metrics = [metric for metric in METRICS if metric not in shared_metrics]
    if missing_metrics:
        print("Metrics missing from at least one group:", ", ".join(missing_metrics))

    upper_i, upper_j = np.triu_indices(len(shared_metrics), k=1)
    names = np.array(shared_metrics)
    comparison = pd.DataFrame({"metrica_1": names[upper_i], "metrica_2": names[upper_j]})

    side = max(6, len(shared_metrics) * 0.37)
    fig, axes = plt.subplots(1, 2, figsize=(2 * side, side + 1), layout="constrained")
    colormap = plt.get_cmap("RdBu_r").copy()
    colormap.set_bad("#cccccc")

    for ax, (group_name, data) in zip(axes, groups.items()):
        # Use the same pairwise-complete Spearman calculation as in the original script.
        values = data[shared_metrics].apply(pd.to_numeric, errors="raise")
        values = values.replace([np.inf, -np.inf], np.nan)
        rho = values.corr(method="spearman", min_periods=2)
        rho.to_csv(args.output_dir / f"spearman_{group_name}.csv", index_label="metrica", na_rep="NA")

        # Number of rows with both metrics present for each pair.
        valid = values.notna().to_numpy(dtype=np.int64)
        counts = valid.T @ valid
        comparison[f"n_{group_name}"] = counts[upper_i, upper_j]
        comparison[f"rho_{group_name}"] = rho.to_numpy()[upper_i, upper_j]

        heatmap = ax.imshow(rho, cmap=colormap, vmin=-1, vmax=1)
        positions = range(len(shared_metrics))
        ax.set_xticks(positions, labels=shared_metrics, rotation=90, fontsize=8)
        ax.set_yticks(positions, labels=shared_metrics, fontsize=8)
        ax.set_title(f"{group_name} ({len(data)} rows)")

    # Signed difference: rho(misclassified) - rho(correctly).
    comparison["delta_rho"] = comparison["rho_misclassified"] - comparison["rho_correctly"]
    comparison.sort_values("delta_rho", key=abs, ascending=False).to_csv(
        args.output_dir / "confronto_coppie.csv", index=False, na_rep="NA"
    )

    fig.colorbar(heatmap, ax=axes, shrink=0.65, label="Spearman ρ")
    fig.suptitle("Correlations by group · Gray = unavailable correlation")
    figure_path = args.output_dir / "spearman_gruppi.png"
    fig.savefig(figure_path, dpi=180)
    print("Saved per-group correlation matrices, pairwise comparison, and heatmap to", args.output_dir)
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
