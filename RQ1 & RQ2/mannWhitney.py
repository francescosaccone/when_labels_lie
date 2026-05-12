import pandas as pd
from scipy.stats import mannwhitneyu


def run_mann_whitney_test(sample_1, sample_2):
    """
    Run the Mann-Whitney U test on two independent samples.

    Parameters
    ----------
    sample_1 : array-like
        First numeric sample.
    sample_2 : array-like
        Second numeric sample.

    Returns
    -------
    tuple
        Mann-Whitney U statistic and p-value.
    """
    stat, p_value = mannwhitneyu(sample_1, sample_2)

    return stat, p_value


csv1 = pd.read_csv("correctly_class.csv")
csv2 = pd.read_csv("miss_class_filtered.csv")

field_1 = csv1["lexical_diversity_less"]
field_2 = csv2["lexical_diversity_less"]

field_1 = field_1.dropna()
field_2 = field_2.dropna()

stat, p_value = run_mann_whitney_test(field_1, field_2)

print(f"U statistic: {stat}")
print(f"P-value: {p_value}")