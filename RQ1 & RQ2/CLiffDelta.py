import numpy as np
import pandas as pd


def cliffs_delta(x, y):
    """
    Calculate Cliff's Delta effect size between two independent samples.

    Cliff's Delta measures how often values from one sample are greater than
    values from another sample, minus the reverse probability. Values range
    from -1 to 1.

    Parameters
    ----------
    x : array-like
        First sample of numeric values.
    y : array-like
        Second sample of numeric values.

    Returns
    -------
    float
        Cliff's Delta effect size.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    count_greater = np.sum(np.greater.outer(x, y))
    count_less = np.sum(np.less.outer(x, y))

    delta = (count_greater - count_less) / (len(x) * len(y))

    return delta


csv1 = pd.read_csv("correctly_class.csv")
csv2 = pd.read_csv("miss_class_filtered.csv")

x = csv1["lexical_diversity_less"]
y = csv2["lexical_diversity_less"]

x = x.dropna()
y = y.dropna()

delta = cliffs_delta(x, y)

print("Cliff's Delta:", delta)