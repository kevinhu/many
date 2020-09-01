import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact


def binary_contingency(a, b, ax=None, heatmap_kwargs={}):
    """
    Plot agreement between two binary variables, along with
    the odds ratio and Fisher's exact test p-value.

    Parameters
    ----------
    a : Boolean series
        boolean series of first variable
    y : Boolean series
        boolean series of second variable
    ax : MatPlotLib axis
        axis to plot in (will create new one if not provided)
    heatmap_kwargs : dictionary
        additional arguments to pass to seaborn.heatmap()

    Returns
    -------
    ax : MatPlotLib axis
        axis with plot data
    """

    # convert, align, and cast
    a, b = pd.Series(a), pd.Series(b)
    a, b = a.dropna(), b.dropna()
    a, b = a.astype(bool), b.astype(bool)

    a, b = a.align(b, join="inner")

    # store names before array conversion
    a_name = a.name
    b_name = b.name

    a, b = np.array(a), np.array(b)

    # compute contingency counts
    xx = np.sum(a & b)
    xy = np.sum(a & ~b)
    yx = np.sum(~a & b)
    yy = np.sum(~a & ~b)

    # create 2x2 contingency table
    contingency = pd.DataFrame(
        [[xx, xy], [yx, yy]],
        columns=["True", "False"],
        index=["True", "False"],
    )

    odds_ratio, p_val = fisher_exact([[xx, xy], [yx, yy]])

    print("Odds ratio:", odds_ratio)
    print("P-value:", p_val)

    # if axis not provided, make figure
    if ax is None:

        plt.figure(figsize=(4, 4))
        ax = plt.subplot(111)

    # plot the contingency table as heatmap
    g = sns.heatmap(
        contingency,
        fmt="d",
        annot=True,
        cbar=False,
        linewidths=2,
        ax=ax,
        **heatmap_kwargs,
    )

    plt.ylabel(a_name)
    plt.xlabel(b_name)

    g.xaxis.tick_top()
    g.xaxis.set_label_position("top")

    return ax
