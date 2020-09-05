from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import (
    gaussian_kde,
    mannwhitneyu,
    pearsonr,
    spearmanr,
    ttest_ind,
)
from sklearn.metrics import (
    auc,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from .utils import as_si


def two_dists(
    binary,
    continuous,
    method,
    summary_type,
    ax=None,
    pal=["#eaeaea", "#a5dee5"],
    annotate=True,
    stripplot=False,
    seaborn_kwargs={},
    stripplot_kwargs={},
):
    """
    Compare the distributions of a continuous variable when grouped
    by a binary one.

    Parameters
    ----------
    binary : Series
        binary values to plot
    continuous : Series
        continuous values to plot
    method : string, "pearson" or "spearman"
        regression method
    summary_type : string
        type of summary plot to use
    ax : matplotlib axis
        axis to plot in (will create new one if not provided)
    pal : list of length two
        colors to use when plotting
    annotate : boolean
        whether or not to show summary statistics
    stripplot : boolean
        whether or not to plot the raw values
    seaborn_kwargs : dictionary
        additional arguments to pass to Seaborn boxplot/violinplot
    stripplot_kwargs : dictionary
        additional arguments to pass to Seaborn stripplot (if stripplot=True)

    Returns
    -------
    ax : matplotlib axis
        axis with plot data
    """

    if ax is None:
        plt.figure(figsize=(3, 4))
        ax = plt.subplot(111)

    # convert, align, and cast
    binary = pd.Series(binary).dropna()
    continuous = pd.Series(continuous).dropna()
    binary = binary.astype(bool)
    continuous, binary = continuous.align(binary, join="inner")

    # extract positive and negative sets
    continuous_pos = continuous[binary]
    continuous_neg = continuous[~binary]

    # add summary statistics
    if annotate:

        # compute summary statistics
        if method == "t_test":
            stat, pval = ttest_ind(continuous_pos, continuous_neg)
            diff = continuous_pos.mean() - continuous_neg.mean()
        elif method == "mw_u":
            stat, pval = mannwhitneyu(
                continuous_pos, continuous_neg, alternative="two-sided"
            )
            diff = continuous_pos.median() - continuous_neg.median()
        else:
            raise ValueError("Method must be 't_test' or 'mw_u'")

        # number of samples
        n = len(continuous)

        # add text of statistics
        diff_text = "Diff = {:.2f}".format(diff)
        pval_text = "P = " + as_si(pval, 2)
        n_text = "n = " + str(n)

        bbox_props = dict(
            boxstyle="round,pad=1",
            fc="lightgrey",
            ec="lightgrey",
            lw=0,
            alpha=0.33,
        )

        ax.text(
            0.1,
            0.9,
            diff_text + "\n" + pval_text + "\n" + n_text,
            ha="left",
            va="top",
            bbox=bbox_props,
            transform=ax.transAxes,
        )

    # cast palette
    pal = sns.color_palette(pal)

    # plot distributions
    if summary_type == "box":

        sns.boxplot(
            binary,
            continuous,
            notch=True,
            palette=pal,
            ax=ax,
            **seaborn_kwargs,
        )

    elif summary_type == "violin":

        sns.violinplot(
            binary,
            continuous,
            inner=None,
            palette=pal,
            ax=ax,
            **seaborn_kwargs,
        )

    else:

        raise ValueError("Method must be 'box' or 'violin'")

    # plot points themselves
    if stripplot:

        sns.stripplot(
            binary,
            continuous,
            linewidth=1,
            palette=pal,
            alpha=0.5,
            size=2,
            ax=ax,
            **stripplot_kwargs,
        )

    # adjust range to fit text
    y_range = plt.ylim()[1] - plt.ylim()[0]
    plt.ylim(plt.ylim()[0], plt.ylim()[1] + y_range * 0.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def multi_dists(
    continuous,
    categorical,
    count_cutoff,
    summary_type,
    ax=None,
    stripplot=False,
    order="ascending",
    newline_counts=False,
    xtick_rotation=45,
    xtick_ha="right",
    seaborn_kwargs={},
    stripplot_kwargs={},
):
    """
    Compare the distributions of a continuous variable when grouped
    by a categorical one.

    Parameters
    ----------
    continuous : Series
        continuous values to plot
    categorical : Series
        categorical values (groups) to plot
    count_cutoff : boolean
        minimum number of samples per groups to include
    summary_type : string, "box" or "violin"
        type of summary plot to make
    ax : MatPlotLib axis
        axis to plot in (will create new one if not provided)
    stripplot : boolean
        whether or not to plot the raw values
    order : "ascending", "descending", or list of categories
        how to sort categories in the plot
    newline_counts : boolean
        whether to add category counts as a separate line
        in the axis labels
    xtick_rotation : float
        how much to rotate the xtick labels by (in degree)
    xtick_ha : string
        horizontal alignment of the xtick labels
    seaborn_kwargs : dictionary
        additional arguments to pass to Seaborn boxplot/violinplot
    stripplot_kwargs : dictionary
        additional arguments to pass to Seaborn stripplot (if stripplot=True)

    Returns
    -------
    ax : MatPlotLib axis
        axis with plot data
    """

    if ax is None:
        ax = plt.subplot(111)

    # remove NaNs and convert continuous
    continuous = pd.Series(continuous).dropna()
    categorical = pd.Series(categorical).dropna().astype(str)

    # series names
    continuous_name = str(continuous.name)
    categorical_name = str(categorical.name)

    # handle cases where series names are missing or identical
    if continuous_name is None:
        continuous_name = "continuous"

    if categorical_name is None:
        categorical_name = "categorical"

    if continuous_name == categorical_name:

        continuous_name += "_continuous"
        categorical_name += "_categorical"

    merged = pd.concat([continuous, categorical], axis=1, join="inner")
    merged.columns = [continuous_name, categorical_name]

    # counts per category, with cutoff
    categorical_counts = Counter(merged[categorical_name])
    merged["count"] = merged[categorical_name].apply(categorical_counts.get)

    merged = merged[merged["count"] >= count_cutoff]

    merged_sorted = (
        merged.groupby([categorical_name])[continuous_name]
        .aggregate(np.median)
        .reset_index()
    )

    # sort categories by mean
    if order == "ascending":

        merged_sorted = merged_sorted.sort_values(
            continuous_name, ascending=True
        )

        order = merged_sorted[continuous_name]

    elif order == "descending":

        merged_sorted = merged_sorted.sort_values(
            continuous_name, ascending=False
        )

        order = merged_sorted[continuous_name]

    else:

        merged_sorted["continuous_idx"] = merged_sorted[
            categorical_name
        ].apply(order.index)

        merged_sorted = merged_sorted.sort_values(
            "continuous_idx", ascending=True
        )

    # recompute category counts after applying cutoff
    counts = merged_sorted[categorical_name].apply(categorical_counts.get)
    counts = counts.astype(str)

    # x-axis labels with counts
    if newline_counts:

        x_labels = merged_sorted[categorical_name] + "\n(" + counts + ")"

    else:

        x_labels = merged_sorted[categorical_name] + " (" + counts + ")"

    if summary_type == "violin":

        sns.violinplot(
            x=categorical_name,
            y=continuous_name,
            data=merged,
            order=merged_sorted[categorical_name],
            inner=None,
            ax=ax,
            **seaborn_kwargs,
        )

    elif summary_type == "box":

        sns.boxplot(
            x=categorical_name,
            y=continuous_name,
            data=merged,
            order=merged_sorted[categorical_name],
            notch=True,
            ax=ax,
            **seaborn_kwargs,
        )

    if stripplot:

        sns.stripplot(
            x=categorical_name,
            y=continuous_name,
            data=merged,
            order=merged_sorted[categorical_name],
            size=2,
            alpha=0.5,
            linewidth=1,
            jitter=0.1,
            edgecolor="black",
            ax=ax,
            **stripplot_kwargs,
        )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax.set_xticklabels(x_labels, rotation=xtick_rotation, ha=xtick_ha)

    return ax


def roc_auc_curve(y, y_pred, ax=None):
    """
    Plot the ROC curve along with the AUC statistic of
    predictions against ground truths.

    Parameters
    ----------
    y : list-like
        ground truth values
    y_pred : list-like
        predicted values
    ax : MatPlotLib axis
        axis to plot in (will create new one if not provided)

    Returns
    -------
    ax : MatPlotLib axis
        axis with plot data
    """

    # create axis if not provided
    if ax is None:
        plt.figure(figsize=(3, 3))
        ax = plt.subplot(111)

    # cast and align
    y = pd.Series(y).dropna()
    y_pred = pd.Series(y_pred).dropna()
    y, y_pred = y.align(y_pred, join="inner")

    # compute false and true positive rates
    fpr, tpr, _ = roc_curve(y, y_pred)
    auroc = auc(fpr, tpr)

    ax.plot(fpr, tpr)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")

    # summary box properties
    bbox_props = dict(
        boxstyle="round,pad=0.5",
        fc="lightgrey",
        ec="lightgrey",
        lw=0,
        alpha=0.33,
    )

    # add summary text
    ax.text(
        0.9,
        0.1,
        f"AUC = {auroc:.3f}",
        ha="right",
        va="bottom",
        bbox=bbox_props,
        transform=ax.transAxes,
    )

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)

    return ax


def pr_curve(y, y_pred, ax=None):
    """
    Plot the precision-recall curve of predictions against ground truths.

    Parameters
    ----------
    y : list-like
        ground truth values
    y_pred : list-like
        predicted values
    ax : MatPlotLib axis
        axis to plot in (will create new one if not provided)

    Returns
    -------
    ax : MatPlotLib axis
        axis with plot data
    """

    # create axis if not provided
    if ax is None:
        plt.figure(figsize=(3, 3))
        ax = plt.subplot(111)

    # cast and align
    y = pd.Series(y).dropna()
    y_pred = pd.Series(y_pred).dropna()
    y, y_pred = y.align(y_pred, join="inner")

    # compute statistics
    precision, recall, thres = precision_recall_curve(y, y_pred)

    ax.plot(recall, precision)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

    ax.set_xlim(-0.025, 1.025)
    ax.set_ylim(-0.025, 1.025)

    ax.set_ylabel("Precision")
    ax.set_xlabel("Recall")

    bbox_props = dict(
        boxstyle="round,pad=0.5",
        fc="lightgrey",
        ec="lightgrey",
        lw=0,
        alpha=0.33,
    )

    ax.text(
        0.9,
        0.1,
        f"Avg. precision = {np.mean(precision):.3f}",
        ha="right",
        va="bottom",
        bbox=bbox_props,
        transform=ax.transAxes,
    )

    return ax


def binary_metrics(y, y_pred):
    """
    Make several plots to evaluate a binary classifier:

        1. Boxplots of predicted values
        2. Violinplots of predicted values
        3. ROC-AUC plot
        4. Precision-recall curve

    Parameters
    ----------
    y : list-like
        ground truth values
    y_pred : list-like
        predicted values

    Returns
    -------
    ax : MatPlotLib axis
        axis with plot data
    """

    plt.figure(figsize=(12, 3))

    # define axis dimensions
    axes_widths = [2, 2, 3, 3]
    total_width = sum(axes_widths)
    cumulative_widths = [sum(axes_widths[:x]) for x in range(len(axes_widths))]
    axes_height = 2

    axes = [
        plt.subplot2grid(
            (axes_height, total_width),
            (0, cumulative_widths[x]),
            colspan=axes_widths[x],
            rowspan=2,
        )
        for x in range(len(axes_widths))
    ]

    # cast and align
    y = pd.Series(y).dropna()
    y_pred = pd.Series(y_pred).dropna()
    y, y_pred = y.align(y_pred, join="inner")

    # boxplot
    ax = axes[0]
    two_dists(
        y, y_pred, ax=ax, method="mw_u", summary_type="box", annotate=False
    )
    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")

    # violinplot
    ax = axes[1]
    two_dists(
        y, y_pred, ax=ax, method="mw_u", summary_type="violin", annotate=False
    )
    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")

    # roc-auc curve
    ax = axes[2]
    roc_auc_curve(y, y_pred, ax)

    # precision-recall curve
    ax = axes[3]
    pr_curve(y, y_pred, ax)

    plt.subplots_adjust(wspace=2.5)
