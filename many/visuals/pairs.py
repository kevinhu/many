from collections import Counter

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import (
    fisher_exact,
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


def binary_contingency(a, b, ax=None, heatmap_kwargs={}):
    """
    Plot agreement between two binary variables, along with
    the odds ratio and Fisher's exact test p-value.

    Parameters
    ----------
    a : Boolean series
        Boolean series of first variable
    y : Boolean series
        Boolean series of second variable
    ax : MatPlotLib axis
        axis to plot in (will create new one if not provided)

    Returns
    -------
    ax : MatPlotLib axis
        axis with plot data
    """
    a, b = pd.Series(a), pd.Series(b)
    a, b = a.dropna(), b.dropna()
    a, b = a.astype(bool), b.astype(bool)

    a, b = a.align(b, join="inner")

    a_name = a.name
    b_name = b.name

    a, b = np.array(a), np.array(b)

    xx = np.sum(a & b)
    xy = np.sum(a & ~b)
    yx = np.sum(~a & b)
    yy = np.sum(~a & ~b)

    contingency = pd.DataFrame(
        [[xx, xy], [yx, yy]],
        columns=["True", "False"],
        index=["True", "False"],
    )

    odds_ratio, p_val = fisher_exact([[xx, xy], [yx, yy]])

    print("Odds ratio:", odds_ratio)
    print("P-value:", p_val)

    if ax is None:

        plt.figure(figsize=(4, 4))
        ax = plt.subplot(111)

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


def regression(
    x, y, method, ax=None, alpha=0.5, text_pos=(0.1, 0.9), scatter_kwargs={}
):
    """
    Plot two sets of points with regression coefficient

    Parameters
    ----------
    x : Series or 1-dimensional array
        x-coordinate values to plot
    y : Series or 1-dimensional array
        y-coordinate values to plot
    method : string, "pearson" or "spearman"
        regression method
    ax : MatPlotLib axis
        axis to plot in (will create new one if not provided)
    alpha : float
        opacity of plotted points
    text_pos : (float, float)
        (x,y) relative position to place regression statistics
    scatter_kwargs : dictionary
        additional arguments to pass to plt.scatter()

    Returns
    -------
    ax : MatPlotLib axis
        axis with plot data
    """

    # check that method is valid
    if method not in ["pearson", "spearman"]:
        raise ValueError("Method must be 'pearson' or 'spearman'.")

    if method == "pearson":
        r, pval = pearsonr(x, y)
    elif method == "spearman":
        r, pval = spearmanr(x, y)

    if ax is None:
        ax = plt.subplot(111)

    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()

    x, y = x.align(y, join="inner")

    # number of samples
    n = len(x)

    # add text of statistics
    r_text = "r = {:.2f}".format(r)
    pval_text = "P = " + as_si(pval, 2)
    n_text = "n = " + str(n)

    bbox_props = dict(
        boxstyle="round,pad=0.5",
        fc="lightgrey",
        ec="lightgrey",
        lw=0,
        alpha=0.33,
    )

    ax.text(
        text_pos[0],
        text_pos[1],
        r_text + "\n" + pval_text + "\n" + n_text,
        ha="left",
        va="top",
        bbox=bbox_props,
        transform=ax.transAxes,
    )

    # plot points
    ax.scatter(
        x, y, linewidth=0, alpha=alpha, rasterized=True, **scatter_kwargs
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


def dense_regression(
    x,
    y,
    method,
    ax=None,
    palette="Blues",
    cmap_offset=0,
    text_pos=(0.1, 0.9),
    scatter_kwargs={},
):
    """
    Plot two sets of points with regression coefficient with density-based coloring

    Parameters
    ----------
    x : Series or 1-dimensional array
        x-coordinate values to plot
    y : Series or 1-dimensional array
        y-coordinate values to plot
    method : string, "pearson" or "spearman"
        regression method
    ax : matplotlib axis
        axis to plot in (will create new one if not provided)
    palette : MatPlotLib color map
        color map to color points by
    cmap_offset : float
        Value to add to KDE to offset colormap
    text_pos : (float, float)
        (x,y) relative position to place regression statistics
    scatter_kwargs : dictionary
        additional arguments to pass to plt.scatter()

    Returns
    -------
    ax : matplotlib axis
        axis with plot data
    """

    # check that method is valid
    if method not in ["pearson", "spearman"]:
        raise ValueError("Method must be 'pearson' or 'spearman'.")

    if method == "pearson":
        r, pval = pearsonr(x, y)
    elif method == "spearman":
        r, pval = spearmanr(x, y)

    if ax is None:
        ax = plt.subplot(111)

    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()

    x, y = x.align(y, join="inner")

    xy = np.vstack([x, y])

    z = gaussian_kde(xy)(xy)
    z = np.arcsinh(z)

    n = len(x)

    # add text of statistics
    r_text = "r = {:.2f}".format(r)
    pval_text = "P = " + as_si(pval, 2)
    n_text = "n = " + str(n)

    bbox_props = dict(
        boxstyle="round,pad=0.5",
        fc="lightgrey",
        ec="lightgrey",
        lw=0,
        alpha=0.33,
    )

    ax.text(
        text_pos[0],
        text_pos[1],
        r_text + "\n" + pval_text + "\n" + n_text,
        ha="left",
        va="top",
        bbox=bbox_props,
        transform=ax.transAxes,
    )

    # plot points
    ax.scatter(
        x,
        y,
        c=z,
        linewidth=0,
        rasterized=True,
        cmap=palette,
        vmin=min(z) + cmap_offset,
        **scatter_kwargs,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return ax


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
    Plot two sets of points, one as a binary variable

    Parameters
    ----------
    binary : Series
        binary values to plot
    continuous : Series
        continuous values to plot
    method : string, "pearson" or "spearman"
        regression method
    ax : matplotlib axis
        axis to plot in (will create new one if not provided)
    pal : list of length two
        colors to use when plotting
    summary_type : string
        type of summary plot to use
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

    binary = pd.Series(binary).dropna()
    continuous = pd.Series(continuous).dropna()

    binary = binary.astype(bool)

    continuous, binary = continuous.align(binary, join="inner")

    # extract positive and negative sets
    continuous_pos = continuous[binary]
    continuous_neg = continuous[~binary]

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

    if annotate:

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

    # plot points
    pal = sns.color_palette(pal)

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
    Plot two sets of points, one as a categorical variable and the other
    as a continuous one.

    Parameters
    ----------
    continuous : Series
        continuous values to plot
    categorical : Series
        categorical values (groups) to plot
    method : string, "pearson" or "spearman"
        regression method
    ax : MatPlotLib axis
        axis to plot in (will create new one if not provided)
    summary_type : string, "box" or "violin"
        type of summary plot to make
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

    # Series names
    continuous_name = str(continuous.name)
    categorical_name = str(categorical.name)

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

    if ax is None:
        plt.figure(figsize=(3, 3))
        ax = plt.subplot(111)

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

    if ax is None:
        plt.figure(figsize=(3, 3))
        ax = plt.subplot(111)

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

    plt.figure(figsize=(12, 3))

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

    y_s = pd.Series(y.reshape(-1))
    y_pred_s = pd.Series(y_pred.reshape(-1))

    # boxplot
    ax = axes[0]
    two_dists(y_pred_s, y_s, ax=ax, summary_type="box", annotate=False)
    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")

    # violinplot
    ax = axes[1]
    two_dists(y_pred_s, y_s, ax=ax, summary_type="violin", annotate=False)
    ax.set_xlabel("Truth")
    ax.set_ylabel("Prediction")

    # roc-auc curve
    ax = axes[2]
    roc_auc_curve(y, y_pred, ax)

    # precision-recall curve
    ax = axes[3]
    pr_curve(y, y_pred, ax)

    plt.subplots_adjust(wspace=2.5)
