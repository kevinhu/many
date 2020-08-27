import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from scipy.stats import gaussian_kde, pearsonr, spearmanr

from .utils import as_si


def corrfunc(x, y, **kwargs):
    nans = np.isnan(x) | np.isnan(y)
    x, y = x[~nans], y[~nans]
    r, pval = pearsonr(x, y)
    n = len(x)
    ax = plt.gca()

    r_text = "r = {:.2f}".format(r)
    pval_text = "P = {:.2e}".format(pval).replace("e", "×10")
    n_text = "n = " + str(n)

    ax.annotate(
        r_text + "\n" + pval_text + "\n" + n_text,
        xy=(0.075, 0.8),
        xycoords=ax.transAxes,
    )


def nan_dist(x, **kwargs):
    nans = np.isnan(x)
    x = x[~nans]
    ax = plt.gca()
    sns.distplot(x, ax=ax, kde=False, norm_hist=True, bins=25, color="#364f6b")


def scatter(x, y, ci=99, **kwargs):
    nans = np.isnan(x) | np.isnan(y)
    x, y = x[~nans], y[~nans]

    ax = sns.regplot(
        x, y, color="#364f6b", scatter_kws={"alpha": 0.5, "s": 8}, ci=ci
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def scatter_grid(df):
    """
    Plot two sets of points, coloring by density and inserting labels given a
    set of significant value masks

    Parameters
    ----------
    df : Pandas DataFrame
        collection of pairs to plot as columns

    Returns
    -------
    ax : matplotlib axis
        axis with plot data
    """

    g = sns.PairGrid(df)
    g.map_upper(scatter)
    g.map_lower(scatter)
    g.map_diag(nan_dist, kde=False)
    g.map_upper(corrfunc)

    return g


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


def dense_plot(
    x,
    y,
    text_adjust: bool,
    labels_mask=None,
    labels=None,
    colormap=None,
    cmap_offset=0,
    ax=None,
    scatter_kwargs={},
    x_offset=0,
    y_offset=0,
):
    """
    Plot two sets of points, coloring by density and inserting labels given a
    set of significant value masks. Density estimated by Gaussian KDE.

    Parameters
    ----------
    x : Series or 1-dimensional array
        x-coordinate values to plot
    y : Series or 1-dimensional array
        y-coordinate values to plot
    labels_mask: Boolean Series or 1-dimensional array
        Boolean mask for values to label
    labels: Series or 1-dimensional array
        Text labels to plot based on labels_mask
    colormap: String or Matplotlib colormap
        Colormap to color point density by. Leave empty for all black.
    adjust_text: Boolean
        Whether or not to adjust the label positions automatically
    x_offset: Float
        offset to use when plotting x text labels
    y_offset: Float
        offset to use when plotting y text labels

    Returns
    -------
    ax : matplotlib axis
        axis with plot data
    """

    # cast and align
    x = pd.Series(x).dropna()
    y = pd.Series(y).dropna()
    x, y = x.align(y, axis=0, join="inner")

    # align labels if specified
    if labels is not None:

        _, labels = x.align(labels, axis=0, join="left")

    if ax is None:

        ax = plt.subplot(111)

    if colormap is not None:

        xy = np.vstack([x, y])

        z = gaussian_kde(xy)(xy)
        z = np.arcsinh(z)

        ax.scatter(
            x,
            y,
            c=z,
            cmap=colormap,
            lw=0,
            rasterized=True,
            vmin=min(z) - cmap_offset,
            **scatter_kwargs,
        )

    else:
        ax.scatter(x, y, c="black", lw=0, rasterized=True, **scatter_kwargs)

    xlims = ax.get_xlim()
    ylims = ax.get_ylim()

    ax.set_xlim(xlims[0] * 1.25, xlims[1] * 1.25)
    ax.set_ylim(ylims[0], ylims[1] * 1.1)

    ax.tick_params(axis="both", which="both", length=5)

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    if labels is not None:

        texts = []

        for x_pos, y_pos, label in zip(
            x[labels_mask], y[labels_mask], labels[labels_mask]
        ):

            ax.scatter(x_pos, y_pos, c="red", s=24)

            if text_adjust:

                texts.append(
                    ax.text(x_pos, y_pos, label, ha="center", va="center")
                )
            else:

                if x_pos <= 0:
                    ax.text(
                        x_pos + x_offset,
                        y_pos + y_offset,
                        label,
                        ha="left",
                        va="center",
                    )

                elif x_pos > 0:
                    ax.text(
                        x_pos + x_offset,
                        y_pos + y_offset,
                        label,
                        ha="right",
                        va="center",
                    )

        if text_adjust:

            adjust_text(
                texts,
                autoalign="",
                arrowprops=dict(arrowstyle="-", color="black"),
            )

    return ax
