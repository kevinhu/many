import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, norm, rankdata, tiecorrect
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm_notebook as tqdm

from . import config
from .utils import precheck_align

try:
    import cupy as cp
    from cupyx.scipy.special import ndtr
except ModuleNotFoundError:
    cupy = None
    ndtr = None


def melt_mwu(effects, pvals, pos_ns, neg_ns, effect):
    """
    Flatten matrix-form outputs to column-form.

    Parameters
    ----------
    effects: Pandas DataFrame
        effect sizes matrix
    pvals: Pandas DataFrame
        p-values matrix
    pos_ns, neg_ns: Pandas DataFrames
        sample group counts
    effect: "mean", "median", or "rank_biserial"
        the effect statistic

    Returns
    -------
    series form statistics
    """

    melted = pd.DataFrame()
    melted[effect] = effects.unstack()
    melted["pval"] = pvals.unstack()
    melted["qval"] = multipletests(
        10 ** (-melted["pval"]),
        alpha=config.MULTIPLETESTS_ALPHA,
        method=config.MULTIPLETESTS_METHOD,
    )[1]

    melted["qval"] = -np.log10(melted["qval"])

    melted["pos_n"] = pos_ns.unstack()
    melted["neg_n"] = neg_ns.unstack()

    melted = melted.sort_values(by="pval", ascending=False)

    melted.index.set_names(["b_col", "a_col"], inplace=True)
    melted.index = melted.index.swaplevel(0, 1)

    return melted


def melt_biserial(effects, pos_ns, neg_ns, effect):
    """
    Flatten matrix-form outputs to column-form.

    Parameters
    ----------
    effects: Pandas DataFrame
        effect sizes matrix
    pos_ns, neg_ns: Pandas DataFrames
        sample group counts
    effect: "mean", "median", or "rank_biserial"
        the effect statistic

    Returns
    -------
    series form statistics
    """

    melted = pd.DataFrame()
    melted[effect] = effects.unstack()

    melted["pos_n"] = pos_ns.unstack()
    melted["neg_n"] = neg_ns.unstack()

    melted = melted.sort_values(by=effect, ascending=False)

    melted.index.set_names(["b_col", "a_col"], inplace=True)
    melted.index = melted.index.swaplevel(0, 1)

    return melted


def mat_mwu_naive(
    a_mat,
    b_mat,
    melt: bool,
    effect: str,
    use_continuity=True,
    pbar=False,
):
    """
    Compute rank-biserial correlations and Mann-Whitney statistics
    between every column-column pair of a_mat (continuous) and b_mat (binary)
    using a double for loop.

    In the case that a_mat or b_mat has a single column, the results are
    re-formatted with the multiple hypothesis-adjusted q-value also returned.

    Parameters
    ----------
    a_mat: Pandas DataFrame
        Continuous set of observations, with rows as samples and columns
        as labels.
    b_mat: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    melt: boolean
        Whether or not to melt the outputs into columns.
    use_continuity: bool
        Whether or not to use a continuity correction. True by default.
    pbar: Boolean
        Whether or not to show a progress bar.
    effect: "mean", "median", or "rank_biserial"
        The effect statistic.

    Returns
    -------
    effects: rank-biserial correlations
    pvals: -log10 p-values of correlations
    """

    if effect not in ["mean", "median", "rank_biserial"]:

        raise ValueError("effect must be 'mean', 'median', or 'rank_biserial'")

    a_mat, b_mat = precheck_align(a_mat, b_mat, np.float64, np.float64)

    a_names = a_mat.columns
    b_names = b_mat.columns

    a_num_cols = a_mat.shape[1]  # number of variables in A
    b_num_cols = b_mat.shape[1]  # number of variables in B

    effects = np.zeros((a_num_cols, b_num_cols))  # null value of r = 0
    pvals = np.zeros((a_num_cols, b_num_cols)) + 1  # null value of p=1

    pos_ns = np.zeros((a_num_cols, b_num_cols))
    neg_ns = np.zeros((a_num_cols, b_num_cols))

    if pbar:
        sys.stderr.flush()
        progress = tqdm(total=a_num_cols * b_num_cols)

    for a_col_idx, a_col_name in enumerate(a_names):
        for b_col_idx, b_col_name in enumerate(b_names):

            a_col = a_mat[a_col_name].dropna()
            b_col = b_mat[b_col_name].dropna()

            a_col, b_col = a_col.align(b_col, join="inner")

            b_pos = b_col == 1
            b_neg = b_col == 0

            pos_n = b_pos.sum()
            neg_n = b_neg.sum()

            pos_ns[a_col_idx][b_col_idx] = pos_n
            neg_ns[a_col_idx][b_col_idx] = neg_n

            if pos_n >= 1 and neg_n >= 1:

                a_pos = a_col[b_pos]
                a_neg = a_col[b_neg]

                # handle identical values cases
                if np.std(np.concatenate([a_pos, a_neg])) == 0:

                    pvals[a_col_idx][b_col_idx] = 1

                else:

                    try:

                        U2, pval = mannwhitneyu(
                            a_pos,
                            a_neg,
                            use_continuity=use_continuity,
                            alternative="two-sided",
                        )

                        if effect == "rank_biserial":
                            effects[a_col_idx][b_col_idx] = (
                                2 * U2 / (len(a_pos) * len(a_neg)) - 1
                            )
                        elif effect == "median":
                            pos_med = a_pos.median()
                            neg_med = a_neg.median()
                            effects[a_col_idx][b_col_idx] = pos_med - neg_med
                        elif effect == "mean":
                            pos_mean = a_pos.mean()
                            neg_mean = a_neg.mean()
                            effects[a_col_idx][b_col_idx] = pos_mean - neg_mean

                        pvals[a_col_idx][b_col_idx] = pval

                    # Catch "All numbers are identical" errors
                    except ValueError:
                        effects[a_col_idx][b_col_idx] = np.nan

                        pvals[a_col_idx][b_col_idx] = 1

            if pbar:
                progress.update(1)

    if pbar:
        progress.close()

    # account for small p-values rounding to 0
    pvals[pvals == 0] = np.finfo(np.float64).tiny

    effects = pd.DataFrame(effects, index=a_names, columns=b_names)
    pvals = pd.DataFrame(pvals, index=a_names, columns=b_names)
    pos_ns = pd.DataFrame(pos_ns, index=a_names, columns=b_names)
    neg_ns = pd.DataFrame(neg_ns, index=a_names, columns=b_names)

    effects = effects.fillna(0)
    pvals = pvals.fillna(1)

    pvals = -np.log10(pvals)

    if melt:

        return melt_mwu(effects, pvals, pos_ns, neg_ns, effect)

    return effects, pvals


def mat_mwu(a_mat, b_mat, melt: bool, effect: str, use_continuity=True):
    """
    Compute rank-biserial correlations and Mann-Whitney statistics
    between every column-column pair of a_mat (continuous) and b_mat (binary).

    In the case that a_mat or b_mat has a single column, the results are
    re-formatted with the multiple hypothesis-adjusted q-value also returned.

    Parameters
    ----------
    a_mat: Pandas DataFrame
        Continuous set of observations, with rows as samples and columns
        as labels.
    b_mat: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    melt: boolean
        Whether or not to melt the outputs into columns.
    use_continuity: bool
        Whether or not to use a continuity correction. True by default.
    effect: "mean", "median", or "rank_biserial"
        The effect statistic.

    Returns
    -------
    effects: rank-biserial correlations
    pvals: -log10 p-values of correlations
    """

    if effect not in ["rank_biserial"]:

        raise ValueError("effect must be 'rank_biserial'")

    a_nan = a_mat.isna().sum().sum() == 0
    b_nan = b_mat.isna().sum().sum() == 0

    if not a_nan and not b_nan:

        raise ValueError("a_mat and b_mat cannot have missing values")

    a_mat, b_mat = precheck_align(a_mat, b_mat, np.float64, np.bool)

    a_names = a_mat.columns
    b_names = b_mat.columns

    a_ranks = a_mat.apply(rankdata)
    a_ties = a_ranks.apply(tiecorrect)

    a_mat, b_mat = np.array(a_mat), np.array(b_mat)
    b_mat = b_mat.astype(np.bool)

    a_num_cols = a_mat.shape[1]  # number of variables in A
    b_num_cols = b_mat.shape[1]  # number of variables in B

    a_mat = a_mat.astype(np.float64)
    b_pos = b_mat.astype(np.float64)
    b_neg = (~b_mat).astype(np.float64)

    pos_ns = b_pos.sum(axis=0)
    neg_ns = b_neg.sum(axis=0)

    pos_ns = np.vstack([pos_ns] * a_num_cols)
    neg_ns = np.vstack([neg_ns] * a_num_cols)

    pos_ranks = np.dot(a_ranks.T, b_pos)

    u1 = pos_ns * neg_ns + (pos_ns * (pos_ns + 1)) / 2.0 - pos_ranks
    u2 = pos_ns * neg_ns - u1

    # temporarily mask zeros
    n_prod = pos_ns * neg_ns
    zero_prod = n_prod == 0
    n_prod[zero_prod] = 1

    effects = 2 * u2 / (pos_ns * neg_ns) - 1

    # set zeros to nan
    effects[zero_prod] = 0

    a_ties = np.vstack([np.array(a_ties)] * b_num_cols).T

    #     if T == 0:
    #         raise ValueError('All numbers are identical in mannwhitneyu')

    sd = np.sqrt(a_ties * pos_ns * neg_ns * (pos_ns + neg_ns + 1) / 12.0)

    meanrank = pos_ns * neg_ns / 2.0 + 0.5 * use_continuity
    bigu = np.maximum(u1, u2)

    # temporarily mask zeros
    sd_0 = sd == 0
    sd[sd_0] = 1

    z = (bigu - meanrank) / sd

    z[sd_0] = 0

    # compute p values
    pvals = 2 * np.vectorize(norm.sf)(np.abs(z))

    # account for small p-values rounding to 0
    pvals[pvals == 0] = np.finfo(np.float64).tiny

    pvals = pd.DataFrame(pvals, columns=b_names, index=a_names)
    effects = pd.DataFrame(effects, columns=b_names, index=a_names)
    pos_ns = pd.DataFrame(pos_ns, columns=b_names, index=a_names)
    neg_ns = pd.DataFrame(neg_ns, columns=b_names, index=a_names)

    effects = effects.fillna(0)
    pvals = pvals.fillna(1)

    pvals = -np.log10(pvals)

    if melt:

        return melt_mwu(effects, pvals, pos_ns, neg_ns, effect)

    return effects, pvals


def mat_mwu_gpu(a_mat, b_mat, melt: bool, effect: str, use_continuity=True):
    """
    Compute rank-biserial correlations and Mann-Whitney statistics
    between every column-column pair of a_mat (continuous) and b_mat (binary).

    Parameters
    ----------
    a_mat: Pandas DataFrame
        Continuous set of observations, with rows as samples and columns as labels.
    b_mat: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    melt: boolean
        Whether or not to melt the outputs into columns.
    use_continuity: bool
        Whether or not to use a continuity correction. True by default.
    effect: "mean", "median", or "rank_biserial"
        The effect statistic.

    Returns
    -------
    effects: rank-biserial correlations
    pvals: -log10 p-values of correlations
    """

    if effect not in ["rank_biserial"]:

        raise ValueError("effect must be 'rank_biserial'")

    a_nan = a_mat.isna().sum().sum() == 0
    b_nan = b_mat.isna().sum().sum() == 0

    if not a_nan and not b_nan:

        raise ValueError("a_mat and b_mat cannot have missing values")

    a_mat, b_mat = precheck_align(a_mat, b_mat, np.float64, np.bool)

    a_names = a_mat.columns
    b_names = b_mat.columns

    a_ranks = a_mat.apply(rankdata)
    a_ties = a_ranks.apply(tiecorrect)

    a_ranks = cp.array(a_ranks)

    a_mat, b_mat = cp.array(a_mat), cp.array(b_mat)
    b_mat = b_mat.astype(cp.bool)

    a_num_cols = a_mat.shape[1]  # number of variables in A
    b_num_cols = b_mat.shape[1]  # number of variables in B

    a_mat = cp.array(a_mat).astype(cp.float64)
    b_pos = b_mat.astype(cp.float64)
    b_neg = (~b_mat).astype(cp.float64)

    pos_ns = b_pos.sum(axis=0)
    neg_ns = b_neg.sum(axis=0)

    pos_ns = cp.vstack([pos_ns] * a_num_cols)
    neg_ns = cp.vstack([neg_ns] * a_num_cols)

    pos_ranks = cp.dot(a_ranks.T, b_pos)

    u1 = pos_ns * neg_ns + (pos_ns * (pos_ns + 1)) / 2.0 - pos_ranks
    u2 = pos_ns * neg_ns - u1

    # temporarily mask zeros
    n_prod = pos_ns * neg_ns
    zero_prod = n_prod == 0
    n_prod[zero_prod] = 1

    effects = 2 * u2 / (pos_ns * neg_ns) - 1

    # set zeros to nan
    effects[zero_prod] = 0

    a_ties = cp.vstack([cp.array(a_ties)] * b_num_cols).T

    #     if T == 0:
    #         raise ValueError('All numbers are identical in mannwhitneyu')

    sd = cp.sqrt(a_ties * pos_ns * neg_ns * (pos_ns + neg_ns + 1) / 12.0)

    meanrank = pos_ns * neg_ns / 2.0 + 0.5 * use_continuity
    bigu = cp.maximum(u1, u2)

    # temporarily mask zeros
    sd_0 = sd == 0
    sd[sd_0] = 1

    z = (bigu - meanrank) / sd

    z[sd_0] = 0

    # compute p values
    pvals = 2 * (1 - ndtr(cp.abs(z)))

    # account for small p-values rounding to 0
    pvals[pvals == 0] = cp.finfo(cp.float64).tiny

    pvals = -cp.log10(pvals)

    pvals = pd.DataFrame(pvals, columns=b_names, index=a_names)
    effects = pd.DataFrame(effects, columns=b_names, index=a_names)
    pos_ns = pd.DataFrame(pos_ns, columns=b_names, index=a_names)
    neg_ns = pd.DataFrame(neg_ns, columns=b_names, index=a_names)

    effects = effects.fillna(0)
    pvals = pvals.fillna(1)

    if melt:

        return melt_mwu(effects, pvals, pos_ns, neg_ns, effect)

    return effects, pvals


def biserial_continuous_nan(a_mat, b_mat, melt: bool, effect: str):

    """
    Compute biserial (point or rank) correlations for every column-column pair of
    a_mat (continuous) and b_mat (binary). Allows for missing values in a_mat.

    Parameters
    ----------
    a_mat: Pandas DataFrame
        Continuous set of observations, with rows as samples and columns
        as labels.
    b_mat: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    melt: boolean
        Whether or not to melt the outputs into columns.
    effect: "point_biserial" or "rank_biserial"
        The effect statistic.

    Returns
    -------
    biserial: biserial correlations
    pos_ns: number of positive group observations per variable pair
    neg_ns: number of negative group observations per variable pair
    """

    if effect not in ["point_biserial", "rank_biserial"]:

        raise ValueError("effect must be 'point_biserial' or 'rank_biserial'")

    a_names = a_mat.columns
    b_names = b_mat.columns

    if effect == "rank_biserial":
        a_mat = a_mat.rank(method="average")

    a_mat, b_mat = precheck_align(a_mat, b_mat, np.float64, np.bool)

    a_mat = a_mat.values
    b_mat = b_mat.values

    a_mat = np.ma.masked_invalid(a_mat)
    a_mat_valid = (~a_mat.mask).astype(np.float32)

    # overall standard deviations for A
    a_stdev = np.std(a_mat, axis=0)

    # sample numbers and grouped means
    pos_ns = np.dot(a_mat_valid.T, b_mat)
    neg_ns = np.dot(a_mat_valid.T, 1 - b_mat)
    pos_means = np.ma.dot(a_mat.T, b_mat) / pos_ns
    neg_means = np.ma.dot(a_mat.T, 1 - b_mat) / neg_ns

    if effect == "point_biserial":

        # number of total samples
        num_total = np.sum(a_mat_valid, axis=0)
        # fraction of positive groups
        pos_frac = pos_ns / num_total[:, None]

        biserial = (
            (pos_means - neg_means)
            * np.sqrt(pos_frac * (1 - pos_frac))
            / a_stdev[:, None]
        )

    elif effect == "rank_biserial":

        biserial = 2 * (pos_means - neg_means) / (pos_ns + neg_ns)

    # cast to DataFrames
    biserial = pd.DataFrame(biserial, columns=b_names, index=a_names)
    pos_ns = pd.DataFrame(pos_ns, columns=b_names, index=a_names)
    neg_ns = pd.DataFrame(neg_ns, columns=b_names, index=a_names)

    if melt:

        return melt_biserial(biserial, pos_ns, neg_ns, effect)

    return biserial, pos_ns, neg_ns
