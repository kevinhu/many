import sys

import numpy as np
import pandas as pd
import scipy.special as special
from scipy.stats import (
    pearsonr,
    spearmanr,
)
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm_notebook as tqdm

from . import config
from .utils import precheck_align


def melt_corr(corrs, pvals, sample_counts, method):
    """
    Flatten matrix-form outputs to column-form.

    Parameters
    ----------
    corrs: Pandas DataFrame
        correlations matrix
    pvals: Pandas DataFrame
        p-values matrix
    pos_ns neg_ns: Pandas DataFrames
        sample group counts
    a_num_cols: int
        number of columns in first observations matrix
    b_num_cols: int
        number of columns in second observations matrix
    method: String, "pearson" or "spearman"
        the correlation method

    Returns
    -------
    series form statistics
    """

    melted = pd.DataFrame()
    melted[method] = corrs.unstack()
    melted["pval"] = pvals.unstack()
    melted["qval"] = multipletests(
        10 ** (-melted["pval"]),
        alpha=config.MULTIPLETESTS_ALPHA,
        method=config.MULTIPLETESTS_METHOD,
    )[1]

    melted["qval"] = -np.log10(melted["qval"])
    melted["n"] = sample_counts.unstack()

    melted = melted.sort_values(by="pval", ascending=False)

    melted.index.set_names(["b_col", "a_col"], inplace=True)
    melted.index = melted.index.swaplevel(0, 1)

    return melted


def mat_corr_naive(a_mat, b_mat, melt: bool, method: str, pbar=False):
    """
    Compute correlations between every column-column pair of A and B
    using a double for loop.

    In the case that A or B has a single column, the results are re-formatted
    with the multiple hypothesis-adjusted q-value also returned.

    Parameters
    ----------
    A: Pandas DataFrame
        First set of observations, with rows as samples and columns as labels
    B: Pandas DataFrame
        Second set of observations, with rows as samples and columns as labels
    method: String, "pearson" or "spearman"
        Correlation method to use
    pbar: Boolean
        Whether or not to show a progress bar.

    Returns
    -------
    corrs: Pearson/Spearman correlation coefficients
    pvals: -log10 p-values of correlations
    """

    a_mat, b_mat = precheck_align(a_mat, b_mat, np.float64, np.float64)

    # store names before array conversion
    a_names = a_mat.columns
    b_names = b_mat.columns

    a_num_cols = len(a_names)  # number of variables in A
    b_num_cols = len(b_names)  # number of variables in B

    # initialize arrays for correlations and p-values
    corrs = np.zeros((a_num_cols, b_num_cols))
    pvals = np.zeros((a_num_cols, b_num_cols))
    sample_counts = np.zeros((a_num_cols, b_num_cols))

    if pbar:
        sys.stderr.flush()
        progress = tqdm(total=a_num_cols * b_num_cols)

    for a_col_idx, a_col_name in enumerate(a_names):
        for b_col_idx, b_col_name in enumerate(b_names):
            # select columns to correlate
            a_col = a_mat[a_col_name].dropna()
            b_col = b_mat[b_col_name].dropna()

            a_col, b_col = a_col.align(b_col, join="inner", axis=0)

            num_samples = len(a_col)

            if num_samples > 2:
                if method == "pearson":
                    corr, pval = pearsonr(a_col, b_col)
                elif method == "spearman":
                    corr, pval = spearmanr(a_col, b_col)

            elif num_samples <= 2:
                corr, pval = np.nan, np.nan

            # add in correlation
            corrs[a_col_idx][b_col_idx] = corr
            pvals[a_col_idx][b_col_idx] = pval
            sample_counts[a_col_idx][b_col_idx] = num_samples

            if pbar:
                progress.update(1)

    if pbar:
        progress.close()

    # account for small p-values rounding to 0
    pvals[pvals == 0] = np.finfo(np.float64).tiny

    # convert correlation arrays to named DataFrames
    corrs = pd.DataFrame(corrs, index=a_names, columns=b_names)

    sample_counts = pd.DataFrame(sample_counts, index=a_names, columns=b_names)

    pvals = pd.DataFrame(pvals, index=a_names, columns=b_names)

    corrs = corrs.fillna(0)
    pvals = pvals.fillna(1)

    pvals = -np.log10(pvals)

    if melt:
        return melt_corr(corrs, pvals, sample_counts, method)

    return corrs, pvals


def mat_corr(a_mat, b_mat, melt: bool, method: str):
    """
    Compute correlations between every column-column pair of A and B

    In the case that A or B has a single column, the results are re-formatted
    with the multiple hypothesis-adjusted q-value also returned.

    Parameters
    ----------
    A: Pandas DataFrame
        First set of observations, with rows as samples and columns as labels
    B: Pandas DataFrame
        Second set of observations, with rows as samples and columns as labels
    method: String, "pearson" or "spearman"
        Correlation method to use

    Returns
    -------
    corrs: Pearson/Spearman correlation coefficients
    pvals: -log10 p-values of correlations
    """

    a_mat, b_mat = precheck_align(a_mat, b_mat, np.float64, np.float64)

    a_names = a_mat.columns
    b_names = b_mat.columns
    num_samples = len(a_mat.index)  # number of samples

    a_nan = a_mat.isna().sum().sum() == 0
    b_nan = b_mat.isna().sum().sum() == 0

    if not a_nan and not b_nan:
        raise ValueError("A and B cannot have missing values")

    # Compute column ranks, as Spearman correlation is equivalent
    # to Pearson correlation between ranks
    if method == "spearman":
        a_mat = a_mat.rank(method="average")
        b_mat = b_mat.rank(method="average")

    a_mat, b_mat = np.array(a_mat), np.array(b_mat)

    # Subtract column means
    residuals_a = a_mat - a_mat.mean(axis=0)
    residuals_b = b_mat - b_mat.mean(axis=0)

    # Sum squares across columns
    sums_a = (residuals_a**2).sum(axis=0)
    sums_b = (residuals_b**2).sum(axis=0)

    # Compute correlations
    residual_products = np.dot(residuals_a.T, residuals_b)
    sum_products = np.sqrt(np.dot(sums_a[:, None], sums_b[None]))

    sum_zeros = sum_products == 0
    sum_products[sum_zeros] = 1

    corrs = residual_products / sum_products

    corrs[sum_zeros] = 0

    # Compute significance values
    ab = num_samples / 2 - 1

    def beta(corr):
        return 2 * special.btdtr(ab, ab, 0.5 * (1 - abs(np.float64(corr))))

    beta = np.vectorize(beta)

    pvals = beta(corrs)

    # account for small p-values rounding to 0
    pvals[pvals == 0] = np.finfo(np.float64).tiny

    # Store correlations in DataFrames
    corrs = pd.DataFrame(corrs, index=a_names, columns=b_names)
    pvals = pd.DataFrame(pvals, index=a_names, columns=b_names)
    sample_counts = pd.DataFrame(num_samples, index=a_names, columns=b_names)

    pvals = -np.log10(pvals)

    if melt:
        return melt_corr(corrs, pvals, sample_counts, method)

    return corrs, pvals


def pearson_significance(row):
    corr = row["corr"]

    if corr == 0:
        return 1

    ab = row["n"] / 2 - 1

    beta = 2 * special.btdtr(ab, ab, 0.5 * (1 - abs(corr)))

    # account for small p-values rounding to 0
    beta = max(np.finfo(np.float64).tiny, beta)

    return beta


def mat_corr_nan(a_mat, b_mat, melt: bool, method: str):
    """
    Compute correlations between A and every column of B. A must be
    a Series for this method to work.

    Parameters
    ----------
    A: Pandas Series
        First set of observations, with rows as samples
    B: Pandas DataFrame
        Second set of observations, with rows as samples and columns as labels
    method: String, "pearson" or "spearman"
        Correlation method to use

    Returns
    -------
    corr_df: DataFrame
        DataFrame with rows for each variable of B, and columns indicating
        correlation coefficient, p-value, and q-value
    """

    a_mat, b_mat = precheck_align(a_mat, b_mat, np.float64, np.float64)

    a_name = a_mat.columns[0]

    if len(a_mat.columns) != 1:
        raise ValueError("A must contain only a single variable.")

    b_names = b_mat.columns
    b_nan = b_mat.isna()

    b_num_cols = len(b_names)

    num_samples = len(a_mat.index)  # number of samples

    # compute column ranks, as Spearman correlation is equivalent
    # to Pearson correlation between ranks
    if method == "spearman":
        b_nan = b_mat.isna()
        b_mat = b_mat.rank(na_option="top", method="average")
        b_mat[b_mat <= b_nan.sum()] = np.nan
        b_mat = b_mat - b_nan.sum()

        # construct mirrored A matrix
        b_num_cols = b_mat.shape[1]
        a_nan = np.repeat(np.array(a_mat), b_num_cols, axis=1)
        a_nan[b_nan] = np.nan
        a_nan = pd.DataFrame(a_nan, columns=b_names)

        # rank mirrored A matrix
        a_nan = a_nan.rank(na_option="top", method="average")
        a_nan[a_nan <= b_nan.sum()] = np.nan
        a_nan = a_nan - b_nan.sum()
        a_nan = np.ma.array(np.array(a_nan), mask=b_nan)

    elif method == "pearson":
        a_repeat = np.repeat(np.array(a_mat), b_num_cols, axis=1)
        a_nan = np.ma.array(a_repeat, mask=b_nan)

    # convert to arrays
    a_mat, b_mat = np.array(a_mat), np.array(b_mat)
    nan_sums = np.isnan(b_mat).sum(axis=0)

    # make masked arrays
    a_mat = np.ma.array(a_mat, mask=np.isnan(a_mat))
    b_mat = np.ma.array(b_mat, mask=np.isnan(b_mat))

    # subtract column means
    residuals_b = b_mat - np.ma.mean(b_mat, axis=0)
    residuals_a_nan = a_nan - np.ma.mean(a_nan, axis=0)

    # sum squares across columns
    sums_b = np.ma.sum(residuals_b**2, axis=0)
    sums_a_nan = np.ma.sum(residuals_a_nan**2, axis=0)

    # compute correlations
    residual_products = np.ma.sum(residuals_a_nan * residuals_b, axis=0)
    sum_products = np.sqrt(sums_a_nan * sums_b)

    corrs = np.array(residual_products / sum_products).reshape(-1)

    corrs_index = pd.MultiIndex.from_arrays(
        [b_names, [a_name] * b_num_cols], names=("b_col", "a_col")
    )

    corr_df = pd.DataFrame(index=corrs_index)

    corr_df["corr"] = corrs
    corr_df["n"] = num_samples - nan_sums
    corr_df["pval"] = corr_df.apply(pearson_significance, axis=1)
    corr_df["qval"] = multipletests(
        corr_df["pval"],
        alpha=0.01,
        method="fdr_bh",
    )[1]

    # rename 'corr' column with name of method used
    corr_df = corr_df.rename({"corr": method}, axis=1)

    corr_df["pval"] = -np.log10(corr_df["pval"])
    corr_df["qval"] = -np.log10(corr_df["qval"])

    corr_df = corr_df.sort_values(by="pval", ascending=False)

    return corr_df


def mat_corr_subtyped(
    a_mat,
    b_mat,
    subtypes,
    min_count: int,
    stack: bool,
    mat_method,
    pbar=False,
    **kwargs,
):
    """
    Compute correlations between a_mat and every column of b_mat, within
    each subsample specified by subtypes. a_mat must be a Series for this
    method to work. Allows for missing values in b_mat.

    Parameters
    ----------
    a_mat: Pandas Series
        First set of observations, with rows as samples
    b_mat: Pandas DataFrame
        Second set of observations, with rows as samples and columns as labels
    subtypes: Pandas Series
        Categories to compute correlations within
    min_count: integer
        Minimum number of samples per subtype to keep for consideration
    stack: boolean
        Whether or not to aggregate statistics into single DataFrame
    mat_method: string
        Correlation method to use, either "mat_corr_naive" or "mat_corr_nan"
    pbar: boolean
        Whether or not to show a progress bar with subtypes
    **kwargs: additional parameters
        Arguments to pass to mat_corrs_nan()

    Returns
    -------

    if stack is False:

        subtype_corrs: DataFrame
            DataFrame of correlations between a_mat and each variable
            of b_mat (rows) within each subtype (columns)
        subtype_ns: DataFrame
            DataFrame of sample sizes between a_mat and each variable
            of b_mat (rows) within each subtype (columns)
        subtype_pvals: DataFrame
            DataFrame of p-values between a_mat and each variable of
            b_mat (rows) within each subtype (columns)

    if stack is True:

        stacked: DataFrame
            DataFrame of correlations between a_mat and each variable
            of b_mat within each subtype, along with sample sizes and p-values,
            with each value in a column
    """

    # remove missing values in A
    a_mat = a_mat.dropna(how="any")

    # common rows between all
    common = set(a_mat.index) & set(b_mat.index) & set(subtypes.index)
    common = sorted(list(common))

    a_mat = a_mat.loc[common]
    b_mat = b_mat.loc[common]
    subtypes = subtypes.loc[common]

    # select subtypes with sufficient sample size
    subtype_counts = subtypes.value_counts()
    count_mask = subtypes.apply(lambda x: subtype_counts[x] >= min_count)
    subtypes = subtypes.loc[count_mask]

    # list of unique subtypes for iterating over
    unique_subtypes = sorted(list(set(subtypes)))

    subtype_res = []

    if pbar:
        sys.stderr.flush()
        progress = tqdm(total=len(unique_subtypes))

    # compute correlation within each subtype
    for subtype in unique_subtypes:
        subtype_rows = list(subtypes[subtypes == subtype].index)

        a_subset = a_mat.loc[subtype_rows]
        b_subset = b_mat.loc[subtype_rows]

        if mat_method == "mat_corr_naive":
            res = mat_corr_naive(a_subset, b_subset, **kwargs, melt=True)

        elif mat_method == "mat_corr_nan":
            res = mat_corr_nan(a_subset, b_subset, **kwargs, melt=True)

        else:
            error = "mat_method must be 'mat_corr_naive' or 'mat_corr_nan'"

            raise ValueError(error)

        # rename columns for merging
        res.columns = [subtype + "_" + x for x in res.columns]

        subtype_res.append(res)

        if pbar:
            progress.update(1)

    if pbar:
        progress.close()

    # extract corrs, ns, and pvals
    subtype_corrs = [x.iloc[:, 0] for x in subtype_res]
    subtype_pvals = [x.iloc[:, 1] for x in subtype_res]
    subtype_ns = [x.iloc[:, 3] for x in subtype_res]

    subtype_corrs = pd.concat(subtype_corrs, axis=1, sort=True, join="outer")
    subtype_ns = pd.concat(subtype_ns, axis=1, sort=True, join="outer")
    subtype_pvals = pd.concat(subtype_pvals, axis=1, sort=True, join="outer")

    # clean up columns
    subtype_corrs.columns = unique_subtypes
    subtype_ns.columns = unique_subtypes
    subtype_pvals.columns = unique_subtypes

    if stack:
        stacked = pd.concat(
            [
                subtype_corrs.stack().rename("corr"),
                subtype_ns.stack().rename("n"),
                subtype_pvals.stack().rename("pval"),
            ],
            axis=1,
            sort=True,
        )

        stacked.index = stacked.index.rename(["a_col", "b_col", subtypes.name])
        stacked = stacked.reset_index()

        return stacked

    return subtype_corrs, subtype_ns, subtype_pvals
