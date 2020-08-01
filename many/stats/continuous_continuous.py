import numpy as np
import pandas as pd
import scipy.special as special
from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm_notebook as tqdm
import sys


def precheck_align(A, B, method):
    """
    Perform basic checks and alignment on A, B.

    Parameters
    ----------
    A: Pandas DataFrame
        First set of observations, with rows as samples and columns as labels
    B: Pandas DataFrame
        Second set of observations, with rows as samples and columns as labels
    method: string
        Correlation method to use

    Returns
    -------
    A, B: reformatted and aligned versions of inputs
    """

    # check method
    if method != "pearson" and method != "spearman":
        raise ValueError("Method must be 'pearson' or 'spearman'")

    # cast to DataFrame in case either is a Series
    A = pd.DataFrame(A, dtype=np.float64)
    B = pd.DataFrame(B, dtype=np.float64)

    # drop samples with all missing values
    A = A.dropna(how="all", axis=0)
    B = B.dropna(how="all", axis=0)

    # align samples
    A, B = A.align(B, axis=0, join="inner")

    # check sample sizes
    n = A.shape[0]  # number of samples for each variable
    if n < 2:
        raise ValueError("x and y must have length at least 2.")

    return A, B


def mat_corrs_naive(A, B, method="pearson", pbar=False):
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

    A, B = precheck_align(A, B, method)

    # store names before array conversion
    A_names = A.columns
    B_names = B.columns

    p = len(A_names)  # number of variables in A
    q = len(B_names)  # number of variables in B

    # initialize arrays for correlations and p-values
    corrs = np.zeros((p, q))
    pvals = np.zeros((p, q))
    ns = np.zeros((p, q))

    if pbar:
        sys.stderr.flush()
        progress = tqdm(total=p * q)

    for A_col_idx, A_col in enumerate(A_names):
        for B_col_idx, B_col in enumerate(B_names):

            # select columns to correlate
            a = A[A_col].dropna()
            b = B[B_col].dropna()

            a, b = a.align(b, join="inner", axis=0)

            n = len(a)

            if n > 2:

                if method == "pearson":
                    corr, pval = pearsonr(a, b)
                elif method == "spearman":
                    corr, pval = spearmanr(a, b)

            elif n <= 2:

                corr, pval = np.nan, np.nan

            # add in correlation
            corrs[A_col_idx][B_col_idx] = corr
            pvals[A_col_idx][B_col_idx] = pval
            ns[A_col_idx][B_col_idx] = n

            if pbar:
                progress.update(1)

    if pbar:
        progress.close()

    # account for small p-values rounding to 0
    pvals[pvals == 0] = np.finfo(np.float64).tiny

    # convert correlation arrays to named DataFrames
    corrs = pd.DataFrame(corrs, index=A_names, columns=B_names)

    ns = pd.DataFrame(ns, index=A_names, columns=B_names)

    pvals = pd.DataFrame(pvals, index=A_names, columns=B_names)

    # if one of the matrices is a single variable,
    # return correlation results in series form
    if p == 1 or q == 1:

        if p == 1:
            corrs = pd.Series(corrs.iloc[0])
            pvals = pd.Series(pvals.iloc[0])
            ns = pd.Series(ns.iloc[0])

        elif q == 1:
            corrs = pd.Series(corrs.iloc[:, 0])
            pvals = pd.Series(pvals.iloc[:, 0])
            ns = pd.Series(ns.iloc[0])

        merged = pd.DataFrame()
        merged["corr"] = corrs
        merged["n"] = ns
        merged["pval"] = pvals

        # drop correlations with less than 3 samples
        merged = merged.dropna(how="any")

        merged["qval"] = multipletests(merged["pval"], alpha=0.01, method="fdr_bh")[1]

        merged["pval"] = -np.log10(merged["pval"])
        merged["qval"] = -np.log10(merged["qval"])

        merged = merged.sort_values(by="pval", ascending=False)

        return merged

    else:

        pvals = -np.log10(pvals)
        return corrs, pvals


def mat_corrs(A, B, method="pearson"):
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

    A, B = precheck_align(A, B, method)

    A_names = A.columns
    B_names = B.columns
    p = len(A_names)  # number of variables in A
    q = len(B_names)  # number of variables in B
    n = len(A.index)  # number of samples

    a_nan = A.isna().sum().sum() == 0
    b_nan = B.isna().sum().sum() == 0

    if not a_nan and not b_nan:
        raise ValueError("A and B cannot have missing values")

    # Compute column ranks, as Spearman correlation is equivalent
    # to Pearson correlation between ranks
    if method == "spearman":
        A = A.rank(method="min")
        B = B.rank(method="min")

    A, B = np.array(A), np.array(B)

    # Subtract column means
    residuals_A = A - A.mean(axis=0)
    residuals_B = B - B.mean(axis=0)

    # Sum squares across columns
    sums_A = (residuals_A ** 2).sum(axis=0)
    sums_B = (residuals_B ** 2).sum(axis=0)

    # Compute correlations
    residual_products = np.dot(residuals_A.T, residuals_B)
    sum_products = np.sqrt(np.dot(sums_A[:, None], sums_B[None]))
    corrs = residual_products / sum_products

    # Compute significance values
    ab = n / 2 - 1

    def beta(r):
        return 2 * special.btdtr(ab, ab, 0.5 * (1 - abs(np.float64(r))))

    beta = np.vectorize(beta)

    pvals = beta(corrs)

    # account for small p-values rounding to 0
    pvals[pvals == 0] = np.finfo(np.float64).tiny

    # Store correlations in DataFrames
    corrs = pd.DataFrame(corrs, index=A_names, columns=B_names)
    pvals = pd.DataFrame(pvals, index=A_names, columns=B_names)

    if p == 1 or q == 1:

        if p == 1:
            corrs = pd.Series(corrs.iloc[0])
            pvals = pd.Series(pvals.iloc[0])

        elif q == 1:
            corrs = pd.Series(corrs.iloc[:, 0])
            pvals = pd.Series(pvals.iloc[:, 0])

        merged = pd.DataFrame()
        merged["corr"] = corrs
        merged["n"] = n
        merged["pval"] = pvals
        merged["qval"] = multipletests(merged["pval"], alpha=0.01, method="fdr_bh")[1]

        merged["pval"] = -np.log10(merged["pval"])
        merged["qval"] = -np.log10(merged["qval"])

        merged = merged.sort_values(by="pval", ascending=False)

        return merged

    else:

        pvals = -np.log10(pvals)
        return corrs, pvals


def pearson_significance(row):
    r = row["corr"]
    ab = row["n"] / 2 - 1

    beta = 2 * special.btdtr(ab, ab, 0.5 * (1 - abs(r)))

    # account for small p-values rounding to 0
    beta = max(np.finfo(np.float64).tiny, beta)

    return beta


def mat_corrs_nan(A, B, method="pearson"):
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

    A, B = precheck_align(A, B, method)

    if len(A.columns) != 1:
        raise ValueError("A must contain only a single variable.")

    B_names = B.columns
    B_nan = B.isna()

    m = len(B_names)

    n = len(A.index)  # number of samples

    # compute column ranks, as Spearman correlation is equivalent
    # to Pearson correlation between ranks
    if method == "spearman":

        B_nan = B.isna()
        B = B.rank(na_option="top", method="min")
        B[B <= B_nan.sum()] = np.nan
        B = B - B_nan.sum()

        # construct mirrored A matrix
        m = B.shape[1]
        A_nan = np.repeat(np.array(A), m, axis=1)
        A_nan[B_nan] = np.nan
        A_nan = pd.DataFrame(A_nan)

        # rank mirrored A matrix
        A_nan = A_nan.rank(na_option="top", method="min")
        A_nan[A_nan <= B_nan.sum()] = np.nan
        A_nan = A_nan - B_nan.sum()
        A_nan = np.ma.array(np.array(A_nan), mask=B_nan)

    elif method == "pearson":
        A_nan = np.ma.array(np.repeat(np.array(A), m, axis=1), mask=B_nan)

    # convert to arrays
    A, B = np.array(A), np.array(B)
    nan_sums = np.isnan(B).sum(axis=0)

    # make masked arrays
    A = np.ma.array(A, mask=np.isnan(A))
    B = np.ma.array(B, mask=np.isnan(B))

    # subtract column means
    residuals_B = B - np.ma.mean(B, axis=0)
    residuals_A_nan = A_nan - np.ma.mean(A_nan, axis=0)

    # sum squares across columns
    sums_B = np.ma.sum(residuals_B ** 2, axis=0)
    sums_A_nan = np.ma.sum(residuals_A_nan ** 2, axis=0)

    # compute correlations
    residual_products = np.ma.sum(residuals_A_nan * residuals_B, axis=0)
    sum_products = np.sqrt(sums_A_nan * sums_B)

    corrs = np.array(residual_products / sum_products).reshape(-1)

    corr_df = pd.DataFrame(index=B_names)

    corr_df["corr"] = corrs
    corr_df["n"] = n - nan_sums
    corr_df["pval"] = corr_df.apply(pearson_significance, axis=1)
    corr_df["qval"] = multipletests(corr_df["pval"], alpha=0.01, method="fdr_bh")[1]

    corr_df["pval"] = -np.log10(corr_df["pval"])
    corr_df["qval"] = -np.log10(corr_df["qval"])

    corr_df = corr_df.sort_values(by="pval", ascending=False)

    return corr_df


def mat_corrs_subtyped(
    A,
    B,
    subtypes,
    min_count=5,
    pbar=False,
    stack=False,
    mat_method="mat_corrs_naive",
    **kwargs
):
    """
    Compute correlations between A and every column of B. A must be
    a Series for this method to work. Allows for missing values in B.

    Parameters
    ----------
    A: Pandas Series
        First set of observations, with rows as samples
    B: Pandas DataFrame
        Second set of observations, with rows as samples and columns as labels
    subtypes: Pandas Series
        Categories to compute correlations within
    min_count: integer
        Minimum number of samples per subtype to keep for consideration
    pbar: boolean
        Whether or not to show a progress bar with subtypes
    **kwargs: additional parameters
        Arguments to pass to mat_corrs_nan()

    Returns
    -------

    if stack is False:

        subtype_corrs: DataFrame
            DataFrame of correlations between A and each variable of B (rows) within
            each subtype (columns)
        subtype_ns: DataFrame
            DataFrame of sample sizes between A and each variable of B (rows) within
            each subtype (columns)
        subtype_pvals: DataFrame
            DataFrame of p-values between A and each variable of B (rows) within
            each subtype (columns)

    if stack is True:

        stacked: DataFrame
            DataFrame of correlations between A and each variable of B within each
            subtypes, along with sample sizes, and p-values, with each value in a
            column
    """

    # remove missing values in A
    A = A.dropna(how="any")

    # common rows between all
    common = set(A.index) & set(B.index) & set(subtypes.index)
    common = sorted(list(common))

    A = A.loc[common]
    B = B.loc[common]
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

        A_subset = A.loc[subtype_rows]
        B_subset = B.loc[subtype_rows]

        if mat_method == "mat_corrs_naive":
            res = mat_corrs_naive(A_subset, B_subset, **kwargs)

        elif mat_method == "mat_corrs_nan":

            res = mat_corrs_nan(A_subset, B_subset, **kwargs)

        elif mat_method == "mat_corrs":

            res = mat_corrs(A_subset, B_subset, **kwargs)

        # rename columns for merging
        res.columns = [subtype + "_" + x for x in res.columns]

        subtype_res.append(res)

        if pbar:
            progress.update(1)

    if pbar:
        progress.close()

    # extract corrs, ns, and pvals
    subtype_corrs = [x.iloc[:, 0] for x in subtype_res]
    subtype_ns = [x.iloc[:, 1] for x in subtype_res]
    subtype_pvals = [x.iloc[:, 2] for x in subtype_res]

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

        stacked.index = stacked.index.rename([A.name, subtypes.name])
        stacked = stacked.reset_index()

        return stacked

    return subtype_corrs, subtype_ns, subtype_pvals
