import numpy as np
import pandas as pd
import scipy.special as special
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm_notebook as tqdm
import sys

from .fisher import mlog10Test1t


def precheck_align(A, B):
    """
    Perform basic checks and alignment on A, B.

    Parameters
    ----------
    A: Pandas DataFrame
        First set of observations, with rows as samples and columns as labels
    B: Pandas DataFrame
        Second set of observations, with rows as samples and columns as labels

    Returns
    -------
    A, B: reformatted and aligned versions of inputs
    """

    # cast to DataFrame in case either is a Series
    A = pd.DataFrame(A)
    B = pd.DataFrame(B)

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


def mat_fishers_naive(A, B, pbar=False):
    """
    Compute odds ratios and Fisher's exact test 
    between every column-column pair of A (binary) and B (binary),
    using a double for loop.

    In the case that A or B has a single column, the results are re-formatted
    with the multiple hypothesis-adjusted q-value also returned.

    Parameters
    ----------
    A: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    B: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    pbar: Boolean
        Whether or not to show a progress bar.

    Returns
    -------
    oddsrs: contingency table odds ratios
    fishers: -log10 p-values of Fisher exact test
    """

    A, B = precheck_align(A, B)

    A_names = A.columns
    B_names = B.columns

    p = len(A_names)
    q = len(B_names)

    oddsrs = np.zeros((p, q))
    pvals = np.zeros((p, q))

    AB = np.zeros((p, q), dtype=np.int64)
    Ab = np.zeros((p, q), dtype=np.int64)
    aB = np.zeros((p, q), dtype=np.int64)
    ab = np.zeros((p, q), dtype=np.int64)

    if pbar:
        sys.stderr.flush()
        progress = tqdm(total=p * q)

    for A_col_idx, A_col in enumerate(A_names):
        for B_col_idx, B_col in enumerate(B_names):

            a = A[A_col].dropna().astype(bool)
            b = B[B_col].dropna().astype(bool)

            a, b = a.align(b, join="inner")

            XY = (a & b).sum()
            Xy = (a & (~b)).sum()
            xY = ((~a) & b).sum()
            xy = ((~a) & (~b)).sum()

            AB[A_col_idx][B_col_idx] = XY
            Ab[A_col_idx][B_col_idx] = Xy
            aB[A_col_idx][B_col_idx] = xY
            ab[A_col_idx][B_col_idx] = xy

            oddsr = (XY / Xy) / (xY / xy)
            pval = mlog10Test1t(XY, Xy, xY, xy)

            oddsrs[A_col_idx][B_col_idx] = oddsr
            pvals[A_col_idx][B_col_idx] = pval

            if pbar:
                progress.update(1)

    if pbar:
        progress.close()

    pvals = pd.DataFrame(pvals, index=A_names, columns=B_names)

    oddsrs = pd.DataFrame(oddsrs, index=A_names, columns=B_names)

    if p == 1 or q == 1:

        if p == 1:

            oddsrs = pd.Series(oddsrs.iloc[0])
            pvals = pd.Series(pvals.iloc[0])

        elif q == 1:

            oddsrs = pd.Series(oddsrs.iloc[:, 0])
            pvals = pd.Series(pvals.iloc[:, 0])

        merged = pd.DataFrame()
        merged["oddsr"] = oddsrs
        merged["pval"] = pvals
        merged["qval"] = multipletests(
            10 ** (-merged["pval"]), alpha=0.01, method="fdr_bh"
        )[1]

        merged["qval"] = -np.log10(merged["qval"])
        merged["AB"] = AB.reshape(-1)
        merged["Ab"] = Ab.reshape(-1)
        merged["aB"] = aB.reshape(-1)
        merged["ab"] = ab.reshape(-1)

        merged = merged.sort_values(by="pval", ascending=False)

        return merged

    else:

        return oddsrs, pvals


def fisher_arr(x):
    """
    Fisher's exact test for flattened contingency table

    Parameters
    ----------
    x: list-like
        List of contingency table coefficients, in order of
        XY, Xy, xY, xy

    Returns
    -------
    p: log10 p-value of Fisher's exact test
    """

    return mlog10Test1t(x[0], x[1], x[2], x[3])


def mat_fishers(A, B):
    """
    Compute odds ratios and Fisher's exact test 
    between every column-column pair of A (binary) and B (binary),
    under the assumption that neither array contains missing values
    (np.nan).

    In the case that A or B has a single column, the results are re-formatted
    with the multiple hypothesis-adjusted q-value also returned.

    Parameters
    ----------
    A: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    B: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.

    Returns
    -------
    oddsrs: contingency table odds ratios
    fishers: -log10 p-values of Fisher exact test
    """

    A, B = precheck_align(A, B)

    A_names = A.columns
    B_names = B.columns

    p = len(A_names)  # number of variables in A
    q = len(B_names)  # number of variables in B

    a_nan = A.isna().sum().sum() == 0
    b_nan = B.isna().sum().sum() == 0

    if not a_nan and not b_nan:
        raise ValueError("A and B cannot have missing values")

    A = A.astype(np.int64)
    B = B.astype(np.int64)

    A, B = np.array(A), np.array(B)

    A_pos = A
    A_neg = 1 - A
    B_pos = B
    B_neg = 1 - B

    AB = np.dot(A_pos.T, B_pos)
    Ab = np.dot(A_pos.T, B_neg)
    aB = np.dot(A_neg.T, B_pos)
    ab = np.dot(A_neg.T, B_neg)

    comb = np.stack([AB, Ab, aB, ab])

    pvals = np.apply_along_axis(fisher_arr, 0, comb)
    pvals = pd.DataFrame(pvals, index=A_names, columns=B_names)

    oddsrs = (AB / Ab) / (aB / ab)
    oddsrs = pd.DataFrame(oddsrs, index=A_names, columns=B_names)

    if p == 1 or q == 1:

        if p == 1:

            oddsrs = pd.Series(oddsrs.iloc[0])
            pvals = pd.Series(pvals.iloc[0])

        elif q == 1:
            oddsrs = pd.Series(oddsrs.iloc[:, 0])
            pvals = pd.Series(pvals.iloc[:, 0])

        merged = pd.DataFrame()
        merged["oddsr"] = oddsrs
        merged["pval"] = pvals
        merged["qval"] = multipletests(
            10 ** (-merged["pval"]), alpha=0.01, method="fdr_bh"
        )[1]

        merged["qval"] = -np.log10(merged["qval"])
        merged["AB"] = AB.reshape(-1)
        merged["Ab"] = Ab.reshape(-1)
        merged["aB"] = aB.reshape(-1)
        merged["ab"] = ab.reshape(-1)

        merged = merged.sort_values(by="pval", ascending=False)

        return merged

    else:

        return oddsrs, pvals


def mat_fishers_nan(A, B):
    """
    Compute odds ratios and Fisher's exact test 
    between every column-column pair of A (binary) and B (binary),
    allowing for missing values.

    In the case that A or B has a single column, the results are re-formatted
    with the multiple hypothesis-adjusted q-value also returned.

    Parameters
    ----------
    A: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    B: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.

    Returns
    -------
    oddsrs: contingency table odds ratios
    fishers: -log10 p-values of Fisher exact test
    """

    A, B = precheck_align(A, B)

    A_names = A.columns
    B_names = B.columns

    A, B = np.array(A), np.array(B)

    p = A.shape[1]  # number of variables in A
    q = B.shape[1]  # number of variables in B

    A_pos = A
    A_neg = 1 - A
    B_pos = B
    B_neg = 1 - B

    A_nan = np.isnan(A)
    B_nan = np.isnan(B)

    A_pos = np.ma.array(A_pos, mask=A_nan, dtype=np.int64)
    A_neg = np.ma.array(A_neg, mask=A_nan, dtype=np.int64)
    B_pos = np.ma.array(B_pos, mask=B_nan, dtype=np.int64)
    B_neg = np.ma.array(B_neg, mask=B_nan, dtype=np.int64)

    AB = np.ma.dot(A_pos.T, B_pos)
    Ab = np.ma.dot(A_pos.T, B_neg)
    aB = np.ma.dot(A_neg.T, B_pos)
    ab = np.ma.dot(A_neg.T, B_neg)

    comb = np.stack([AB, Ab, aB, ab])

    pvals = np.apply_along_axis(fisher_arr, 0, comb)
    pvals = pd.DataFrame(pvals, index=A_names, columns=B_names)

    oddsrs = (AB / Ab) / (aB / ab)
    oddsrs = pd.DataFrame(oddsrs, index=A_names, columns=B_names)

    if p == 1 or q == 1:

        if p == 1:

            oddsrs = pd.Series(oddsrs.iloc[0])
            pvals = pd.Series(pvals.iloc[0])

        elif q == 1:

            oddsrs = pd.Series(oddsrs.iloc[:, 0])
            pvals = pd.Series(pvals.iloc[:, 0])

        merged = pd.DataFrame()
        merged["oddsr"] = oddsrs
        merged["pval"] = pvals
        merged["qval"] = multipletests(
            10 ** (-merged["pval"]), alpha=0.01, method="fdr_bh"
        )[1]

        merged["qval"] = -np.log10(merged["qval"])
        merged["AB"] = AB.reshape(-1)
        merged["Ab"] = Ab.reshape(-1)
        merged["aB"] = aB.reshape(-1)
        merged["ab"] = ab.reshape(-1)

        merged = merged.sort_values(by="pval", ascending=False)

        return merged

    else:

        return oddsrs, pvals
