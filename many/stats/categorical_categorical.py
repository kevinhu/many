import sys

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm_notebook as tqdm

from . import config
from .fisher import mlog10Test1t
from .utils import precheck_align


def melt_fisher(
    oddsrs, pvals, AB, Ab, aB, ab, a_num_cols: int, b_num_cols: int
):
    """
    Flatten matrix-form outputs to column-form.

    Parameters
    ----------
    oddsrs: Pandas DataFrame
        odds ratios matrix
    pvals: Pandas DataFrame
        p-values matrix
    AB, Ab, aB, ab: Pandas DataFrames
        contingency table counts
    a_num_cols: int
        number of columns in first observations matrix
    b_num_cols: int
        number of columns in second observations matrix

    Returns
    -------
    series form statistics
    """

    melted = pd.DataFrame()
    melted["oddsr"] = oddsrs.unstack()
    melted["pval"] = pvals.unstack()
    melted["qval"] = multipletests(
        10 ** (-melted["pval"]),
        alpha=config.MULTIPLETESTS_ALPHA,
        method=config.MULTIPLETESTS_METHOD,
    )[1]

    melted["qval"] = -np.log10(melted["qval"])

    AB = pd.DataFrame(AB, index=oddsrs.index, columns=oddsrs.columns)
    Ab = pd.DataFrame(Ab, index=oddsrs.index, columns=oddsrs.columns)
    aB = pd.DataFrame(aB, index=oddsrs.index, columns=oddsrs.columns)
    ab = pd.DataFrame(ab, index=oddsrs.index, columns=oddsrs.columns)

    melted["AB"] = AB.unstack()
    melted["Ab"] = Ab.unstack()
    melted["aB"] = aB.unstack()
    melted["ab"] = ab.unstack()

    melted = melted.sort_values(by="pval", ascending=False)

    melted.index.set_names(["b_col", "a_col"], inplace=True)
    melted.index = melted.index.swaplevel(0, 1)

    return melted


def mat_fisher_naive(a_mat, b_mat, melt: bool, pseudocount=0, pbar=False):
    """
    Compute odds ratios and Fisher's exact test
    between every column-column pair of a_mat (binary) and b_mat (binary),
    using a double for loop.

    Parameters
    ----------
    a_mat: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    b_mat: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    melt: boolean
        Whether or not to melt the outputs into columns.
    pseudocount: float
        value to add to all contingency cells (0 by default)
    pbar: Boolean
        Whether or not to show a progress bar.

    Returns
    -------
    oddsrs: contingency table odds ratios
    fishers: -log10 p-values of Fisher exact test
    """

    a_mat, b_mat = precheck_align(a_mat, b_mat, np.int64, np.int64)

    a_names = a_mat.columns
    b_names = b_mat.columns

    a_num_cols = len(a_names)
    b_num_cols = len(b_names)

    oddsrs = np.zeros((a_num_cols, b_num_cols))
    pvals = np.zeros((a_num_cols, b_num_cols))

    # contingency table counts
    AB = np.zeros((a_num_cols, b_num_cols), dtype=np.float64)
    Ab = np.zeros((a_num_cols, b_num_cols), dtype=np.float64)
    aB = np.zeros((a_num_cols, b_num_cols), dtype=np.float64)
    ab = np.zeros((a_num_cols, b_num_cols), dtype=np.float64)

    if pbar:
        sys.stderr.flush()
        progress = tqdm(total=a_num_cols * b_num_cols)

    for a_col_idx, a_col_name in enumerate(a_names):
        for b_col_idx, b_col_name in enumerate(b_names):

            a_col = a_mat[a_col_name].dropna().astype(bool)
            b_col = b_mat[b_col_name].dropna().astype(bool)

            a_col, b_col = a_col.align(b_col, join="inner")

            XY = (a_col & b_col).sum() + pseudocount
            Xy = (a_col & (~b_col)).sum() + pseudocount
            xY = ((~a_col) & b_col).sum() + pseudocount
            xy = ((~a_col) & (~b_col)).sum() + pseudocount

            AB[a_col_idx][b_col_idx] = XY
            Ab[a_col_idx][b_col_idx] = Xy
            aB[a_col_idx][b_col_idx] = xY
            ab[a_col_idx][b_col_idx] = xy

            numer = XY * xy
            denom = xY * Xy

            # dummy values for odds ratio
            if denom == 0:
                oddsr = -1
            else:
                oddsr = numer / denom

            pval = mlog10Test1t(XY, Xy, xY, xy)

            oddsrs[a_col_idx][b_col_idx] = oddsr
            pvals[a_col_idx][b_col_idx] = pval

            if pbar:
                progress.update(1)

    if pbar:
        progress.close()

    pvals = pd.DataFrame(pvals, index=a_names, columns=b_names)
    oddsrs = pd.DataFrame(oddsrs, index=a_names, columns=b_names)

    pvals = pvals.fillna(0)
    oddsrs = oddsrs.fillna(1)

    if melt:

        return melt_fisher(
            oddsrs, pvals, AB, Ab, aB, ab, a_num_cols, b_num_cols
        )

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


def mat_fisher(a_mat, b_mat, melt: bool, pseudocount=0):
    """
    Compute odds ratios and Fisher's exact test
    between every column-column pair of a_mat (binary) and b_mat (binary),
    under the assumption that neither array contains missing values
    (np.nan).

    Parameters
    ----------
    a_mat: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    b_mat: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    melt: boolean
        Whether or not to melt the outputs into columns.
    pseudocount: integer
        value to add to all contingency cells (0 by default)

    Returns
    -------
    oddsrs: contingency table odds ratios
    fishers: -log10 p-values of Fisher exact test
    """

    a_mat, b_mat = precheck_align(a_mat, b_mat, np.int64, np.int64)

    a_names = a_mat.columns
    b_names = b_mat.columns

    a_num_cols = len(a_names)  # number of variables in a_mat
    b_num_cols = len(b_names)  # number of variables in b_mat

    a_nan = a_mat.isna().sum().sum() == 0
    b_nan = b_mat.isna().sum().sum() == 0

    if not a_nan and not b_nan:
        raise ValueError("A and B cannot have missing values")

    a_mat, b_mat = np.array(a_mat), np.array(b_mat)

    A_pos = a_mat
    A_neg = 1 - a_mat
    B_pos = b_mat
    B_neg = 1 - b_mat

    AB = np.dot(A_pos.T, B_pos) + pseudocount
    Ab = np.dot(A_pos.T, B_neg) + pseudocount
    aB = np.dot(A_neg.T, B_pos) + pseudocount
    ab = np.dot(A_neg.T, B_neg) + pseudocount

    comb = np.stack([AB, Ab, aB, ab]).astype(np.int64)

    pvals = np.apply_along_axis(fisher_arr, 0, comb)
    pvals = pd.DataFrame(pvals, index=a_names, columns=b_names)

    numer = AB * ab
    denom = aB * Ab

    zero_denom = denom == 0
    denom[zero_denom] = 1

    oddsrs = numer / denom
    oddsrs[zero_denom] = -1
    oddsrs = pd.DataFrame(oddsrs, index=a_names, columns=b_names)

    pvals = pvals.fillna(0)
    oddsrs = oddsrs.fillna(1)

    if melt:

        return melt_fisher(
            oddsrs, pvals, AB, Ab, aB, ab, a_num_cols, b_num_cols
        )

    return oddsrs, pvals


def mat_fisher_nan(a_mat, b_mat, melt: bool, pseudocount=0):
    """
    Compute odds ratios and Fisher's exact test
    between every column-column pair of a_mat (binary) and b_mat (binary),
    allowing for missing values.

    Parameters
    ----------
    a_mat: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    b_mat: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    melt: boolean
        Whether or not to melt the outputs into columns.
    pseudocount: integer
        value to add to all contingency cells (0 by default)

    Returns
    -------
    oddsrs: contingency table odds ratios
    fishers: -log10 p-values of Fisher exact test
    """

    a_mat, b_mat = precheck_align(a_mat, b_mat, np.int64, np.int64)

    a_names = a_mat.columns
    b_names = b_mat.columns

    a_mat, b_mat = np.array(a_mat), np.array(b_mat)

    a_num_cols = a_mat.shape[1]  # number of variables in a_mat
    b_num_cols = b_mat.shape[1]  # number of variables in b_mat

    a_pos = a_mat
    a_neg = 1 - a_mat
    b_pos = b_mat
    b_neg = 1 - b_mat

    a_nan = np.isnan(a_mat)
    b_nan = np.isnan(b_mat)

    a_pos = np.ma.array(a_pos, mask=a_nan, dtype=np.int64)
    a_neg = np.ma.array(a_neg, mask=a_nan, dtype=np.int64)
    b_pos = np.ma.array(b_pos, mask=b_nan, dtype=np.int64)
    b_neg = np.ma.array(b_neg, mask=b_nan, dtype=np.int64)

    AB = np.ma.dot(a_pos.T, b_pos) + pseudocount
    Ab = np.ma.dot(a_pos.T, b_neg) + pseudocount
    aB = np.ma.dot(a_neg.T, b_pos) + pseudocount
    ab = np.ma.dot(a_neg.T, b_neg) + pseudocount

    comb = np.stack([AB, Ab, aB, ab])

    pvals = np.apply_along_axis(fisher_arr, 0, comb)
    pvals = pd.DataFrame(pvals, index=a_names, columns=b_names)

    numer = AB * ab
    denom = aB * Ab

    zero_denom = denom == 0
    denom[zero_denom] = 1

    oddsrs = numer / denom
    oddsrs[zero_denom] = -1
    oddsrs = pd.DataFrame(oddsrs, index=a_names, columns=b_names)

    pvals = pvals.fillna(0)
    oddsrs = oddsrs.fillna(1)

    if melt:

        return melt_fisher(
            oddsrs, pvals, AB, Ab, aB, ab, a_num_cols, b_num_cols
        )

    return oddsrs, pvals
