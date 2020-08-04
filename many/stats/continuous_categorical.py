import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, tiecorrect, norm, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm_notebook as tqdm

from .utils import precheck_align


def mat_mwus_naive(
    a_mat, b_mat, use_continuity=True, pbar=False, effect="rank_biserial",
):
    """
    Compute rank-biserial correlations and Mann-Whitney statistics 
    between every column-column pair of A (continuous) and B (binary)
    using a double for loop.

    In the case that A or B has a single column, the results are re-formatted
    with the multiple hypothesis-adjusted q-value also returned.

    Parameters
    ----------
    A: Pandas DataFrame
        Continuous set of observations, with rows as samples and columns as labels.
    B: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    pbar: Boolean
        Whether or not to show a progress bar.
    effect: "mean", "median", or "rank_biserial"

    Returns
    -------
    corrs: rank-biserial correlations
    pvals: -log10 p-values of correlations
    """

    a_mat, b_mat = precheck_align(a_mat, b_mat)

    a_names = a_mat.columns
    b_names = b_mat.columns

    a_num_cols = a_mat.shape[1]  # number of variables in A
    b_num_cols = b_mat.shape[1]  # number of variables in B

    corrs = np.zeros((a_num_cols, b_num_cols))  # null value of r = 0
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

                a_pos = a_mat[b_pos]
                a_neg = a_mat[b_neg]

                # handle identical values cases
                if np.std(np.concatenate([a_pos, a_neg])) == 0:

                    pvals[a_col_idx][b_col_idx] = 1

                else:

                    U2, pval = mannwhitneyu(
                        a_pos, a_neg, use_continuity=use_continuity, alternative="two-sided"
                    )

                    if effect == "rank_biserial":
                        corrs[a_col_idx][b_col_idx] = 2 * U2 / (len(a_pos) * len(a_neg)) - 1
                    elif effect == "median":
                        corrs[a_col_idx][b_col_idx] = a_pos.median() - a_neg.median()
                    elif effect == "mean":
                        corrs[a_col_idx][b_col_idx] = a_pos.mean() - a_neg.mean()

                    pvals[a_col_idx][b_col_idx] = pval

            if pbar:
                progress.update(1)

    if pbar:
        progress.close()

    # account for small p-values rounding to 0
    pvals[pvals == 0] = np.finfo(np.float64).tiny

    corrs = pd.DataFrame(corrs, index=a_names, columns=b_names)
    pvals = pd.DataFrame(pvals, index=a_names, columns=b_names)
    pos_ns = pd.DataFrame(pos_ns, index=a_names, columns=b_names)
    neg_ns = pd.DataFrame(neg_ns, index=a_names, columns=b_names)

    if a_num_cols == 1 or b_num_cols == 1:

        if a_num_cols == 1:

            corrs = pd.Series(corrs.iloc[0])
            pvals = pd.Series(pvals.iloc[0])
            pos_ns = pd.Series(pos_ns.iloc[0])
            neg_ns = pd.Series(neg_ns.iloc[0])

        elif b_num_cols == 1:

            corrs = pd.Series(corrs.iloc[:, 0])
            pvals = pd.Series(pvals.iloc[:, 0])
            pos_ns = pd.Series(pos_ns.iloc[:, 0])
            neg_ns = pd.Series(neg_ns.iloc[:, 0])

        merged = pd.DataFrame()
        merged["corr"] = corrs
        merged["pval"] = pvals
        merged["pos_n"] = pos_ns
        merged["neg_n"] = neg_ns

        merged = merged[(merged["pos_n"] >= 1) & (merged["neg_n"] >= 1)]

        merged["qval"] = multipletests(merged["pval"], alpha=0.01, method="fdr_bh")[1]

        merged["pval"] = -np.log10(merged["pval"])
        merged["qval"] = -np.log10(merged["qval"])

        merged = merged.sort_values(by="pval", ascending=False)

        return merged

    pvals = -np.log10(pvals)
    return corrs, pvals


def mat_mwus(a_mat, b_mat, use_continuity=True):
    """
    Compute rank-biserial correlations and Mann-Whitney statistics 
    between every column-column pair of A (continuous) and B (binary).

    In the case that A or B has a single column, the results are re-formatted
    with the multiple hypothesis-adjusted q-value also returned.

    Parameters
    ----------
    A: Pandas DataFrame
        Continuous set of observations, with rows as samples and columns as labels.
    B: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.

    Returns
    -------
    corrs: rank-biserial correlations
    pvals: -log10 p-values of correlations
    """

    a_mat, b_mat = precheck_align(a_mat, b_mat)

    a_names = a_mat.columns
    b_names = b_mat.columns

    b_mat = b_mat.astype(bool)

    a_ranks = a_mat.apply(rankdata)
    a_ties = a_ranks.apply(tiecorrect)

    a_nan = a_mat.isna().sum().sum() == 0
    b_nan = b_mat.isna().sum().sum() == 0

    if not a_nan and not b_nan:
        raise ValueError("A and B cannot have missing values")

    a_mat, b_mat = np.array(a_mat), np.array(b_mat)

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
    neg_ranks = np.dot(a_ranks.T, b_neg)

    u1 = pos_ns * neg_ns + (pos_ns * (pos_ns + 1)) / 2.0 - pos_ranks
    u2 = pos_ns * neg_ns - u1

    corrs = 2 * u2 / (pos_ns * neg_ns) - 1

    a_ties = np.vstack([np.array(a_ties)] * b_num_cols).T

    #     if T == 0:
    #         raise ValueError('All numbers are identical in mannwhitneyu')

    sd = np.sqrt(a_ties * pos_ns * neg_ns * (pos_ns + neg_ns + 1) / 12.0)

    meanrank = pos_ns * neg_ns / 2.0 + 0.5 * use_continuity
    bigu = np.maximum(u1, u2)
    z = (bigu - meanrank) / sd

    pvals = 2 * np.vectorize(norm.sf)(np.abs(z))

    # account for small p-values rounding to 0
    pvals[pvals == 0] = np.finfo(np.float64).tiny

    pvals = pd.DataFrame(pvals, columns=b_names, index=a_names)
    corrs = pd.DataFrame(corrs, columns=b_names, index=a_names)
    pos_ns = pd.DataFrame(pos_ns, columns=b_names, index=a_names)
    neg_ns = pd.DataFrame(neg_ns, columns=b_names, index=a_names)

    if a_num_cols == 1 or b_num_cols == 1:

        if a_num_cols == 1:

            corrs = pd.Series(corrs.iloc[0])
            pvals = pd.Series(pvals.iloc[0])
            pos_ns = pd.Series(pos_ns.iloc[0])
            neg_ns = pd.Series(neg_ns.iloc[0])

        elif b_num_cols == 1:

            corrs = pd.Series(corrs.iloc[:, 0])
            pvals = pd.Series(pvals.iloc[:, 0])
            pos_ns = pd.Series(pos_ns.iloc[:, 0])
            neg_ns = pd.Series(neg_ns.iloc[:, 0])

        merged = pd.DataFrame()
        merged["corr"] = corrs
        merged["pval"] = pvals
        merged["pos_n"] = pos_ns
        merged["neg_n"] = neg_ns

        merged = merged[(merged["pos_n"] >= 1) & (merged["neg_n"] >= 1)]

        merged["qval"] = multipletests(merged["pval"], alpha=0.01, method="fdr_bh")[1]

        merged["pval"] = -np.log10(merged["pval"])
        merged["qval"] = -np.log10(merged["qval"])

        merged = merged.sort_values(by="pval", ascending=False)

        return merged

    pvals = -np.log10(pvals)
    return corrs, pvals
