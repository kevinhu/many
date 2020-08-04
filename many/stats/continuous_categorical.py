import numpy as np
import pandas as pd
from scipy.stats import rankdata, tiecorrect, norm, mannwhitneyu
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm_notebook as tqdm
import sys


def precheck_align(A, B):
    """
    Perform basic checks and alignment on A, B.

    Parameters
    ----------
    A: Pandas DataFrame
        Continuous set of observations, with rows as samples and columns as labels.
    B: Pandas DataFrame
        Binary set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.

    Returns
    -------
    A, B: reformatted and aligned versions of inputs
    """

    # cast to DataFrame in case either is a Series
    A = pd.DataFrame(A, dtype=np.float64)
    B = pd.DataFrame(B, dtype=np.float16)

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


def mat_mwus_naive(
    A, B, use_continuity=True, pbar=False, effect="rank_biserial",
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

    A, B = precheck_align(A, B)

    A_names = A.columns
    B_names = B.columns

    p = A.shape[1]  # number of variables in A
    q = B.shape[1]  # number of variables in B

    corrs = np.zeros((p, q))  # null value of r = 0
    pvals = np.zeros((p, q)) + 1  # null value of p=1

    pos_ns = np.zeros((p, q))
    neg_ns = np.zeros((p, q))

    if pbar:
        sys.stderr.flush()
        progress = tqdm(total=p * q)

    for A_col_idx, A_col in enumerate(A_names):
        for B_col_idx, B_col in enumerate(B_names):

            a = A[A_col].dropna()
            b = B[B_col].dropna()

            a, b = a.align(b, join="inner")

            b_pos = b == 1
            b_neg = b == 0

            pos_n = b_pos.sum()
            neg_n = b_neg.sum()

            pos_ns[A_col_idx][B_col_idx] = pos_n
            neg_ns[A_col_idx][B_col_idx] = neg_n

            if pos_n >= 1 and neg_n >= 1:

                a1 = a[b_pos]
                a2 = a[b_neg]

                # handle identical values cases
                if np.std(np.concatenate([a1, a2])) == 0:

                    pvals[A_col_idx][B_col_idx] = 1

                else:

                    U2, pval = mannwhitneyu(
                        a1, a2, use_continuity=use_continuity, alternative="two-sided"
                    )

                    if effect == "rank_biserial":
                        corrs[A_col_idx][B_col_idx] = 2 * U2 / (len(a1) * len(a2)) - 1
                    elif effect == "median":
                        corrs[A_col_idx][B_col_idx] = a1.median() - a2.median()
                    elif effect == "mean":
                        corrs[A_col_idx][B_col_idx] = a1.mean() - a2.mean()

                    pvals[A_col_idx][B_col_idx] = pval

            if pbar:
                progress.update(1)

    if pbar:
        progress.close()

    # account for small p-values rounding to 0
    pvals[pvals == 0] = np.finfo(np.float64).tiny

    corrs = pd.DataFrame(corrs, index=A_names, columns=B_names)
    pvals = pd.DataFrame(pvals, index=A_names, columns=B_names)
    pos_ns = pd.DataFrame(pos_ns, index=A_names, columns=B_names)
    neg_ns = pd.DataFrame(neg_ns, index=A_names, columns=B_names)

    if p == 1 or q == 1:

        if p == 1:

            corrs = pd.Series(corrs.iloc[0])
            pvals = pd.Series(pvals.iloc[0])
            pos_ns = pd.Series(pos_ns.iloc[0])
            neg_ns = pd.Series(neg_ns.iloc[0])

        elif q == 1:

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

    else:

        pvals = -np.log10(pvals)
        return corrs, pvals


def mat_mwus(A, B, use_continuity=True):
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

    A, B = precheck_align(A, B)

    A_names = A.columns
    B_names = B.columns

    B = B.astype(bool)

    A_r = A.apply(rankdata)
    A_T = A_r.apply(tiecorrect)

    a_nan = A.isna().sum().sum() == 0
    b_nan = B.isna().sum().sum() == 0

    if not a_nan and not b_nan:
        raise ValueError("A and B cannot have missing values")

    A, B = np.array(A), np.array(B)

    p = A.shape[1]  # number of variables in A
    q = B.shape[1]  # number of variables in B

    A = A.astype(np.float64)
    B_pos = B.astype(np.float64)
    B_neg = (~B).astype(np.float64)

    pos_ns = B_pos.sum(axis=0)
    neg_ns = B_neg.sum(axis=0)

    pos_ns = np.vstack([pos_ns] * p)
    neg_ns = np.vstack([neg_ns] * p)

    pos_ranks = np.dot(A_r.T, B_pos)
    neg_ranks = np.dot(A_r.T, B_neg)

    u1 = pos_ns * neg_ns + (pos_ns * (pos_ns + 1)) / 2.0 - pos_ranks
    u2 = pos_ns * neg_ns - u1

    corrs = 2 * u2 / (pos_ns * neg_ns) - 1

    A_T = np.vstack([np.array(A_T)] * q).T

    #     if T == 0:
    #         raise ValueError('All numbers are identical in mannwhitneyu')

    sd = np.sqrt(A_T * pos_ns * neg_ns * (pos_ns + neg_ns + 1) / 12.0)

    meanrank = pos_ns * neg_ns / 2.0 + 0.5 * use_continuity
    bigu = np.maximum(u1, u2)
    z = (bigu - meanrank) / sd

    pvals = 2 * np.vectorize(norm.sf)(np.abs(z))

    # account for small p-values rounding to 0
    pvals[pvals == 0] = np.finfo(np.float64).tiny

    pvals = pd.DataFrame(pvals, columns=B_names, index=A_names)
    corrs = pd.DataFrame(corrs, columns=B_names, index=A_names)
    pos_ns = pd.DataFrame(pos_ns, columns=B_names, index=A_names)
    neg_ns = pd.DataFrame(neg_ns, columns=B_names, index=A_names)

    if p == 1 or q == 1:

        if p == 1:

            corrs = pd.Series(corrs.iloc[0])
            pvals = pd.Series(pvals.iloc[0])
            pos_ns = pd.Series(pos_ns.iloc[0])
            neg_ns = pd.Series(neg_ns.iloc[0])

        elif q == 1:

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

    else:

        pvals = -np.log10(pvals)
        return corrs, pvals
