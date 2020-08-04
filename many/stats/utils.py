import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def precheck_align(a_mat, b_mat):
    """
    Perform basic checks and alignment on A, B.

    Parameters
    ----------
    A: Pandas DataFrame
        First set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    B: Pandas DataFrame
        Second set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.

    Returns
    -------
    A, B: reformatted and aligned versions of inputs
    """

    # cast to DataFrame in case either is a Series
    a_mat = pd.DataFrame(a_mat)
    b_mat = pd.DataFrame(b_mat)

    # drop samples with all missing values
    a_mat = a_mat.dropna(how="all", axis=0)
    b_mat = b_mat.dropna(how="all", axis=0)

    # align samples
    a_mat, b_mat = a_mat.align(b_mat, axis=0, join="inner")

    # check sample sizes
    num_samples = a_mat.shape[0]  # number of samples for each variable
    if num_samples < 2:
        raise ValueError("x and y must have length at least 2.")

    return a_mat, b_mat


def generate_test(
    n_samples, A_n_cols, B_n_cols, A_type="continuous", B_type="continuous", nan=False
):
    """
    Generates randomly initialized matrix pairs for testing and benchmarking.

    Parameters
    ----------
    n_samples: int
        Number of samples per matrix (equivalent to number of rows)
    A_n_cols: int
        Number of variables for A (equivalent to number of columns in A)
    B_n_cols: int
        Number of variables for B (equivalent to number of columns in B) 
    A_type: string, "continuous" or "categorical"
        Type of variables in A
    B_type: string, "continuous" or "categorical"
        Type of variables in B
    nan: boolean
        whether or not to simulate missing (NaN) values in A and B

    Returns
    -------
    A_test, B_test: randomly initialized test matrices
    """

    if A_type == "categorical":
        A_test = np.random.randint(0, 2, (n_samples, A_n_cols))
    elif A_type == "continuous":
        A_test = np.random.random((n_samples, A_n_cols))
    else:
        raise ValueError("'A_type' must be 'categorical' or 'continuous'")

    if B_type == "categorical":
        B_test = np.random.randint(0, 2, (n_samples, B_n_cols))
    elif B_type == "continuous":
        B_test = np.random.random((n_samples, B_n_cols))
    else:
        raise ValueError("'B_type' must be 'categorical' or 'continuous'")

    if nan:

        A_test = A_test.astype(np.float16)
        B_test = B_test.astype(np.float16)

        A_nan = np.random.randint(0, 2, (n_samples, A_n_cols))
        B_nan = np.random.randint(0, 2, (n_samples, B_n_cols))

        A_test[A_nan == 1] = np.nan
        B_test[B_nan == 1] = np.nan

    A_test = pd.DataFrame(A_test)
    B_test = pd.DataFrame(B_test)

    return A_test, B_test


def test_plot_dataframes(v_effects, v_pvals, n_effects, n_pvals):
    """
    Plots effect sizes and p-values of two methods, intended for comparing
    vectorized and naive implementations.

    Parameters
    ----------
    v_effects: DataFrame
        Effect sizes between variable pairs (rows and columns),
        produced from vectorized implementation
    v_pvals: DataFrame
        P-values between variable pairs (rows and columns),
        produced from vectorized implementation
    n_effects: DataFrame
        Effect sizes between variable pairs (rows and columns),
        produced from naive implementation
    n_pvals: DataFrame
        P-values between variable pairs (rows and columns),
        produced from naive implementation
    """

    v_effects = np.array(v_effects).reshape(-1)
    v_pvals = np.array(v_pvals).reshape(-1)

    n_effects = np.array(n_effects).reshape(-1)
    n_pvals = np.array(n_pvals).reshape(-1)

    plt.figure(figsize=(6, 3))

    plt.subplot(121)
    sns.scatterplot(v_effects, n_effects)
    plt.xlabel("")
    plt.ylabel("")
    plt.axis("scaled")

    plt.subplot(122)
    sns.scatterplot(v_pvals, n_pvals)
    plt.xlabel("")
    plt.ylabel("")
    plt.axis("scaled")

    plt.subplots_adjust(wspace=0.25)

    plt.show()


def test_plot_series(v_res, n_res, effect_name, pval_name):
    """
    Plots effect sizes and p-values of two methods, intended for comparing
    vectorized and naive implementations. This method handles the results
    when one of A, B is a Series, which causes a different result
    format to be returned.

    Parameters
    ----------
    v_res: DataFrame
        Results (effect size, p-value, and q-value) with each row
        representing a variable's association values, calculated
        using the vectorized implementation
    n_res: DataFrame
        Results (effect size, p-value, and q-value) with each row
        representing a variable's association values, calculated
        using the naive implementation
    effect_name: string
        Name of the effect size column
    pval_name: string
        Name of the p-value column
    """

    plt.figure(figsize=(6, 3))

    plt.subplot(121)
    sns.scatterplot(v_res[effect_name], n_res[effect_name])
    plt.xlabel("")
    plt.ylabel("")
    plt.axis("scaled")

    plt.subplot(122)
    sns.scatterplot(v_res[pval_name], n_res[pval_name])
    plt.xlabel("")
    plt.ylabel("")
    plt.axis("scaled")

    plt.subplots_adjust(wspace=0.25)

    plt.show()
