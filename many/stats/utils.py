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
    n_samples,
    a_num_cols,
    b_num_cols,
    a_type="continuous",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
):
    """
    Generates randomly initialized matrix pairs for testing and benchmarking.

    Parameters
    ----------
    n_samples: int
        Number of samples per matrix (equivalent to number of rows)
    a_num_cols: int
        Number of variables for A (equivalent to number of columns in A)
    b_num_cols: int
        Number of variables for B (equivalent to number of columns in B) 
    a_type: string, "continuous", "categorical", or "zero"
        Type of variables in A
    b_type: string, "continuous", "categorical", or "zero"
        Type of variables in B
    nan: boolean
        whether or not to simulate missing (NaN) values in A and B

    Returns
    -------
    a_test, b_test: test DataFrames
    """

    if a_type == "categorical":
        a_test = np.random.randint(0, 2, (n_samples, a_num_cols))
    elif a_type == "continuous":
        a_test = np.random.random((n_samples, a_num_cols))
    elif a_type == "zero":
        a_test = np.zeros((n_samples, a_num_cols))
    else:
        raise ValueError("'a_type' must be 'categorical', 'continuous', or 'zero'")

    if b_type == "categorical":
        b_test = np.random.randint(0, 2, (n_samples, b_num_cols))
    elif b_type == "continuous":
        b_test = np.random.random((n_samples, b_num_cols))
    elif b_type == "zero":
        b_test = np.zeros((n_samples, b_num_cols))
    else:
        raise ValueError("'b_type' must be 'categorical', 'continuous', or 'zero'")

    if a_nan:

        a_test = a_test.astype(np.float64)

        A_nan = np.random.randint(0, 2, (n_samples, a_num_cols))

        a_test[A_nan == 1] = np.nan

    if b_nan:

        b_test = b_test.astype(np.float64)

        B_nan = np.random.randint(0, 2, (n_samples, b_num_cols))

        b_test[B_nan == 1] = np.nan

    a_test = pd.DataFrame(a_test)
    b_test = pd.DataFrame(b_test)

    return a_test, b_test
