import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def precheck_align(a_mat, b_mat, a_cast, b_cast):
    """
    Perform basic checks and alignment on a_mat and b_mat.

    Parameters
    ----------
    a_mat: Pandas DataFrame
        First set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    b_mat: Pandas DataFrame
        Second set of observations, with rows as samples and columns as labels.
        Required to be castable to boolean datatype.
    a_cast: NumPy DataType
        DataType to cast a_mat to
    b_cast: Numpy DataType
        DataType to cast b_mat to

    Returns
    -------
    Reformatted and aligned versions of inputs
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
