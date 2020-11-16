import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("log")

# Correlation functions from https://github.com/cmap/cmapPy/tree/master/cmapPy/math
"""
BSD 3-Clause License

Copyright (c) 2017, Connectivity Map (CMap) at the Broad Institute, Inc.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


def _fast_dot_divide(x, y, destination):
    """
    Helper method for use within the _fast_cov method - carry out the dot product and
    subsequent division to generate the covariance values.  For use when there are no
    missing values.
    """
    np.dot(x.T, y, out=destination)
    np.divide(destination, (x.shape[0] - 1), out=destination)


def calculate_non_mask_overlaps(x_mask, y_mask):
    """
    For two mask arrays (x_mask, y_mask - boolean arrays) determine the number of
    entries in common there would be for each entry if their dot product were taken
    """
    x_is_not_nan = 1 * ~x_mask
    y_is_not_nan = 1 * ~y_mask

    r = np.dot(x_is_not_nan.T, y_is_not_nan)
    return r


def _nan_dot_divide(x, y, destination):
    """
    Helper method for use within the _fast_cov method - carry out the dot
    product and subsequent division to generate the covariance values.
    For use when there are missing values.
    """
    np.ma.dot(x.T, y, out=destination)

    divisor = calculate_non_mask_overlaps(x.mask, y.mask) - 1

    np.ma.divide(destination, divisor, out=destination)


def fast_cov(x, y=None, destination=None):
    """
    Calculate the covariance matrix for the columns of x (MxN), or optionally, the
    covariance matrix between the columns of x and and the columns of y (MxP).  (In the
    language of statistics, the columns are variables, the rows are observations).

    Args:
        x
            (np array-like) MxN in shape
        y
            (np array-like) MxP in shape
        destination
            (np array-like) optional location where to store the results as they are
            calculated (e.g. a np memmap of a file)

    Returns:
        (np array-like) array of the covariance values
            for defaults (y=None), shape is NxN
            if y is provided, shape is NxP
    """
    r = _fast_cov(np.mean, _fast_dot_divide, x, y, destination)

    return r


def _fast_cov(mean_method, dot_divide_method, x, y, destination):
    validate_inputs(x, y, destination)

    if y is None:
        y = x

    if destination is None:
        destination = np.zeros((x.shape[1], y.shape[1]))

    mean_x = mean_method(x, axis=0)
    mean_y = mean_method(y, axis=0)

    mean_centered_x = (x - mean_x).astype(destination.dtype)
    mean_centered_y = (y - mean_y).astype(destination.dtype)

    dot_divide_method(mean_centered_x, mean_centered_y, destination)

    return destination


def validate_inputs(x, y, destination):
    error_msg = ""

    if not hasattr(x, "shape"):
        error_msg += f"""x needs to be np array-like but it does not have "shape"
                        attribute - type(x):  {type(x)}\n"""

    if destination is not None and not hasattr(destination, "shape"):
        error_msg += f"""destination needs to be np array-like but it does not have
                         'shape'attribute - type(destination):  {type(destination)}\n"""

    if y is None:
        if destination is not None:
            expected_shape = (x.shape[1], x.shape[1])
            if destination.shape != expected_shape:
                error_msg += f"""x and destination provided, therefore destination must
                                have shape matching number of columns of x but it does
                                not - x.shape:  {x.shape}
                                expected_shape:  {expected_shape}
                                destination.shape:  {destination.shape}\n"""
    else:
        if not hasattr(y, "shape"):
            error_msg += f"""y needs to be np array-like but it does not have "shape"
                            attribute - type(y):  {type(y)}\n"""
        elif x.shape[0] != y.shape[0]:
            error_msg += """the number of rows in the x and y matrices must be the same
                            x.shape:  {x.shape}  y.shape:  {y.shape}\n"""
        elif destination is not None:
            expected_shape = (x.shape[1], y.shape[1])
            if destination.shape != expected_shape:
                error_msg += f"""x, y, and destination provided, therefore destination
                                 must have number of rows matching number of columns of
                                 x and destination needs to have number of columns
                                 matching number of columns of y - x.shape:  {x.shape}
                                 y.shape:  {y.shape}  expected_shape:  {expected_shape}
                                 destination.shape:  {destination.shape}\n"""

    if error_msg != "":
        raise error_msg


def nan_fast_cov(x, y=None, destination=None):
    """
    Calculate the covariance matrix (ignoring nan values) for the columns of x (MxN),
    or optionally, the covariance matrix between the columns of x and and the columns of
    y (MxP).  (In the language of statistics, the columns are variables, the rows are
    observations).

    Args:
        x
            (np array-like) MxN in shape
        y
            (np array-like) MxP in shape
        destination
            (np masked array-like) optional location where to store the results as they
            are calculated (e.g. a np memmap of a file)

    Returns:
        (np array-like) array of the covariance values
            for defaults (y=None), shape is NxN
            if y is provided, shape is NxP
    """
    x_masked = np.ma.array(x, mask=np.isnan(x))

    if y is None:
        y_masked = x_masked
    else:
        y_masked = np.ma.array(y, mask=np.isnan(y))

    dest_was_None = False
    if destination is None:
        destination = np.ma.zeros((x_masked.shape[1], y_masked.shape[1]))
        dest_was_None = True

    r = _fast_cov(np.nanmean, _nan_dot_divide, x_masked, y_masked, destination)

    r[np.isinf(r)] = np.nan

    r = np.ma.filled(r, fill_value=np.nan) if dest_was_None else r

    return r


def fast_corr(x, y=None, destination=None):
    """
    Calculate the pearson correlation matrix for the columns of x (with dimensions MxN),
    or optionally, the pearson correlaton matrix between x and y (with dimensions OxP).
    If destination is provided, put the results there. In the language of statistics the
    columns are the variables and the rows are the observations.

    Args:
        x
            (np array-like) MxN in shape
        y
            (optional, np array-like) OxP in shape.  M (# rows in x) must equal O
            (# rows in y)
        destination
            (np array-like) optional location where to store the results as they are
            calculated (e.g. a np memmap of a file)

    Returns:
        (np array-like) array of the covariance values
            for defaults (y=None), shape is NxN
            if y is provied, shape is NxP
    """
    if y is None:
        y = x

    r = fast_cov(x, y, destination=destination)

    std_x = np.std(x, axis=0, ddof=1)
    std_y = np.std(y, axis=0, ddof=1)

    np.divide(r, std_x[:, np.newaxis], out=r)
    np.divide(r, std_y[np.newaxis, :], out=r)

    return r


def calculate_moments_with_additional_mask(x, mask):
    """
    Calculate the moments (y, y^2, and variance) of the columns of x, excluding masked
    within x, for each of the masking columns in mask.
    Number of rows in x and mask must be the same.

    Args:
        x
            (np.ma.array like)
        mask
            (np array-like boolean)
    """
    non_mask_overlaps = calculate_non_mask_overlaps(x.mask, mask)

    unmask = 1.0 * ~mask

    expect_x = np.ma.dot(x.T, unmask) / non_mask_overlaps
    expect_x = expect_x.T

    expect_x_squared = np.ma.dot(np.power(x, 2.0).T, unmask) / non_mask_overlaps
    expect_x_squared = expect_x_squared.T

    var_x = (
        (expect_x_squared - np.power(expect_x, 2.0))
        * non_mask_overlaps.T
        / (non_mask_overlaps.T - 1)
    )

    return expect_x, expect_x_squared, var_x


def nan_fast_corr(x, y=None, destination=None):
    """
    Calculate the pearson correlation matrix (ignoring nan values) for the columns of x
    (with dimensions MxN), or optionally, the pearson correlaton matrix between x and y
    (with dimensions OxP).  If destination is provided, put the results there. In the
    language of statistics the columns are the variables and the rows are the
    observations.

    Args:
        x
            (np array-like) MxN in shape
        y
            (optional, np array-like) OxP in shape.  M (# rows in x) must equal O
            (# rows in y)
        destination
            (np array-like) optional location where to store the results as they are
            calculated (e.g. a np memmap of a file)

    Returns:
        (np array-like) array of the covariance values
            for defaults (y=None), shape is NxN
            if y is provied, shape is NxP
    """
    x_masked = np.ma.array(x, mask=np.isnan(x))

    if y is None:
        y_masked = x_masked
    else:
        y_masked = np.ma.array(y, mask=np.isnan(y))

    r = nan_fast_cov(x_masked, y_masked, destination=destination)

    # calculate the standard deviation of the columns of each matrix, given the masking from the other
    _, _, var_x = calculate_moments_with_additional_mask(x_masked, y_masked.mask)
    std_x = np.sqrt(var_x)

    _, _, var_y = calculate_moments_with_additional_mask(y_masked, x_masked.mask)
    std_y = np.sqrt(var_y)

    r = r / (std_x.T * std_y)
    r = r.clip(-1, 1)

    return r


def fast_spearman(x, y=None, destination=None):
    """
    Calculate the spearman correlation matrix for the columns of x
    (with dimensions MxN), or optionally, the spearman correlaton matrix between the
    columns of x and the columns of y (with dimensions OxP).  If destination is
    provided, put the results there. In the language of statistics the columns are the
    variables and the rows are the observations.

    Args:
        x
            (np array-like) MxN in shape
        y
            (optional, np array-like) OxP in shape.  M (# rows in x) must equal O
            (# rows in y)
        destination
            (np array-like) optional location where to store the results as they are
            calculated (e.g. a np memmap of a file)

    Returns:
        (np array-like) array of the covariance values
            for defaults (y=None), shape is NxN
            if y is provied, shape is NxP
    """
    r = _fast_spearman(fast_corr, x, y, destination)
    return r


def _fast_spearman(corr_method, x, y, destination):
    """
    Internal method for calculating spearman correlation, allowing subsititution of
    methods for calculationg correlation (corr_method), allowing to choose methods that
    are fast (fast_corr) or tolerant of nan's (nan_fast_corr) to be used
    """
    logger.debug("x.shape:  {}".format(x.shape))
    if hasattr(y, "shape"):
        logger.debug("y.shape:  {}".format(y.shape))

    x_ranks = pd.DataFrame(x).rank(method="average", na_option="keep").values
    logger.debug(
        "some min and max ranks of x_ranks:\n{}\n{}".format(
            np.min(x_ranks[:10], axis=0), np.max(x_ranks[:10], axis=0)
        )
    )

    y_ranks = (
        pd.DataFrame(y).rank(method="average", na_option="keep").values
        if y is not None
        else None
    )

    return corr_method(x_ranks, y_ranks, destination=destination)


def nan_fast_spearman(x, y=None, destination=None):
    """
    Calculate the spearman correlation matrix (ignoring nan values) for the columns of x
    (with dimensions MxN), or optionally, the spearman correlation matrix between the
    columns of x and the columns of y (with dimensions OxP).  If destination is
    provided, put the results there. In the language of statistics the columns are the
    variables and the rows are the observations. Note that the ranks will be slightly
    miscalculated in the masked situations leading to slight errors in the spearman rho
    alue.

    Args:
        x
            (np array-like) MxN in shape
        y
            (optional, np array-like) OxP in shape.  M (# rows in x) must equal O
            (# rows in y)
        destination
            (np array-like) optional location where to store the results as they are
            calculated (e.g. a np memmap of a file)

    Returns:
        (np array-like) array of the covariance values
            for defaults (y=None), shape is NxN
            if y is provied, shape is NxP
    """
    r = _fast_spearman(nan_fast_corr, x, y, destination)
    return r
