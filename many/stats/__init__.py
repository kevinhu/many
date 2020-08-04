from .categorical_categorical import (
    mat_fishers_naive,
    fisher_arr,
    mat_fishers,
    mat_fishers_nan,
)

from .continuous_categorical import mat_mwus_naive, mat_mwus

from .continuous_continuous import (
    mat_corrs_naive,
    mat_corrs,
    mat_corrs_nan,
    mat_corrs_subtyped,
)

from .utils import generate_test, test_plot_dataframes, test_plot_series
