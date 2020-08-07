from itertools import product

import many

a_types = ["continuous", "zero"]
b_types = ["continuous", "zero"]
methods = ["pearson", "spearman"]

mat_corr_param_combos = product(a_types, b_types, methods)

params = []

for a_type, b_type, corr_method in mat_corr_param_combos:

    explicit_params = [
        # mat_corr, full-size comparison
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr,
            100,
            10,
            25,
            a_type,
            b_type,
            False,
            False,
            {"method": corr_method},
            ["corrs", "pvals"],
        ],
        # mat_corr, 1-d a_mat
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr,
            100,
            1,
            25,
            a_type,
            b_type,
            False,
            False,
            {"method": corr_method},
            ["merged"],
        ],
        # mat_corr, 1-d b_mat
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr,
            100,
            10,
            1,
            a_type,
            b_type,
            False,
            False,
            {"method": corr_method},
            ["merged"],
        ],
        # mat_corr_nan, 1-d both
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr,
            100,
            1,
            1,
            a_type,
            b_type,
            False,
            False,
            {"method": corr_method},
            ["merged"],
        ],
        # mat_corr_nan, no nans
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr_nan,
            100,
            1,
            100,
            a_type,
            b_type,
            False,
            False,
            {"method": corr_method},
            ["merged"],
        ],
        # mat_corr_nan, nans in a
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr_nan,
            100,
            1,
            100,
            a_type,
            b_type,
            True,
            False,
            {"method": corr_method},
            ["merged"],
        ],
        # mat_corr_nan, nans in b
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr_nan,
            100,
            1,
            100,
            a_type,
            b_type,
            False,
            True,
            {"method": corr_method},
            ["merged"],
        ],
        # mat_corr_nan, nans in both
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr_nan,
            100,
            1,
            100,
            a_type,
            b_type,
            True,
            True,
            {"method": corr_method},
            ["merged"],
        ],
    ]

    params = params + explicit_params
