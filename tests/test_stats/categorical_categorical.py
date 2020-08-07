from itertools import product

import many

a_col_counts = [1, 10]
b_col_counts = [1, 25]
a_types = ["categorical", "zero"]
b_types = ["categorical", "zero"]

mat_fisher_param_combos = product(a_col_counts, b_col_counts, a_types, b_types)

params = []

for a_col_count, b_col_count, a_type, b_type in mat_fisher_param_combos:

    if a_col_count == 1 or b_col_count == 1:

        explicit_params = [
            # mat_fisher, single-dim comparison
            [
                many.stats.mat_fisher_naive,
                many.stats.mat_fisher,
                100,
                a_col_count,
                b_col_count,
                a_type,
                b_type,
                False,
                False,
                {},
                ["merged"],
            ],
            # mat_fisher_nan, single-dim comparison
            [
                many.stats.mat_fisher_naive,
                many.stats.mat_fisher_nan,
                100,
                a_col_count,
                b_col_count,
                a_type,
                b_type,
                True,
                True,
                {},
                ["merged"],
            ],
        ]

    else:

        explicit_params = [
            # mat_fisher, full-size comparison
            [
                many.stats.mat_fisher_naive,
                many.stats.mat_fisher,
                100,
                a_col_count,
                b_col_count,
                a_type,
                b_type,
                False,
                False,
                {},
                ["corrs", "pvals"],
            ],
            # mat_fisher_nan, single-dim comparison
            [
                many.stats.mat_fisher_naive,
                many.stats.mat_fisher_nan,
                100,
                a_col_count,
                b_col_count,
                a_type,
                b_type,
                True,
                True,
                {},
                ["corrs", "pvals"],
            ],
        ]

    params = params + explicit_params
