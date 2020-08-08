from itertools import product

import many

a_col_counts = [1, 10]
b_col_counts = [1, 25]
a_types = ["categorical", "zero"]
b_types = ["categorical", "zero"]
melts = [False, True]

mat_fisher_param_combos = product(a_col_counts, b_col_counts, a_types, b_types, melts)

params = []

for a_col_count, b_col_count, a_type, b_type, melt in mat_fisher_param_combos:

    if melt:

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
                {"melt":melt},
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
                {"melt":melt},
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
                {"melt":melt},
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
                {"melt":melt},
                ["corrs", "pvals"],
            ],
        ]

    params = params + explicit_params
