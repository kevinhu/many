from itertools import product

import many

a_types = ["continuous", "zero"]
b_types = ["categorical", "zero"]
effects = ["rank_biserial"]

mat_mwu_param_combos = product(a_types, b_types, effects)

params = []

for a_type, b_type, effect in mat_mwu_param_combos:

    explicit_params = [  # mat_mwu, full-size comparison
        [
            many.stats.mat_mwu_naive,
            many.stats.mat_mwu,
            100,
            10,
            25,
            a_type,
            b_type,
            False,
            False,
            {"effect": effect},
            ["corrs", "pvals"],
        ],
        # mat_mwu, 1-d a_mat
        [
            many.stats.mat_mwu_naive,
            many.stats.mat_mwu,
            100,
            1,
            25,
            a_type,
            b_type,
            False,
            False,
            {"effect": effect},
            ["merged"],
        ],
        # mat_mwu, 1-d b_mat
        [
            many.stats.mat_mwu_naive,
            many.stats.mat_mwu,
            100,
            10,
            1,
            a_type,
            b_type,
            False,
            False,
            {"effect": effect},
            ["merged"],
        ],
        # mat_mwu, 1-d both
        [
            many.stats.mat_mwu_naive,
            many.stats.mat_mwu,
            100,
            1,
            1,
            a_type,
            b_type,
            False,
            False,
            {"effect": effect},
            ["merged"],
        ],
    ]

    params = params + explicit_params
