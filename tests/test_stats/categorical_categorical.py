from itertools import product

import many

a_col_counts = [1, 10]
b_col_counts = [1, 25]
a_types = ["binary", "zero"]
b_types = ["binary", "zero"]
melts = [False, True]
pseudocounts = [0, 1]

mat_fisher_param_combos = product(
    a_col_counts, b_col_counts, a_types, b_types, melts, pseudocounts
)

params = []

for (
    a_col_count,
    b_col_count,
    a_type,
    b_type,
    melt,
    pseudocount,
) in mat_fisher_param_combos:

    if melt:

        output_names = ["melted"]

    else:

        output_names = ["corrs", "pvals"]

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
            {"melt": melt, "pseudocount": pseudocount},
            output_names,
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
            {"melt": melt, "pseudocount": pseudocount},
            output_names,
        ],
    ]

    params = params + explicit_params
