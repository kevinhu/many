import many

b_num_cols = [10, 100, 1000]

params = []

for b_num_col in b_num_cols:

    params.append(
        [
            many.stats.mat_mwu_naive,
            many.stats.mat_mwu,
            1000,
            100,
            b_num_col,
            "continuous",
            "binary",
            False,
            False,
            {"effect": "rank_biserial", "melt": False},
            ["effects", "pvals"],
            True,
        ]
    )
