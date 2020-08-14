import many

b_num_cols = [10, 100, 1000]

params = []

for b_num_col in b_num_cols:

    params.append(
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr,
            1000,
            100,
            b_num_col,
            "continuous",
            "binary",
            False,
            False,
            {"method": "pearson", "melt": False},
            ["cirrs", "pvals"],
            True,
        ]
    )

for b_num_col in b_num_cols:

    params.append(
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr,
            1000,
            100,
            b_num_col,
            "continuous",
            "binary",
            False,
            False,
            {"method": "spearman", "melt": False},
            ["cirrs", "pvals"],
            True,
        ]
    )
