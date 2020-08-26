import many

b_num_cols = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

params = []

for b_num_col in b_num_cols:

    params.append(
        [
            many.stats.mat_fisher_naive,
            many.stats.mat_fisher,
            1000,
            100,
            b_num_col,
            "binary",
            "binary",
            False,
            False,
            {"melt": False},
            ["corrs", "pvals"],
            True,
        ]
    )

for b_num_col in b_num_cols:

    params.append(
        [
            many.stats.mat_fisher_naive,
            many.stats.mat_fisher_nan,
            1000,
            100,
            b_num_col,
            "binary",
            "binary",
            True,
            True,
            {"melt": False},
            ["corrs", "pvals"],
            True,
        ]
    )
