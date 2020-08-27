import many

b_num_cols = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

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
            ["corrs", "pvals"],
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
            ["corrs", "pvals"],
            True,
        ]
    )

for b_num_col in b_num_cols:

    params.append(
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr_nan,
            1000,
            1,
            b_num_col * 100,
            "continuous",
            "binary",
            False,
            False,
            {"method": "pearson", "melt": True},
            ["melted"],
            True,
        ]
    )

for b_num_col in b_num_cols:

    params.append(
        [
            many.stats.mat_corr_naive,
            many.stats.mat_corr_nan,
            1000,
            1,
            b_num_col * 100,
            "continuous",
            "binary",
            False,
            False,
            {"method": "spearman", "melt": True},
            ["melted"],
            True,
        ]
    )
