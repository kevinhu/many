import numpy as np

import sys
sys.path.append("../")

import many

# preset dimensions of A and B

n_samples = 100
A_mult_n = 10
B_mult_n = 25

# mat_corrs with full-size matrices

print("Testing mat_corrs (method = pearson):")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="continuous", B_type="continuous", nan=False
)

v_corrs, v_pvals = many.stats.mat_corrs(A_test, B_test, method="pearson")
n_corrs, n_pvals = many.stats.mat_corrs_naive(A_test, B_test, method="pearson")

print("Max corr deviation: {0:.1E}".format((np.abs(v_corrs - n_corrs)).max().max()))
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_pvals - n_pvals)).max().max()
    )
)
print()

print("Testing mat_corrs (method = spearman):")

v_corrs, v_pvals = many.stats.mat_corrs(A_test, B_test, method="spearman")
n_corrs, n_pvals = many.stats.mat_corrs_naive(A_test, B_test, method="spearman")

print("Max corr deviation: {0:.1E}".format((np.abs(v_corrs - n_corrs)).max().max()))
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_pvals - n_pvals)).max().max()
    )
)
print()

# mat_corrs with 1-dimensional A matrix

print("Testing mat_corrs (method = pearson, 1-dimensional A):")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="continuous", B_type="continuous", nan=False
)

A_test = A_test.iloc[:, 0]

v_res = many.stats.mat_corrs(A_test, B_test, method="pearson")
n_res = many.stats.mat_corrs_naive(A_test, B_test, method="pearson")

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["corr"] - n_res["corr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()

print("Testing mat_corrs (method = spearman, 1-dimensional A):")

v_res = many.stats.mat_corrs(A_test, B_test, method="spearman")
n_res = many.stats.mat_corrs_naive(A_test, B_test, method="spearman")

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["corr"] - n_res["corr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()

# mat_corrs with 1-dimensional B matrix

print("Testing mat_corrs (method = pearson, 1-dimensional B):")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="continuous", B_type="continuous", nan=False
)

B_test = B_test.iloc[:, 0]

v_res = many.stats.mat_corrs(A_test, B_test, method="pearson")
n_res = many.stats.mat_corrs_naive(A_test, B_test, method="pearson")

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["corr"] - n_res["corr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()

print("Testing mat_corrs (method = spearman, 1-dimensional B):")

v_res = many.stats.mat_corrs(A_test, B_test, method="spearman")
n_res = many.stats.mat_corrs_naive(A_test, B_test, method="spearman")

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["corr"] - n_res["corr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()

# mat_corrs_nan

print("Testing mat_corrs_nan (method = pearson):")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="continuous", B_type="continuous", nan=True
)

A_test = A_test.iloc[:, 0]

v_res = many.stats.mat_corrs_nan(A_test, B_test, method="pearson")
n_res = many.stats.mat_corrs_naive(A_test, B_test, method="pearson")

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["corr"] - n_res["corr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()


print("Testing mat_corrs_nan (method = spearman):")

v_res = many.stats.mat_corrs_nan(A_test, B_test, method="spearman")
n_res = many.stats.mat_corrs_naive(A_test, B_test, method="spearman")

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["corr"] - n_res["corr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()

# mat_mwus with full-size matrices

print("Testing mat_mwus:")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="continuous", B_type="categorical", nan=False
)

v_corrs, v_pvals = many.stats.mat_mwus(A_test, B_test)
n_corrs, n_pvals = many.stats.mat_mwus_naive(A_test, B_test)

print("Max corr deviation: {0:.1E}".format((np.abs(v_corrs - n_corrs)).max().max()))
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_pvals - n_pvals)).max().max()
    )
)
print()

# mat_mwus with 1-dimensional A matrix

print("Testing mat_mwus (1-dimensional A):")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="continuous", B_type="categorical", nan=False
)

A_test = A_test[0]

v_res = many.stats.mat_mwus(A_test, B_test)
n_res = many.stats.mat_mwus_naive(A_test, B_test)

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["corr"] - n_res["corr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()

# mat_mwus with 1-dimensional B matrix

print("Testing mat_mwus (1-dimensional B):")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="continuous", B_type="categorical", nan=False
)

B_test = B_test[0]

v_res = many.stats.mat_mwus(A_test, B_test)
n_res = many.stats.mat_mwus_naive(A_test, B_test)

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["corr"] - n_res["corr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()

# mat_fishers with full-size matrices

print("Testing mat_fishers:")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="categorical", B_type="categorical", nan=False
)

v_corrs, v_pvals = many.stats.mat_fishers(A_test, B_test)
n_corrs, n_pvals = many.stats.mat_fishers_naive(A_test, B_test)

print(
    "Max odds ratio deviation: {0:.1E}".format((np.abs(v_corrs - n_corrs)).max().max())
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_pvals - n_pvals)).max().max()
    )
)
print()

# mat_fishers with 1-dimensional A matrix

print("Testing mat_fishers (1-dimensional A):")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="categorical", B_type="categorical", nan=False
)

A_test = A_test[0]

v_res = many.stats.mat_fishers(A_test, B_test)
n_res = many.stats.mat_fishers_naive(A_test, B_test)

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["oddsr"] - n_res["oddsr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()

# mat_fishers with 1-dimensional B matrix

print("Testing mat_fishers (1-dimensional B):")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="categorical", B_type="categorical", nan=False
)

B_test = B_test[0]

v_res = many.stats.mat_fishers(A_test, B_test)
n_res = many.stats.mat_fishers_naive(A_test, B_test)

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["oddsr"] - n_res["oddsr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()

# mat_fishers with full-size matrices

print("Testing mat_fishers:")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="categorical", B_type="categorical", nan=True
)

v_corrs, v_pvals = many.stats.mat_fishers_nan(A_test, B_test)
n_corrs, n_pvals = many.stats.mat_fishers_naive(A_test, B_test)

print(
    "Max odds ratio deviation: {0:.1E}".format((np.abs(v_corrs - n_corrs)).max().max())
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_pvals - n_pvals)).max().max()
    )
)
print()

# mat_fishers with 1-dimensional A matrix

print("Testing mat_fishers (1-dimensional A):")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="categorical", B_type="categorical", nan=True
)

A_test = A_test[0]

v_res = many.stats.mat_fishers_nan(A_test, B_test)
n_res = many.stats.mat_fishers_naive(A_test, B_test)

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["oddsr"] - n_res["oddsr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()

# mat_fishers with 1-dimensional B matrix

print("Testing mat_fishers (1-dimensional B):")

A_test, B_test = many.stats.generate_test(
    n_samples, A_mult_n, B_mult_n, A_type="categorical", B_type="categorical", nan=True
)

B_test = B_test[0]

v_res = many.stats.mat_fishers_nan(A_test, B_test)
n_res = many.stats.mat_fishers_naive(A_test, B_test)

print(
    "Max corr deviation: {0:.1E}".format(
        (np.abs(v_res["oddsr"] - n_res["oddsr"])).max().max()
    )
)
print(
    "Max -log10(pval) deviation: {0:.1E}".format(
        (np.abs(v_res["pval"] - n_res["pval"])).max().max()
    )
)
print()