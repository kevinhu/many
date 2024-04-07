# import argparse
# from pathlib import Path

# import importlib

import config
import matplotlib.pyplot as plt
import numpy as np
import utils

import many

DPI = 256

# p = Path("./test_stats").glob("*.py")
# all_submodules = [x.stem for x in p if x.is_file()]
# all_submodules = [x for x in all_submodules if x != "__init__"]

# parser = argparse.ArgumentParser(description="Test statistical methods")
# parser.add_argument(
#     "-s",
#     "--submodules",
#     nargs="+",
#     help=f"submodules to test: any of {all_submodules} (leave empty for all)",
#     default=None,
#     required=False,
# )

print(many.visuals.as_si(0.0000032493, decimals=4))
print(many.visuals.as_si(5.493491349e30, decimals=8))


print(many.visuals.colorize([0, 0, 0, 0, 1, 1, 1, 2, 2], cmap="Blues"))
print(
    many.visuals.colorize(
        ["a", "b", "c", "a", "a", "b", "c", "d", "b", "d", "a"],
        cmap="Blues",
    )
)

# ------------
# scatter_grid
# ------------

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=4,
    b_num_cols=1,
    a_type="continuous",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)
a[1] = a[0] + np.random.normal(size=100)
a[2] = a[1] + np.random.normal(size=100)
a[3] = a[2] + np.random.normal(size=100)

many.visuals.scatter_grid(a)
plt.savefig(config.PLOTS_DIR / "scatter_grid.png", bbox_inches="tight", dpi=DPI)
plt.clf()

# ----------
# regression
# ----------

plt.figure(figsize=(4, 4))
ax = plt.subplot(111)

x = np.random.normal(size=1000)
y = x + np.random.normal(size=1000)

many.visuals.regression(x, y, method="pearson")
plt.savefig(config.PLOTS_DIR / "regression_pearson.png", bbox_inches="tight", dpi=DPI)
plt.clf()

# -----------
# dense_plot
# -----------

plt.figure(figsize=(4, 4))
ax = plt.subplot(111)

x = np.random.normal(size=1000)
y = x + np.random.normal(size=1000)

many.visuals.dense_plot(
    x, y, text_adjust=False, ax=ax, colormap="Blues", cmap_offset=0.1
)
plt.savefig(
    config.PLOTS_DIR / "dense_plot_default.png",
    bbox_inches="tight",
    dpi=DPI,
)
plt.clf()

# -----------------
# dense_regression
# -----------------

plt.figure(figsize=(4, 4))
ax = plt.subplot(111)

x = np.random.normal(size=1000)
y = x + np.random.normal(size=1000)

many.visuals.dense_regression(x, y, method="pearson", colormap="Blues", cmap_offset=0.1)
plt.savefig(
    config.PLOTS_DIR / "dense_regression_pearson.png",
    bbox_inches="tight",
    dpi=DPI,
)
plt.clf()

# ---------
# two_dists
# ---------

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="binary",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

b[0] = b[0] + a[0]

many.visuals.two_dists(a[0], b[0], method="t_test", summary_type="box", stripplot=True)
plt.savefig(config.PLOTS_DIR / "two_dists_t_test_box.png", bbox_inches="tight", dpi=DPI)
plt.clf()

# -----------
# multi_dists
# -----------

plt.figure(figsize=(8, 4))
ax = plt.subplot(111)

a, b = utils.generate_test(
    n_samples=500,
    a_num_cols=1,
    b_num_cols=1,
    a_type="continuous",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

a[0] = np.random.normal(size=500)
b = (b * 25).astype(int)

many.visuals.multi_dists(a[0], b[0], count_cutoff=0, summary_type="box", ax=ax)
plt.savefig(config.PLOTS_DIR / "multi_dists_box.png", bbox_inches="tight", dpi=DPI)
plt.clf()

# -------------
# roc_auc_curve
# -------------

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="binary",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

b[0] = b[0] + a[0] * np.random.random(size=100)

many.visuals.roc_auc_curve(a[0], b[0])
plt.savefig(config.PLOTS_DIR / "roc_auc_curve.png", bbox_inches="tight", dpi=DPI)
plt.clf()

# --------
# pr_curve
# --------

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="binary",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

b[0] = b[0] + a[0] * np.random.random(size=100)

many.visuals.pr_curve(a[0], b[0])
plt.savefig(config.PLOTS_DIR / "pr_curve.png", bbox_inches="tight", dpi=DPI)
plt.clf()

# --------------
# binary_metrics
# --------------

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="binary",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

b[0] = b[0] + a[0] * np.random.random(size=100)

many.visuals.binary_metrics(a[0], b[0])
plt.savefig(config.PLOTS_DIR / "binary_metrics.png", bbox_inches="tight", dpi=DPI)
plt.clf()

# ------------------
# binary_contingency
# ------------------

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="binary",
    b_type="binary",
    a_nan=False,
    b_nan=False,
)

many.visuals.binary_contingency(a[0], b[0])
plt.savefig(
    config.PLOTS_DIR / "binary_contingency_default.png",
    bbox_inches="tight",
    dpi=DPI,
)
plt.clf()
