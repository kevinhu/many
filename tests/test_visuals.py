# import argparse
# from pathlib import Path

# import importlib

import config
import many
import matplotlib.pyplot as plt
import utils

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
        ["a", "b", "c", "a", "a", "b", "c", "d", "b", "d", "a"], cmap="Blues"
    )
)

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="continuous",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

many.visuals.dense_plot(a[0], b[0], text_adjust=False)
plt.savefig(config.PLOTS_DIR / "dense_plot_default.pdf", bbox_inches="tight")
plt.clf()

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
    config.PLOTS_DIR / "binary_contingency_default.pdf", bbox_inches="tight"
)
plt.clf()

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="continuous",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

many.visuals.regression(a[0], b[0], method="pearson")
plt.savefig(config.PLOTS_DIR / "regression_pearson.pdf", bbox_inches="tight")
plt.clf()

a, b = utils.generate_test(
    n_samples=1000,
    a_num_cols=1,
    b_num_cols=1,
    a_type="continuous",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

many.visuals.dense_regression(a[0], b[0], method="pearson")
plt.savefig(
    config.PLOTS_DIR / "dense_regression_pearson.pdf", bbox_inches="tight"
)
plt.clf()

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="binary",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

many.visuals.two_dists(a[0], b[0], method="t_test", summary_type="box")
plt.savefig(config.PLOTS_DIR / "two_dists_t_test_box.pdf", bbox_inches="tight")
plt.clf()

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="continuous",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

b = (b * 10).astype(int)

many.visuals.multi_dists(a[0], b[0], count_cutoff=0, summary_type="box")
plt.savefig(config.PLOTS_DIR / "multi_dists_box.pdf", bbox_inches="tight")
plt.clf()

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="binary",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

many.visuals.roc_auc_curve(a[0], b[0])
plt.savefig(config.PLOTS_DIR / "roc_auc_curve.pdf", bbox_inches="tight")
plt.clf()

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="binary",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

many.visuals.pr_curve(a[0], b[0])
plt.savefig(config.PLOTS_DIR / "pr_curve.pdf", bbox_inches="tight")
plt.clf()

a, b = utils.generate_test(
    n_samples=100,
    a_num_cols=1,
    b_num_cols=1,
    a_type="binary",
    b_type="continuous",
    a_nan=False,
    b_nan=False,
)

many.visuals.binary_metrics(a[0], b[0])
plt.savefig(config.PLOTS_DIR / "binary_metrics.pdf", bbox_inches="tight")
plt.clf()
