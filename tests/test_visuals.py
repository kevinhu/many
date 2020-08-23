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
