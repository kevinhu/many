import argparse
import importlib
from pathlib import Path

import config
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

p = Path("./stats_benchmark_params").glob("*.py")
all_submodules = [x.stem for x in p if x.is_file()]
all_submodules = [x for x in all_submodules if x != "__init__"]

parser = argparse.ArgumentParser(description="Benchmark statistical methods")
parser.add_argument(
    "-s",
    "--submodules",
    nargs="+",
    help=f"submodules to benchmark: any of {all_submodules} (leave empty for all)",
    default=None,
    required=False,
)

args = parser.parse_args()
submodules = args.submodules

if submodules is None:

    submodules = all_submodules

for submodule in submodules:

    benchmarks_df = pd.read_csv(
        config.BENCHMARK_DATA_DIR / f"{submodule}.txt", sep="\t", index_col=0
    )

    methods = list(set(benchmarks_df["method"]))
    method_kwargs = list(set(benchmarks_df["method_kwargs"]))

    for method in methods:

        for method_kwarg in method_kwargs:

            method_benchmarks = benchmarks_df[
                benchmarks_df["method"] == method
            ]
            method_benchmarks = method_benchmarks[
                method_benchmarks["method_kwargs"] == method_kwarg
            ]

            method_benchmarks["num_comparisons"] = (
                method_benchmarks["a_num_cols"]
                * method_benchmarks["b_num_cols"]
            )

            ax = plt.subplot(111)

            ax.scatter(
                method_benchmarks["num_comparisons"],
                method_benchmarks["base_times"],
            )
            ax.scatter(
                method_benchmarks["num_comparisons"],
                method_benchmarks["method_times"],
            )

            ax.set_xscale("log")
            ax.set_yscale("log")

            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            plt.savefig(
                config.BENCHMARK_PLOTS_DIR
                / f"{submodule}_{method}_{method_kwarg}.png",
                dpi=256,
                bbox_inches="tight",
                transparent=True,
            )

            plt.clf()
