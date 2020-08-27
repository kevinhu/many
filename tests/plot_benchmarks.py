import argparse
import importlib
from pathlib import Path

import config
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SUBPLOT_COLS = 3
SUBPLOT_ROWS = 3

p = Path("./stats_benchmark_params").glob("*.py")
submodules = [x.stem for x in p if x.is_file()]
submodules = [x for x in submodules if x != "__init__"]

fig = plt.figure(figsize=(15, 10))

position = 1

for submodule in submodules:

    benchmarks_df = pd.read_csv(
        config.BENCHMARK_DATA_DIR / f"{submodule}.txt", sep="\t", index_col=0
    )

    benchmarks_df["num_comparisons"] = (
        benchmarks_df["a_num_cols"] * benchmarks_df["b_num_cols"]
    )

    benchmarks_df["method_id"] = (
        benchmarks_df["method"] + "-" + benchmarks_df["method_kwargs"]
    )

    method_ids = list(set(benchmarks_df["method_id"]))

    for method_id in method_ids:

        method_benchmarks = benchmarks_df[
            benchmarks_df["method_id"] == method_id
        ]

        ax = fig.add_subplot(SUBPLOT_ROWS, SUBPLOT_COLS, position)

        ax.scatter(
            method_benchmarks["num_comparisons"],
            method_benchmarks["base_times"],
            c="#222831",
            label="Naive",
        )
        ax.scatter(
            method_benchmarks["num_comparisons"],
            method_benchmarks["method_times"],
            c="#71c9ce",
            label="Vectorized",
        )

        plt.legend()

        for x, y, ratio in zip(
            method_benchmarks["num_comparisons"],
            method_benchmarks["method_times"],
            method_benchmarks["ratios"],
        ):
            plt.text(x * 1.1, y, f"×{int(ratio)}", va="center")

        ax.set_xscale("log")
        ax.set_yscale("log")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.set_xlabel("Number of compared variables")
        ax.set_ylabel("Speed, seconds")

        # plt.savefig(
        #     config.BENCHMARK_PLOTS_DIR / f"{submodule}_{method_id}.png",
        #     dpi=256,
        #     bbox_inches="tight",
        #     transparent=True,
        # )

        # plt.clf()

        position += 1

plt.subplots_adjust(hspace=0.4, wspace=0.4)

plt.savefig(
    config.BENCHMARK_PLOTS_DIR / "all_benchmarks.png",
    dpi=256,
    bbox_inches="tight",
)
