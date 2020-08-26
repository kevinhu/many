import argparse
import importlib
import inspect
from pathlib import Path

import config
import pandas as pd
from utils import compare

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

    params = importlib.import_module(
        f"stats_benchmark_params.{submodule}"
    ).params

    base_times = []
    method_times = []
    ratios = []

    for param_set in params:

        base_result, result, benchmark_results = compare(*param_set)

        base_time = benchmark_results["base_time"]
        method_time = benchmark_results["method_time"]

        ratio = base_time / method_time

        base_times.append(base_time)
        method_times.append(method_time)
        ratios.append(ratio)

    benchmarks_df = pd.DataFrame(
        params, columns=inspect.getfullargspec(compare)[0]
    )

    benchmarks_df["base_method"] = benchmarks_df["base_method"].apply(
        lambda x: x.__name__
    )
    benchmarks_df["method"] = benchmarks_df["method"].apply(
        lambda x: x.__name__
    )

    benchmarks_df["base_times"] = base_times
    benchmarks_df["method_times"] = method_times
    benchmarks_df["ratios"] = ratios

    benchmarks_df.to_csv(
        config.BENCHMARK_DATA_DIR / f"{submodule}.txt", sep="\t"
    )
