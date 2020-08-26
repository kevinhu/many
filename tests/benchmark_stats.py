import argparse
import importlib
from pathlib import Path
import inspect

import pandas as pd

from utils import compare

p = Path("./benchmark_stats").glob("*.py")
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

    params = importlib.import_module(f"benchmark_stats.{submodule}").params

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

        break

    # print(inspect.getfullargspec(compare))

    benchmarks_df = pd.DataFrame(
        params, columns=inspect.getfullargspec(compare)[0]
    )
    benchmarks_df["base_times"] = base_times
    benchmarks_df["method_times"] = method_times
    benchmarks_df["ratios"] = ratios

    print(benchmarks_df)
