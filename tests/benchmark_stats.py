import argparse
from pathlib import Path
import importlib

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

    for param_set in params:

        compare(*param_set)