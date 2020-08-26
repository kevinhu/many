import time
from typing import List

import many
import numpy as np
import pandas as pd
from config import TOLERANCE


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class OutputMismatchError(Exception):
    pass


class TypeMismatchError(Exception):
    pass


class ShapeMismatchError(Exception):
    pass


def generate_test(
    n_samples: int,
    a_num_cols: int,
    b_num_cols: int,
    a_type: str,
    b_type: str,
    a_nan: bool,
    b_nan: bool,
):
    """
    Generates randomly initialized matrix pairs for testing and benchmarking.

    Parameters
    ----------
    n_samples: int
        Number of samples per matrix (equivalent to number of rows)
    a_num_cols: int
        Number of variables for a_mat (equivalent to number of columns in a_mat)
    b_num_cols: int
        Number of variables for b_mat (equivalent to number of columns in b_mat)
    a_type: string, "continuous", "binary", or "zero"
        Type of variables in a_mat
    b_type: string, "continuous", "binary", or "zero"
        Type of variables in b_mat
    a_nan: boolean
        whether or not to simulate missing (NaN) values in a_mat
    b_nan: boolean
        whether or not to simulate missing (NaN) values in b_mat

    Returns
    -------
    a_test, b_test: test DataFrames
    """

    if a_type == "binary":
        a_test = np.random.randint(0, 2, (n_samples, a_num_cols))
    elif a_type == "continuous":
        a_test = np.random.random((n_samples, a_num_cols))
    elif a_type == "zero":
        a_test = np.zeros((n_samples, a_num_cols))
    else:
        raise ValueError("'a_type' must be 'binary', 'continuous', or 'zero'")

    if b_type == "binary":
        b_test = np.random.randint(0, 2, (n_samples, b_num_cols))
    elif b_type == "continuous":
        b_test = np.random.random((n_samples, b_num_cols))
    elif b_type == "zero":
        b_test = np.zeros((n_samples, b_num_cols))
    else:
        raise ValueError("'b_type' must be 'binary', 'continuous', or 'zero'")

    if a_nan:

        a_test = a_test.astype(np.float64)

        A_nan = np.random.randint(0, 2, (n_samples, a_num_cols))

        a_test[A_nan == 1] = np.nan

    if b_nan:

        b_test = b_test.astype(np.float64)

        B_nan = np.random.randint(0, 2, (n_samples, b_num_cols))

        b_test[B_nan == 1] = np.nan

    a_test = pd.DataFrame(a_test)
    b_test = pd.DataFrame(b_test)

    return a_test, b_test


def compare(
    base_method,
    method,
    num_samples: int,
    a_num_cols: int,
    b_num_cols: int,
    a_type: str,
    b_type: str,
    a_nan: bool,
    b_nan: bool,
    method_kwargs,
    output_names: List[str],
    report_benchmark: bool,
):

    """
    General test handler for comparing two methods.

    Parameters
    ----------
    base_method: function
        first method to compare
    method: function
        second method to compare
    num_samples: int
        number of samples per column (i.e. row count)
    a_num_cols: int
        number of columns in a_mat
    b_num_cols: int
        number of columns in b_mat
    a_type: string, "continuous", "binary", or "zero"
        Type of variables in a_mat
    b_type: string, "continuous", "binary", or "zero"
        Type of variables in b_mat
    a_nan: boolean
        whether or not to simulate missing (NaN) values in a_mat
    b_nan: boolean
        whether or not to simulate missing (NaN) values in b_mat
    method_kwargs: dict
        arguments to pass to base_method and method
    output_names: list of strings
        names of the output variables (for logging)
    report_benchmark: bool
        whether to print comparison of runtimes

    """

    # announce comparison methods
    print(
        f"Comparing {bcolors.BOLD}{bcolors.HEADER}{method.__name__}{bcolors.ENDC} ",
        end="",
    )

    # announce method parameters
    print(
        f" with {bcolors.BOLD}{bcolors.HEADER}{base_method.__name__}{bcolors.ENDC}"
    )

    args_string = ", ".join(
        f"{bcolors.BOLD}{key}{bcolors.ENDC} = {value}"
        for key, value in [
            ["num_samples", num_samples],
            ["a_num_cols", a_num_cols],
            ["b_num_cols", b_num_cols],
            ["a_type", a_type],
            ["b_type", b_type],
            ["a_nan", a_nan],
            ["b_nan", b_nan],
        ]
    )
    print(f"\twith {args_string}")

    # announce kwargs
    kwargs_string = ", ".join(
        f"{bcolors.BOLD}{bcolors.OKBLUE}{key}{bcolors.ENDC} = {value}"
        for key, value in method_kwargs.items()
    )
    print(f"\tand {kwargs_string}")

    # generate test cases
    a_test, b_test = generate_test(
        num_samples, a_num_cols, b_num_cols, a_type, b_type, a_nan, b_nan
    )

    # compute outputs
    start = time.time()
    base_result = base_method(a_test, b_test, **method_kwargs)
    end = time.time()
    base_time = end - start

    start = time.time()
    result = method(a_test, b_test, **method_kwargs)
    end = time.time()
    method_time = end - start

    if report_benchmark:
        print(f"\tNaive speed: {bcolors.BOLD}{base_time:.2f}s{bcolors.ENDC}")
        print(
            f"\tVectorized speed: {bcolors.BOLD}{method_time:.2f}s{bcolors.ENDC}"
        )
        print(
            f"\tSpeedup: {bcolors.BOLD}{base_time/method_time:.2f}x{bcolors.ENDC}"
        )

    benchmark_results = {"base_time": base_time, "method_time": method_time}

    # cast single outputs to lists
    if len(output_names) == 1:

        base_result = [base_result]
        result = [result]

    if len(base_result) != len(result):
        raise OutputMismatchError("\tOutputs have different lengths")

    # check outputs one-by-one
    for i, output_name in enumerate(output_names):

        base_output = base_result[i]
        output = result[i]

        output_name = output_names[i]

        print(
            f"\tChecking {bcolors.BOLD}{output_name}{bcolors.ENDC} outputs: ",
            end="",
        )

        if type(base_output) != type(output):
            raise TypeMismatchError("Outputs have different types")

        if base_output.shape != output.shape:
            raise ShapeMismatchError(
                "Outputs have different shapes",
                base_output.shape,
                output.shape,
            )

        max_deviation = np.abs(base_output - output).max().max()
        max_deviation_str = "{0:.3E}".format(max_deviation)

        if max_deviation < TOLERANCE:

            print(
                f"max deviation is {bcolors.OKGREEN}{max_deviation_str}{bcolors.ENDC}"
            )

        else:

            print(
                f"max deviation is {bcolors.FAIL}{max_deviation_str}{bcolors.ENDC}"
            )

    return base_result, result, benchmark_results
