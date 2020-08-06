import many
import numpy as np

import sys

sys.path.append("../")


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


TOLERANCE = 1e-8


class OutputMismatchError(Exception):
    pass


class TypeMismatchError(Exception):
    pass


class ShapeMismatchError(Exception):
    pass


def compare(
    base_method,
    method,
    num_samples,
    a_num_cols,
    b_num_cols,
    a_type,
    b_type,
    a_nan,
    b_nan,
    method_kwargs,
    output_names,
):

    print(
        f"Comparing {bcolors.BOLD}{bcolors.HEADER}{method.__name__}{bcolors.ENDC} ",
        end="",
    )

    print(f" with {bcolors.BOLD}{bcolors.HEADER}{base_method.__name__}{bcolors.ENDC}")

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

    kwargs_string = ", ".join(
        f"{bcolors.BOLD}{bcolors.OKBLUE}{key}{bcolors.ENDC} = {value}"
        for key, value in method_kwargs.items()
    )
    print(f"\tand {kwargs_string}")

    a_test, b_test = many.stats.generate_test(
        num_samples, a_num_cols, b_num_cols, a_type, b_type, a_nan, b_nan
    )

    base_result = base_method(a_test, b_test, **method_kwargs)
    result = method(a_test, b_test, **method_kwargs)

    if len(output_names) == 1:

        base_result = [base_result]
        result = [result]

    if len(base_result) != len(result):
        raise OutputMismatchError("\tOutputs have different lengths")

    for i in range(len(output_names)):

        base_output = base_result[i]
        output = result[i]

        output_name = output_names[i]

        print(f"\tChecking {bcolors.BOLD}{output_name}{bcolors.ENDC} outputs: ", end="")

        if type(base_output) != type(output):
            raise TypeMismatchError("Outputs have different types")

        if base_output.shape != output.shape:
            raise ShapeMismatchError(
                "Outputs have different shapes", base_output.shape, output.shape
            )

        max_deviation = np.abs(base_output - output).max().max()

        max_deviation_str = "{0:.3E}".format(max_deviation)

        if max_deviation < TOLERANCE:

            print(
                f"max deviation is {bcolors.OKGREEN}{max_deviation_str}{bcolors.ENDC}"
            )

        else:

            print(f"max deviation is {bcolors.FAIL}{max_deviation_str}{bcolors.ENDC}")


# mat_corrs, full-size comparison
compare(
    many.stats.mat_corrs_naive,
    many.stats.mat_corrs,
    100,
    10,
    25,
    "continuous",
    "continuous",
    False,
    False,
    {"method": "pearson"},
    ["corrs", "pvals"],
)

mat_corr_methods = ["pearson", "spearman"]

for corr_method in mat_corr_methods:

    # mat_corrs, 1-d a_mat
    compare(
        many.stats.mat_corrs_naive,
        many.stats.mat_corrs,
        100,
        1,
        25,
        "continuous",
        "continuous",
        False,
        False,
        {"method": corr_method},
        ["merged"],
    )

    # mat_corrs, 1-d b_mat
    compare(
        many.stats.mat_corrs_naive,
        many.stats.mat_corrs,
        100,
        10,
        1,
        "continuous",
        "continuous",
        False,
        False,
        {"method": corr_method},
        ["merged"],
    )

    # mat_corrs_nan, 1-d both
    compare(
        many.stats.mat_corrs_naive,
        many.stats.mat_corrs,
        100,
        1,
        1,
        "continuous",
        "continuous",
        False,
        False,
        {"method": corr_method},
        ["merged"],
    )

    # mat_corrs_nan, no nans
    compare(
        many.stats.mat_corrs_naive,
        many.stats.mat_corrs_nan,
        100,
        1,
        100,
        "continuous",
        "continuous",
        False,
        False,
        {"method": corr_method},
        ["merged"],
    )

    # mat_corrs_nan, nans in a
    compare(
        many.stats.mat_corrs_naive,
        many.stats.mat_corrs_nan,
        100,
        1,
        100,
        "continuous",
        "continuous",
        True,
        False,
        {"method": corr_method},
        ["merged"],
    )

    # mat_corrs_nan, nans in b
    compare(
        many.stats.mat_corrs_naive,
        many.stats.mat_corrs_nan,
        100,
        1,
        100,
        "continuous",
        "continuous",
        False,
        True,
        {"method": corr_method},
        ["merged"],
    )

    # mat_corrs_nan, nans in both
    compare(
        many.stats.mat_corrs_naive,
        many.stats.mat_corrs_nan,
        100,
        1,
        100,
        "continuous",
        "continuous",
        True,
        True,
        {"method": corr_method},
        ["merged"],
    )
