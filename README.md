# many

This package serves as a general-use toolkit for frequently-implemented statistical and visual methods.

## Installation

```bash
pip install many
```

## Components

### Statistical methods

The statistical methods comprise several functions for association mining between variable pairs. The methods used here are optimized for `pandas` DataFrames and are inspired by the `corrcoef` function provided by `numpy`.

Because these functions rely on native matrix-level operations provided by `numpy`, many are orders of magnitude faster than naive looping-based alternatives. This makes them useful for constructing large association networks or for feature extraction, which have important uses in areas such as biomarker discovery. All methods also return estimates of statistical significance.

In certain cases such as the computation of correlation coefficients, **these vectorized methods come with the caveat of [numerical instability](https://stats.stackexchange.com/questions/94056/instability-of-one-pass-algorithm-for-correlation-coefficient)**. As a compromise, "naive" loop-based implementations are also provided for testing and comparison. It is recommended that any significant results obtained with the vectorized methods be verified with these base methods.

The current functions available are listed below by variable comparison type. Benchmarks are also provided with comparisons to the equivalent looping-based method. In all methods, a `melt` option is provided to return the outputs as a set of row-column variable-variable pair statistic matrices or as a single `DataFrame` with each statistic melted to a column.

#### Continuous vs. continuous

```python
mat_corr(a_mat, b_mat, melt: bool, method: str)
```

Computes pairwise Pearson or Spearman correlations between columns of `a_mat` and `b_mat`, provided that there are no missing values in either matrix. `method` can be either "pearson" or "spearman".

```python
mat_corr_nan(a_mat, b_mat, melt: bool, method: str)
```

Computes pairwise Pearson or Spearman correlations between `a_mat` and the columns of `b_mat`, provided that `a_mat` is a `Series` and `b_mat` is a `DataFrame` that may or may not contain some missing values. `method` can be either "pearson" or "spearman".

```python
mat_corr_naive(a_mat, b_mat, melt: bool, method: str, pbar=False)
```

Same functionality as `mat_corr`, but uses a double loop for direct computation of statistics. `method` can be either "pearson" or "spearman".

#### Continuous vs. categorical

```python
mat_mwu(a_mat, b_mat, melt: bool, effect: str, use_continuity=True)
```

Computes pairwise Mann-Whitney U tests between columns of `a_mat` (continuous samples) and `b_mat` (binary samples). Assumes that `a_mat` and `b_mat` both do not contain any missing values. `effect` can only be `rank_biserial`. `use_continuity` specifies whether a continuity correction should be applied.

```python
mat_mwu_naive( a_mat, b_mat, melt: bool, effect: str, use_continuity=True, pbar=False)
```

Same functionality as `mat_mwu`, but uses a double loop for direct computation of statistics. Unlike `mat_mwus, ` `effect` parameters of "mean", "median", and "rank_biserial" are all supported.

#### Categorical vs. categorical

```python
mat_fisher(a_mat, b_mat, melt: bool, pseudocount=0)
```

Computes pairwise Fisher's exact tests between columns of `a_mat` and `b_mat`, provided that both are boolean-castable matrices and do not contain any missing values. The `pseudocount` parameter (which must be an integer) specifies the value that should be added to all cells of the contingency matrices.

```python
mat_fisher_nan(a_mat, b_mat, melt: bool, pseudocount=0)
```

Computes pairwise Fisher's exact tests between columns of `a_mat` and `b_mat`, provided that both are boolean-castable matrices and may or may not contain missing values.

```python
mat_fisher_naive(a_mat, b_mat, melt: bool, pseudocount=0, pbar=False)
```

Same functionality as `mat_fisher`, but uses a double loop for direct computation of statistics.

#### Benchmarks

Benchmarks were run with 1,000 samples per variable (i.e. setting each input matrix to have 1,000 rows). The number of variables in `a_mat` was set to 100, and the number of variables in `b_mat` was varied as shown below. The number of pairwise comparisons (equivalent to the product of the column counts of `a_mat` and `b_mat`) is also indicated.

Benchmarks were run on an i7-7700K with 16GB of 2133 MHz RAM.

##### `mat_corr` (Pearson)

<p align="center">
  <img width=512 src="https://github.com/kevinhu/many/raw/master/tests/benchmark_plots/continuous_continuous_mat_corr_%7B'method':%20'pearson'%2C%20'melt':%20False%7D.png">
</p>

##### `mat_corr` (Spearman)

<p align="center">
  <img width=512 src="https://github.com/kevinhu/many/raw/master/tests/benchmark_plots/continuous_continuous_mat_corr_%7B'method':%20'spearman'%2C%20'melt':%20False%7D.png">
</p>

##### `mat_mwu`

<p align="center">
  <img width=512 src="https://github.com/kevinhu/many/raw/master/tests/benchmark_plots/continuous_categorical_mat_mwu_%7B'effect':%20'rank_biserial'%2C%20'melt':%20False%7D.png">
</p>

##### `mat_fisher`

<p align="center">
  <img width=512 src="https://github.com/kevinhu/many/raw/master/tests/benchmark_plots/categorical_categorical_mat_fisher_%7B'melt':%20False%7D.png">
</p>

##### `mat_fisher_nan`

<p align="center">
  <img width=512 src="https://github.com/kevinhu/many/raw/master/tests/benchmark_plots/categorical_categorical_mat_fisher_nan_%7B'melt':%20False%7D.png">
</p>

### Visual methods

## Development

1. Install dependencies with `poetry install`
2. Initialize environment with `poetry shell`
3. Initialize pre-commit hooks with `pre-commit install`
