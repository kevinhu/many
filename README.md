# many

This package serves as a general-use toolkit for statistical and visual methods that I frequently implement.

## Installation

```bash
pip install many
```

## Components

### Statistical methods

The statistical methods comprise several functions for explorative data analysis with a focus on association mining between variable pairs. The methods used here are optimized for `pandas` DataFrames and are inspired by the `corrcoef` function provided by `numpy`.

Because these functions rely on native matrix-level operations provided by `numpy`, many are orders of magnitude faster than naive looping-based alternatives. This makes them useful for constructing large association networks or for feature extraction, which have important uses in areas such as biomarker discovery. 

In certain cases, such as the computation of correlation coefficients, **these vectorized methods come with the caveat of [numerical instability](https://stats.stackexchange.com/questions/94056/instability-of-one-pass-algorithm-for-correlation-coefficient)**. As a compromise, "naive" loop-based implementations are also provided for testing and comparison.

The current functions available are listed below by variable comparison type. Benchmarks are also provided with comparisons to an equivalent looping-based method. In all methods, a `melt` option is provided to return the outputs as a set of variable-variable pair statistic matrices or as a single `DataFrame` with each statistic melted to a column.

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
mat_mwus(a_mat, b_mat, melt: bool, effect:str, use_continuity=True)
```

Computes pairwise Mann-Whitney U tests between columns of `a_mat` (continuous samples) and `b_mat` (binary samples). Assumes that `a_mat` and `b_mat` both do not contain any missing values. `effect` can only be `rank_biserial`. `use_continuity` specifies whether a continuity correction should be applied.

```python
mat_mwus_naive( a_mat, b_mat, melt: bool, effect:str, use_continuity=True, pbar=False)
```

Same functionality as `mat_mwus`, but uses a double loop for direct computation of statistics. Unlike `mat_mwus`, `effect` parameters of "mean", "median", and "rank_biserial" are all supported.

#### Categorical vs. categorical

```python
mat_fishers(a_mat, b_mat, melt: bool, pseudocount=0)
```

Computes pairwise Fisher's exact tests between columns of `a_mat` and `b_mat`, provided that both are boolean-castable matrices and do not contain any missing values. The `pseudocount` parameter (which must be an integer) specifies the value that should be added to all cells of the contingency matrices.

```python
mat_fishers_nan(a_mat, b_mat, melt: bool, pseudocount=0)
```

Computes pairwise Fisher's exact tests between columns of `a_mat` and `b_mat`, provided that both are boolean-castable matrices and may or may not contain missing values.

```python
mat_fishers_naive(a_mat, b_mat, melt: bool, pseudocount=0, pbar=False)
```

Same functionality as `mat_fishers`, but uses a double loop for direct computation of statistics.

#### Benchmarks

Benchmarks were run with 1,000 samples per variable (i.e. setting each input matrix to have 1,000 rows). The number of variables in `a_mat` was set to 100, and the number of variables in `b_mat` was varied as shown below. The number of pairwise comparisons (equivalent to the product of the column counts of `a_mat` and `b_mat`) is also indicated.

Benchmarks were run on an i7-7700K with 16GB of 2133 MHz RAM.

##### `mat_corr` (Pearson)

| Column count of `b_mat` | Total comparisons | Runtime, `mat_corr_naive`, seconds | Runtime, `mat_corr`, seconds | Speedup    |
| ----------------------- | ----------------- | ---------------------------------- | ---------------------------- | ---------- |
| 10                      | 1,000             | 0.29                               | **0.01**                     | **×25.86** |
| 100                     | 10,000            | 2.67                               | **0.07**                     | **×37.37** |
| 1,000                   | 100,000           | 26.36                              | **0.35**                     | **×74.47** |

##### `mat_corr` (Spearman)

| Column count of `b_mat` | Total comparisons | Runtime, `mat_corr_naive`, seconds | Runtime, `mat_corr`, seconds | Speedup     |
| ----------------------- | ----------------- | ---------------------------------- | ---------------------------- | ----------- |
| 10                      | 1,000             | 0.74                               | **0.02**                     | **×46.74**  |
| 100                     | 10,000            | 7.28                               | **0.08**                     | **×94.21**  |
| 1,000                   | 100,000           | 72.06                              | **0.38**                     | **×190.58** |

##### `mat_mwu`

| Column count of `b_mat` | Total comparisons | Runtime, `mat_mwu_naive`, seconds | Runtime, `mat_mwu`, seconds | Speedup    |
| ----------------------- | ----------------- | --------------------------------- | --------------------------- | ---------- |
| 10                      | 1,000             | 1.06                              | **0.14**                    | **×7.55**  |
| 100                     | 10,000            | 10.38                             | **0.71**                    | **×14.62** |
| 1,000                   | 100,000           | 103.17                            | **6.52**                    | **×15.82** |

##### `mat_fisher`

| Column count of `b_mat` | Total comparisons | Runtime, `mat_fisher_naive`, seconds | Runtime, `mat_fisher`, seconds | Speedup   |
| ----------------------- | ----------------- | ------------------------------------ | ------------------------------ | --------- |
| 10                      | 1,000             | 1.18                                 | **0.24**                       | **×5.01** |
| 100                     | 10,000            | 11.72                                | **2.42**                       | **×4.83** |
| 1,000                   | 100,000           | 116.48                               | **23.77**                      | **×4.90** |

##### `mat_fisher_nan`

| Column count of `b_mat` | Total comparisons | Runtime, `mat_fisher_naive`, seconds | Runtime, `mat_fisher_nan`, seconds | Speedup   |
| ----------------------- | ----------------- | ------------------------------------ | ---------------------------------- | --------- |
| 10                      | 1,000             | 1.23                                 | **0.14**                           | **×8.56** |
| 100                     | 10,000            | 12.04                                | **1.43**                           | **×8.40** |
| 1,000                   | 100,000           | 120.20                               | **14.46**                          | **×8.31** |

### Visual methods

