# many

This package serves as a general-use toolkit for frequently-implemented statistical and visual methods.

## Installation

```bash
pip install many
```

Note: if you want to use CUDA-accelerated statistical methods (i.e. `many.stats.mat_mwu_gpu`), you must also independently install the corresponding version of [cupy](https://github.com/cupy/cupy).

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
mat_mwu_gpu(a_mat, b_mat, melt: bool, effect: str, use_continuity=True)
```

Exact same behavior as `mat_mwu`, with the exception that computation is accelerated via cupy.

```python
mat_mwu_naive(a_mat, b_mat, melt: bool, effect: str, use_continuity=True, pbar=False)
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

Benchmarks were run on an i7-7700K with 16GB of 2133 MHz RAM. GPU benchmarks were performed on a GTX 1080.

<p align="center">
  <img src="https://github.com/kevinhu/many/raw/master/tests/benchmark_plots/all_benchmarks.png">
</p>

### Visual methods

#### Continuous vs. continuous

```python
scatter_grid(dataframe)
```

Plot relationships between columns in a DataFrame, coloring by density and inserting labels given a set of significant value masks.

<p align="center">
  <img width=480 src="https://github.com/kevinhu/many/raw/master/tests/output_plots/scatter_grid.png">
</p>

```python
regression(
    x, y, method, ax=None, alpha=0.5, text_pos=(0.1, 0.9), scatter_kwargs={}
)
```

Plot two sets of points with along with their regression coefficient.

<p align="center">
  <img width=480 src="https://github.com/kevinhu/many/raw/master/tests/output_plots/regression_pearson.png">
</p>

```python
dense_regression(
    x,
    y,
    method,
    ax=None,
    colormap=None,
    cmap_offset=0,
    text_pos=(0.1, 0.9),
    scatter_kwargs={},
)
```

Plot two sets of points and their regression coefficient, along with density-based coloring.

<p align="center">
  <img width=480 src="https://github.com/kevinhu/many/raw/master/tests/output_plots/dense_regression_pearson.png">
</p>

```python
dense_plot(
    x,
    y,
    text_adjust: bool,
    ax=None,
    labels_mask=None,
    labels=None,
    colormap=None,
    cmap_offset=0,
    scatter_kwargs={},
    x_offset=0,
    y_offset=0,
)
```

Plot two sets of points, coloring by density and inserting labels given a set of significant value masks. Density estimated by Gaussian KDE.

<p align="center">
  <img width=480 src="https://github.com/kevinhu/many/raw/master/tests/output_plots/dense_plot_default.png">
</p>

#### Continuous vs. categorical

```python
two_dists(
    binary,
    continuous,
    method,
    summary_type,
    ax=None,
    pal=["#eaeaea", "#a5dee5"],
    annotate=True,
    stripplot=False,
    seaborn_kwargs={},
    stripplot_kwargs={},
)
```

Compare the distributions of a continuous variable when grouped by a binary one.
    
<p align="center">
  <img width=480 src="https://github.com/kevinhu/many/raw/master/tests/output_plots/two_dists_t_test_box.png">
</p>

```python
multi_dists(
    continuous,
    categorical,
    count_cutoff,
    summary_type,
    ax=None,
    stripplot=False,
    order="ascending",
    newline_counts=False,
    xtick_rotation=45,
    xtick_ha="right",
    seaborn_kwargs={},
    stripplot_kwargs={},
)
```

Compare the distributions of a continuous variable when grouped by a categorical one.

<p align="center">
  <img src="https://github.com/kevinhu/many/raw/master/tests/output_plots/multi_dists_box.png">
</p>

```python
roc_auc_curve(y, y_pred, ax=None)
```

Plot the ROC curve along with the AUC statistic of predictions against ground truths.

<p align="center">
  <img width=480 src="https://github.com/kevinhu/many/raw/master/tests/output_plots/roc_auc_curve.png">
</p>

```python
pr_curve(y, y_pred, ax=None)
```

Plot the precision-recall curve of predictions against ground truths.

<p align="center">
  <img width=480 src="https://github.com/kevinhu/many/raw/master/tests/output_plots/pr_curve.png">
</p>

```python
binary_metrics(y, y_pred)
```

Make several plots to evaluate a binary classifier:

    1. Boxplots of predicted values
    2. Violinplots of predicted values
    3. ROC-AUC plot
    4. Precision-recall curve

<p align="center">
  <img src="https://github.com/kevinhu/many/raw/master/tests/output_plots/binary_metrics.png">
</p>

#### Categorical vs. categorical

```python
binary_contingency(a, b, ax=None, heatmap_kwargs={})
```

Plot agreement between two binary variables, along with the odds ratio and Fisher's exact test p-value.

<p align="center">
  <img width=480 src="https://github.com/kevinhu/many/raw/master/tests/output_plots/binary_contingency_default.png">
</p>

## Development

1. Install dependencies with `poetry install`
2. Initialize environment with `poetry shell`
3. Initialize pre-commit hooks with `pre-commit install`
