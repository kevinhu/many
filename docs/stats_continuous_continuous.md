## Continuous vs. continuous

### mat_corr

```python
mat_corr(a_mat, b_mat, melt: bool, method: str)
```

Computes pairwise Pearson or Spearman correlations between columns of `a_mat` and `b_mat`, provided that there are no missing values in either matrix. `method` can be either "pearson" or "spearman".

### mat_corr_nan

```python
mat_corr_nan(a_mat, b_mat, melt: bool, method: str)
```

Computes pairwise Pearson or Spearman correlations between `a_mat` and the columns of `b_mat`, provided that `a_mat` is a `Series` and `b_mat` is a `DataFrame` that may or may not contain some missing values. `method` can be either "pearson" or "spearman".

### mat_corr_naive

```python
mat_corr_naive(a_mat, b_mat, melt: bool, method: str, pbar=False)
```

Same functionality as `mat_corr`, but uses a double loop for direct computation of statistics. `method` can be either "pearson" or "spearman".