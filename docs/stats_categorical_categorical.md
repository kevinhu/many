## Categorical vs. categorical

### many.stats.mat_fisher

```python
many.stats.mat_fisher(a_mat, b_mat, melt: bool, pseudocount=0)
```

Computes pairwise Fisher's exact tests between columns of `a_mat` and `b_mat`, provided that both are boolean-castable matrices and do not contain any missing values. The `pseudocount` parameter (which must be an integer) specifies the value that should be added to all cells of the contingency matrices.

### many.stats.mat_fisher_nan

```python
many.stats.mat_fisher_nan(a_mat, b_mat, melt: bool, pseudocount=0)
```

Computes pairwise Fisher's exact tests between columns of `a_mat` and `b_mat`, provided that both are boolean-castable matrices and may or may not contain missing values.

### many.stats.mat_fisher_naive

```python
many.stats.mat_fisher_naive(a_mat, b_mat, melt: bool, pseudocount=0, pbar=False)
```

Same functionality as `mat_fisher`, but uses a double loop for direct computation of statistics.