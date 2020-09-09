## Continuous vs. categorical

### mat_mwu

```python
mat_mwu(a_mat, b_mat, melt: bool, effect: str, use_continuity=True)
```

Computes pairwise Mann-Whitney U tests between columns of `a_mat` (continuous samples) and `b_mat` (binary samples). Assumes that `a_mat` and `b_mat` both do not contain any missing values. `effect` can only be `rank_biserial`. `use_continuity` specifies whether a continuity correction should be applied.

### mat_mwu_gpu

```python
mat_mwu_gpu(a_mat, b_mat, melt: bool, effect: str, use_continuity=True)
```

Exact same behavior as `mat_mwu`, with the exception that computation is accelerated via cupy.

### mat_mwu_naive

```python
mat_mwu_naive(a_mat, b_mat, melt: bool, effect: str, use_continuity=True, pbar=False)
```

Same functionality as `mat_mwu`, but uses a double loop for direct computation of statistics. Unlike `mat_mwus, ` `effect` parameters of "mean", "median", and "rank_biserial" are all supported.