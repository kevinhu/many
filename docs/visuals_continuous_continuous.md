## Continuous vs. continuous

### many.visuals.scatter_grid

```python
many.visuals.scatter_grid(dataframe)
```

Plot relationships between columns in a DataFrame, coloring by density and inserting labels given a set of significant value masks.

<p align="center">
  <img width=480 src="https://raw.githubusercontent.com/kevinhu/many/master/tests/output_plots/scatter_grid.png">
</p>

### many.visuals.regression

```python
many.visuals.regression(
    x, y, method, ax=None, alpha=0.5, text_pos=(0.1, 0.9), scatter_kwargs={}
)
```

Plot two sets of points with along with their regression coefficient.

<p align="center">
  <img width=480 src="https://raw.githubusercontent.com/kevinhu/many/master/tests/output_plots/regression_pearson.png">
</p>

### many.visuals.dense_regression

```python
many.visuals.dense_regression(
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
  <img width=480 src="https://raw.githubusercontent.com/kevinhu/many/master/tests/output_plots/dense_regression_pearson.png">
</p>

### many.visuals.dense_plot

```python
many.visuals.dense_plot(
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
  <img width=480 src="https://raw.githubusercontent.com/kevinhu/many/master/tests/output_plots/dense_plot_default.png">
</p>
