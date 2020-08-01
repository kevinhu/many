import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


def as_si(x, ndp):
    """
    Convert a number to scientific notation

    Parameters
    ----------
    x : float
        number to convert
    ndp: float
        number of decimal places

    Returns
    -------
    x_si : string
        x formatted in scientific notation
    """

    s = "{x:0.{ndp:d}e}".format(x=x, ndp=ndp)
    m, e = s.split("e")
    x_si = r"{m:s} × $10^{{{e:d}}}$".format(m=m, e=int(e))

    return x_si


def get_clustermap_colors(s, cmap="Blues"):
    """
    Map a Pandas series to colors, useful for
    col_colors/row_colors in Seaborn clustermaps

    Parameters
    ----------
    s : Pandas series
        series of values to colorize
    cmap: string, Matplotlib color map, or Seaborn palette
        colormap to generate colors from

    Returns
    -------
    colors : Pandas series
        color-mapped series
    mapping : dictionary
        mapping scheme from each unique element of s
        to corresponding color
    """

    pal = sns.color_palette(cmap, len(s.unique()))
    mapping = dict(zip(s.unique(), pal))
    colors = pd.Series(s).map(mapping)

    return colors, mapping
