import pandas as pd
import seaborn as sns


def as_si(x: float, decimals: int) -> str:
    """
    Convert a number to scientific notation

    Parameters
    ----------
    x : float
        number to convert
    decimals: float
        number of decimal places

    Returns
    -------
    x_si : string
        x formatted in scientific notation
    """

    s = "{x:0.{ndp:d}e}".format(x=x, ndp=decimals)
    m, e = s.split("e")
    x_si = r"{m:s} × $10^{{{e:d}}}$".format(m=m, e=int(e))

    return x_si


def colorize(s, cmap):
    """
    Map an array-like to colors, useful for
    col_colors/row_colors in Seaborn clustermaps

    Parameters
    ----------
    s : Pandas series
        series of values to colorize
    cmap: string, MatPlotLib color map, or Seaborn palette
        colormap to generate colors from

    Returns
    -------
    colors : Pandas series
        color-mapped series
    mapping : dictionary
        mapping scheme from each unique element of s
        to corresponding color
    """

    s = pd.Series(s)

    pal = sns.color_palette(cmap, len(s.unique()))
    mapping = dict(zip(s.unique(), pal))
    colors = pd.Series(s).map(mapping)

    return colors, mapping
