"""
Functions to plot the tracks
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import cartopy.crs as ccrs


def plot_tracks_basic(
    tracks,
    intensity_var=None,
    subplot_kws=dict(projection=ccrs.PlateCarree(180)),
    fig_kws=dict(figsize=(10, 10)),
    scatter_kws=dict(palette="nipy_spectral", s=2, color="k"),
):
    assert "lon" in list(tracks.keys()), "lon is not present in the data"
    assert "lat" in list(tracks.keys()), "lat is not present in the data"

    fig, ax = plt.subplots(subplot_kw=subplot_kws, **fig_kws)
    ax.coastlines()
    sns.scatterplot(
        data=tracks,
        x="lon",
        y="lat",
        hue=intensity_var,
        ax=ax,
        **scatter_kws,
        transform=ccrs.PlateCarree(),
    )

    return fig, ax


def add_tag(x, y, tag, **kwargs):
    """Add a label to a track at each point that it changes

    Parameters
    ----------
    x, y : array_like
    tag : array_like, same length as x and y
    **kwargs

    Returns
    -------
    None

    """
    # Iteration can be awkward with xarray so just convert non numpy arrays to an array
    if not isinstance(tag, np.ndarray):
        tag = np.array(tag)

    prev_tag = ""
    for n, tag_ in enumerate(tag):
        tag_ = str(tag_)
        if tag_ != prev_tag:
            plt.text(x[n], y[n], tag_, **kwargs)
            prev_tag = tag_
