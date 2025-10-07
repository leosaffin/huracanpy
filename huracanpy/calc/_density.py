"""
Module containing function to compute track densities
"""

from metpy.constants import earth_avg_radius
import numpy as np
from sklearn.neighbors import KernelDensity
import xarray as xr


def density(
    lon,
    lat,
    method="histogram",
    bin_size=5,
    lon_range=None,
    lat_range=(-90, 90),
    crop=False,
    function_kws=dict(),
):
    """Function to compute the track density, based on a simple 2D histogram.

    Parameters
    ----------
    lon : array_like
        longitude series
    lat : array_like
        latitude series
    method : str, default="histogram"
        The method used to calculate the density, currently "histogram" or "kde", which
        gives a 2d histogram using `np.histogram2d`
    bin_size : int or float, default=5
        When using histogram, defines the size (in degrees) of the bins.
    lon_range : tuple, default
        The maximum and minimum longitude to calculate the density over. If None, then
        it is set to global: (-180, 180) or (0, 360) depending on the input data
    lat_range : tuple, default=(-90, 90)
        The maximum and minimum latitude to calculate the density over.
    crop : bool, default=False
        If True crop the result to remove any outer bounds that only have zero density
    function_kws : dict
        Keyword arguments passed to the function used for calculating density

        * If method="histogram", `numpy.histogram2d`
        * If method="kde", `sklearn.neighbors.KernelDensity`. Note that the bandwidth
          argument is set to `"scott"` rather than the default of `1.0`

    Raises
    ------
    NotImplementedError
        If method given is not 'histogram' or 'kde'

    Returns
    -------
    xarray.DataArray
        Track density as a 2D map.

    """
    # Define coordinates for mapping
    if lon_range is None:
        if lon.min() < 0:
            lon_range = (-180, 180)
        else:
            lon_range = (0, 360)

    x_edge = np.arange(lon_range[0], lon_range[1] + bin_size, bin_size)
    y_edge = np.arange(lat_range[0], lat_range[1] + bin_size, bin_size)
    x_mid, y_mid = (x_edge[1:] + x_edge[:-1]) / 2, (y_edge[1:] + y_edge[:-1]) / 2

    # Compute density
    if method == "histogram":
        h = _histogram(lon, lat, x_edge, y_edge, function_kws)

        area = (earth_avg_radius**2) * np.outer(
            np.diff(np.sin(np.deg2rad(y_edge))), np.diff(np.deg2rad(x_edge))
        )

        h = h / area
    elif method == "kde":
        h = _kde(lon, lat, x_mid, y_mid, function_kws)
    else:
        raise NotImplementedError(
            f"Method {method} not implemented yet. Use one 'histogram', 'kde'"
        )

    # Turn into xarray
    da = xr.DataArray(
        h,
        dims=["lat", "lon"],
        coords={"lon": x_mid, "lat": y_mid},
    )

    if crop:
        # Crop the map to where there are non-zero points
        has_data = da > 0

        # Keep the band of latitudes between first and lat non-empty row
        # and longitudes between first and lat empty column
        idx_lat = np.where(has_data.any(dim="lon"))[0]
        da = da.isel(lat=slice(idx_lat[0], idx_lat[-1] + 1))

        idx_lon = np.where(has_data.any(dim="lat"))[0]
        da = da.isel(lon=slice(idx_lon[0], idx_lon[-1] + 1))

        return da
    else:
        return da


def _histogram(lon, lat, x_edge, y_edge, function_kws):
    # Compute 2D histogram with numpy
    h, _x, _y = np.histogram2d(lon, lat, bins=[x_edge, y_edge], **function_kws)
    return h.T  # Transpose result


def _kde(lon, lat, x_mid, y_mid, function_kws):
    if "bandwidth" not in function_kws:
        function_kws["bandwidth"] = "scott"

    # engineer positions array for kernel estimation computation
    x_grid, y_grid = np.meshgrid(x_mid, y_mid)
    grid_positions = np.deg2rad(np.array([y_grid.flatten(), x_grid.flatten()]).T)
    track_positions = np.deg2rad([lat, lon]).T

    # Compute kernel density estimate
    kde = KernelDensity(**function_kws).fit(track_positions)

    # Evaluation kernel along positions
    return np.exp(kde.score_samples(grid_positions)).reshape(len(y_mid), len(x_mid))
