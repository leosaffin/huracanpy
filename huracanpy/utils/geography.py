"""
Utils related to geographical attributes
"""

import warnings
from pint.errors import UnitStrippedWarning

import numpy as np
import pandas as pd
from shapely.geometry import Point
import geopandas as gpd
from cartopy.geodesic import Geodesic
from cartopy.io.shapereader import natural_earth
from metpy.xarray import preprocess_and_wrap
from cartopy.crs import Geodetic, PlateCarree

from ._basins import basins_def


_geodesic = Geodesic()


@preprocess_and_wrap(wrap_like="lat")
def get_hemisphere(lat):
    """
    Function to detect which hemisphere each point corresponds to

    Parameters
    ----------
    lat : xarray.DataArray

    Returns
    -------
    xarray.DataArray
        The hemisphere series.
        You can append it to your tracks by running tracks["hemisphere"] = get_hemisphere(tracks)
    """

    return np.where(lat >= 0, "N", "S")


def get_basin(lon, lat, convention="WMO", crs=None):
    """
    Function to determine the basin of each point, according to the selected convention.

    Parameters
    ----------
    lon : xarray.DataArray
        Longitude series
    lat : xarray.DataArray
        Latitude series
    convention : str
        Name of the basin convention you want to use.
            * WMO
    crs : cartopy.crs.CRS, optional
        The coordinate reference system of the lon, lat inputs. The basins are defined
        in PlateCarree (-180, 180), so this will transform lon/lat to this projection
        before checking the basin. If None is given, it will use cartopy.crs.Geodetic
        which is essentially the same, but allows the longitudes to be defined in ranges
        broader than -180, 180

    Returns
    -------
    xarray.DataArray
        The basin series.
        You can append it to your tracks by running tracks["basin"] = get_basin(tracks)
    """
    return _get_natural_earth_feature(
        lon,
        lat,
        feature="basin",
        category="physical",
        name=convention,
        resolution=0,
        crs=crs,
    )


# Running this on lots of tracks was very slow if the file is reopened every time this
# is called
_natural_earth_feature_cache = {
    f"physical_{key}_0_basin": value.rename_axis("basin").reset_index()
    for key, value in basins_def.items()
}


@preprocess_and_wrap(wrap_like="lon")
def _get_natural_earth_feature(lon, lat, feature, category, name, resolution, crs=None):
    key = f"{category}_{name}_{resolution}_{feature}"
    if key in _natural_earth_feature_cache:
        df = _natural_earth_feature_cache[key]
    else:
        fname = natural_earth(resolution=resolution, category=category, name=name)
        df = gpd.read_file(fname)
        df = df[["geometry", feature]]
        _natural_earth_feature_cache[key] = df

    # The metpy wrapper converting to pint causes errors, but I'm still going to use it
    # because it lets me pass different array_like types for lon/lat without writing
    # our own wrapper. For now, just convert anything not a numpy array to a numpy array
    if not isinstance(lon, np.ndarray):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UnitStrippedWarning)
            lon = np.array(lon)
            lat = np.array(lat)

    if crs is None:
        crs = Geodetic()
    xyz = PlateCarree().transform_points(crs, lon, lat)

    # Create dataframe of points coordinates
    points = pd.DataFrame(dict(coords=list(xyz[:, :2])))
    # Transform into Points within a GeoDataFrame
    points = gpd.GeoDataFrame(points.coords.apply(Point), geometry="coords", crs=df.crs)

    result = np.array(
        gpd.tools.sjoin(df, points, how="right", predicate="contains")[feature]
    ).astype(str)

    # Set "nan" as empty
    result[result == "nan"] = ""

    return result


def get_land_or_ocean(lon, lat, resolution="10m", crs=None):
    """
    Detect whether each point is over land or ocean

    Parameters
    ----------
    lon, lat : float or array_like
    resolution : str
        The resolution of the Land/Sea outlines dataset to use. One of

        * 10m (1:10,000,000)
        * 50m (1:50,000,000)
        * 110m (1:110,000,000)

    crs : cartopy.crs.CRS

    Returns
    -------
    array_like
        Array of "Land" or "Ocean" for each lon/lat point. Should return the same type
        of array as the input lon/lat, or a length 1 :py:class:`numpy.ndarray` if
        lon/lat are floats
    """
    is_ocean = _get_natural_earth_feature(
        lon,
        lat,
        feature="featurecla",
        category="physical",
        name="ocean",
        resolution=resolution,
        crs=crs,
    )

    is_ocean[is_ocean == ""] = "Land"

    return is_ocean


def get_country(lon, lat, resolution="10m", crs=None):
    """Detect the country each point is over

    Parameters
    ----------
    lon, lat : float or array_like
    resolution : str
        The resolution of the Land/Sea outlines dataset to use. One of

        * 10m (1:10,000,000)
        * 50m (1:50,000,000)
        * 110m (1:110,000,000)

    crs : cartopy.crs.CRS

    Returns
    -------
    array_like
        Array of country names (or empty string for no country) for each lon/lat point.
        Should return the same type of array as the input lon/lat, or a length 1
        :py:class:`numpy.ndarray` if lon/lat are floats
    """
    return _get_natural_earth_feature(
        lon,
        lat,
        feature="NAME",
        category="cultural",
        name="admin_0_countries",
        resolution=resolution,
        crs=crs,
    )


def get_continent(lon, lat, resolution="10m", crs=None):
    """Detect the continent each point is over

    Parameters
    ----------
    lon, lat : float or array_like
    resolution : str
        The resolution of the Land/Sea outlines dataset to use. One of

        * 10m (1:10,000,000)
        * 50m (1:50,000,000)
        * 110m (1:110,000,000)

    crs : cartopy.crs.CRS

    Returns
    -------
    array_like
        Array of continent names (or empty string for no continent) for each lon/lat
        point. Should return the same type of array as the input lon/lat, or a length 1
        :py:class:`numpy.ndarray` if lon/lat are floats
    """
    return _get_natural_earth_feature(
        lon,
        lat,
        feature="CONTINENT",
        category="cultural",
        name="admin_0_countries",
        resolution=resolution,
        crs=crs,
    )


def get_propagation(lon, lat, track_ids=None):
    """Calculate the distance and angle between successive latitude and longitude points

    Uses geodesic calculations provided by `cartopy.geodesic`

    Parameters
    ----------
    lon, lat : array_like
        1d arrays of longitudes and latitudes (in degrees)
    track_ids : array_like, optional
        A 1d array with the same length as lon/lat which splits the points into separate
        tracks. If this is included, the distances and angles between different tracks
        won't be returned and instead the first point of each track will be the forward
        difference (from 0 to 1) and the last point of each track will be the backward
        different (from n-1 to n). If track_ids is not included, it is assumed that lon
        and lat represent a single track

    Returns
    -------
    distance, bearing : numpy.ndarray
        The distance (in m) and direction (degrees from north [-180, 180]) of travel
        of the track(s) for each lon/lat point
    """
    xy = np.array([lon, lat]).T

    result = _geodesic.inverse(xy[:-1], xy[1:])
    distance = result[:, 0]
    bearing = result[:, 1]

    # Use the forward and backward difference for propagation at each end of the track
    # centered difference in the middle
    distance_mid = np.concatenate(
        [[distance[0]], 0.5 * (distance[:-1] + distance[1:]), [distance[-1]]]
    )

    # More careful averaging of angles
    # Separate into unit x and y vectors and then recombine
    dx = np.sin(np.deg2rad(bearing))
    dy = np.cos(np.deg2rad(bearing))
    bearing_mid = np.rad2deg(
        np.arctan2(0.5 * (dx[:-1] + dx[1:]), 0.5 * (dy[:-1] + dy[1:]))
    )
    bearing_mid = np.concatenate([[bearing[0]], bearing_mid, [bearing[-1]]])

    if track_ids is not None:
        # Find the start/end points of each track to use forward/backward difference
        # instead of averaging across two tracks
        # Ignore first and last point as these haven't been averaged and could be out
        # of bounds when taking +1/-1 points
        idx = np.where(np.array(track_ids[1:-2]) != np.array(track_ids[2:-1]))[0] + 1

        # Backward difference at the end of each track (previous point)
        distance_mid[idx] = distance[idx - 1]
        bearing_mid[idx] = bearing[idx - 1]

        # Forward difference at the start of each track (following point)
        distance_mid[idx + 1] = distance[idx + 1]
        bearing_mid[idx + 1] = bearing[idx + 1]

        # Any tracks of length-1 set as nan
        trid, trid_idx, trid_counts = np.unique(
            track_ids, return_index=True, return_counts=True
        )
        idx_1s = trid_idx[trid_counts == 1]

        distance_mid[idx_1s] = np.nan
        bearing_mid[idx_1s] = np.nan

    return distance_mid, bearing_mid
