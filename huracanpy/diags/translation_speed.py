"""
Module containing functions to compute translation speed
"""

from haversine import haversine
import numpy as np
import xarray as xr


def translation_speed(data, lat_name="lat", lon_name="lon", time_name="time"):
    """
    Compute translation speed along tracks

    Parameters
    ----------
    data : xarray.Dataset

    Returns
    -------
    xarray.Dataset
        Translation speeds. Output is stored for points that correspond to the middle of two consecutive points in the initial dataset.

    """
    data = data.sortby(["track_id", "time"])
    V, lat, lon, t, tid = [], [], [], [], []  # To store results temporarily

    # Convert longitudes beyond 180° (necessary for haversine to work)
    data[lon_name] = xr.DataArray(
        np.where(data[lon_name] > 180, data[lon_name] - 360, data[lon_name]),
        dims=data.dims,
    )

    dims = data.time.dims
    assert len(dims) == 1
    dim = dims[0]
    for i in range(len(data[dim]) - 1):
        p = data.isel(**{dim: i})  # Current point
        q = data.isel(**{dim: i + 1})  # Next point
        if p.track_id == q.track_id:  # If both points belong to the same track
            dt = np.timedelta64((q.time - p.time).values, "s")  # Temporal interval in s
            dx = haversine(
                (p[lat_name].values[()], p[lon_name].values[()]),
                (q[lat_name].values[()], q[lon_name].values[()]),
                unit="m",
            )  # Displacement in m
            v = dx / dt.astype(float)  # translation speed in m/s
            V.append(v)
            # Results will be stored with coordinates corresponding to the middle of p and q
            lat.append((p[lat_name] + q[lat_name]).values / 2)
            lon.append((p[lon_name] + q[lon_name]).values / 2)
            t.append((p[time_name] + (q[time_name] - p[time_name]) / 2).values)
            tid.append(p.track_id.values)

    # Transform into clean dataset
    V = xr.DataArray(V, dims="mid_record", coords={"mid_record": np.arange(len(V))})
    V.attrs["units"] = "m s**-1"
    lon = xr.DataArray(
        lon, dims="mid_record", coords={"mid_record": np.arange(len(lon))}
    )
    lat = xr.DataArray(
        lat, dims="mid_record", coords={"mid_record": np.arange(len(lat))}
    )
    t = xr.DataArray(t, dims="mid_record", coords={"mid_record": np.arange(len(t))})
    tid = xr.DataArray(
        tid, dims="mid_record", coords={"mid_record": np.arange(len(tid))}
    )

    return xr.Dataset(
        {"lon": lon, "lat": lat, "time": t, "track_id": tid, "translation_speed": V}
    )
