"""
Utils related to time
"""

import pandas as pd
import numpy as np
import xarray as xr
from .geography import get_hemisphere


def get_time(year, month, day, hour):
    """
    Get np.datetime64 array corresponding to year, month, day and hour arrays

    Parameters
    ----------
    year (np.array or pd.Series)
    month (np.array or pd.Series)
    day (np.array or pd.Series)
    hour (np.array or pd.Series)

    Returns
    -------
    np.array or pd.Series
        The corresponding np.datetime64
    """
    time = pd.to_datetime(
        year.astype(str)
        + "-"
        + month.astype(str)
        + "-"
        + day.astype(str)
        + " "
        + hour.astype(str)
        + ":00"
    )
    return time


def get_season(track_id, lat, time, convention="long"):
    """


    Parameters
    ----------
    track_id : xr.DataArray
    lat : xr.DataArray
    time : xr.DataArray
    convention : str
        * 'short' : In the southern hemisphere, the season n corresponds to July n-1 to June n
        * 'long' : In the southern hemisphere, the season from July n-1 to June n is named "(n-1)n"

    Raises
    ------
    NotImplementedError
        If convention given is not 'short' or 'long'

    Returns
    -------
    xr.DataArray
        The season series.
        You can append it to your tracks by running tracks["season"] = get_season(tracks.track_id, tracks.lat, tracks.time)
    """

    # Derive values
    hemi = get_hemisphere(lat)
    year = time.dt.year
    month = time.dt.month
    # Store in a dataframe
    df = pd.DataFrame(
        {"hemi": hemi, "year": year, "month": month, "track_id": track_id}
    )
    # Most frequent year, month and hemisphere for each track
    # Grouping is done to avoid labelling differently points in a track that might cross hemisphere or seasons.
    group = df.groupby("track_id")[["year", "month", "hemi"]].agg(
        lambda x: pd.Series.mode(x)[0]
    )

    # Assign season
    if convention == "short":
        season = np.where(group.hemi == "N", group.year, np.nan)
        season = np.where(
            (group.hemi == "S") & (group.month >= 7), group.year + 1, season
        )
        season = np.where((group.hemi == "S") & (group.month <= 6), group.year, season)
    elif convention == "long":
        season = np.where(group.hemi == "N", group.year.astype(str), np.nan)
        season = np.where(
            (group.hemi == "S") & (group.month >= 7),
            group.year.astype(str) + (group.year + 1).astype(str),
            season,
        )
        season = np.where(
            (group.hemi == "S") & (group.month <= 6),
            (group.year - 1).astype(str) + group.year.astype(str),
            season,
        )
    else:
        raise NotImplementedError("Convention not recognized")

    group["season"] = season
    df = df.merge(group[["season"]], on="track_id")

    return xr.DataArray(df.season.values, dims="record", coords={"record": lat.record})