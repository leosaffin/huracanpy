import numpy as np
import pandas as pd
from tropycal.tracks.dataset import TrackDataset

from .category import get_sshs_cat
from .geography import get_basin
from .time import get_season
from ..diags.track_stats import ace_by_point


def to_tropycal(tracks, vmax=None, mslp=None):
    output = TrackDataset.__new__(TrackDataset)

    output.proj = None
    output.basin = ""
    output.atlantic_url = None
    output.pacific_url = None
    output.ibtracs_url = None
    output.source = "huracanpy"
    output.catarina = False
    output.ibtracs_mode = ""
    output.neumann = False

    if "ace" not in tracks:
        if vmax is not None:
            tracks["ace"] = ace_by_point(tracks[vmax])
        else:
            tracks["ace"] = ("record", np.zeros(len(tracks.time)))

    if "basin" not in tracks:
        tracks["basin"] = get_basin(tracks.lon, tracks.lat)

    if "season" not in tracks:
        tracks["season"] = get_season(tracks.track_id, tracks.lat, tracks.time)

    if "category" not in tracks:
        if vmax is not None:
            tracks["category"] = get_sshs_cat(tracks[vmax])
        else:
            tracks["category"] = ("record", np.zeros(len(tracks.time), dtype=int))

    output.data = dict()
    for track_id, track in tracks.groupby("track_id"):
        output.data[track_id] = dict(
            id=track_id,
            operational_id="",
            name="",
            year=track.time.dt.year.values[0],
            season=tracks.season.values[0],
            basin=track.basin.values[0],
            source_info="",
            source="huracanpy",
            time=pd.to_datetime(track.time),
            extra_obs=np.zeros(len(track.time), dtype=int),
            special=np.zeros(len(track.time), dtype="<U1"),
            type=[],
            lat=track.lat.values,
            lon=track.lon.values,
            wmo_basin=track.basin.values,
            ace=np.sum(track.ace.values),
        )

        if vmax is not None:
            output.data[track_id]["vmax"] = track[vmax].values
        else:
            output.data[track_id]["vmax"] = np.zeros(len(track.time))
        if mslp is not None:
            output.data[track_id]["mslp"] = track[mslp].values
        else:
            output.data[track_id]["mslp"] = np.zeros(len(track.time))

    output.keys = list(output.data.keys())
    output.keys_tors = [0 for key in output.keys]
    output.data_tors = {}
    output.data_interp = {}

    output.attrs = dict(
        basin=output.basin,
        source=output.source,
        ibtracs_mode=output.ibtracs_mode,
        start_year=tracks.time.dt.year.values.min(),
        end_year=tracks.time.dt.year.values.max(),
        max_wind=None,
        min_mslp=None,
    )

    if vmax is not None:
        idx = tracks[vmax].argmax()
        point_max_wind = tracks.isel(record=idx)
        max_wind_tuple = (
            point_max_wind.track_id.values[()],
            point_max_wind.time.dt.year.values[()],
        )
        output.attrs["max_wind"] = max_wind_tuple

    if mslp is not None:
        idx = tracks[mslp].argmin()
        point_min_pressure = tracks.isel(record=idx)
        min_mslp_tuple = (
            point_min_pressure.track_id.values[()],
            point_min_pressure.time.dt.year.values[()],
        )
        output.attrs["min_mslp"] = min_mslp_tuple

    return output
