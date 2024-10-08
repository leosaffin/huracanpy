import pytest

import datetime
from collections import namedtuple

import numpy as np
import xarray as xr

import huracanpy


@pytest.fixture()
def tracks_csv():
    return huracanpy.load(huracanpy.example_csv_file)


@pytest.fixture()
def tracks_minus180_plus180():
    return xr.Dataset(
        dict(
            track_id=np.zeros(24),
            time=[datetime.datetime(2000, 1, 1, n) for n in range(24)],
            lon=np.linspace(-180, 180, 24),
            lat=np.linspace(-90, 90, 24),
        )
    )


@pytest.fixture()
def tracks_0_360():
    return xr.Dataset(
        dict(
            track_id=np.zeros(24),
            time=[datetime.datetime(2000, 1, 1, n) for n in range(24)],
            lon=np.linspace(-180, 180, 24) % 360,
            lat=np.linspace(-90, 90, 24),
        )
    )


coords = namedtuple("coords", ["lon", "lat"])


@pytest.fixture()
def tracks_as_list():
    return coords(lon=[0, 20], lat=[0, 0])


@pytest.fixture()
def tracks_as_point():
    return coords(lon=0.0, lat=0.0)
