import pytest

import pathlib

import numpy as np

import huracanpy


testdata = pathlib.Path(__file__).parent / "saved_results"


@pytest.mark.parametrize(
    "data, expected",
    [
        ("tracks_minus180_plus180", np.array(["S"] * 12 + ["N"] * 12)),
        ("tracks_0_360", np.array(["S"] * 12 + ["N"] * 12)),
        ("tracks_csv", np.array(["S"] * 99)),
    ],
)
def test_hemisphere(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.utils.geography.get_hemisphere(data.lat)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.array(["SP"] * 8 + ["SA"] * 4 + ["MED"] * 2 + ["NI"] * 4 + ["WNP"] * 6),
        ),
        (
            "tracks_0_360",
            np.array(["SP"] * 8 + ["SA"] * 4 + ["MED"] * 2 + ["NI"] * 4 + ["WNP"] * 6),
        ),
        ("tracks_csv", np.array(["AUS"] * 51 + ["SI"] * 48)),
    ],
)
def test_basin(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.utils.geography.get_basin(data.lon, data.lat)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.array(
                ["Land"]
                + ["Ocean"] * 6
                + ["Land"] * 2
                + ["Ocean"] * 4
                + ["Land"]
                + ["Ocean"]
                + ["Land"] * 6
                + ["Ocean"] * 3
            ),
        ),
        (
            "tracks_0_360",
            np.array(
                ["Land"]
                + ["Ocean"] * 6
                + ["Land"] * 2
                + ["Ocean"] * 4
                + ["Land"]
                + ["Ocean"]
                + ["Land"] * 6
                + ["Ocean"] * 3
            ),
        ),
        ("tracks_csv", np.array(["Ocean"] * 15 + ["Land"] * 15 + ["Ocean"] * 69)),
    ],
)
def test_get_land_ocean(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.utils.geography.get_land_or_ocean(data.lon, data.lat)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.array(
                ["Antarctica"]
                + [""] * 6
                + ["Argentina"] * 2
                + [""] * 4
                + ["Sudan"]
                + [""]
                + ["Iran"]
                + ["Afghanistan"]
                + ["China"]
                + ["Mongolia"]
                + ["Russia"] * 2
                + [""] * 3
            ),
        ),
        (
            "tracks_0_360",
            np.array(
                ["Antarctica"]
                + [""] * 6
                + ["Argentina"] * 2
                + [""] * 4
                + ["Sudan"]
                + [""]
                + ["Iran"]
                + ["Afghanistan"]
                + ["China"]
                + ["Mongolia"]
                + ["Russia"] * 2
                + [""] * 3
            ),
        ),
        ("tracks_csv", np.array([""] * 15 + ["Australia"] * 15 + [""] * 69)),
    ],
)
def test_get_country(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.utils.geography.get_country(data.lon, data.lat)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            "tracks_minus180_plus180",
            np.array(
                ["Antarctica"]
                + [""] * 6
                + ["South America"] * 2
                + [""] * 4
                + ["Africa"]
                + [""]
                + ["Asia"] * 4
                + ["Europe"] * 2
                + [""] * 3
            ),
        ),
        (
            "tracks_0_360",
            np.array(
                ["Antarctica"]
                + [""] * 6
                + ["South America"] * 2
                + [""] * 4
                + ["Africa"]
                + [""]
                + ["Asia"] * 4
                + ["Europe"] * 2
                + [""] * 3
            ),
        ),
        ("tracks_csv", np.array([""] * 15 + ["Oceania"] * 15 + [""] * 69)),
    ],
)
def test_get_continent(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.utils.geography.get_continent(data.lon, data.lat)
    assert (result == expected).all()


@pytest.mark.parametrize(
    "data, expected",
    [
        ("tracks_minus180_plus180", "get_propagation.npy"),
        ("tracks_0_360", "get_propagation.npy"),
        ("tracks_csv", "get_propagation_csv.npy"),
    ],
)
def test_get_propagation(data, expected, request):
    data = request.getfixturevalue(data)
    result = huracanpy.utils.geography.get_propagation(
        data.lon, data.lat, data.track_id
    )
    expected = np.load(testdata / expected)

    for r_, e_ in zip(result, expected):
        np.testing.assert_allclose(r_, e_, rtol=1e-12)
