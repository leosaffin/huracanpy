import huracanpy

import numpy as np


def test_ace(tracks_csv):
    ace = huracanpy.diags.track_stats.ace_by_track(tracks_csv, tracks_csv.wind10)

    np.testing.assert_allclose(ace, np.array([3.03623809, 2.21637375, 4.83686787]))

    assert isinstance(ace.data, np.ndarray)


def test_pace(tracks_csv):
    # Pass wind values to fit a (quadratic) model to the pressure-wind relationship
    pace, model = huracanpy.diags.track_stats.pace_by_track(
        tracks_csv, tracks_csv.slp, wind=tracks_csv.wind10
    )

    np.testing.assert_allclose(pace, np.array([4.34978137, 2.65410482, 6.09892875]))

    # Call with the already fit model instead of wind values
    pace, _ = huracanpy.diags.track_stats.pace_by_track(
        tracks_csv,
        tracks_csv.slp,
        model=model,
    )

    np.testing.assert_allclose(pace, np.array([4.34978137, 2.65410482, 6.09892875]))


def test_duration():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    d = huracanpy.diags.track_stats.duration(data.time, data.track_id)
    assert d.min() == 126
    assert d.max() == 324
    assert d.mean() == 210


def test_gen_vals():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    G = huracanpy.diags.track_stats.gen_vals(data)
    assert G.day.mean() == 10


def test_extremum_vals():
    data = huracanpy.load(huracanpy.example_csv_file, tracker="csv")
    M = huracanpy.diags.track_stats.extremum_vals(data, "wind10", "max")
    m = huracanpy.diags.track_stats.extremum_vals(data, "slp", "min")
    assert M.day.mean() == 15
    assert m.lat.mean() == -27
