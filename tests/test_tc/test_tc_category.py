import pint
import pytest
import numpy as np
import xarray as xr

import huracanpy


def test_sshs():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    assert huracanpy.tc.saffir_simpson_category(data.wind10).min() == -1
    assert huracanpy.tc.saffir_simpson_category(data.wind10).max() == 0


def test_pressure_cat():
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")
    klotz = huracanpy.tc.pressure_category(data.slp / 100)
    simps = huracanpy.tc.pressure_category(data.slp / 100, convention="Simpson")
    assert klotz.sum() == 62
    assert simps.sum() == -23


@pytest.mark.parametrize("pass_as_numpy", [False, True])
@pytest.mark.parametrize(
    "units, expected",
    [
        ("m s-1", "default"),
        ("cm s-1", np.asarray([-1.0] * 99)),
        ("km s-1", np.asarray([5.0] * 99)),
    ],
)
def test_sshs_units(units, expected, pass_as_numpy):
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")

    if isinstance(expected, str) and expected == "default":
        expected = huracanpy.tc.saffir_simpson_category(data.wind10)

    if pass_as_numpy:
        result = huracanpy.tc.saffir_simpson_category(
            data.wind10.data, wind_units=units
        )
    else:
        data.wind10.attrs["units"] = units
        result = huracanpy.tc.saffir_simpson_category(data.wind10)

    assert (result == expected).all()

    if pass_as_numpy:
        assert isinstance(result, np.ndarray)
    else:
        assert isinstance(result, xr.DataArray)
        assert not isinstance(result.data, pint.Quantity)


@pytest.mark.parametrize("pass_as_numpy", [False, True])
@pytest.mark.parametrize("convention", ["Klotzbach", "Simpson"])
@pytest.mark.parametrize(
    "units, expected",
    [
        ("Pa", "default"),
        ("hPa", np.asarray([-1.0] * 99)),
    ],
)
def test_pressure_cat_units(units, expected, convention, pass_as_numpy):
    data = huracanpy.load(huracanpy.example_csv_file, source="csv")

    if isinstance(expected, str) and expected == "default":
        with pytest.warns(UserWarning, match="Caution, pressure are likely in Pa"):
            expected = huracanpy.tc.pressure_category(data.slp, convention=convention)

    if pass_as_numpy:
        result = huracanpy.tc.pressure_category(
            data.slp.data, convention=convention, slp_units=units
        )
    else:
        data.slp.attrs["units"] = units
        result = huracanpy.tc.pressure_category(data.slp, convention=convention)

    assert (result == expected).all()

    if pass_as_numpy:
        assert isinstance(result, np.ndarray)
    else:
        assert isinstance(result, xr.DataArray)
        assert not isinstance(result.data, pint.Quantity)
