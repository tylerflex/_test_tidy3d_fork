"""Tests heat medium objects."""
import pytest
import pydantic
import tidy3d as td
import matplotlib.pylab as plt


def test_temperature_dependence():

    constant = td.TemperatureDependenceConstant(value=2)
    linear = td.TemperatureDependenceLinear(ref_temp=300, coeff=0.01)
    table = td.TemperatureDependenceTable(temp_points=[250,300,350], func_values=[4,2,5])

    # no negative reference temperature
    with pytest.raises(pydantic.ValidationError):
        _ = td.TemperatureDependenceLinear(ref_temp=-2, coeff=0.01)

    temps = [200, 240, 250, 280]

    constant.plot(temps)
    linear.plot(temps)
    table.plot(temps)

    fig, ax = plt.subplots(1, 1)
    constant.plot(temps, ax=ax)
    linear.plot(temps, ax=ax)
    table.plot(temps, ax=ax)


def test_heat_spec():
    liq = td.HeatSpecLiquid()

    perm_change = td.TemperatureDependenceLinear(ref_temp=300, coeff=0.1)
    cond_change = td.TemperatureDependenceConstant(value=0)
    solid = td.HeatSpecSolid(
        density=1,
        capacity=2,
        conductivity=3,
        permittivity_change=perm_change,
        conductivity_change=cond_change,
    )

    # no negative physical properties
    with pytest.raises(pydantic.ValidationError):
        solid = td.HeatSpecSolid(
            density=-1,
            capacity=2,
            conductivity=3,
            permittivity_change=perm_change,
            conductivity_change=cond_change,
        )

    # no negative physical properties
    with pytest.raises(pydantic.ValidationError):
        solid = td.HeatSpecSolid(
            density=1,
            capacity=0,
            conductivity=3,
            permittivity_change=perm_change,
            conductivity_change=cond_change,
        )

    # no negative physical properties
    with pytest.raises(pydantic.ValidationError):
        solid = td.HeatSpecSolid(
            density=1,
            capacity=4,
            conductivity=-3,
            permittivity_change=perm_change,
            conductivity_change=cond_change,
        )

    solid.plot(temperature=[100, 200, 300])

    fig, ax = plt.subplots(1, 1)
    solid.plot(temperature=[100, 200, 300], ax=ax)
