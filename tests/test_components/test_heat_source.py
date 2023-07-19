"""Tests heat medium objects."""
import pytest
import pydantic
import tidy3d as td
import numpy as np
#import matplotlib.pylab as plt


def test_heat_source():

    nx, ny, nz, nt = 9, 6, 5, 3
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    z = np.linspace(-5, 5, nz)
    t = np.linspace(0, 1, nt)
    rate = np.random.default_rng().uniform(300, 350, (nx, ny, nz, nt))
    coords = dict(x=x, y=y, z=z, t=[0])
    temperature_field = td.ScalarFieldTimeDataArray(T, coords=coords)

    src = td.HeatUniformSource(geometry=td.Box(center=(0, 0, 0), size=(1, 1, 1)), rate=10)

#    # no negative temperature
#    with pytest.raises(pydantic.ValidationError):
#        bc = td.HeatBCTemperature(temperature=-300)

