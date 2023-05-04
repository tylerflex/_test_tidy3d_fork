"""Tests heat data objects."""
import pytest
import pydantic
import numpy as np
import tidy3d as td

from ..utils import SIM_FULL


def test_heat_data():

    heat_sim = td.HeatSimulation(simulation=SIM_FULL)

    nx, ny, nz = 9, 6, 5
    x = np.linspace(-5, 5, nx)
    y = np.linspace(-5, 5, ny)
    z = np.linspace(-5, 5, nz)
    T = np.random.default_rng().uniform(300, 350, (nx, ny, nz, 1))
    coords = dict(x=x, y=y, z=z, t=[0])
    temperature_field = td.ScalarFieldTimeDataArray(T, coords=coords)

    heat_sim_data = td.HeatSimulationData(
        heat_simulation=heat_sim,
        temperature_data=temperature_field,
    )

    new_sim = heat_sim_data.apply_heat_to_sim()
