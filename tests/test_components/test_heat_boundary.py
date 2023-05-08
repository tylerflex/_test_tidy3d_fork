"""Tests heat medium objects."""
import pytest
import pydantic
import tidy3d as td
#import matplotlib.pylab as plt


def test_heat_bc():

    bc = td.HeatBoundaryTemperature(temperature=300)
    bc = td.HeatBoundaryFlux(heat_flux=1)
    bc = td.HeatBoundaryConvection(ambient_temperature=400, transfer_coeff=1)

    # no negative temperature
    with pytest.raises(pydantic.ValidationError):
        bc = td.HeatBoundaryTemperature(temperature=-300)

    # no negative transfer coeff
    with pytest.raises(pydantic.ValidationError):
        bc = td.HeatBoundaryConvection(ambient_temperature=400, transfer_coeff=-1)

    # no negative temperature
    with pytest.raises(pydantic.ValidationError):
        bc = td.HeatBoundaryConvection(ambient_temperature=-400, transfer_coeff=1)


def test_heat_bc_placement():

    bc1 = td.HeatBoundaryTemperature(temperature=300)
    bc2 = td.HeatBoundaryFlux(heat_flux=1)
    bc3 = td.HeatBoundaryConvection(ambient_temperature=400, transfer_coeff=1)

    pl = td.HeatBoundaryPlacementMediumMedium(bc=bc1, mediums=["one", "two"])
    pl = td.HeatBoundaryPlacementStructure(bc=bc2, structure="one")
    pl = td.HeatBoundaryPlacementStructureStructure(bc=bc3, structures=["one", "two"])
    pl = td.HeatBoundaryPlacementSimulation(bc=bc1)
    pl = td.HeatBoundaryPlacementStructureSimulation(bc=bc2, structure="one")

    with pytest.raises(pydantic.ValidationError):
        pl = td.HeatBoundaryPlacementMediumMedium(bc=bc1, mediums=["one"])

    with pytest.raises(pydantic.ValidationError):
        pl = td.HeatBoundaryPlacementStructureStructure(bc=bc2, structures=["one", "two", "three"])

    with pytest.raises(pydantic.ValidationError):
        pl = td.HeatBoundaryPlacementStructure(bc=bc3, structure=["one", "two"])
