"""Tests heat medium objects."""
import pytest
import pydantic
import tidy3d as td
#import matplotlib.pylab as plt


def test_heat_bc():

    bc = td.HeatBCTemperature(temperature=300)
    bc = td.HeatBCFlux(heat_flux=1)
    bc = td.HeatBCConvection(ambient_temperature=400, transfer_coeff=1)

    # no negative temperature
    with pytest.raises(pydantic.ValidationError):
        bc = td.HeatBCTemperature(temperature=-300)

    # no negative transfer coeff
    with pytest.raises(pydantic.ValidationError):
        bc = td.HeatBCConvection(ambient_temperature=400, transfer_coeff=-1)

    # no negative temperature
    with pytest.raises(pydantic.ValidationError):
        bc = td.HeatBCConvection(ambient_temperature=-400, transfer_coeff=1)


def test_heat_bc_placement():

    bc1 = td.HeatBCTemperature(temperature=300)
    bc2 = td.HeatBCFlux(heat_flux=1)
    bc3 = td.HeatBCConvection(ambient_temperature=400, transfer_coeff=1)

    pl = td.HeatBCPlacementMediumMedium(bc=bc1, mediums=["one", "two"])
    pl = td.HeatBCPlacementStructure(bc=bc2, structure="one")
    pl = td.HeatBCPlacementStructureStructure(bc=bc3, structures=["one", "two"])
    pl = td.HeatBCPlacementSimulation(bc=bc1)
    pl = td.HeatBCPlacementStructureSimulation(bc=bc2, structure="one")

    with pytest.raises(pydantic.ValidationError):
        pl = td.HeatBCPlacementMediumMedium(bc=bc1, mediums=["one"])

    with pytest.raises(pydantic.ValidationError):
        pl = td.HeatBCPlacementStructureStructure(bc=bc2, structures=["one", "two", "three"])

    with pytest.raises(pydantic.ValidationError):
        pl = td.HeatBCPlacementStructure(bc=bc3, structure=["one", "two"])
