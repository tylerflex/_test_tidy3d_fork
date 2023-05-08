"""Tests heat simulation objects."""
import pytest
import pydantic
import tidy3d as td

from ..utils import SIM_FULL


def test_heat_sim():

    bc1 = td.HeatBoundaryTemperature(temperature=300)
    bc2 = td.HeatBoundaryFlux(heat_flux=1)
    bc3 = td.HeatBoundaryConvection(ambient_temperature=400, transfer_coeff=1)

    pl1 = td.HeatBoundaryPlacementMediumMedium(bc=bc1, mediums=["dieletric", "lossy_dieletric"])
    pl2 = td.HeatBoundaryPlacementStructure(bc=bc2, structure="dieletric_box")
    pl3 = td.HeatBoundaryPlacementStructureStructure(bc=bc3, structures=["dieletric_box", "lossy_box"])
    pl4 = td.HeatBoundaryPlacementSimulation(bc=bc1)
    pl5 = td.HeatBoundaryPlacementStructureSimulation(bc=bc2, structure="dieletric_mesh")

    heat_sim = td.HeatSimulation(
        simulation=SIM_FULL,
        boundary_conditions=[pl1, pl2, pl3, pl4, pl5],
    )

    # wrong names given
    with pytest.raises(pydantic.ValidationError):
        pl = td.HeatBoundaryPlacementMediumMedium(bc=bc1, mediums=["badname", "lossy_dieletric"])
        heat_sim = td.HeatSimulation(
            simulation=SIM_FULL,
            boundary_conditions=[pl],
        )

    with pytest.raises(pydantic.ValidationError):
        pl = td.HeatBoundaryPlacementStructure(bc=bc2, structure="no_box")
        heat_sim = td.HeatSimulation(
            simulation=SIM_FULL,
            boundary_conditions=[pl],
        )

    with pytest.raises(pydantic.ValidationError):
        pl = td.HeatBoundaryPlacementStructureStructure(bc=bc3, structures=["no_box", "lossy_box"])
        heat_sim = td.HeatSimulation(
            simulation=SIM_FULL,
            boundary_conditions=[pl],
        )

    with pytest.raises(pydantic.ValidationError):
        pl = td.HeatBoundaryPlacementStructureSimulation(bc=bc2, structure="no_mesh")
        heat_sim = td.HeatSimulation(
            simulation=SIM_FULL,
            boundary_conditions=[pl],
        )

