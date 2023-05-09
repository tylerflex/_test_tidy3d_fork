"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Tuple

import pydantic as pd
import numpy as np

from ..base import Tidy3dBaseModel
from ..types import ArrayFloat1D, Ax
from ..viz import add_ax_if_none

from ...exceptions import SetupError
from ...constants import KELVIN, HEAT_FLUX, HEAT_TRANSFER_COEFF


class HeatBoundary(ABC, Tidy3dBaseModel):
    """Abstract thermal boundary conditions."""


class HeatBoundaryTemperature(HeatBoundary):
    """Constant temperature thermal boundary conditions."""

    temperature: pd.PositiveFloat = pd.Field(
        title="Temperature",
        description=f"Temperature value in units of {KELVIN}.",
        units=KELVIN,
    )


class HeatBoundaryFlux(HeatBoundary):
    """Constant flux thermal boundary conditions."""

    heat_flux: float = pd.Field(
        title="Heat Flux",
        description=f"Heat flux value in units of {HEAT_FLUX}.",
        units=HEAT_FLUX,
    )


class HeatBoundaryConvection(HeatBoundary):
    """Convective thermal boundary conditions."""

    ambient_temperature: pd.PositiveFloat = pd.Field(
        title="Ambient Temperature",
        description=f"Ambient temperature value in units of {KELVIN}.",
        units=KELVIN,
    )

    transfer_coeff: pd.NonNegativeFloat = pd.Field(
        title="Heat Transfer Coefficient",
        description=f"Heat flux value in units of {HEAT_TRANSFER_COEFF}.",
        units=HEAT_TRANSFER_COEFF,
    )


HeatBoundaryType = Union[HeatBoundaryTemperature, HeatBoundaryFlux, HeatBoundaryConvection]


class HeatBoundaryPlacement(ABC, Tidy3dBaseModel):
    """Abstract location for thermal boundary conditions."""

    bc: HeatBoundaryType = pd.Field(
        title="Boundary Conditions",
        description="Boundary conditions applied at the selected location.",
    )


class HeatBoundaryPlacementStructure(HeatBoundaryPlacement):
    """Placement of thermal boundary conditions on the structure's boundary."""

    structure: str = pd.Field(
        title="Structure Name",
        description="Name of the structure.",
    )


class HeatBoundaryPlacementStructureStructure(HeatBoundaryPlacement):
    """Placement of thermal boundary conditions between two structures."""

    structures: Tuple[str, Union[str, None]] = pd.Field(
        title="Structures",
        description="Names of two structures.",
    )

    @pd.validator("structures", always=True)
    def unique_names(cls, val, values):
        """Error if the same structure is provided twice"""
        if val[0] == val[1]:
            raise SetupError(
                "The same structure is provided twice in "
                ":class:`HeatBoundaryPlacementStructureStructure`."
            )
        return val


class HeatBoundaryPlacementMediumMedium(HeatBoundaryPlacement):
    """Placement of thermal boundary conditions between two mediums."""

    mediums: Tuple[str, str] = pd.Field(
        title="Mediums",
        description="Names of two mediums.",
    )

    @pd.validator("mediums", always=True)
    def unique_names(cls, val, values):
        """Error if the same structure is provided twice"""
        if val[0] == val[1]:
            raise SetupError(
                "The same medium is provided twice in "
                ":class:`HeatBoundaryPlacementMediumMedium`."
            )
        return val


class HeatBoundaryPlacementSimulation(HeatBoundaryPlacement):
    """Placement of thermal boundary conditions on the simulation box boundary."""


class HeatBoundaryPlacementStructureSimulation(HeatBoundaryPlacement):
    """Placement of thermal boundary conditions on the simulation box boundary."""

    structure: str = pd.Field(
        title="Structure Name",
        description="Name of the structure.",
    )


HeatBoundaryPlacementType = Union[
    HeatBoundaryPlacementStructure,
    HeatBoundaryPlacementStructureStructure,
    HeatBoundaryPlacementMediumMedium,
    HeatBoundaryPlacementSimulation,
    HeatBoundaryPlacementStructureSimulation,
]
