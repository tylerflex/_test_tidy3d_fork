"""Defines heat simulation class"""
from __future__ import annotations

# from abc import ABC, abstractmethod
from typing import Tuple

import pydantic as pd
# import numpy as np

from .heat_source import HeatSourceType
from .heat_boundary import HeatBoundaryPlacementType
from .heat_boundary import HeatBoundaryPlacementStructure, HeatBoundaryPlacementStructureStructure
from .heat_boundary import HeatBoundaryPlacementStructureSimulation
from .heat_boundary import HeatBoundaryPlacementMediumMedium

from ..base import Tidy3dBaseModel
# from ..types import ArrayFloat1D, Ax
# from ..viz import add_ax_if_none
from ..simulation import Simulation

from ...exceptions import SetupError
# from ...constants import KELVIN, DENSITY, SPECIFIC_HEAT_CAPACITY, THERMAL_CONDUCTIVITY, PERMITTIVITY
# from ...constants import CONDUCTIVITY, HEAT_FLUX


class HeatSimulation(Tidy3dBaseModel):
    """Contains all information about heat simulation.

    Example
    -------
    >>>
    """

    simulation: Simulation = pd.Field(
        title="Simulation",
        description="Tidy3D simulation object describing problem geometry.",
    )

    sources: Tuple[HeatSourceType, ...] = pd.Field(
        (),
        title="Heat Sources",
        description="List of heat sources.",
    )

    boundary_conditions: Tuple[HeatBoundaryPlacementType, ...] = pd.Field(
        (),
        title="Boundary Conditions",
        description="List of boundary conditions.",
    )

    @pd.validator("boundary_conditions", always=True)
    def names_exist(cls, val, values):
        """Error if boundary conditions point to non-existing structures/media"""
        structures = values["simulation"].structures
        mediums = values["simulation"].mediums
        structures_names = {s.name for s in structures}
        mediums_names = {m.name for m in mediums}

        for bc in val:
            if isinstance(bc, (HeatBoundaryPlacementStructure, HeatBoundaryPlacementStructureSimulation)):
                if bc.structure not in structures_names:
                    raise SetupError(
                        f"Structure {bc.structure} provided in a :class:`{bc.type}` "
                        "is not found among simulation structures."
                    )
            if isinstance(bc, (HeatBoundaryPlacementStructureStructure)):
                for ind in range(2):
                    if bc.structures[ind] not in structures_names:
                        raise SetupError(
                            f"Structure {bc.structures[ind]} provided in a :class:`{bc.type}` "
                            "is not found among simulation structures."
                        )
            if isinstance(bc, (HeatBoundaryPlacementMediumMedium)):
                for ind in range(2):
                    if bc.mediums[ind] not in mediums_names:
                        raise SetupError(
                            f"Material {bc.mediums[ind]} provided in a :class:`{bc.type}` "
                            "is not found among simulation mediums."
                        )
        return val

#    grid_spec: HeatGridSpec = pd.Field(
#        title="Grid Specification",
#        description="Grid specification for heat simulation.",
#    )

#    heat_domain: Box = pd.Field(
#        ...,
#        title="Heat Simulation Domain",
#        description="Domain in which heat simulation is solved. If ``None`` heat simulation is "
#        "solved in the entire domain of the Tidy3D simulation."
#    )
