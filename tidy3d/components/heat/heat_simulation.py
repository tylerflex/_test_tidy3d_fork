"""Defines heat simulation class"""
from __future__ import annotations

# from abc import ABC, abstractmethod
# from typing import Union

import pydantic as pd
# import numpy as np

from ..base import Tidy3dBaseModel
# from ..types import ArrayFloat1D, Ax
# from ..viz import add_ax_if_none
from ..simulation import Simulation

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

#    sources: Tuple[HeatSourceType, ...] = pd.Field(
#        ...,
#        title="Heat Sources",
#        description="List of heat sources.",
#    )

#    boundary_conditions: Tuple[HeatBoundaryType, ...] = pd.Field(
#        ...,
#        title="Boundary Conditions",
#        description="List of boundary conditions.",
#    )

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
