"""Defines heat material specifications"""
from __future__ import annotations

# from abc import ABC, abstractmethod
from typing import Union

import pydantic as pd
# import numpy as np

from .simulation import HeatSimulation
from .medium import HeatSpecSolid
from .structure import HeatStructure
from .data_array import TemperatureFieldType

from ....components.base import Tidy3dBaseModel
from ....components.simulation import Simulation
from ....components.structure import Structure
# from ..types import ArrayFloat1D, Ax
# from ..viz import add_ax_if_none

from ....constants import KELVIN


class HeatSimulationData(Tidy3dBaseModel):
    """Stores results of a heat simulation.

    Example
    -------
    """

    heat_simulation: HeatSimulation = pd.Field(
        title="Heat Simulation",
        description="``HeatSimulation`` object describing the problem setup.",
    )

    temperature_data: TemperatureFieldType = pd.Field(
        title="Temperature Field",
        description="Temperature field obtained from heat simulation.",
        units=KELVIN,
    )

    def apply_heat_to_sim(self) -> Simulation:
        """Apply heat data to the original Tidy3D simulation (replaces appropriate media with CustomMedia). """

        sim_dict = self.heat_simulation.dict(
            exclude={
                "type",
                "heat_medium",
                "heat_structures",
#                "heat_sources",
                "heat_boundary_conditions",
                "heat_domain",
                "heat_grid_spec",
            }
        )
        structures = self.heat_simulation.heat_structures

        # For each structure that contains HeatSpecSolid sample temperature field and obtain
        # perturbed medium as a CustomMedium. Pack it into a regular Structure.
        # Currently we perform separate interpolation inside the bounding box of each HeatStructure,
        # a more efficient approach could be interpolating once on the entire grid and then
        # selecting subsets of points corresponding to each structure
        new_structures = []
        for s in structures:
            if isinstance(s, HeatStructure):
                med = s.medium
                heat_spec = med.heat_spec
                if isinstance(heat_spec, HeatSpecSolid):
                    # get structure's bounding box
                    bounding_box = s.geometry.bounding_box

                    # get mesh covering region of interest
                    sub_grid = self.heat_simulation.discretize(bounding_box)

                    # sample temperature
                    temp_values = {}

                    for d in "xyz":
                        coords = sub_grid[f"E{d}"]
                        x = coords.x
                        y = coords.y
                        z = coords.z

                        temp_values[d] = self.temperature_data.interp(x=x, y=y, z=z)

                    new_medium = med.to_medium_after_heat(temp_values)
                else:
                    new_medium = med.to_medium()

                new_s = Structure(medium=new_medium, geometry=s.geometry)
                new_structures.append(new_s)

        sim_dict["structures"] = new_structures

        med = self.heat_simulation.heat_medium
        heat_spec = med.heat_spec
        if isinstance(heat_spec, HeatSpecSolid):

            # get mesh covering region of interest
            sub_grid = self.heat_simulation.grid

            # sample temperature
            temp_values = {}

            for d in "xyz":
                coords = sub_grid[f"E{d}"]
                x = coords.x
                y = coords.y
                z = coords.z

                temp_values[d] = self.temperature_data.interp(x=x, y=y, z=z)

            new_medium = med.to_medium_after_heat(temp_values)
        else:
            new_medium = med.to_medium()

        sim_dict["medium"] = new_medium

        return Simulation.parse_obj(sim_dict)

