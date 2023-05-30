"""Defines heat material specifications"""
from __future__ import annotations

# from abc import ABC, abstractmethod
from typing import Union

import pydantic as pd
# import numpy as np

from .simulation import HeatSimulation
from .medium import HeatSpecSolid
from .structure import HeatStructure

from ....components.base import Tidy3dBaseModel
from ....components.simulation import Simulation
from ....components.structure import Structure
from ....components.medium import CustomMedium
from ....components.data.data_array import ScalarFieldTimeDataArray, ScalarFieldDataArray
from ....components.data.dataset import PermittivityDataset
# from ..types import ArrayFloat1D, Ax
# from ..viz import add_ax_if_none

# from ...constants import KELVIN, DENSITY, SPECIFIC_HEAT_CAPACITY, THERMAL_CONDUCTIVITY, PERMITTIVITY
# from ...constants import CONDUCTIVITY, HEAT_FLUX


TemperatureFieldType = Union[ScalarFieldTimeDataArray]


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
    )

    def apply_heat_to_sim(self) -> Simulation:
        """Apply heat data to the original Tidy3D simulation (replaces appropriate media with CustomMedia). """

        sim_dict = self.heat_simulation.dict(
            exclude={
                "type",
                "structures",
                "heat_sources",
                "heat_boundary_conditions",
                "heat_domain",
            }
        )
        sim = Simulation.parse_obj(sim_dict)
        structures = self.heat_simulation.structures

        new_structures = []
        for s in structures:
            if isinstance(s, HeatStructure):
                med = s.medium
                heat_spec = med.heat_spec
                if isinstance(heat_spec, HeatSpecSolid):
                    # get structure's bounding box
                    bounding_box = s.geometry.bounding_box

                    # get mesh covering region of interest
                    sub_grid = sim.discretize(bounding_box)

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

        return sim.updated_copy(structures=new_structures)

