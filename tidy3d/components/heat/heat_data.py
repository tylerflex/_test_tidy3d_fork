"""Defines heat material specifications"""
from __future__ import annotations

# from abc import ABC, abstractmethod
from typing import Union

import pydantic as pd
# import numpy as np

from ..base import Tidy3dBaseModel
from ..simulation import Simulation
from ..medium import CustomMedium
from .heat_simulation import HeatSimulation
from .heat_medium import HeatSpecSolid
from ..data.data_array import ScalarFieldTimeDataArray, ScalarFieldDataArray
from ..data.dataset import PermittivityDataset
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
        sim = self.heat_simulation.simulation
        structures = sim.structures

        new_structures = []
        for s in structures:
            med = s.medium
            heat_spec = med.heat_spec
            if isinstance(heat_spec, HeatSpecSolid):
                # get structure's bounding box
                bounding_box = s.geometry.bounding_box

                # get mesh covering region of interest
                sub_grid = sim.discretize(bounding_box)

                # sample material
                eps_components = {}

                for d in "xyz":
                    coords = sub_grid[f"E{d}"]
                    x = coords.x
                    y = coords.y
                    z = coords.z

                    temp_values = self.temperature_data.interp(x=x, y=y, z=z)
                    perm_values = med.permittivity + heat_spec.permittivity_change.sample(temp_values.data)
                    cond_values = med.conductivity + heat_spec.conductivity_change.sample(temp_values.data)
                    freq = 1
                    eps_values = med.eps_sigma_to_eps_complex(perm_values, cond_values, freq)
                    eps_components[f"eps_{d}{d}"] = ScalarFieldDataArray(eps_values, coords=dict(x=x, y=y, z=z, f=[freq]))

                # pack into a CustomMedium
                eps_dataset = PermittivityDataset(**eps_components)
                new_medium = CustomMedium(eps_dataset=eps_dataset, interp_method="nearest")

                # create a new structure
                new_s = s.updated_copy(medium=new_medium)
                new_structures.append(new_s)
            else:
                new_structures.append(s)

        return sim.updated_copy(structures=new_structures)

