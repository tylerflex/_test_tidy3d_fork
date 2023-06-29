# pylint: disable=invalid-name, too-many-lines
"""Defines perturbations to properties of the medium / materials"""
from __future__ import annotations

#from abc import ABC, abstractmethod
#from typing import Tuple, Union, Callable, Optional, Dict, List
#import functools
#from math import isclose

#import pydantic as pd
#import numpy as np
#import xarray as xr

#from .base import Tidy3dBaseModel, cached_property
#from .grid.grid import Coords, Grid
#from .types import PoleAndResidue, Ax, FreqBound, TYPE_TAG_STR, InterpMethod, Bound, ArrayComplex3D
#from .types import Axis, TensorReal
#from .data.dataset import PermittivityDataset
#from .data.data_array import SpatialDataArray, ScalarFieldDataArray
#from .viz import add_ax_if_none
#from .geometry import Geometry
#from .validators import validate_name_str
#from ..constants import C_0, pec_val, EPSILON_0, LARGE_NUMBER, fp_eps
#from ..constants import HERTZ, CONDUCTIVITY, PERMITTIVITY, RADPERSEC, MICROMETER, SECOND
#from ..exceptions import ValidationError, SetupError
#from ..log import log
#from .transformation import RotationType


from .data.data_array import SpatialDataArray, ScalarFieldDataArray
from abc import ABC, abstractmethod
import pydantic as pd

from .base import Tidy3dBaseModel, cached_property
from typing import Union, Tuple
import pydantic as pd
from .medium import Medium, PoleResidue, PoleAndResidue
from ..constants import KELVIN, CONDUCTIVITY, PERMITTIVITY


class HeatPerturbation(ABC, Tidy3dBaseModel):
    """Abstract class for heat perturbation."""

    temperature_range: Tuple[float, float] = pd.Field(
        None,
        title="Temperature range",
        description="Temparature range in which perturbation model is valid.",
        units=KELVIN,
    )

    @cached_property
    @abstractmethod
    def perturbation_range(self) -> Tuple[float, float]:
        """Perturbation range."""

    @abstractmethod
    def sample(self, temperature: float) -> float:
        """Sample perturbation."""


class LinearHeatPerturbation(HeatPerturbation):

    reference_temperature: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Reference temperature",
        description="Reference temperature.",
        units=KELVIN,
    )

    thermo_optic_coeff: float = pd.Field(
        ...,
        title="Thermo-optic Coefficient",
        description="Thermo-optic Coefficient.",
    )

    @cached_property
    def perturbation_range(self) -> Tuple[float, float]:
        return np.sort(thermo_optic_coeff * (np.array(temperature_range) - reference_temperature))

    def sample(self, temperature: float) -> float:
        """Sample perturbation."""
        return thermo_optic_coeff * (temperature - reference_temperature)


class AbstractMediumPerturbation(ABC, Tidy3dBaseModel):
    """Abstract class for medium perturbation."""

    @abstractmethod
    def apply_heat_and_charge(
        self,
        temperature: SpatialDataArray = None,
        electron_concentration: SpatialDataArray = None,
        hole_concentration: SpatialDataArray = None
    ) -> AbstractCustomMedium:
        """Sample perturbations on provided heat and/or charge data and create a custom medium."""


class MediumPerturbation(Medium, AbstractMediumPerturbation):
    """Dispersionless medium with perturbations.

    Example
    -------
    >>> dielectric = Medium(permittivity=4.0, name='my_medium')
    >>> eps = dielectric.eps_model(200e12)
    """

    permittivity_perturbations: Tuple[PerturbationType, ...] = pd.Field(
        ...,
        title="Permittivity Perturbations",
        description="List of heat and/or charge perturbations to permittivity.",
    )

    conductivity_perturbations: Tuple[PerturbationType, ...] = pd.Field(
        (),
        title="Permittivity Perturbations",
        description="List of heat and/or charge perturbations to permittivity.",
    )

    @pd.validator("permittivity_perturbations", always=True, allow_reuse=True)
    def warn_permittivity_range(cls, val, values):
        min_permittivity = values["permittivity"]

        for perm_change in val:
            perm_change_min, _ = perm_change.perturbation_range
            min_permittivity += perm_change_min

        if min_permittivity < 1:
            log.warning("Permittivity can potentially become less than one for a MediumPerturbation material.")

        return val

    @pd.validator("conductivity_perturbations", always=True, allow_reuse=True)
    def warn_conductivity_range(cls, val, values):
        min_conductivity = values["conductivity"]

        for cond_change in val:
            cond_change_min, _ = perm_change.perturbation_range
            min_conductivity += cond_change_min

        if min_conductivity < 0:
            log.warning("Conductivity can potentially become less than zero for a MediumPerturbation material.")

        return val

    def apply_heat_and_charge(
        self,
        temperature: SpatialDataArray = None,
        electron_concentration: SpatialDataArray = None,
        hole_concentration: SpatialDataArray = None
    ) -> CustomMedium:
        """Sample perturbations on provided heat and/or charge data and create a custom medium."""

        if temperature:
            permittivity_field = self.permittivity * xr.ones_like(temperature)
        elif electron_concentration:
            permittivity_field = self.permittivity * xr.ones_like(electron_concentration)
        elif hole_concentration:
            permittivity_field = self.permittivity * xr.ones_like(hole_concentration)





medium = td.MediumPerturbation(
    permittivity=1,
    permittivity_perturbation=[td.LinearHeatPerturbation(...)]
)

class ChargePerturbation(ABC, Tidy3dBaseModel):
    """Abstract class for charge perturbation."""

    hole_concentration_range: Tuple[float, float] = pd.Field(
        None,
        title="Hole concentration range",
        description="Hole concentration range in which perturbation model is valid.",
    )

    electron_concentration_range: Tuple[float, float] = pd.Field(
        None,
        title="Electron concentration range",
        description="Electron concentration range in which perturbation model is valid.",
    )


class MediumPerturbation(ABC, Tidy3dBaseModel):
    """Abstract class for non-dispersive medium perturbation."""


    @cached_property
    @abstractmethod
    def permittivity_change_range(self) -> Tuple[float, float]:
        """Range of permittivity perturbation."""

    @cached_property
    @abstractmethod
    def conductivity_change_range(self) -> Tuple[float, float]:
        """A spatially varying medium."""


#class MediumChargePerturbation(ABC, Tidy3dBaseModel):

#    hole_concentration_range: Tuple[float, float] = pd.Field(
#        None,
#        title="Hole concentration range",
#        description="Hole concentration range in which perturbation model is valid.",
#    )

#    electron_concentration_range: Tuple[float, float] = pd.Field(
#        None,
#        title="Electron concentration range",
#        description="Electron concentration range in which perturbation model is valid.",
#    )

#    @abstractmethod
#    @cached_property
#    def permittivity_change_range(self) -> Tuple[float, float]:

#    @abstractmethod
#    @cached_property
#    def conductivity_change_range(self) -> Tuple[float, float]:


class LinearMediumHeatPerturbation(MediumPerturbation, HeatPerturbation):
    """Linear heat perturbation of a non-dispersive medium."""

    reference_temperature: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Reference temperature",
        description="Reference temperature.",
        units=KELVIN,
    )

    permittivity_coeff: float = pd.Field(
        ...,
        title="Permittivity thermo-optic coefficient",
        description="Permittivity thermo-optic coefficient.",
        units=f"{PERMITTIVITY}/{KELVIN}",
    )

    conductivity_coeff: float = pd.Field(
        None,
        title="Conductivity thermo-optic coefficient",
        description="Conductivity thermo-optic coefficient.",
        units=f"{CONDUCTIVITY}/{KELVIN}",
    )

    @cached_property
    def permittivity_change_range(self) -> Tuple[float, float]:
        return np.sort(permittivity_coeff * (np.array(temperature_range) - reference_temperature))

    @cached_property
    def conductivity_change_range(self) -> Tuple[float, float]:
        if conductivity_coeff:
            return np.sort(conductivity_coeff * (np.array(temperature_range) - reference_temperature))

        return (0, 0)


class CustomMediumHeatPerturbation(MediumPerturbation, HeatPerturbation):
    """Custom heat perturbation of a non-dispersive medium."""

    permittivity_change: HeatDataArray = pd.Field(
        ...,
        title="Permittivity perturbation",
        description="Permittivity thermo-optic coefficient.",
        units=PERMITTIVITY,
    )

    conductivity_change: HeatDataArray = pd.Field(
        None,
        title="Conductivity perturbation",
        description="Conductivity thermo-optic coefficient.",
        units=CONDUCTIVITY,
    )

    @cached_property
    def permittivity_change_range(self) -> Tuple[float, float]:

        return np.min(permittivity_change), np.max(permittivity_change)

    @cached_property
    def conductivity_change_range(self) -> Tuple[float, float]:
        if conductivity_coeff:
            return np.min(conductivity_change), np.max(conductivity_change)
        return (0, 0)

class PoleResiduePerturbation(ABC, Tidy3dBaseModel):
