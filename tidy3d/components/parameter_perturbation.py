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

from abc import ABC, abstractmethod
from typing import Callable, Union, Tuple, List
import functools

import pydantic as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from .data.data_array import SpatialDataArray, HeatDataArray, ChargeDataArray
from .base import Tidy3dBaseModel, cached_property
from ..constants import KELVIN, inf
#from ..exceptions import SetupError
from ..log import log
from ..components.types import Ax, ArrayLike, Complex, FieldVal
from ..components.viz import add_ax_if_none

""" Generic perturbation classes """


class AbstractPerturbation(ABC, Tidy3dBaseModel):
    """Abstract class for a generic perturbation."""

    @cached_property
    @abstractmethod
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[Complex, Complex]]:
        """Perturbation range."""

    @cached_property
    @abstractmethod
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""

    @staticmethod
    def _linear_range(interval: Tuple[float, float], ref: float, coeff: Union[float, Complex]):
        """Find value range for a linear perturbation."""
        if coeff == 0 or coeff == 0j:  # to avoid 0*inf
            return np.array([0, 0])
        return np.sort(coeff * (np.array(interval) - ref))

    @staticmethod
    def _get_val(
        field: Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray], val: FieldVal
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Get specified value from a field."""

        if val == "real":
            return np.real(field)

        elif val == "imag":
            return np.imag(field)

        elif val == "abs":
            return np.abs(field)

        elif val == "abs^2":
            return np.abs(field) ** 2

        elif val == "phase":
            return np.arctan2(np.real(field), np.imag(field))

        else:
            raise ValueError(
                "Unknown 'val' key. Argument 'val' can take values 'real', 'imag', 'abs', "
                "'abs^2', or 'phase'."
            )


""" Elementary heat perturbation classes """


def ensure_temp_in_range(
    sample: Callable[
        Union[ArrayLike[float], SpatialDataArray],
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray],
    ]
) -> Callable[
        Union[ArrayLike[float], SpatialDataArray],
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray],
]:
    """Decorate ``sample`` to log warning if temperature supplied is out of bounds."""

    @functools.wraps(sample)
    def _sample(
        self, temperature: Union[ArrayLike[float], SpatialDataArray]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """New sample function."""

        # if temperature range not present just return original function
        if self.temperature_range is None:
            return sample(self, temperature)

        temh_min, temh_max = self.temperature_range
        if np.any(temperature < temh_min) or np.any(temperature > temh_max):
            log.warning(
                "temperature passed to 'HeatPerturbation.sample()'"
                f"is outside of 'HeatPerturbation.temperature_range' = {self.temperature_range}"
            )
        return sample(self, temperature)

    return _sample


class HeatPerturbation(AbstractPerturbation):
    """Abstract class for heat perturbation."""

    temperature_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        (0, inf),
        title="Temperature range",
        description="Temparature range in which perturbation model is valid.",
        units=KELVIN,
    )

    @abstractmethod
    def sample(
        self, temperature: Union[ArrayLike[float], SpatialDataArray]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation.

        Parameters
        ----------
        temperature : Union[ArrayLike[float], SpatialDataArray]
            Temperature sample point(s).

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """

    @add_ax_if_none
    def plot(
        self,
        temperature: ArrayLike[float],
        val: FieldVal = "real",
        ax: Ax = None,
    ) -> Ax:
        """Plot perturbation using provided temperature sample points.

        Parameters
        ----------
        temperature : ArrayLike[float]
            Array of temperature sample points.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        temperature_numpy = np.array(temperature)

        values = self.sample(temperature_numpy)
        values = self._get_val(values, val)

        ax.plot(temperature_numpy, values)
        ax.set_xlabel("temperature (K)")
        ax.set_ylabel(f"{val}(perturbation value)")
        ax.set_title("temperature dependence")
        ax.set_aspect("auto")

        return ax


class LinearHeatPerturbation(HeatPerturbation):
    """Linear heat perturbation."""

    temperature_ref: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Reference temperature",
        description="Reference temperature.",
        units=KELVIN,
    )

    coeff: Union[float, Complex] = pd.Field(
        ...,
        title="Thermo-optic Coefficient",
        description="Thermo-optic Coefficient.",
    )

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[Complex, Complex]]:
        """Perturbation range."""
        return self._linear_range(self.temperature_range, self.temperature_ref, self.coeff)

    @ensure_temp_in_range
    def sample(
        self, temperature: Union[ArrayLike[float], SpatialDataArray]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation.

        Parameters
        ----------
        temperature : Union[ArrayLike[float], SpatialDataArray]
            Temperature sample point(s).

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """

        return self.coeff * (temperature - self.temperature_ref)

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""
        return np.iscomplex(self.coeff)


class CustomHeatPerturbation(HeatPerturbation):
    """Custom heat perturbation."""

    perturbation_values: HeatDataArray = pd.Field(
        ...,
        title="Perturbation Values",
        description="Sampled perturbation values.",
    )

    temperature_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        None,
        title="Temperature range",
        description="Temparature range in which perturbation model is valid. For "
            ":class:`.CustomHeatPerturbation` this field is computed automatically based on "
            "provided ``perturbation_values``",
        units=KELVIN,
    )

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[Complex, Complex]]:
        """Perturbation range."""
        return np.min(self.perturbation_values), np.max(self.perturbation_values)

    @pd.root_validator(skip_on_failure=True)
    def compute_temperature_range(cls, values):
        """Compute and set temperature range based on provided ``perturbation_values``."""
        if values["temperature_range"] is not None:
            log.warning(
                "Temperature range for 'CustomHeatPerturbation' is calculated automatically "
                "based on provided 'perturbation_values'. Provided 'temperature_range' will be overwritten."
            )

        perturbation_values = values["perturbation_values"]

        temperature_range = (
            float(np.min(perturbation_values.coords["T"])),
            float(np.max(perturbation_values.coords["T"])),
        )

        values.update({"temperature_range": temperature_range})

        return values

    @ensure_temp_in_range
    def sample(
        self, temperature: Union[ArrayLike[float], SpatialDataArray]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation.

        Parameters
        ----------
        temperature : Union[ArrayLike[float], SpatialDataArray]
            Temperature sample point(s).

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """

        t_range = self.temperature_range
        temperature_clip = np.clip(temperature, t_range[0], t_range[1])
        return self.perturbation_values.interp(T=temperature_clip).drop_vars("T")

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""
        return np.iscomplexobj(self.perturbation_values.data)


HeatPerturbationType = Union[LinearHeatPerturbation, CustomHeatPerturbation]


""" Elementary charge perturbation classes """


def ensure_charge_in_range(
        sample: Callable[
        [Union[ArrayLike[float], SpatialDataArray], Union[ArrayLike[float], SpatialDataArray]],
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray],
    ]
) -> Callable[
        [Union[ArrayLike[float], SpatialDataArray], Union[ArrayLike[float], SpatialDataArray]],
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray],
]:
    """Decorate ``sample`` to log warning if charge supplied is out of bounds."""

    @functools.wraps(sample)
    def _sample(
        self,
        electron_density: Union[ArrayLike[float], SpatialDataArray],
        hole_density: Union[ArrayLike[float], SpatialDataArray]
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """New sample function."""

        if self.electron_range:
            e_min, e_max = self.electron_range

            # if temperature range not present just return original function
            if np.any(electron_density < e_min) or np.any(electron_density > e_max):
                log.warning(
                    "Electron density values passed to 'ChargePerturbation.sample()'"
                    f"is outside of 'ChargePerturbation.electron_range' = {self.electron_range}"
                )

        if self.hole_range:
            h_min, h_max = self.hole_range

            # if temperature range not present just return original function
            if np.any(hole_density < h_min) or np.any(hole_density > h_max):
                log.warning(
                    "Hole density values passed to 'ChargePerturbation.sample()'"
                    f"is outside of 'ChargePerturbation.hole_range' = {self.hole_range}"
                )

        return sample(self, electron_density, hole_density)

    return _sample


class ChargePerturbation(AbstractPerturbation):
    """Abstract class for charge perturbation."""

    electron_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        (0, inf),
        title="Electron Density Range",
        description="Range of electrons densities in which perturbation model is valid.",
    )

    hole_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        (0, inf),
        title="Hole Density Range",
        description="Range of holes densities in which perturbation model is valid.",
    )

    @abstractmethod
    def sample(
        self,
        electron_density: Union[ArrayLike[float], SpatialDataArray],
        hole_density: Union[ArrayLike[float], SpatialDataArray],
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation.

        Parameters
        ----------
        electron_density : Union[ArrayLike[float], SpatialDataArray]
            Electron density sample point(s).
        hole_density : Union[ArrayLike[float], SpatialDataArray]
            Hole density sample point(s).

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """

    @add_ax_if_none
    def plot(
        self,
        electron_density: ArrayLike[float],
        hole_density: ArrayLike[float],
        val: FieldVal = "real",
        ax: Ax = None
    ) -> Ax:
        """Plot perturbation using provided charge sample points.

        Parameters
        ----------
        electron_density : Union[ArrayLike[float], SpatialDataArray]
            Array of electron density sample points.
        hole_density : Union[ArrayLike[float], SpatialDataArray]
            Array of hole density sample points.
        val : Literal['real', 'imag', 'abs', 'abs^2', 'phase'] = 'real'
            Which part of the field to plot.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        values = self.sample(electron_density, hole_density)
        values = self._get_val(values, val)

        e_mesh, h_mesh = np.meshgrid(electron_density, hole_density, indexing='ij')
        pc = ax.pcolormesh(e_mesh, h_mesh, values, shading="gouraud")
        plt.colorbar(pc, ax=ax)
        ax.set_xlabel("electron density (1/cm^3)")
        ax.set_ylabel("hole density (1/cm^3)")
        ax.set_title(f"charge dependence of {val}(perturbation value)")
        ax.set_aspect("auto")

        return ax


class LinearChargePerturbation(ChargePerturbation):
    """Linear charge perturbation."""

    electron_ref: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Reference Electron Density",
        description="Electron density reference value.",
    )

    hole_ref: pd.NonNegativeFloat = pd.Field(
        ...,
        title="Reference Hole Density",
        description="Hole density reference value.",
    )

    electron_coeff: float = pd.Field(
        ...,
        title="Sensitivity to Electron Density",
        description="Sensitivity to electron density values.",
    )

    hole_coeff: float = pd.Field(
        ...,
        title="Sensitivity to Hole Density",
        description="Sensitivity to hole density values.",
    )

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[Complex, Complex]]:
        """Perturbation range."""

        range_from_e = self._linear_range(self.electron_range, self.electron_ref, self.electron_coeff)
        range_from_h = self._linear_range(self.hole_range, self.hole_ref, self.hole_coeff)

        return range_from_e + range_from_h

    @ensure_charge_in_range
    def sample(
        self,
        electron_density: Union[ArrayLike[float], SpatialDataArray],
        hole_density: Union[ArrayLike[float], SpatialDataArray],
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation.

        Parameters
        ----------
        electron_density : Union[ArrayLike[float], SpatialDataArray]
            Electron density sample point(s).
        hole_density : Union[ArrayLike[float], SpatialDataArray]
            Hole density sample point(s).

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """
        e_mesh, h_mesh = np.meshgrid(electron_density, hole_density, indexing='ij')
        return (
            self.electron_coeff * (e_mesh - self.electron_ref) +
            self.hole_coeff * (h_mesh - self.hole_ref)
        )

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""
        return np.iscomplex(self.electron_coeff) and np.iscomplex(self.hole_coeff)


class CustomChargePerturbation(ChargePerturbation):
    """Custom charge perturbation."""

    perturbation_values: ChargeDataArray = pd.Field(
        ...,
        title="Petrubation Values",
        description="2D array of perturbation values.",
    )

    temperature_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        None,
        title="Temperature range",
        description="Temparature range in which perturbation model is valid. For "
        ":class:`.CustomHeatPerturbation` this field is computed automatically based on provided "
        "``perturbation_values``",
        units=KELVIN,
    )

    electron_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        None,
        title="Electron Density Range",
        description="Range of electrons densities in which perturbation model is valid. For "
            ":class:`.CustomChargePerturbation` this field is computed automatically based on "
            "provided ``perturbation_values``",
    )

    hole_range: Tuple[pd.NonNegativeFloat, pd.NonNegativeFloat] = pd.Field(
        None,
        title="Hole Density Range",
        description="Range of holes densities in which perturbation model is valid. For "
            ":class:`.CustomChargePerturbation` this field is computed automatically based on "
            "provided ``perturbation_values``",
    )

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[complex, complex]]:
        """Perturbation range."""
        return np.min(self.perturbation_values), np.max(self.perturbation_values)

    @pd.root_validator(skip_on_failure=True)
    def compute_eh_ranges(cls, values):
        """Compute and set electron and hole density ranges based on provided
        ``perturbation_values``.
        """
        if values["electron_range"] is not None:
            log.warning(
                "Electron density range for 'CustomChargePerturbation' is calculated automatically "
                "based on provided 'perturbation_values'. Provided 'electron_range' will be overwritten."
            )

        if values["hole_range"] is not None:
            log.warning(
                "Hole density range for 'CustomChargePerturbation' is calculated automatically "
                "based on provided 'perturbation_values'. Provided 'hole_range' will be overwritten."
            )

        perturbation_values = values["perturbation_values"]

        electron_range = (
            float(np.min(perturbation_values.coords["n"])),
            float(np.max(perturbation_values.coords["n"])),
        )

        hole_range = (
            float(np.min(perturbation_values.coords["p"])),
            float(np.max(perturbation_values.coords["p"])),
        )

        values.update({"electron_range": electron_range, "hole_range": hole_range})

        return values

    @ensure_charge_in_range
    def sample(
        self,
        electron_density: Union[ArrayLike[float], SpatialDataArray],
        hole_density: Union[ArrayLike[float], SpatialDataArray],
    ) -> Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]:
        """Sample perturbation.

        Parameters
        ----------
        electron_density : Union[ArrayLike[float], SpatialDataArray]
            Electron density sample point(s).
        hole_density : Union[ArrayLike[float], SpatialDataArray]
            Hole density sample point(s).

        Returns
        -------
        Union[ArrayLike[float], ArrayLike[Complex], SpatialDataArray]
            Sampled perturbation value(s).
        """
        e_clip = np.clip(electron_density, self.electron_range[0], self.electron_range[1])
        h_clip = np.clip(hole_density, self.hole_range[0], self.hole_range[1])
        return self.perturbation_values.interp(n=e_clip, p=h_clip).drop_vars(["n", "p"])

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""
        return np.iscomplexobj(self.perturbation_values.data)


ChargePerturbationType = Union[LinearChargePerturbation, CustomChargePerturbation]


PerturbationType = Union[HeatPerturbationType, ChargePerturbationType]


class ParameterPerturbation(Tidy3dBaseModel):
    """Class that summarizes parameter perturbation."""

    heat: HeatPerturbationType = pd.Field(
        None,
        title="Heat Perturbation",
        description="Heat perturbation to apply.",
    )

    charge: ChargePerturbationType = pd.Field(
        None,
        title="Charge Perturbation",
        description="Charge perturbation to apply.",
    )

    @cached_property
    def perturbation_list(self) -> List[PerturbationType]:
        """List of provided perturbations."""
        list = []
        for p in [self.heat, self.charge]:
            if p is not None:
                list.append(p)
        return list

    @cached_property
    def perturbation_range(self) -> Union[Tuple[float, float], Tuple[Complex, Complex]]:
        """Perturbation range."""
        prange = np.zeros(2)
        for p in self.perturbation_list:
            prange += p.perturbation_range
        return prange

    @staticmethod
    def _zeros_like(
        T: SpatialDataArray = None,
        n: SpatialDataArray = None,
        p: SpatialDataArray = None,
    ):
        """Check that fields have the same coordinates and return an array field with zeros."""
        template = None
        for field in [T, n, p]:
            if field is not None:
                if template is not None and field.coords != template.coords:
                    raise ValueError(
                        "temperature, electron_density, and hole_density must have the same "
                        "coordinates if provided."
                    )
                template = field

        if template is None:
            raise ValueError(
                "At least one of temperature, electron_density, or hole_density must be provided"
            )

        return xr.zeros_like(template)

    def apply_data(
        self,
        temperature: SpatialDataArray = None,
        electron_density: SpatialDataArray = None,
        hole_density: SpatialDataArray = None,
    ) -> SpatialDataArray:
        """Sample perturbations on provided heat and/or charge data. At least one of temperature,
        electron_density, and hole_density must be not 'None'. All provided fields must have
        identical coords.

        Parameters
        ----------
        temperature : SpatialDataArray = None
            Temperature field data.
        electron_density : SpatialDataArray = None
            Electron density field data.
        hole_density : SpatialDataArray = None
            Hole density field data.

        Returns
        -------
        SpatialDataArray
            Sampled perturbation field.
        """

        result = self._zeros_like(temperature, electron_density, hole_density)

        if self.is_complex:
            result = result + 0j

        if temperature is not None and self.heat is not None:
            result += self.heat.sample(temperature)

        if (electron_density is not None or hole_density is not None) and self.charge is not None:

            if electron_density is None:
                electron_density = 0

            if hole_density is None:
                hole_density = 0

            result += self.charge.sample(electron_density, hole_density)

        return result

    @cached_property
    def is_complex(self) -> bool:
        """Whether perturbation is complex valued."""

        return np.any([p.is_complex for p in self.perturbation_list])

