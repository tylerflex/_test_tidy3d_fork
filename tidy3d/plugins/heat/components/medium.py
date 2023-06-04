"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union, Dict

import pydantic as pd
import numpy as np

from ....components.base import Tidy3dBaseModel
from ....components.types import ArrayFloat1D, Ax
from ....components.viz import add_ax_if_none
from ....components.medium import Medium, CustomMedium
from ....components.data.data_array import ScalarFieldDataArray
from ....components.data.dataset import PermittivityDataset


from ....constants import KELVIN, DENSITY, SPECIFIC_HEAT_CAPACITY, THERMAL_CONDUCTIVITY, PERMITTIVITY
from ....constants import CONDUCTIVITY

from .data_array import TemperatureFieldType


class TemperatureDependence(ABC, Tidy3dBaseModel):
    """Abstract temperature dependence function."""

    @abstractmethod
    def sample(self, temperature: ArrayFloat1D) -> ArrayFloat1D:
        """Sample function at provided temperature sample points.

        Parameters
        ----------
        temperature : ArrayFloat1D
            Array of temperature sample points.

        Returns
        -------
        ArrayFloat1D
            Function values measured at sample points.
        """

    @add_ax_if_none
    def plot(self, temperature: ArrayFloat1D, ax: Ax = None) -> Ax:
        """Plot function using provided temperature sample points.

        Parameters
        ----------
        temperature : ArrayFloat1D
            Array of temperature sample points.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        temp_numpy = np.array(temperature)

        values = self.sample(temp_numpy)

        ax.plot(temp_numpy, values)
        ax.set_xlabel("temperature (K)")
        ax.set_title("Temperature dependence")
        ax.set_aspect("auto")

        return ax


class TemperatureDependenceConstant(TemperatureDependence):
    """Constant function of temperature.

    Example
    -------
    >>> const_func = TemperatureDependenceConstant(value=1)
    """

    value: float = pd.Field(
        title="Function value",
        description="Value of the constant function.",
    )

    def sample(self, temperature: ArrayFloat1D) -> ArrayFloat1D:
        """Sample function at provided temperature sample points."""
        return self.value * np.ones_like(temperature)


class TemperatureDependenceLinear(TemperatureDependence):
    """Linear function of temperature.

    Example
    -------
    >>> linear_func = TemperatureDependenceLinear(temp_ref=300, coeff=0.1)
    """

    ref_temp: pd.NonNegativeFloat = pd.Field(
        title="Reference temperature",
        description=f"Reference temperature in units of {KELVIN}.",
        units=KELVIN,
    )

    coeff: float = pd.Field(
        title="Linear heat coefficient",
        description="Linear heat coefficient.",
    )

    def sample(self, temperature: ArrayFloat1D) -> ArrayFloat1D:
        """Sample function at provided temperature sample points."""
        return self.coeff * (temperature - self.ref_temp)


class TemperatureDependenceTable(TemperatureDependence):
    """Function of temperature as a look-up table.

    Example
    -------
    >>> temp_points = [200, 300, 400]
    >>> func_values = [0, 2, 1]
    >>> table_func = TemperatureDependenceTable(temp_points=temp_points, func_values=func_values)
    """

    temp_points: ArrayFloat1D = pd.Field(
        title="Temperature values",
        description=f"Temperature sample points in units of {KELVIN}.",
        units=KELVIN,
    )

    func_values: ArrayFloat1D = pd.Field(
        title="Function values",
        description="Function values measured at temperature sample points.",
    )

    def sample(self, temperature: ArrayFloat1D) -> ArrayFloat1D:
        """Sample function at provided temperature sample points."""
        return np.interp(temperature, self.temp_points, self.func_values)


TemperatureDependenceType = Union[
    TemperatureDependenceConstant,
    TemperatureDependenceLinear,
    TemperatureDependenceTable
]


class HeatSpec(ABC, Tidy3dBaseModel):
    """Abstract heat material specification."""


# Liquid class
class HeatSpecFluid(HeatSpec):
    """Liquid heat material class."""


# Solid class
class HeatSpecSolid(HeatSpec):
    """Solid heat material class.

    Example
    -------
    >>> perm_change = TemperatureDependenceLinear(temp_ref=300, coeff=0.1)
    >>> cond_change = TemperatureDependenceConstant(value=0)
    >>> heat_spec = HeatSpecSolid(
    >>>     density=1,
    >>>     capacity=2,
    >>>     conductivity=3,
    >>>     permittivity_change=perm_change,
    >>>     conductivity_change=cond_change,
    >>> )
    """

    density: pd.PositiveFloat = pd.Field(
        title="Material density",
        description=f"Material density in units of {DENSITY}.",
        units=DENSITY,
    )

    capacity: pd.PositiveFloat = pd.Field(
        title="Specific heat capacity",
        description=f"Material density in units of {SPECIFIC_HEAT_CAPACITY}.",
        units=SPECIFIC_HEAT_CAPACITY,
    )

    conductivity: pd.PositiveFloat = pd.Field(
        title="Thermal conductivity",
        description=f"Thermal conductivity of material in units of {THERMAL_CONDUCTIVITY}.",
        units=THERMAL_CONDUCTIVITY,
    )

    permittivity_change: TemperatureDependenceType = pd.Field(
        title="Permittivity change",
        description="Change in permittivity of material due to heat effects as a function of temperature.",
        units=PERMITTIVITY,
    )

    conductivity_change: TemperatureDependenceType = pd.Field(
        title="Conductivity change",
        description="Change in conductivity of material due to heat effects as a function of temperature.",
        units=CONDUCTIVITY,
    )

    @add_ax_if_none
    def plot(self, temperature: ArrayFloat1D, ax: Ax = None) -> Ax:
        """Plot changes in permittivity and conductivity as functions of temperatue.

        Parameters
        ----------
        temperature : ArrayFloat1D
            Array of temperature sample points.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        self.permittivity_change.plot(temperature=temperature, ax=ax)
        self.conductivity_change.plot(temperature=temperature, ax=ax)
        ax.lines[-2].set_label("Permettivity change")
        ax.lines[-1].set_label("Conductivity change")
        ax.legend()

        return ax


HeatSpecType = Union[HeatSpecFluid, HeatSpecSolid]


class HeatMedium(Medium):

    heat_spec: HeatSpecType = pd.Field(
        None,
        title="Heat Specification",
        description="Specification of thermal properties for the medium. If ``None``, this material"
        " will be excluded from heat simulations."
    )

    def to_medium(self):
        """Create an analogous Tidy3d medium without heat specs. """

        med_dict = self.dict(
            exclude={
                "type",
                "heat_spec",
            }
        )

        return Medium.parse_obj(med_dict)

    def to_medium_after_heat(self, temperature_data: Dict[str, TemperatureFieldType]) -> CustomMedium:
        """Apply heat data sample on Yee grid to the medium and create the resulting CustomMedium. """

        if isinstance(self.heat_spec, HeatSpecFluid):
            return self.to_medium()

        # sample material
        eps_components = {}

        for d in "xyz":

            temp_values = temperature_data[d]
            x = temp_values.coords["x"]
            y = temp_values.coords["y"]
            z = temp_values.coords["z"]

            perm_values = self.permittivity + self.heat_spec.permittivity_change.sample(temp_values.data)
            cond_values = self.conductivity + self.heat_spec.conductivity_change.sample(temp_values.data)
            freq = 1
            eps_values = self.eps_sigma_to_eps_complex(perm_values, cond_values, freq)
            eps_components[f"eps_{d}{d}"] = ScalarFieldDataArray(eps_values, coords=dict(x=x, y=y, z=z, f=[freq]))

        # pack into a CustomMedium
        eps_dataset = PermittivityDataset(**eps_components)
        return CustomMedium(eps_dataset=eps_dataset, interp_method="nearest")


HeatMediumType = Union[HeatMedium]
