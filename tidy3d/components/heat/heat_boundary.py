"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Union

import pydantic as pd
import numpy as np

from .base import Tidy3dBaseModel
from .types import ArrayFloat1D, Ax
from .viz import add_ax_if_none

from ..constants import KELVIN, DENSITY, SPECIFIC_HEAT_CAPACITY, THERMAL_CONDUCTIVITY, PERMITTIVITY
from ..constants import CONDUCTIVITY, HEAT_FLUX


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

    heat_flux: pd.PositiveFloat = pd.Field(
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

    transfer_coeff: pd.PositiveFloat = pd.Field(
        title="Heat Transfer Coefficient",
        description=f"Heat flux value in units of {HEAT_FLUX}.",
        units=HEAT_FLUX,
    )



