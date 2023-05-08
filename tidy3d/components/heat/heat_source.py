"""Defines heat material specifications"""
from __future__ import annotations

from abc import ABC
from typing import Union

import pydantic as pd

from ..base import Tidy3dBaseModel
# from ..types import ArrayFloat1D, Ax
# from ..viz import add_ax_if_none
from ..geometry import GeometryType
from ..data.data_array import TimeDataArray, ScalarFieldTimeDataArray

from ...constants import VOLUMETRIC_HEAT_RATE


class HeatSource(ABC, Tidy3dBaseModel):
    """Abstract heat source."""


class HeatUniformSource(HeatSource):
    """Volumetric heat source.

    Example
    -------
    >>> const_func = TemperatureDependenceConstant(value=1)
    """

    geometry: GeometryType = pd.Field(
    title="Source Geometry",
        description="Geometry of the heat source.",
    )

    rate: Union[float, TimeDataArray] = pd.Field(
        title="Volumetric Heat Rate",
        description=f"Volumetric rate of heating or cooling (if negative) in units of {VOLUMETRIC_HEAT_RATE}.",
        units=VOLUMETRIC_HEAT_RATE,
    )


class HeatCustomSource(HeatSource):
    """Spatially dependent volumetric heat source.

    Example
    -------
    >>> const_func = TemperatureDependenceConstant(value=1)
    """

    geometry: GeometryType = pd.Field(
        title="Source Geometry",
        description="Geometry of the heat source.",
    )

    rate: ScalarFieldTimeDataArray = pd.Field(
        title="Volumetric Heat Rate",
        description="Spatially dependent volumetric rate of heating or cooling (if negative) in units of {VOLUMETRIC_HEAT_RATE}.",
        units=VOLUMETRIC_HEAT_RATE,
    )


HeatSourceType = Union[
    HeatUniformSource,
    HeatCustomSource,
]
