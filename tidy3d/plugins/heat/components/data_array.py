"""Defines heat material specifications"""
from __future__ import annotations

from typing import Union
from ....components.data.data_array import ScalarFieldTimeDataArray, SpatialDataArray


TemperatureFieldType = Union[ScalarFieldTimeDataArray, SpatialDataArray]

# TODO: unstructured data?
