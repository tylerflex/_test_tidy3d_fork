"""Defines a heat solver compatible structure."""
from __future__ import annotations

import pydantic as pd
from typing import Union

from ....components.structure import Structure

from .medium import HeatMediumType


class HeatStructure(Structure):
    """A :class:`.Structure` registered with jax."""

    medium: HeatMediumType = pd.Field(
        ...,
        title="Medium",
        description="Medium of the structure, which is heat solver compatible.",
    )


HeatStructureType = Union[Structure, HeatStructure]
