"""Tests heat simulation objects."""
import pytest
import pydantic
import tidy3d as td

from ..utils import SIM_FULL


def test_heat_sim():

    heat_sim = td.HeatSimulation(simulation=SIM_FULL)
