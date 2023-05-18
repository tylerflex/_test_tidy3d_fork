"""Defines heat simulation class"""
from __future__ import annotations

# from abc import ABC, abstractmethod
from typing import Tuple, List

import pydantic as pd
# import numpy as np

from shapely.plotting import plot_line
from shapely import LineString, MultiLineString, GeometryCollection

from .heat_source import HeatSourceType
from .heat_boundary import HeatBCTemperature, HeatBCFlux, HeatBCConvection
from .heat_boundary import HeatBCPlacementType
from .heat_boundary import HeatBCPlacementStructure, HeatBCPlacementStructureStructure
from .heat_boundary import HeatBCPlacementStructureSimulation, HeatBCPlacementSimulation
from .heat_boundary import HeatBCPlacementMediumMedium

from ..base import Tidy3dBaseModel, cached_property
from ..types import Ax, Shapely
from ..viz import add_ax_if_none, equal_aspect, plot_params_heat_bc, PlotParams
from ..viz import HEAT_BC_COLOR_TEMPERATURE, HEAT_BC_COLOR_FLUX, HEAT_BC_COLOR_CONVECTION
from ..simulation import Simulation
from ..structure import Structure
from ..geometry import Box

from ...exceptions import SetupError
from ...constants import inf
# from ...constants import KELVIN, DENSITY, SPECIFIC_HEAT_CAPACITY, THERMAL_CONDUCTIVITY, PERMITTIVITY
# from ...constants import CONDUCTIVITY, HEAT_FLUX

HEAT_BACK_STRUCTURE_STR = "<<<HEAT_BACKGROUND_STRUCTURE>>>"


class HeatSimulation(Tidy3dBaseModel):
    """Contains all information about heat simulation.

    Example
    -------
    >>> FIXME
    """

    simulation: Simulation = pd.Field(
        title="Simulation",
        description="Tidy3D simulation object describing problem geometry.",
    )

    sources: Tuple[HeatSourceType, ...] = pd.Field(
        (),
        title="Heat Sources",
        description="List of heat sources.",
    )

    boundary_conditions: Tuple[HeatBCPlacementType, ...] = pd.Field(
        (),
        title="Boundary Conditions",
        description="List of boundary conditions.",
    )

#    grid_spec: HeatGridSpec = pd.Field(
#        title="Grid Specification",
#        description="Grid specification for heat simulation.",
#    )

    heat_domain: Box = pd.Field(
        None,
        title="Heat Simulation Domain",
        description="Domain in which heat simulation is solved. If ``None`` heat simulation is "
        "solved in the entire domain of the Tidy3D simulation."
    )

    @pd.validator("boundary_conditions", always=True)
    def names_exist(cls, val, values):
        """Error if boundary conditions point to non-existing structures/media"""
        structures = values["simulation"].structures
        mediums = values["simulation"].mediums
        structures_names = {s.name for s in structures}
        mediums_names = {m.name for m in mediums}

        for bc in val:
            if isinstance(bc, (HeatBCPlacementStructure, HeatBCPlacementStructureSimulation)):
                if bc.structure not in structures_names:
                    raise SetupError(
                        f"Structure {bc.structure} provided in a :class:`{bc.type}` "
                        "is not found among simulation structures."
                    )
            if isinstance(bc, (HeatBCPlacementStructureStructure)):
                for ind in range(2):
                    if bc.structures[ind] and bc.structures[ind] not in structures_names:
                        raise SetupError(
                            f"Structure {bc.structures[ind]} provided in a :class:`{bc.type}` "
                            "is not found among simulation structures."
                        )
            if isinstance(bc, (HeatBCPlacementMediumMedium)):
                for ind in range(2):
                    if bc.mediums[ind] not in mediums_names:
                        raise SetupError(
                            f"Material {bc.mediums[ind]} provided in a :class:`{bc.type}` "
                            "is not found among simulation mediums."
                        )
        return val

    @cached_property
    def background_structure(self) -> Structure:
        """Returns structure representing the background of the :class:`.HeatSimulation`."""

        # Unlike the FDTD Simulation.background_structure, the current one is also used to provide/
        # information about domain in which heat simulation is solved. Thus, we set its boundaries
        # either to self.heat_domain or, if None, to bounding box of self.simulation
        if self.heat_domain:
            heat_domain_actual = self.heat_domain
        else:
            heat_domain_actual = self.simulation.bounding_box

        fdtd_background = self.simulation.background_structure
        return fdtd_background.updated_copy(geometry=heat_domain_actual, name=HEAT_BACK_STRUCTURE_STR)

    @equal_aspect
    @add_ax_if_none
    def plot_heat_boundaries(  # pylint: disable=too-many-arguments,too-many-locals
        self,
        x: float = None,
        y: float = None,
        z: float = None,
        ax: Ax = None,
    ) -> Ax:
        """Plot each of simulation's boundary conditions on a plane defined by one nonzero x,y,z
        coordinate.

        Parameters
        ----------
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.
        ax : matplotlib.axes._subplots.Axes = None
            Matplotlib axes to plot on, if not specified, one is created.

        Returns
        -------
        matplotlib.axes._subplots.Axes
            The supplied or created matplotlib axes.
        """

        # get structure list
        structures = [self.background_structure]
        structures += list(self.simulation.structures)

        # construct slicing plane
        axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
        center = Box.unpop_axis(position, (0, 0), axis=axis)
        size = Box.unpop_axis(0, (inf, inf), axis=axis)
        plane = Box(center=center, size=size)

        # get boundary conditions in the plane
        boundaries = self._construct_heat_boundaries(
            structures=structures,
            plane=plane,
            boundary_conditions=self.boundary_conditions,
        )

        # plot boundary conditions
        for (bc, bdry) in boundaries:
            ax = self._plot_boundary_condition(boundary=bdry, condition=bc, ax=ax)

        ax = self.simulation._set_plot_bounds(ax=ax, x=x, y=y, z=z)

        # clean up the axis display
        axis, position = self.simulation.parse_xyz_kwargs(x=x, y=y, z=z)
        ax = self.simulation.add_ax_labels_lims(axis=axis, ax=ax)
        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

        return ax

    def _get_bc_plot_params(self, condition: HeatBCPlacementType) -> PlotParams:
        """Constructs the plot parameters for given boundary conditions."""

        plot_params = plot_params_heat_bc
        bc = condition.bc

        if isinstance(bc, HeatBCTemperature):
            plot_params = plot_params.updated_copy(edgecolor=HEAT_BC_COLOR_TEMPERATURE)
        elif isinstance(bc, HeatBCFlux):
            plot_params = plot_params.updated_copy(edgecolor=HEAT_BC_COLOR_FLUX)
        elif isinstance(bc, HeatBCConvection):
            plot_params = plot_params.updated_copy(edgecolor=HEAT_BC_COLOR_CONVECTION)

        return plot_params

    def _plot_boundary_condition(self, boundary: Shapely, condition: HeatBCPlacementType, ax: Ax) -> Ax:
        """Plot a structure's cross section shape for a given boundary condition."""
        plot_params_bc = self._get_bc_plot_params(condition=condition)
        ax = self.plot_line(line=boundary, plot_params=plot_params_bc, ax=ax)
        return ax

    # FIXME: probably needs revision
    def plot_line(self, line: Shapely, plot_params: PlotParams, ax: Ax) -> Ax:
        """Defines how a line is plotted on a matplotlib axes."""

        if isinstance(line, (MultiLineString, GeometryCollection)):
            lines = line.geoms
        elif isinstance(line, LineString):
            lines = [line]

        for l in lines:
            plot_line(l, ax=ax, add_points=False, color=plot_params.edgecolor, linewidth=plot_params.linewidth)
            # ax.add_artist(patch)
        return ax

    @staticmethod
    def _construct_heat_boundaries(# pylint:disable=too-many-locals
        structures: List[Structure],
        plane: Box,
        boundary_conditions: List[HeatBCPlacementType],
    ) -> List[Tuple[HeatBCPlacementType, Shapely]]:
        """Compute list of boundary lines to plot on plane.

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            list of structures to filter on the plane.
        plane : :class:`.Box`
            target plane.
        boundary_conditions : List[HeatBCPlacementType]
            list of boundary conditions associated with structures.

        Returns
        -------
        List[Tuple[:class:`.HeatBCPlacementType`, shapely.geometry.base.BaseGeometry]]
            List of boundary lines and boundary conditions on the plane after merging.
        """

        # get structures in the plane and present named structures and media
        shapes = []
        named_structures_present = set()
        named_mediums_present = set()
        for structure in structures:

            # get list of Shapely shapes that intersect at the plane
            shapes_plane = structure.geometry.intersections_2dbox(plane)

            # append each of them and their medium information to the list of shapes
            for shape in shapes_plane:
                shapes.append((structure.name, structure.medium, shape, shape.bounds))

            # also keep track of present named structures and media
            if structure.name:
                named_structures_present.add(structure.name)

            if structure.medium.name:
                named_mediums_present.add(structure.medium.name)

        background_structure_shape = shapes[0][2]

        # construct an inverse mapping structure -> bc for present structures
        struct_to_bc = {}
        for bc in boundary_conditions:
            if isinstance(bc, (HeatBCPlacementStructure, HeatBCPlacementStructureSimulation)) and bc.structure in named_structures_present:
                if bc.structure in struct_to_bc:
                    struct_to_bc[bc.structure] += [bc]
                else:
                    struct_to_bc[bc.structure] = [bc]

            if isinstance(bc, HeatBCPlacementStructureStructure):
                for structure in bc.structures:
                    if structure in named_structures_present:
                        if structure in struct_to_bc:
                            struct_to_bc[structure] += [bc]
                        else:
                            struct_to_bc[structure] = [bc]

            if isinstance(bc, HeatBCPlacementSimulation):
                struct_to_bc[HEAT_BACK_STRUCTURE_STR] = [bc]

        # construct an inverse mapping medium -> bc for present mediums
        med_to_bc = {}
        for bc in boundary_conditions:
            if isinstance(bc, HeatBCPlacementMediumMedium):
                for med in bc.mediums:
                    if med in named_mediums_present:
                        if med in med_to_bc:
                            med_to_bc[med] += [bc]
                        else:
                            med_to_bc[med] = [bc]

        # construct boundaries in 2 passes:

        # 1. forward foop to take care of Simulation, StructureSimulation, Structure, and MediumMediums
        boundaries = []
        background_shapes = []
        for name, medium, shape, bounds in shapes:

            minx, miny, maxx, maxy = bounds

            # intersect existing boundaries (both structure based and medium based)
            for index, (_bc, _name, _bdry, _bounds) in enumerate(boundaries):

                # simulation bc is overriden only by HeatBCPlacementStructureSimulation
                if isinstance(_bc, HeatBCPlacementSimulation):
                    if name not in struct_to_bc:
                        continue
                    if any(not isinstance(bc, HeatBCPlacementStructureSimulation) for bc in struct_to_bc[name]):
                        continue

                _minx, _miny, _maxx, _maxy = _bounds

                # do a bounding box check to see if any intersection to do anything about
                if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                    continue

                # look more closely to see if intersected.
                if _bdry.is_empty or not shape.intersects(_bdry):
                    continue

                diff_shape = _bdry - shape

                boundaries[index] = (_bc, _name, diff_shape, diff_shape.bounds)

            # create new srtucture based boundary

            if name in struct_to_bc:
                for bc in struct_to_bc[name]:

                    if isinstance(bc, HeatBCPlacementStructure):
                        bdry = shape.exterior
                        bdry = bdry.intersection(background_structure_shape)
                        boundaries.append((bc, name, bdry, bdry.bounds))

                    if isinstance(bc, HeatBCPlacementSimulation):
                        boundaries.append((bc, name, shape.exterior, shape.exterior.bounds))

                    if isinstance(bc, HeatBCPlacementStructureSimulation):
                        bdry = background_structure_shape.exterior
                        bdry = bdry.intersection(shape)
                        boundaries.append((bc, name, bdry, bdry.bounds))

            # create new medium based boundary, and cut or merge relevant background shapes

            # loop through background_shapes (note: all background are non-intersecting or merged)
            # this is similar to _filter_structures_plane but only mediums participating in BCs are tracked
            for index, (_medium, _shape, _bounds) in enumerate(background_shapes):

                _minx, _miny, _maxx, _maxy = _bounds

                # do a bounding box check to see if any intersection to do anything about
                if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                    continue

                # look more closely to see if intersected.
                if _shape.is_empty or not shape.intersects(_shape):
                    continue

                diff_shape = _shape - shape

                # different medium, remove intersection from background shape
                if medium != _medium and len(diff_shape.bounds) > 0:
                    background_shapes[index] = (_medium, diff_shape, diff_shape.bounds)

                    # in case when there is a bc between two media
                    # create a new boudnary segment
                    for bc in med_to_bc[_medium.name]:
                        if medium.name in bc.mediums:
                            bdry = shape.exterior.intersection(_shape)
                            boundaries.append((bc, name, bdry, bdry.bounds))

                # same medium, add diff shape to this shape and mark background shape for removal
                # note: this only happens if this medium is listed in BCs
                else:
                    shape = shape | diff_shape
                    background_shapes[index] = None

            # after doing this with all background shapes, add this shape to the background
            # but only if this medium is listed in BCs
            if medium.name in med_to_bc:
                background_shapes.append((medium, shape, shape.bounds))

            # remove any existing background shapes that have been marked as 'None'
            background_shapes = [b for b in background_shapes if b is not None]

        # filter out empty geometries
        boundaries = [(bc, bdry) for (bc, name, bdry, _) in boundaries if bdry]

        # 2. backward foop to take care of StructureStructure
        # we do it in this way because we define the boundary between
        # two overlapping structures A and B, where A comes before B, as
        # boundary(B) intersected by A
        # So, in this loop as we go backwards through the structures we:
        # - (1) when come upon B, create boundary(B)
        # - (2) cut away from it by other structures
        # - (3) when come upon A, intersect it with A and mark it as complete,
        #   that is, no more further modifications
        boundaries_reverse = []

        for name, _, shape, bounds in shapes[:0:-1]:

            minx, miny, maxx, maxy = bounds

            # intersect existing boundaries
            for index, (_bc, _name, _bdry, _bounds, _completed) in enumerate(boundaries_reverse):

                if not _completed:

                    _minx, _miny, _maxx, _maxy = _bounds

                    # do a bounding box check to see if any intersection to do anything about
                    if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
                        continue

                    # look more closely to see if intersected.
                    if _bdry.is_empty or not shape.intersects(_bdry):
                        continue

                    # event (3) from above
                    if name in _bc.structures:
                        new_bdry = _bdry.intersection(shape)
                        boundaries_reverse[index] = (_bc, _name, new_bdry, diff_shape.bounds, True)

                    # event (2) from above
                    else:
                        new_bdry = _bdry - shape
                        boundaries_reverse[index] = (_bc, _name, new_bdry, diff_shape.bounds, _completed)

            # create new boundary (event (1) from above)
            if name in struct_to_bc:
                for bc in struct_to_bc[name]:
                    if isinstance(bc, HeatBCPlacementStructureStructure):
                        bdry = shape.exterior
                        bdry = bdry.intersection(background_structure_shape)
                        boundaries_reverse.append((bc, name, bdry, bdry.bounds, False))

        # filter and append completed boundaries to main list
        for bc, _, bdry, _, is_completed in boundaries_reverse:
            if bdry and is_completed:
                boundaries.append((bc, bdry))

        return boundaries
