"""Defines heat simulation class"""
from __future__ import annotations

# from abc import ABC, abstractmethod
from typing import Tuple, List

import pydantic as pd
# import numpy as np

from .heat_source import HeatSourceType
from .heat_boundary import HeatBoundaryPlacementType
from .heat_boundary import HeatBoundaryPlacementStructure, HeatBoundaryPlacementStructureStructure
from .heat_boundary import HeatBoundaryPlacementStructureSimulation
from .heat_boundary import HeatBoundaryPlacementMediumMedium

from ..base import Tidy3dBaseModel
from ..types import Ax, Shapely
from ..viz import add_ax_if_none, equal_aspect
from ..simulation import Simulation
from ..structure import Structure
from ..geometry import Box

from ...exceptions import SetupError
# from ...constants import KELVIN, DENSITY, SPECIFIC_HEAT_CAPACITY, THERMAL_CONDUCTIVITY, PERMITTIVITY
# from ...constants import CONDUCTIVITY, HEAT_FLUX


class HeatSimulation(Tidy3dBaseModel):
    """Contains all information about heat simulation.

    Example
    -------
    >>>
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

    boundary_conditions: Tuple[HeatBoundaryPlacementType, ...] = pd.Field(
        (),
        title="Boundary Conditions",
        description="List of boundary conditions.",
    )

    @pd.validator("boundary_conditions", always=True)
    def names_exist(cls, val, values):
        """Error if boundary conditions point to non-existing structures/media"""
        structures = values["simulation"].structures
        mediums = values["simulation"].mediums
        structures_names = {s.name for s in structures}
        mediums_names = {m.name for m in mediums}

        for bc in val:
            if isinstance(bc, (HeatBoundaryPlacementStructure, HeatBoundaryPlacementStructureSimulation)):
                if bc.structure not in structures_names:
                    raise SetupError(
                        f"Structure {bc.structure} provided in a :class:`{bc.type}` "
                        "is not found among simulation structures."
                    )
            if isinstance(bc, (HeatBoundaryPlacementStructureStructure)):
                for ind in range(2):
                    if bc.structures[ind] and bc.structures[ind] not in structures_names:
                        raise SetupError(
                            f"Structure {bc.structures[ind]} provided in a :class:`{bc.type}` "
                            "is not found among simulation structures."
                        )
            if isinstance(bc, (HeatBoundaryPlacementMediumMedium)):
                for ind in range(2):
                    if bc.mediums[ind] not in mediums_names:
                        raise SetupError(
                            f"Material {bc.mediums[ind]} provided in a :class:`{bc.type}` "
                            "is not found among simulation mediums."
                        )
        return val

#    @equal_aspect
#    @add_ax_if_none
#    def plot_structures_eps(  # pylint: disable=too-many-arguments,too-many-locals
#        self,
#        x: float = None,
#        y: float = None,
#        z: float = None,
#        freq: float = None,
#        alpha: float = None,
#        cbar: bool = True,
#        reverse: bool = False,
#        ax: Ax = None,
#    ) -> Ax:
#        """Plot each of simulation's structures on a plane defined by one nonzero x,y,z coordinate.
#        The permittivity is plotted in grayscale based on its value at the specified frequency.

#        Parameters
#        ----------
#        x : float = None
#            position of plane in x direction, only one of x, y, z must be specified to define plane.
#        y : float = None
#            position of plane in y direction, only one of x, y, z must be specified to define plane.
#        z : float = None
#            position of plane in z direction, only one of x, y, z must be specified to define plane.
#        freq : float = None
#            Frequency to evaluate the relative permittivity of all mediums.
#            If not specified, evaluates at infinite frequency.
#        reverse : bool = False
#            If ``False``, the highest permittivity is plotted in black.
#            If ``True``, it is plotteed in white (suitable for black backgrounds).
#        cbar : bool = True
#            Whether to plot a colorbar for the relative permittivity.
#        alpha : float = None
#            Opacity of the structures being plotted.
#            Defaults to the structure default alpha.
#        ax : matplotlib.axes._subplots.Axes = None
#            Matplotlib axes to plot on, if not specified, one is created.

#        Returns
#        -------
#        matplotlib.axes._subplots.Axes
#            The supplied or created matplotlib axes.
#        """

#        structures = self.structures

#        # alpha is None just means plot without any transparency
#        if alpha is None:
#            alpha = 1

#        if alpha <= 0:
#            return ax

#        if alpha < 1 and not isinstance(self.medium, CustomMedium):
#            axis, position = Box.parse_xyz_kwargs(x=x, y=y, z=z)
#            center = Box.unpop_axis(position, (0, 0), axis=axis)
#            size = Box.unpop_axis(0, (inf, inf), axis=axis)
#            plane = Box(center=center, size=size)
#            medium_shapes = self._filter_structures_plane(structures=structures, plane=plane)
#        else:
#            structures = [self.background_structure] + list(structures)
#            medium_shapes = self._get_structures_plane(structures=structures, x=x, y=y, z=z)

#        eps_min, eps_max = self.eps_bounds(freq=freq)
#        for (medium, shape) in medium_shapes:
#            # if the background medium is custom medium, it needs to be rendered separately
#            if medium == self.medium and alpha < 1 and not isinstance(medium, CustomMedium):
#                continue
#            # no need to add patches for custom medium
#            if not isinstance(medium, CustomMedium):
#                ax = self._plot_shape_structure_eps(
#                    freq=freq,
#                    alpha=alpha,
#                    medium=medium,
#                    eps_min=eps_min,
#                    eps_max=eps_max,
#                    reverse=reverse,
#                    shape=shape,
#                    ax=ax,
#                )
#            else:
#                # For custom medium, apply pcolormesh clipped by the shape.
#                self._pcolormesh_shape_custom_medium_structure_eps(
#                    x, y, z, freq, alpha, medium, eps_min, eps_max, reverse, shape, ax
#                )

#        if cbar:
#            self._add_cbar(eps_min=eps_min, eps_max=eps_max, ax=ax)
#        ax = self._set_plot_bounds(ax=ax, x=x, y=y, z=z)

#        # clean up the axis display
#        axis, position = self.parse_xyz_kwargs(x=x, y=y, z=z)
#        ax = self.add_ax_labels_lims(axis=axis, ax=ax)
#        ax.set_title(f"cross section at {'xyz'[axis]}={position:.2f}")

#        return ax

    def _construct_heat_boundaries(self,  # pylint:disable=too-many-locals
        structures: List[Structure], plane: Box
    #) -> List[Tuple[Heat, Shapely]]:
    ) -> List[Shapely]:
        """Compute list of shapes to plot on plane specified by {x,y,z}.
        Overlaps are removed or merged depending on medium.

        Parameters
        ----------
        structures : List[:class:`.Structure`]
            list of structures to filter on the plane.
        x : float = None
            position of plane in x direction, only one of x, y, z must be specified to define plane.
        y : float = None
            position of plane in y direction, only one of x, y, z must be specified to define plane.
        z : float = None
            position of plane in z direction, only one of x, y, z must be specified to define plane.

        Returns
        -------
        List[Tuple[:class:`.AbstractMedium`, shapely.geometry.base.BaseGeometry]]
            List of shapes and mediums on the plane after merging.
        """

        shapes = []
        named_shapes = {}
        for structure in structures:

            # get list of Shapely shapes that intersect at the plane
            shapes_plane = structure.geometry.intersections_2dbox(plane)

            # Append each of them and their medium information to the list of shapes
            for shape in shapes_plane:
                shapes.append((structure.name, shape, shape.bounds))

            # also keep track of named structures
            if structure.name:
                named_shapes[structure.name] = shapes_plane

        # get potential boundaries
        bcs_present = set()
        for bc in self.boundary_conditions:
            if isinstance(bc, HeatBoundaryPlacementStructure) and bc.structure in named_shapes:
                bcs_present.add(bc.structure)

        background_shapes = []
        boundaries = []

        # intersect them with other geometries

        for name, shape, bounds in shapes:

            minx, miny, maxx, maxy = bounds

            # intersect existing boundaries

            # create new boundary
            if name in bcs_present:
                boundaries.apprend(shape.exterior)

#            # loop through background_shapes (note: all background are non-intersecting or merged)
#            for index, (_medium, _shape, _bounds) in enumerate(background_shapes):

#                _minx, _miny, _maxx, _maxy = _bounds

#                # do a bounding box check to see if any intersection to do anything about
#                if minx > _maxx or _minx > maxx or miny > _maxy or _miny > maxy:
#                    continue

#                # look more closely to see if intersected.
#                if _shape.is_empty or not shape.intersects(_shape):
#                    continue

#                diff_shape = _shape - shape

#                # different medium, remove intersection from background shape
#                if medium != _medium and len(diff_shape.bounds) > 0:
#                    background_shapes[index] = (_medium, diff_shape, diff_shape.bounds)

#                # same medium, add diff shape to this shape and mark background shape for removal
#                else:
#                    shape = shape | diff_shape
#                    background_shapes[index] = None

#            # after doing this with all background shapes, add this shape to the background
#            background_shapes.append((medium, shape, shape.bounds))

#            # remove any existing background shapes that have been marked as 'None'
#            background_shapes = [b for b in background_shapes if b is not None]

        # filter out any remaining None or empty shapes (shapes with area completely removed)
        return boundaries
#        return [(medium, shape) for (medium, shape, _) in background_shapes if shape]

#    grid_spec: HeatGridSpec = pd.Field(
#        title="Grid Specification",
#        description="Grid specification for heat simulation.",
#    )

#    heat_domain: Box = pd.Field(
#        ...,
#        title="Heat Simulation Domain",
#        description="Domain in which heat simulation is solved. If ``None`` heat simulation is "
#        "solved in the entire domain of the Tidy3D simulation."
#    )
