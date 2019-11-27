from itertools import product, chain, groupby

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator

import numpy as np

from traits.api import Array, HasStrictTraits, Tuple, Float, Unicode, List, Int


def split_interval(relative_areas, radius_range):
    """ Split an interval by relative area.

    Parameters
    ----------
    relative_area - Array
        The relative proportions to split the interval by.
        Needs to be sorted in descending order.
    radius_range - Tuple
        The min/max bounds of the interval.

    Returns
    -------
    min/max values - generator
        Generates the intervals with a tuple for each segment.
    """

    _radius_range = radius_range[::-1]
    # _radius_range = radius_range

    delta = _radius_range[1] - _radius_range[0]

    # as a percentage
    _relative_areas = relative_areas / np.sum(relative_areas)

    # find coords for each section
    minvals = [
        i for i in np.cumsum(_relative_areas) * delta + _radius_range[0]
    ]
    maxvals = [_radius_range[0]] + [i for i in minvals[:-1]]

    return list(zip(minvals, maxvals))


def split_angle_range(angles, angle_range, centered=False):
    """ Split an diagram by angle. Angles in the transform are bounded
        between [0,180.) this is assumed. For higher order coefficients,
        the angles are embedded in the angle range of the previous
        transform.

        As a special case, at the lowest order, the result is centered
        not bounded by the angle range.

    Parameters
    ----------
    angles - Array
        The angles used in the transform, bounded [0,180) degrees.
    angle_range - Tuple
        The min/max bounds of the interval.


    Returns
    -------
    min/max values - generator
        Generates the intervals with a tuple for each segment.
    """

    # angles are bounded
    _angles = list(angles) + [180]

    relative_areas = np.diff(_angles)
    # as a percentage
    relative_areas /= np.sum(relative_areas)
    delta = angle_range[1] - angle_range[0]

    # find coords for each section
    maxvals = [i for i in np.cumsum(relative_areas) * delta + angle_range[0]]
    minvals = [angle_range[0]] + [i for i in maxvals[:-1]]

    if centered:
        _maxvals = [
            (0.5 * (maxvals[i + 1] + maxvals[i]))
            for i, _ in enumerate(maxvals[:-1])
        ]
        _minvals = [
            (0.5 * (minvals[i + 1] + minvals[i]))
            for i, _ in enumerate(minvals[:-1])
        ]

        maxvals = [_minvals[0]] + _maxvals
        minvals = [-_minvals[0]] + _minvals
    else:
        indx = len(_angles) // 2

        minvals = np.roll(minvals, indx)
        maxvals = np.roll(maxvals, indx)

    return list(zip(minvals, maxvals))


def find_order(node):
    _node = node
    counter = 0
    while _node:
        _node = _node.parent
        counter += 1
    # -1 while condition
    return counter - 1


class PlotElement(HasStrictTraits):
    """ An element in the rose plot it parameterized
        by min/max ranges in polar coordinates, radius
        and angles in degrees.
    """

    #:
    name = Unicode

    #: min/max ranges of the element
    radius_range = Tuple(Float, Float)

    #: min/max ranges of the angles (in degrees)
    angle_range = Tuple(Float, Float)

    #:
    order = Int

    def __str__(self):
        return "({},{}: {},{},{},{})".format(
            self.name,
            self.order,
            self.radius_range[0],
            self.radius_range[1],
            self.angle_range[0],
            self.angle_range[1],
        )


class Visualize2D(HasStrictTraits):
    angles = Tuple

    plot_elements = List

    _num_ang_cell = Int(980)
    _num_r_cell = Int(100)

    theta = Array
    radii = Array

    def __init__(self, **traits):
        super().__init__(**traits)

        ang_min = -self.angles[1] // 2
        self.theta, self.radii = np.meshgrid(
            np.linspace(ang_min, ang_min + 360, self._num_ang_cell),
            np.linspace(0, 1, self._num_r_cell),
        )

    def _render_element(self, pixel, z, do_log=False):

        plot_element, data = pixel

        angle_range = plot_element.angle_range
        radius_range = plot_element.radius_range

        indx = np.logical_and(
            np.logical_and(
                self.radii >= radius_range[0], self.radii <= radius_range[1]
            ),
            np.logical_and(
                self.theta >= angle_range[0], self.theta <= angle_range[1]
            ),
        )

        if do_log:
            z[indx] = np.log(data)
        else:
            z[indx] = data

        def reflect(angle_range):
            return (angle_range[0] + 180, angle_range[1] + 180)

        _angle = reflect(angle_range)
        indx = np.logical_and(
            np.logical_and(
                self.radii >= radius_range[0], self.radii <= radius_range[1]
            ),
            np.logical_and(self.theta >= _angle[0], self.theta <= _angle[1]),
        )

        if do_log:
            z[indx] = np.log(data)
        else:
            z[indx] = data

        if do_log:
            z[z == 0] = np.min(np.log(data))

    def plot(self, coeffs, ax1, order, do_log=False, cmap="cividis"):
        z = np.zeros(self.radii.shape)

        # fill in values
        [
            self._render_element(pixel, z, do_log=do_log)
            for pixel in zip(self.plot_elements, coeffs)
            if pixel[0].order == order
        ]

        cmap = plt.get_cmap(cmap)
        levels = MaxNLocator(nbins=100).tick_values(z.min(), z.max())
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        ax1.pcolormesh(
            np.deg2rad(self.theta), self.radii, z, cmap=cmap, norm=norm
        )
        ax1.set_theta_zero_location("N")
        ax1.set_yticklabels([])
        ax1.set_xticklabels([])

        ax1.grid(False)

        return ax1

    def apply_to_cell(self, children, root_elements, order):
        _plot_elements = list(
            chain(
                *[
                    self._extract_elements(children, i, order)
                    for i in root_elements
                ]
            )
        )
        # reorganize them, one list of elements for each name, one name per
        # child.
        # i.e. on list of elements per child.
        _plot_elements = sorted(_plot_elements, key=lambda x: x.name)
        _plot_elements = [
            list(val) for _, val in groupby(_plot_elements, lambda x: x.name)
        ]

        return [(i.children, j) for i, j in zip(children, _plot_elements)]

    def recurse(self, root_node, root_element, max_order=2):
        current_layer = [(root_node.children, [root_element])]
        for order in range(1, max_order + 1):
            next_layer = []
            for layer in current_layer:
                children, plot_elements = layer
                next_layer += self.apply_to_cell(
                    children, plot_elements, order
                )
            current_layer = next_layer

            self.plot_elements += list(
                chain([ele for layer in next_layer for ele in layer[1]])
            )

    def _extract_elements(self, children, element, order):
        """ For a given list of children, extract elements:
        """
        if len(children) < 1:
            return []

        relative_area = np.array([np.sqrt(2 ** -i.scale) for i in children])
        # relative_area = np.array([1 for i in children])

        radius_intervals = split_interval(relative_area, element.radius_range)
        # associate the names with the radius interval
        names = [i.name for i in children]

        angle_intervals = split_angle_range(
            self.angles, element.angle_range, centered=(order == 1)
        )

        # create a tuple (name, radius, range, angle range)
        # this creates cell for each name/radius at each angle
        cells = product(zip(names, radius_intervals), angle_intervals)
        # unpack to name, radius range, angle range
        cells = [(i[0][0], i[0][1], i[1]) for i in cells]

        elements = [
            PlotElement(
                name=i[0],
                order=order,
                radius_range=tuple(i[1]),
                angle_range=tuple(i[2]),
            )
            for i in cells
        ]

        return elements
