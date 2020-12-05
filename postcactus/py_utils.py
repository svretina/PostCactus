#!/usr/bin/env python3

# Copyright (C) 2020 Gabriele Bozzola
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# this program; if not, see <https://www.gnu.org/licenses/>.

"""The :py:mod:`~.py_utiles` module provides generic helper functions for
several common operations in Python.

"""

import numpy as np


def apply_to_points(function, array):
    """Apply a function on all the multidimensional points of array.

    Array can have any shape, but it is intended to be a collection
    of multimensional points. For example, in 2D, array could be
    (0, 0), or [(0, 0)], or [(0, 0), (1, 1)], or complicated nested
    arrays. This function applies a given function to each single
    point while preserving the original shape.

    :param function: Function that takes a point as input
    :type function: callable
    :param array: Collection of points
    :type array: numpy array

    :returns: function applied on each single point
    :rtype: numpy array
    """

    return np.apply_along_axis(function, -1, np.asarray(array))


def ndarray_to_tuple(ndarray):
    """Convert a nested numpy array into a nested tuple."""
    return tuple(
        ndarray_to_tuple(i) if isinstance(i, np.ndarray) else i
        for i in ndarray
    )
