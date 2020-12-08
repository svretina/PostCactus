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


def apply_to_points(function, array, num_dimensions, ufunc=False):
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
    :param num_dimensions: Dimensionality of each point
    :type num_dimensions: int
    :param ufunc: function is a numpy universal function and can
                  be applied at multiple points at the same time.
    :type ufunc: bool

    :returns: function applied on each single point
    :rtype: numpy array
    """

    # TODO: This function is very inelegant

    # The case with 1D scalar data is tricky because it is not an array
    input_is_scalar = not hasattr(array, "__len__")

    # Now we make sure we have an array. This does not change the dimensions
    # in the case with 2 or more dimensions, but add one "level" to
    # scalar inputs.
    array_np = np.atleast_1d(array)

    # We have to handle the case with only one point
    input_is_one_point = array_np.shape == (num_dimensions, )

    # With this we add one layer to the array, so for all the dimensions
    # we have added one level when the input is a single point. We will
    # strip this level later.
    if input_is_one_point and not input_is_scalar:
        array_np = np.asarray([array_np])

    # The simplest way to achieve what we want is very slow, so we
    # reserve that for cases that we cannot vectorize
    if (not ufunc):
        ret = np.apply_along_axis(function, -1, array_np)
    else:
        # In the case of universal functions we have to get our hands dirty to
        # get massive speedups.
        #
        # We determine what is the shape of the points forgetting about
        # their dimensionality (which we don't need). We use this reshape
        # the output.
        points_shape = array_np.shape[:-1]

        # Next, we reshape up to the last axis, which means that
        # now we have a collection of points
        array_np = array_np.reshape(-1, array_np.shape[-1])
        ret = function(array_np)
        # Now we have to reconstruct the correct return shape.
        # First, we determine what is the dimensionality of the output
        # of function
        shape_function_return = ret[0].shape
        # And append this to the shape of the points
        ret_shape = tuple(points_shape + shape_function_return)
        # Finally, we reshape
        ret = ret.reshape(ret_shape)

    return ret[0] if input_is_one_point else ret


def ndarray_to_tuple(ndarray):
    """Convert a nested numpy array into a nested tuple.

    It can be slow."""
    return tuple(
        ndarray_to_tuple(i) if isinstance(i, np.ndarray) else i
        for i in ndarray
    )
