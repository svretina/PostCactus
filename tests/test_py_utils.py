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


import unittest

import numpy as np

from postcactus import py_utils as put


class TestPyUtils(unittest.TestCase):
    def test_apply_to_points(self):

        # We test with 2D points with two functions, one
        # a vector and a scalar one
        def vec_func(x):
            # Take a vector, return a vector
            return 2 * x

        def scal_func(x):
            return np.sum(x, axis=-1)

        # We run the same test suite with ufunc = Ture and ufunc = False
        def test_ufunc(ufunc):

            # Only one point
            self.assertTrue(
                np.array_equal(
                    put.apply_to_points(vec_func, [2, 3], 2, ufunc), [4, 6]
                )
            )

            self.assertTrue(
                np.array_equal(
                    put.apply_to_points(scal_func, [2, 3], 2, ufunc), 5
                )
            )

            # Test two points
            arr1 = [[2, 3], [10, 20]]
            exp_arr1 = [[4, 6], [20, 40]]
            self.assertTrue(
                np.array_equal(
                    put.apply_to_points(vec_func, arr1, 2, ufunc), exp_arr1
                )
            )

            exp_arr1_scal = [5, 30]
            self.assertTrue(
                np.array_equal(
                    put.apply_to_points(scal_func, arr1, 2, ufunc),
                    exp_arr1_scal,
                )
            )

            # Now three sets of three points
            arr2 = [
                [[2, 3], [10, 20], [-4, -6]],
                [[5, 1], [1, 2], [8, 11]],
                [[0, 1], [2, 3], [-1, -2]],
            ]
            exp_arr2 = [
                [[4, 6], [20, 40], [-8, -12]],
                [[10, 2], [2, 4], [16, 22]],
                [[0, 2], [4, 6], [-2, -4]],
            ]
            self.assertTrue(
                np.array_equal(
                    put.apply_to_points(vec_func, arr2, 2, ufunc), exp_arr2
                )
            )

            exp_arr2_scal = [[5, 30, -10], [6, 3, 19], [1, 5, -3]]
            self.assertTrue(
                np.array_equal(
                    put.apply_to_points(scal_func, arr2, 2, ufunc),
                    exp_arr2_scal,
                )
            )

            # 1D data
            # Scalar
            arr3 = 3.5
            self.assertEqual(put.apply_to_points(vec_func, arr3, 1, ufunc), 7)
            self.assertTrue(
                isinstance(
                    put.apply_to_points(vec_func, arr3, 1, ufunc), np.float
                )
            )
            # Vector
            arr4 = [5.5]
            self.assertTrue(
                np.array_equal(
                    put.apply_to_points(vec_func, arr4, 1, ufunc), [11]
                )
            )
            self.assertTrue(
                isinstance(
                    put.apply_to_points(vec_func, arr4, 1, ufunc), np.ndarray
                )
            )
            # Collection of scalars
            arr5 = [1, 2, 3]
            self.assertTrue(
                np.array_equal(
                    put.apply_to_points(vec_func, arr5, 1, ufunc), [2, 4, 6]
                )
            )
            # Collection of "vectors"
            arr6 = [[1], [2], [3]]
            self.assertTrue(
                np.array_equal(
                    put.apply_to_points(vec_func, arr6, 1, ufunc),
                    [[2], [4], [6]],
                )
            )

        for ufunc in (False, True):
            with self.subTest(ufunc=ufunc):
                test_ufunc(ufunc)

    def test_ndarray_to_tuple(self):

        arr1 = np.atleast_1d(2)
        exp_arr1 = (2,)

        self.assertEqual(put.ndarray_to_tuple(arr1), exp_arr1)

        arr2 = np.array([2, 3])
        exp_arr2 = (2, 3)

        self.assertEqual(put.ndarray_to_tuple(arr2), exp_arr2)

        arr3 = np.array([[[0, 0], [1, 1], [2, 2]], [[0, 0], [1, 1], [2, 5]]])
        exp_arr3 = (((0, 0), (1, 1), (2, 2)), ((0, 0), (1, 1), (2, 5)))

        self.assertEqual(put.ndarray_to_tuple(arr3), exp_arr3)
