#!/usr/bin/env python3

# Copyright (C) 2020 Gabriele Bozzola, Wolfgang Kastaun
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

"""The :py:mod:`~.visualize` module provides functions to plot ``PostCactus``
objects with matplotlib.

"""
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable

from postcactus import grid_data as gd
from postcactus.cactus_grid_functions import BaseOneGridFunction


def setup_matplotlib():
    """Setup matplotlib with some reasonable defaults for better plots.

    Matplotlib behaves differently on different machines. With this, we make
    sure that we set all the relevant paramters that we care of.

    This is highly opinionated.
    """

    matplotlib.rcParams.update(
        {
            "lines.markersize": 4,
            "axes.labelsize": 16,
            "font.weight": "light",
            "font.size": 16,
            "legend.fontsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "axes.formatter.limits": [-3, 3],
            "xtick.minor.visible": True,
            "ytick.minor.visible": True,
            "image.cmap": "inferno",
        }
    )


def _preprocess_plot(func):
    """Decorator to set-up plot functions.

    When we plot anything, there is always some boilerplate that has to be
    executed. For example, we want to provide an axis keyword so that the user
    can specify where to plot, but if the keyword is not provided, we want to
    plot on the current figure.

    Essentially, this decorator sets default values. Why don't we do
    axis=plt.gca() then? The problem is that the default values are set when
    the function is defined, not when it is called. So, this will not work.

    This decorator takes care of everything.

    1. It handles the axis keyword setting it to plt.gca() if it was not
       provided.
    2. It handles the figure keyword setting it to plt.gcf() if it was not
       provided.

    func has to take as keyword arguments:
    1. 'axis=None', where the plot will be plot, or plt.gca() if None
    2. 'figure=None', where the plot will be plot, or plt.gcf() if None

    """

    def inner(data, *args, **kwargs):
        # Setdetault addes the key if it is not already there
        kwargs.setdefault("axis", plt.gca())
        kwargs.setdefault("figure", plt.gcf())
        return func(data, *args, **kwargs)

    return inner


def _preprocess_plot_grid(func):
    """Decorator to set-up plot functions that plot grid data.

    This dectorator exends _preprocess_plot for specific functions.

    1. It handles differt types to plot what intuitively one would want to
       plot.
    1a. If the data is a numpy array with shape 2, just pass the data,
        otherwise raise an error
    1b. If the data is a numpy array, just pass the data.
    1c. If data is UniformGridData, pass the data and the coordinates.
    1d. If data is HierarchicalGridData, read resample it to the given grid,
        then pass do 1c.
    1e. If data is a BaseOneGridFunction, we read the iteration and pass to
        1d.

    func has to take as keyword arguments (in addition to the ones in
    _preprocess_plot):
    1. 'data'. data will be passed as a numpy array, unless it is
               already so.
    2. 'coordinates=None'. coordinates will be passed as a list of numpy
                           arrays, unless it is not None. Each numpy
                           array is the coordinates along one axis.

    """

    @_preprocess_plot
    def inner(data, *args, **kwargs):
        # The flow is: We check if data is BaseOneGridFunction or derived. If
        # yes, we read the requested iteration. Then, we check if data is
        # HierachicalGridData, if yes, we resample to UniformGridData. Then we
        # work with UniformGridData and handle coordinates, finally we work
        # with numpy arrays, which is what we pass to the function.

        def not_in_kwargs_or_None(attr):
            """This is a helper function to see if the user passed an attribute
            or if the attribute is None
            """
            return attr not in kwargs or kwargs[attr] is None

        if isinstance(data, BaseOneGridFunction):
            if not_in_kwargs_or_None("iteration"):
                raise TypeError(
                    "Data has multiple iterations, specify what do you want to plot"
                )

            # Overwrite data with HierarchicalGridData
            data = data[kwargs["iteration"]]

        if isinstance(data, gd.HierarchicalGridData):
            if not_in_kwargs_or_None("shape"):
                raise TypeError(
                    "The data must be resampled but the shape was not provided"
                )

            # If x0 or x1 are None, we use the ones of the grid
            if not_in_kwargs_or_None("x0"):
                x0 = data.x0
            else:
                x0 = kwargs["x0"]

            if not_in_kwargs_or_None("x1"):
                x1 = data.x1
            else:
                x1 = kwargs["x1"]

            if not_in_kwargs_or_None("resample"):
                resample = False
            else:
                resample = kwargs["resample"]

            # Overwrite data with UniformGridData
            data = data.to_UniformGridData(
                shape=kwargs["shape"], x0=x0, x1=x1, resample=resample
            )

        if isinstance(data, gd.UniformGridData):
            # We check if the user has passed coordinates too.
            if "coordinates" in kwargs and kwargs["coordinates"] is not None:
                warnings.warn(
                    "Ignoring provided coordinates (data is UniformGridData)."
                    " To specify boundaries use x0 and x1."
                )
            resampling = False

            if not_in_kwargs_or_None("x0"):
                x0 = data.x0
            else:
                x0 = kwargs["x0"]
                resampling = True

            if not_in_kwargs_or_None("x1"):
                x1 = data.x1
            else:
                x1 = kwargs["x1"]
                resampling = True

            if resampling and not_in_kwargs_or_None("shape"):
                raise TypeError(
                    "The data must be resampled but the shape was not provided"
                )

            if resampling:
                new_grid = gd.UniformGrid(shape=kwargs["shape"], x0=x0, x1=x1)
                data = data.resampled(new_grid)

            kwargs["coordinates"] = data.coordinates_from_grid()
            # Overwrite data with numpy array
            data = data.data_xyz

        if isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise ValueError("Only 2-dimensional data can be plotted")

        # TODO: Check that coordinates are good

        # We remove what we don't need from kwargs, so that it is not
        # accidentally passed to the function
        if "shape" in kwargs:
            del kwargs["shape"]
        if "x0" in kwargs:
            del kwargs["x0"]
        if "x1" in kwargs:
            del kwargs["x1"]
        if "iteration" in kwargs:
            del kwargs["iteration"]
        if "resample" in kwargs:
            del kwargs["resample"]
        return func(data, *args, **kwargs)

    return inner


def _vmin_vmax_extend(data, vmin=None, vmax=None):

    colorbar_extend = "neither"

    if vmin is None:
        vmin = data.min()

    if data.min() < vmin:
        colorbar_extend = "min"

    if vmax is None:
        vmax = data.max()

    if data.max() > vmax:
        if colorbar_extend == "min":
            colorbar_extend = "both"
        else:
            colorbar_extend = "max"

    return vmin, vmax, colorbar_extend


# All the difficult stuff is in _preprocess_plot_grid
@_preprocess_plot_grid
def plot_contourf(
    data,
    figure=None,
    axis=None,
    coordinates=None,
    xlabel=None,
    ylabel=None,
    colorbar=False,
    label=None,
    logscale=False,
    vmin=None,
    vmax=None,
    aspect_ratio="equal",
    **kwargs,
):
    """Plot 2D grid from numpy array, UniformGridData, HierarhicalGridData,
    or OneGridFunction.

    Read the full documentation to see how to use this function.

    If logscale is True, vmin and vmax are the log10 of the variable.
    """

    # Considering all the effort put in _preprocess_plot_grid, we we can plot
    # as we were plotting normal numpy arrays.
    if logscale:
        # We mask the values that are smaller or equal than 0
        data = np.ma.log10(data)

    vmin, vmax, colorbar_extend = _vmin_vmax_extend(data, vmin=vmin, vmax=vmax)

    # To implement vmin and vmax, we clamp the data to vmin and vmax instead of
    # using the options in matplotlib. This greatly simplifies handling things
    # like colormaps.
    data = np.clip(data, vmin, vmax)

    axis.set_aspect(aspect_ratio)

    if coordinates is None:
        cf = axis.imshow(data, **kwargs)
    else:
        cf = axis.contourf(
            *coordinates, data, extend=colorbar_extend, **kwargs
        )
    if xlabel is not None:
        axis.set_xlabel(xlabel)
    if ylabel is not None:
        axis.set_ylabel(ylabel)
    if colorbar:
        plot_colorbar(cf, axis=axis, label=label)
    return cf


@_preprocess_plot
def plot_colorbar(mpl_artist, figure=None, axis=None, label=None, **kwargs):
    """Add a colorbar to an existing image (as produced by plot_contourf)."""
    # The next two lines guarantee that the colorbar is the same size as
    # the plot. From https://stackoverflow.com/a/18195921
    divider = make_axes_locatable(axis)
    cax = divider.append_axes("right", size="5%", pad=0.25)
    cb = plt.colorbar(mpl_artist, cax=cax, **kwargs)
    if label is not None:
        cb.set_label(label)
    return cb


@_preprocess_plot
def add_text_to_figure_corner(text, figure=None, axis=None):
    """Add text to the bottom right corner of a figure."""

    return axis.text(
        0.98,
        0.02,
        text,
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=figure.transFigure,
    )


@_preprocess_plot
def save(
    outputpath,
    figure_extension,
    as_tikz=False,
    figure=None,
    axis=None,
    **kwargs,
):
    """Save figure to outputpath.

    If as_tikz is True, save it as TikZ file.

    :param outputpath:  Output path without extension.
    :type outputpath:  str
    :param figure_extension: Extension of the figure to save.
                             This is ignored when as_tikz=True.
    :type figure_extension:  str
    :param as_tikz: Save figure with tikzplotlib instead of
                    matplotlib. Output will have extension .tikz
    :type as_tikz:  bool


    """

    if as_tikz:
        figurepath = f"{outputpath}.tikz"
        tikzplotlib.save(figurepath, **kwargs)
    else:
        figurepath = f"{outputpath}.{figure_extension}"
        plt.savefig(figurepath, **kwargs)
