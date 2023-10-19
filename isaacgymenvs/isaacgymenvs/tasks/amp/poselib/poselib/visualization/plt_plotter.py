# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""
The matplotlib plotter implementation for all the primitive tasks (in our case: lines and
dots)
"""
from typing import Any, Callable, Dict, List

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3

import numpy as np

from .core import BasePlotter, BasePlotterTask


class Matplotlib2DPlotter(BasePlotter):
    _fig: plt.figure  # plt figure
    _ax: plt.axis  # plt axis
    # stores artist objects for each task (task name as the key)
    _artist_cache: Dict[str, Any]
    # callables for each task primitives
    _create_impl_callables: Dict[str, Callable]
    _update_impl_callables: Dict[str, Callable]

    def __init__(self, task: "BasePlotterTask") -> None:
        fig, ax = plt.subplots()
        self._fig = fig
        self._ax = ax
        self._artist_cache = {}

        self._create_impl_callables = {
            "Draw2DLines": self._lines_create_impl,
            "Draw2DDots": self._dots_create_impl,
            "Draw2DTrail": self._trail_create_impl,
        }
        self._update_impl_callables = {
            "Draw2DLines": self._lines_update_impl,
            "Draw2DDots": self._dots_update_impl,
            "Draw2DTrail": self._trail_update_impl,
        }
        self._init_lim()
        super().__init__(task)

    @property
    def ax(self):
        return self._ax

    @property
    def fig(self):
        return self._fig

    def show(self):
        plt.show()

    def _min(self, x, y):
        if x is None:
            return y
        if y is None:
            return x
        return min(x, y)

    def _max(self, x, y):
        if x is None:
            return y
        if y is None:
            return x
        return max(x, y)

    def _init_lim(self):
        self._curr_x_min = None
        self._curr_y_min = None
        self._curr_x_max = None
        self._curr_y_max = None

    def _update_lim(self, xs, ys):
        self._curr_x_min = self._min(np.min(xs), self._curr_x_min)
        self._curr_y_min = self._min(np.min(ys), self._curr_y_min)
        self._curr_x_max = self._max(np.max(xs), self._curr_x_max)
        self._curr_y_max = self._max(np.max(ys), self._curr_y_max)

    def _set_lim(self):
        if not (
            self._curr_x_min is None
            or self._curr_x_max is None
            or self._curr_y_min is None
            or self._curr_y_max is None
        ):
            self._ax.set_xlim(self._curr_x_min, self._curr_x_max)
            self._ax.set_ylim(self._curr_y_min, self._curr_y_max)
        self._init_lim()

    @staticmethod
    def _lines_extract_xy_impl(index, lines_task):
        return lines_task[index, :, 0], lines_task[index, :, 1]

    @staticmethod
    def _trail_extract_xy_impl(index, trail_task):
        return (trail_task[index : index + 2, 0], trail_task[index : index + 2, 1])

    def _lines_create_impl(self, lines_task):
        color = lines_task.color
        self._artist_cache[lines_task.task_name] = [
            self._ax.plot(
                *Matplotlib2DPlotter._lines_extract_xy_impl(i, lines_task),
                color=color,
                linewidth=lines_task.line_width,
                alpha=lines_task.alpha
            )[0]
            for i in range(len(lines_task))
        ]

    def _lines_update_impl(self, lines_task):
        lines_artists = self._artist_cache[lines_task.task_name]
        for i in range(len(lines_task)):
            artist = lines_artists[i]
            xs, ys = Matplotlib2DPlotter._lines_extract_xy_impl(i, lines_task)
            artist.set_data(xs, ys)
            if lines_task.influence_lim:
                self._update_lim(xs, ys)

    def _dots_create_impl(self, dots_task):
        color = dots_task.color
        self._artist_cache[dots_task.task_name] = self._ax.plot(
            dots_task[:, 0],
            dots_task[:, 1],
            c=color,
            linestyle="",
            marker=".",
            markersize=dots_task.marker_size,
            alpha=dots_task.alpha,
        )[0]

    def _dots_update_impl(self, dots_task):
        dots_artist = self._artist_cache[dots_task.task_name]
        dots_artist.set_data(dots_task[:, 0], dots_task[:, 1])
        if dots_task.influence_lim:
            self._update_lim(dots_task[:, 0], dots_task[:, 1])

    def _trail_create_impl(self, trail_task):
        color = trail_task.color
        trail_length = len(trail_task) - 1
        self._artist_cache[trail_task.task_name] = [
            self._ax.plot(
                *Matplotlib2DPlotter._trail_extract_xy_impl(i, trail_task),
                color=trail_task.color,
                linewidth=trail_task.line_width,
                alpha=trail_task.alpha * (1.0 - i / (trail_length - 1))
            )[0]
            for i in range(trail_length)
        ]

    def _trail_update_impl(self, trail_task):
        trails_artists = self._artist_cache[trail_task.task_name]
        for i in range(len(trail_task) - 1):
            artist = trails_artists[i]
            xs, ys = Matplotlib2DPlotter._trail_extract_xy_impl(i, trail_task)
            artist.set_data(xs, ys)
            if trail_task.influence_lim:
                self._update_lim(xs, ys)

    def _create_impl(self, task_list):
        for task in task_list:
            self._create_impl_callables[task.task_type](task)
        self._draw()

    def _update_impl(self, task_list):
        for task in task_list:
            self._update_impl_callables[task.task_type](task)
        self._draw()

    def _set_aspect_equal_2d(self, zero_centered=True):
        xlim = self._ax.get_xlim()
        ylim = self._ax.get_ylim()

        if not zero_centered:
            xmean = np.mean(xlim)
            ymean = np.mean(ylim)
        else:
            xmean = 0
            ymean = 0

        plot_radius = max(
            [
                abs(lim - mean_)
                for lims, mean_ in ((xlim, xmean), (ylim, ymean))
                for lim in lims
            ]
        )

        self._ax.set_xlim([xmean - plot_radius, xmean + plot_radius])
        self._ax.set_ylim([ymean - plot_radius, ymean + plot_radius])

    def _draw(self):
        self._set_lim()
        self._set_aspect_equal_2d()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.00001)


class Matplotlib3DPlotter(BasePlotter):
    _fig: plt.figure  # plt figure
    _ax: p3.Axes3D  # plt 3d axis
    # stores artist objects for each task (task name as the key)
    _artist_cache: Dict[str, Any]
    # callables for each task primitives
    _create_impl_callables: Dict[str, Callable]
    _update_impl_callables: Dict[str, Callable]

    def __init__(self, task: "BasePlotterTask") -> None:
        self._fig = plt.figure()
        self._ax = p3.Axes3D(self._fig)
        self._artist_cache = {}

        self._create_impl_callables = {
            "Draw3DLines": self._lines_create_impl,
            "Draw3DDots": self._dots_create_impl,
            "Draw3DTrail": self._trail_create_impl,
        }
        self._update_impl_callables = {
            "Draw3DLines": self._lines_update_impl,
            "Draw3DDots": self._dots_update_impl,
            "Draw3DTrail": self._trail_update_impl,
        }
        self._init_lim()
        super().__init__(task)

    @property
    def ax(self):
        return self._ax

    @property
    def fig(self):
        return self._fig

    def show(self):
        plt.show()

    def _min(self, x, y):
        if x is None:
            return y
        if y is None:
            return x
        return min(x, y)

    def _max(self, x, y):
        if x is None:
            return y
        if y is None:
            return x
        return max(x, y)

    def _init_lim(self):
        self._curr_x_min = None
        self._curr_y_min = None
        self._curr_z_min = None
        self._curr_x_max = None
        self._curr_y_max = None
        self._curr_z_max = None

    def _update_lim(self, xs, ys, zs):
        self._curr_x_min = self._min(np.min(xs), self._curr_x_min)
        self._curr_y_min = self._min(np.min(ys), self._curr_y_min)
        self._curr_z_min = self._min(np.min(zs), self._curr_z_min)
        self._curr_x_max = self._max(np.max(xs), self._curr_x_max)
        self._curr_y_max = self._max(np.max(ys), self._curr_y_max)
        self._curr_z_max = self._max(np.max(zs), self._curr_z_max)

    def _set_lim(self):
        if not (
            self._curr_x_min is None
            or self._curr_x_max is None
            or self._curr_y_min is None
            or self._curr_y_max is None
            or self._curr_z_min is None
            or self._curr_z_max is None
        ):
            self._ax.set_xlim3d(self._curr_x_min, self._curr_x_max)
            self._ax.set_ylim3d(self._curr_y_min, self._curr_y_max)
            self._ax.set_zlim3d(self._curr_z_min, self._curr_z_max)
        self._init_lim()

    @staticmethod
    def _lines_extract_xyz_impl(index, lines_task):
        return lines_task[index, :, 0], lines_task[index, :, 1], lines_task[index, :, 2]

    @staticmethod
    def _trail_extract_xyz_impl(index, trail_task):
        return (
            trail_task[index : index + 2, 0],
            trail_task[index : index + 2, 1],
            trail_task[index : index + 2, 2],
        )

    def _lines_create_impl(self, lines_task):
        color = lines_task.color
        self._artist_cache[lines_task.task_name] = [
            self._ax.plot(
                *Matplotlib3DPlotter._lines_extract_xyz_impl(i, lines_task),
                color=color,
                linewidth=lines_task.line_width,
                alpha=lines_task.alpha
            )[0]
            for i in range(len(lines_task))
        ]

    def _lines_update_impl(self, lines_task):
        lines_artists = self._artist_cache[lines_task.task_name]
        for i in range(len(lines_task)):
            artist = lines_artists[i]
            xs, ys, zs = Matplotlib3DPlotter._lines_extract_xyz_impl(i, lines_task)
            artist.set_data(xs, ys)
            artist.set_3d_properties(zs)
            if lines_task.influence_lim:
                self._update_lim(xs, ys, zs)

    def _dots_create_impl(self, dots_task):
        color = dots_task.color
        self._artist_cache[dots_task.task_name] = self._ax.plot(
            dots_task[:, 0],
            dots_task[:, 1],
            dots_task[:, 2],
            c=color,
            linestyle="",
            marker=".",
            markersize=dots_task.marker_size,
            alpha=dots_task.alpha,
        )[0]

    def _dots_update_impl(self, dots_task):
        dots_artist = self._artist_cache[dots_task.task_name]
        dots_artist.set_data(dots_task[:, 0], dots_task[:, 1])
        dots_artist.set_3d_properties(dots_task[:, 2])
        if dots_task.influence_lim:
            self._update_lim(dots_task[:, 0], dots_task[:, 1], dots_task[:, 2])

    def _trail_create_impl(self, trail_task):
        color = trail_task.color
        trail_length = len(trail_task) - 1
        self._artist_cache[trail_task.task_name] = [
            self._ax.plot(
                *Matplotlib3DPlotter._trail_extract_xyz_impl(i, trail_task),
                color=trail_task.color,
                linewidth=trail_task.line_width,
                alpha=trail_task.alpha * (1.0 - i / (trail_length - 1))
            )[0]
            for i in range(trail_length)
        ]

    def _trail_update_impl(self, trail_task):
        trails_artists = self._artist_cache[trail_task.task_name]
        for i in range(len(trail_task) - 1):
            artist = trails_artists[i]
            xs, ys, zs = Matplotlib3DPlotter._trail_extract_xyz_impl(i, trail_task)
            artist.set_data(xs, ys)
            artist.set_3d_properties(zs)
            if trail_task.influence_lim:
                self._update_lim(xs, ys, zs)

    def _create_impl(self, task_list):
        for task in task_list:
            self._create_impl_callables[task.task_type](task)
        self._draw()

    def _update_impl(self, task_list):
        for task in task_list:
            self._update_impl_callables[task.task_type](task)
        self._draw()

    def _set_aspect_equal_3d(self):
        xlim = self._ax.get_xlim3d()
        ylim = self._ax.get_ylim3d()
        zlim = self._ax.get_zlim3d()

        xmean = np.mean(xlim)
        ymean = np.mean(ylim)
        zmean = np.mean(zlim)

        plot_radius = max(
            [
                abs(lim - mean_)
                for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
                for lim in lims
            ]
        )

        self._ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
        self._ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
        self._ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])

    def _draw(self):
        self._set_lim()
        self._set_aspect_equal_3d()
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.00001)
