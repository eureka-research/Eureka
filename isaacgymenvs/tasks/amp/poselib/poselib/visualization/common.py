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

import os

from ..core import logger
from .plt_plotter import Matplotlib3DPlotter
from .skeleton_plotter_tasks import Draw3DSkeletonMotion, Draw3DSkeletonState


def plot_skeleton_state(skeleton_state, task_name=""):
    """
    Visualize a skeleton state

    :param skeleton_state:
    :param task_name:
    :type skeleton_state: SkeletonState
    :type task_name: string, optional
    """
    logger.info("plotting {}".format(task_name))
    task = Draw3DSkeletonState(task_name=task_name, skeleton_state=skeleton_state)
    plotter = Matplotlib3DPlotter(task)
    plotter.show()


def plot_skeleton_states(skeleton_state, skip_n=1, task_name=""):
    """
    Visualize a sequence of skeleton state. The dimension of the skeleton state must be 1

    :param skeleton_state:
    :param task_name:
    :type skeleton_state: SkeletonState
    :type task_name: string, optional
    """
    logger.info("plotting {} motion".format(task_name))
    assert len(skeleton_state.shape) == 1, "the state must have only one dimension"
    task = Draw3DSkeletonState(task_name=task_name, skeleton_state=skeleton_state[0])
    plotter = Matplotlib3DPlotter(task)
    for frame_id in range(skeleton_state.shape[0]):
        if frame_id % skip_n != 0:
            continue
        task.update(skeleton_state[frame_id])
        plotter.update()
    plotter.show()


def plot_skeleton_motion(skeleton_motion, skip_n=1, task_name=""):
    """
    Visualize a skeleton motion along its first dimension.

    :param skeleton_motion:
    :param task_name:
    :type skeleton_motion: SkeletonMotion
    :type task_name: string, optional
    """
    logger.info("plotting {} motion".format(task_name))
    task = Draw3DSkeletonMotion(
        task_name=task_name, skeleton_motion=skeleton_motion, frame_index=0
    )
    plotter = Matplotlib3DPlotter(task)
    for frame_id in range(len(skeleton_motion)):
        if frame_id % skip_n != 0:
            continue
        task.update(frame_id)
        plotter.update()
    plotter.show()


def plot_skeleton_motion_interactive_base(skeleton_motion, task_name=""):
    class PlotParams:
        def __init__(self, total_num_frames):
            self.current_frame = 0
            self.playing = False
            self.looping = False
            self.confirmed = False
            self.playback_speed = 4
            self.total_num_frames = total_num_frames

        def sync(self, other):
            self.current_frame = other.current_frame
            self.playing = other.playing
            self.looping = other.current_frame
            self.confirmed = other.confirmed
            self.playback_speed = other.playback_speed
            self.total_num_frames = other.total_num_frames

    task = Draw3DSkeletonMotion(
        task_name=task_name, skeleton_motion=skeleton_motion, frame_index=0
    )
    plotter = Matplotlib3DPlotter(task)

    plot_params = PlotParams(total_num_frames=len(skeleton_motion))
    print("Entered interactive plot - press 'n' to quit, 'h' for a list of commands")

    def press(event):
        if event.key == "x":
            plot_params.playing = not plot_params.playing
        elif event.key == "z":
            plot_params.current_frame = plot_params.current_frame - 1
        elif event.key == "c":
            plot_params.current_frame = plot_params.current_frame + 1
        elif event.key == "a":
            plot_params.current_frame = plot_params.current_frame - 20
        elif event.key == "d":
            plot_params.current_frame = plot_params.current_frame + 20
        elif event.key == "w":
            plot_params.looping = not plot_params.looping
            print("Looping: {}".format(plot_params.looping))
        elif event.key == "v":
            plot_params.playback_speed *= 2
            print("playback speed: {}".format(plot_params.playback_speed))
        elif event.key == "b":
            if plot_params.playback_speed != 1:
                plot_params.playback_speed //= 2
            print("playback speed: {}".format(plot_params.playback_speed))
        elif event.key == "n":
            plot_params.confirmed = True
        elif event.key == "h":
            rows, columns = os.popen("stty size", "r").read().split()
            columns = int(columns)
            print("=" * columns)
            print("x: play/pause")
            print("z: previous frame")
            print("c: next frame")
            print("a: jump 10 frames back")
            print("d: jump 10 frames forward")
            print("w: looping/non-looping")
            print("v: double speed (this can be applied multiple times)")
            print("b: half speed (this can be applied multiple times)")
            print("n: quit")
            print("h: help")
            print("=" * columns)

        print(
            'current frame index: {}/{} (press "n" to quit)'.format(
                plot_params.current_frame, plot_params.total_num_frames - 1
            )
        )

    plotter.fig.canvas.mpl_connect("key_press_event", press)
    while True:
        reset_trail = False
        if plot_params.confirmed:
            break
        if plot_params.playing:
            plot_params.current_frame += plot_params.playback_speed
        if plot_params.current_frame >= plot_params.total_num_frames:
            if plot_params.looping:
                plot_params.current_frame %= plot_params.total_num_frames
                reset_trail = True
            else:
                plot_params.current_frame = plot_params.total_num_frames - 1
        if plot_params.current_frame < 0:
            if plot_params.looping:
                plot_params.current_frame %= plot_params.total_num_frames
                reset_trail = True
            else:
                plot_params.current_frame = 0
        yield plot_params
        task.update(plot_params.current_frame, reset_trail)
        plotter.update()


def plot_skeleton_motion_interactive(skeleton_motion, task_name=""):
    """
    Visualize a skeleton motion along its first dimension interactively.

    :param skeleton_motion:
    :param task_name:
    :type skeleton_motion: SkeletonMotion
    :type task_name: string, optional
    """
    for _ in plot_skeleton_motion_interactive_base(skeleton_motion, task_name):
        pass


def plot_skeleton_motion_interactive_multiple(*callables, sync=True):
    for _ in zip(*callables):
        if sync:
            for p1, p2 in zip(_[:-1], _[1:]):
                p2.sync(p1)


# def plot_skeleton_motion_interactive_multiple_same(skeleton_motions, task_name=""):

