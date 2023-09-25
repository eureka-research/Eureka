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
This is where all skeleton related complex tasks are defined (skeleton state and skeleton
motion)
"""
import numpy as np

from .core import BasePlotterTask
from .simple_plotter_tasks import Draw3DDots, Draw3DLines, Draw3DTrail


class Draw3DSkeletonState(BasePlotterTask):
    _lines_task: Draw3DLines  # sub-task for drawing lines
    _dots_task: Draw3DDots  # sub-task for drawing dots

    def __init__(
        self,
        task_name: str,
        skeleton_state,
        joints_color: str = "red",
        lines_color: str = "blue",
        alpha=1.0,
    ) -> None:
        super().__init__(task_name=task_name, task_type="3DSkeletonState")
        lines, dots = Draw3DSkeletonState._get_lines_and_dots(skeleton_state)
        self._lines_task = Draw3DLines(
            self.get_scoped_name("bodies"), lines, joints_color, alpha=alpha
        )
        self._dots_task = Draw3DDots(
            self.get_scoped_name("joints"), dots, lines_color, alpha=alpha
        )

    @property
    def name(self):
        return "3DSkeleton"

    def update(self, skeleton_state) -> None:
        self._update(*Draw3DSkeletonState._get_lines_and_dots(skeleton_state))

    @staticmethod
    def _get_lines_and_dots(skeleton_state):
        """Get all the lines and dots needed to draw the skeleton state
        """
        assert (
            len(skeleton_state.tensor.shape) == 1
        ), "the state has to be zero dimensional"
        dots = skeleton_state.global_translation.numpy()
        skeleton_tree = skeleton_state.skeleton_tree
        parent_indices = skeleton_tree.parent_indices.numpy()
        lines = []
        for node_index in range(len(skeleton_tree)):
            parent_index = parent_indices[node_index]
            if parent_index != -1:
                lines.append([dots[node_index], dots[parent_index]])
        lines = np.array(lines)
        return lines, dots

    def _update(self, lines, dots) -> None:
        self._lines_task.update(lines)
        self._dots_task.update(dots)

    def __iter__(self):
        yield from self._lines_task
        yield from self._dots_task


class Draw3DSkeletonMotion(BasePlotterTask):
    def __init__(
        self,
        task_name: str,
        skeleton_motion,
        frame_index=None,
        joints_color="red",
        lines_color="blue",
        velocity_color="green",
        angular_velocity_color="purple",
        trail_color="black",
        trail_length=10,
        alpha=1.0,
    ) -> None:
        super().__init__(task_name=task_name, task_type="3DSkeletonMotion")
        self._trail_length = trail_length
        self._skeleton_motion = skeleton_motion
        # if frame_index is None:
        curr_skeleton_motion = self._skeleton_motion.clone()
        if frame_index is not None:
            curr_skeleton_motion.tensor = self._skeleton_motion.tensor[frame_index, :]
        # else:
        #     curr_skeleton_motion = self._skeleton_motion[frame_index, :]
        self._skeleton_state_task = Draw3DSkeletonState(
            self.get_scoped_name("skeleton_state"),
            curr_skeleton_motion,
            joints_color=joints_color,
            lines_color=lines_color,
            alpha=alpha,
        )
        vel_lines, avel_lines = Draw3DSkeletonMotion._get_vel_and_avel(
            curr_skeleton_motion
        )
        self._com_pos = curr_skeleton_motion.root_translation.numpy()[
            np.newaxis, ...
        ].repeat(trail_length, axis=0)
        self._vel_task = Draw3DLines(
            self.get_scoped_name("velocity"),
            vel_lines,
            velocity_color,
            influence_lim=False,
            alpha=alpha,
        )
        self._avel_task = Draw3DLines(
            self.get_scoped_name("angular_velocity"),
            avel_lines,
            angular_velocity_color,
            influence_lim=False,
            alpha=alpha,
        )
        self._com_trail_task = Draw3DTrail(
            self.get_scoped_name("com_trail"),
            self._com_pos,
            trail_color,
            marker_size=2,
            influence_lim=True,
            alpha=alpha,
        )

    @property
    def name(self):
        return "3DSkeletonMotion"

    def update(self, frame_index=None, reset_trail=False, skeleton_motion=None) -> None:
        if skeleton_motion is not None:
            self._skeleton_motion = skeleton_motion

        curr_skeleton_motion = self._skeleton_motion.clone()
        if frame_index is not None:
            curr_skeleton_motion.tensor = curr_skeleton_motion.tensor[frame_index, :]
        if reset_trail:
            self._com_pos = curr_skeleton_motion.root_translation.numpy()[
                np.newaxis, ...
            ].repeat(self._trail_length, axis=0)
        else:
            self._com_pos = np.concatenate(
                (
                    curr_skeleton_motion.root_translation.numpy()[np.newaxis, ...],
                    self._com_pos[:-1],
                ),
                axis=0,
            )
        self._skeleton_state_task.update(curr_skeleton_motion)
        self._com_trail_task.update(self._com_pos)
        self._update(*Draw3DSkeletonMotion._get_vel_and_avel(curr_skeleton_motion))

    @staticmethod
    def _get_vel_and_avel(skeleton_motion):
        """Get all the velocity and angular velocity lines
        """
        pos = skeleton_motion.global_translation.numpy()
        vel = skeleton_motion.global_velocity.numpy()
        avel = skeleton_motion.global_angular_velocity.numpy()

        vel_lines = np.stack((pos, pos + vel * 0.02), axis=1)
        avel_lines = np.stack((pos, pos + avel * 0.01), axis=1)
        return vel_lines, avel_lines

    def _update(self, vel_lines, avel_lines) -> None:
        self._vel_task.update(vel_lines)
        self._avel_task.update(avel_lines)

    def __iter__(self):
        yield from self._skeleton_state_task
        yield from self._vel_task
        yield from self._avel_task
        yield from self._com_trail_task


class Draw3DSkeletonMotions(BasePlotterTask):
    def __init__(self, skeleton_motion_tasks) -> None:
        self._skeleton_motion_tasks = skeleton_motion_tasks

    @property
    def name(self):
        return "3DSkeletonMotions"

    def update(self, frame_index) -> None:
        list(map(lambda x: x.update(frame_index), self._skeleton_motion_tasks))

    def __iter__(self):
        yield from self._skeleton_state_tasks
