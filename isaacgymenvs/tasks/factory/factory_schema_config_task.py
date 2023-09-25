# Copyright (c) 2021-2023, NVIDIA Corporation
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

"""Factory: schema for task class configurations.

Used by Hydra. Defines template for task class YAML files. Not enforced.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Sim:
    use_gpu_pipeline: bool  # use GPU pipeline
    up_axis: str  # up-down axis {x, y, z}
    dt: float  # timestep size
    gravity: list[float]  # gravity vector

    disable_gravity: bool  # disable gravity for all actors


@dataclass
class Env:
    numObservations: int  # number of observations per env; camel case required by VecTask
    numActions: int  # number of actions per env; camel case required by VecTask
    numEnvs: int  # number of envs; camel case required by VecTask


@dataclass
class Randomize:
    franka_arm_initial_dof_pos: list[float]  # initial Franka arm DOF position (7)


@dataclass
class RL:
    pos_action_scale: list[float]  # scale on pos displacement targets (3), to convert [-1, 1] to +- x m
    rot_action_scale: list[float]  # scale on rot displacement targets (3), to convert [-1, 1] to +- x rad
    force_action_scale: list[float]  # scale on force targets (3), to convert [-1, 1] to +- x N
    torque_action_scale: list[float]  # scale on torque targets (3), to convert [-1, 1] to +- x Nm

    clamp_rot: bool  # clamp small values of rotation actions to zero
    clamp_rot_thresh: float  # smallest acceptable value

    max_episode_length: int  # max number of timesteps in each episode


@dataclass
class All:
    jacobian_type: str  # map between joint space and task space via geometric or analytic Jacobian {geometric, analytic}
    gripper_prop_gains: list[float]  # proportional gains on left and right Franka gripper finger DOF position (2)
    gripper_deriv_gains: list[float]  # derivative gains on left and right Franka gripper finger DOF position (2)


@dataclass
class GymDefault:
    joint_prop_gains: list[int]  # proportional gains on Franka arm DOF position (7)
    joint_deriv_gains: list[int]  # derivative gains on Franka arm DOF position (7)


@dataclass
class JointSpaceIK:
    ik_method: str  # use Jacobian pseudoinverse, Jacobian transpose, damped least squares or adaptive SVD {pinv, trans, dls, svd}
    joint_prop_gains: list[int]
    joint_deriv_gains: list[int]


@dataclass
class JointSpaceID:
    ik_method: str
    joint_prop_gains: list[int]
    joint_deriv_gains: list[int]


@dataclass
class TaskSpaceImpedance:
    motion_ctrl_axes: list[bool]  # axes for which to enable motion control {0, 1} (6)
    task_prop_gains: list[float]  # proportional gains on Franka fingertip pose (6)
    task_deriv_gains: list[float]  # derivative gains on Franka fingertip pose (6)


@dataclass
class OperationalSpaceMotion:
    motion_ctrl_axes: list[bool]
    task_prop_gains: list[float]
    task_deriv_gains: list[float]


@dataclass
class OpenLoopForce:
    force_ctrl_axes: list[bool]  # axes for which to enable force control {0, 1} (6)


@dataclass
class ClosedLoopForce:
    force_ctrl_axes: list[bool]
    wrench_prop_gains: list[float]  # proportional gains on Franka finger force (6)


@dataclass
class HybridForceMotion:
    motion_ctrl_axes: list[bool]
    task_prop_gains: list[float]
    task_deriv_gains: list[float]
    force_ctrl_axes: list[bool]
    wrench_prop_gains: list[float]


@dataclass
class Ctrl:
    ctrl_type: str # {gym_default,
                   #  joint_space_ik,
                   #  joint_space_id,
                   #  task_space_impedance,
                   #  operational_space_motion,
                   #  open_loop_force,
                   #  closed_loop_force,
                   #  hybrid_force_motion}
    gym_default: GymDefault
    joint_space_ik: JointSpaceIK
    joint_space_id: JointSpaceID
    task_space_impedance: TaskSpaceImpedance
    operational_space_motion: OperationalSpaceMotion
    open_loop_force: OpenLoopForce
    closed_loop_force: ClosedLoopForce
    hybrid_force_motion: HybridForceMotion


@dataclass
class FactorySchemaConfigTask:
    name: str
    physics_engine: str
    sim: Sim
    env: Env
    rl: RL
    ctrl: Ctrl
