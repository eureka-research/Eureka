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

"""Factory: schema for base class configuration.

Used by Hydra. Defines template for base class YAML file.
"""

from dataclasses import dataclass


@dataclass
class Mode:
    export_scene: bool  # export scene to USD
    export_states: bool  # export states to NPY


@dataclass
class PhysX:
    solver_type: int  # default = 1 (Temporal Gauss-Seidel)
    num_threads: int
    num_subscenes: int
    use_gpu: bool
    num_position_iterations: int  # number of position iterations for solver (default = 4)
    num_velocity_iterations: int  # number of velocity iterations for solver (default = 1)
    contact_offset: float  # default = 0.02
    rest_offset: float  # default = 0.001
    bounce_threshold_velocity: float  # default = 0.01
    max_depenetration_velocity: float  # default = 100.0
    friction_offset_threshold: float  # default = 0.04
    friction_correlation_distance: float  # default = 0.025
    max_gpu_contact_pairs: int  # default = 1024 * 1024
    default_buffer_size_multiplier: float
    contact_collection: int  # 0: CC_NEVER (do not collect contact info), 1: CC_LAST_SUBSTEP (collect contact info on last substep), 2: CC_ALL_SUBSTEPS (collect contact info at all substeps)


@dataclass
class Sim:
    dt: float  # timestep size (default = 1.0 / 60.0)
    num_substeps: int  # number of substeps (default = 2)
    up_axis: str
    use_gpu_pipeline: bool
    gravity: list  # gravitational acceleration vector
    add_damping: bool  # add damping to stabilize gripper-object interactions
    physx: PhysX


@dataclass
class Env:
    env_spacing: float  # lateral offset between envs
    franka_depth: float  # depth offset of Franka base relative to env origin
    table_height: float  # height of table
    franka_friction: float  # coefficient of friction associated with Franka
    table_friction: float  # coefficient of friction associated with table


@dataclass
class FactorySchemaConfigBase:
    mode: Mode
    sim: Sim
    env: Env
