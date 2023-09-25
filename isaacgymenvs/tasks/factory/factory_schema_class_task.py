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

"""Factory: abstract base class for task classes.

Inherits ABC class. Inherited by task classes. Defines template for task classes.
"""

from abc import ABC, abstractmethod


class FactoryABCTask(ABC):

    @abstractmethod
    def __init__(self):
        """Initialize instance variables. Initialize environment superclass."""
        pass

    @abstractmethod
    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""
        pass

    @abstractmethod
    def _acquire_task_tensors(self):
        """Acquire tensors."""
        pass

    @abstractmethod
    def _refresh_task_tensors(self):
        """Refresh tensors."""
        pass

    @abstractmethod
    def pre_physics_step(self):
        """Reset environments. Apply actions from policy as controller targets. Simulation step called after this method."""
        pass

    @abstractmethod
    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward."""
        pass

    @abstractmethod
    def compute_observations(self):
        """Compute observations."""
        pass

    @abstractmethod
    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""
        pass

    @abstractmethod
    def _update_rew_buf(self):
        """Compute reward at current timestep."""
        pass

    @abstractmethod
    def _update_reset_buf(self):
        """Assign environments for reset if successful or failed."""
        pass

    @abstractmethod
    def reset_idx(self):
        """Reset specified environments."""
        pass

    @abstractmethod
    def _reset_franka(self):
        """Reset DOF states and DOF targets of Franka."""
        pass

    @abstractmethod
    def _reset_object(self):
        """Reset root state of object."""
        pass

    @abstractmethod
    def _reset_buffers(self):
        """Reset buffers."""
        pass

    @abstractmethod
    def _set_viewer_params(self):
        """Set viewer parameters."""
        pass
