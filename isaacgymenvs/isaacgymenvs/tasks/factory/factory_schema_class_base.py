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
"""Factory: abstract base class for base class.

Inherits ABC class. Inherited by base class. Defines template for base class.
"""

from abc import ABC, abstractmethod


class FactoryABCBase(ABC):

    @abstractmethod
    def __init__(self):
        """Initialize instance variables. Initialize VecTask superclass."""
        pass

    @abstractmethod
    def _get_base_yaml_params(self):
        """Initialize instance variables from YAML files."""
        pass

    @abstractmethod
    def create_sim(self):
        """Set sim and PhysX params. Create sim object, ground plane, and envs."""
        pass

    @abstractmethod
    def _create_ground_plane(self):
        """Set ground plane params. Add plane."""
        pass

    @abstractmethod
    def import_franka_assets(self):
        """Set Franka and table asset options. Import assets."""
        pass

    @abstractmethod
    def acquire_base_tensors(self):
        """Acquire and wrap tensors. Create views."""
        pass

    @abstractmethod
    def refresh_base_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.
        pass

    @abstractmethod
    def parse_controller_spec(self):
        """Parse controller specification into lower-level controller configuration."""
        pass

    @abstractmethod
    def generate_ctrl_signals(self):
        """Get Jacobian. Set Franka DOF position targets or DOF torques."""
        pass

    @abstractmethod
    def enable_gravity(self):
        """Enable gravity."""
        pass

    @abstractmethod
    def disable_gravity(self):
        """Disable gravity."""
        pass

    @abstractmethod
    def export_scene(self):
        """Export scene to USD."""
        pass

    @abstractmethod
    def extract_poses(self):
        """Extract poses of all bodies."""
        pass
