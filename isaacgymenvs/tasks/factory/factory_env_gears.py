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

"""Factory: class for gears env.

Inherits base class and abstract environment class. Inherited by gear task class. Not directly executed.

Configuration defined in FactoryEnvGears.yaml. Asset info defined in factory_asset_info_gears.yaml.
"""

import hydra
import numpy as np
import os
import torch

from isaacgym import gymapi
from isaacgymenvs.tasks.factory.factory_base import FactoryBase
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from isaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv


class FactoryEnvGears(FactoryBase, FactoryABCEnv):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass. Acquire tensors."""

        self._get_env_yaml_params()

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.acquire_base_tensors()  # defined in superclass
        self._acquire_env_tensors()
        self.refresh_base_tensors()  # defined in superclass
        self.refresh_env_tensors()

    def _get_env_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_env', node=FactorySchemaConfigEnv)

        config_path = 'task/FactoryEnvGears.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_gears.yaml'  # relative to Hydra search path (cfg dir)
        self.asset_info_gears = hydra.compose(config_name=asset_info_path)
        self.asset_info_gears = self.asset_info_gears['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting


    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        gear_small_asset, gear_medium_asset, gear_large_asset, base_asset = self._import_env_assets()
        self._create_actors(lower, upper, num_per_row, franka_asset, gear_small_asset, gear_medium_asset,
                            gear_large_asset, base_asset, table_asset)

    def _import_env_assets(self):
        """Set gear and base asset options. Import assets."""

        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'factory', 'urdf')
        gear_small_file = 'factory_gear_small.urdf'
        gear_medium_file = 'factory_gear_medium.urdf'
        gear_large_file = 'factory_gear_large.urdf'
        if self.cfg_env.env.tight_or_loose == 'tight':
            base_file = 'factory_gear_base_tight.urdf'
        elif self.cfg_env.env.tight_or_loose == 'loose':
            base_file = 'factory_gear_base_loose.urdf'

        gear_options = gymapi.AssetOptions()
        gear_options.flip_visual_attachments = False
        gear_options.fix_base_link = False
        gear_options.thickness = 0.0  # default = 0.02
        gear_options.density = self.cfg_env.env.gears_density  # default = 1000.0
        gear_options.armature = 0.0  # default = 0.0
        gear_options.use_physx_armature = True
        gear_options.linear_damping = 0.0  # default = 0.0
        gear_options.max_linear_velocity = 1000.0  # default = 1000.0
        gear_options.angular_damping = 0.0  # default = 0.5
        gear_options.max_angular_velocity = 64.0  # default = 64.0
        gear_options.disable_gravity = False
        gear_options.enable_gyroscopic_forces = True
        gear_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        gear_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            gear_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        base_options = gymapi.AssetOptions()
        base_options.flip_visual_attachments = False
        base_options.fix_base_link = True
        base_options.thickness = 0.0  # default = 0.02
        base_options.density = self.cfg_env.env.base_density  # default = 1000.0
        base_options.armature = 0.0  # default = 0.0
        base_options.use_physx_armature = True
        base_options.linear_damping = 0.0  # default = 0.0
        base_options.max_linear_velocity = 1000.0  # default = 1000.0
        base_options.angular_damping = 0.0  # default = 0.5
        base_options.max_angular_velocity = 64.0  # default = 64.0
        base_options.disable_gravity = False
        base_options.enable_gyroscopic_forces = True
        base_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        base_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            base_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        gear_small_asset = self.gym.load_asset(self.sim, urdf_root, gear_small_file, gear_options)
        gear_medium_asset = self.gym.load_asset(self.sim, urdf_root, gear_medium_file, gear_options)
        gear_large_asset = self.gym.load_asset(self.sim, urdf_root, gear_large_file, gear_options)
        base_asset = self.gym.load_asset(self.sim, urdf_root, base_file, base_options)

        return gear_small_asset, gear_medium_asset, gear_large_asset, base_asset

    def _create_actors(self, lower, upper, num_per_row, franka_asset, gear_small_asset, gear_medium_asset,
                       gear_large_asset, base_asset, table_asset):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        franka_pose = gymapi.Transform()
        franka_pose.p.x = self.cfg_base.env.franka_depth
        franka_pose.p.y = 0.0
        franka_pose.p.z = 0.0
        franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        gear_pose = gymapi.Transform()
        gear_pose.p.x = 0.0
        gear_pose.p.y = self.cfg_env.env.gears_lateral_offset
        gear_pose.p.z = self.cfg_base.env.table_height
        gear_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        base_pose = gymapi.Transform()
        base_pose.p.x = 0.0
        base_pose.p.y = 0.0
        base_pose.p.z = self.cfg_base.env.table_height
        base_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = 0.0
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.franka_handles = []
        self.gear_small_handles = []
        self.gear_medium_handles = []
        self.gear_large_handles = []
        self.base_handles = []
        self.table_handles = []
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.gear_small_actor_ids_sim = []  # within-sim indices
        self.gear_medium_actor_ids_sim = []  # within-sim indices
        self.gear_large_actor_ids_sim = []  # within-sim indices
        self.base_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i + self.num_envs, 0, 0)
            else:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i, 0, 0)
            self.franka_actor_ids_sim.append(actor_count)
            actor_count += 1

            gear_small_handle = self.gym.create_actor(env_ptr, gear_small_asset, gear_pose, 'gear_small', i, 0, 0)
            self.gear_small_actor_ids_sim.append(actor_count)
            actor_count += 1

            gear_medium_handle = self.gym.create_actor(env_ptr, gear_medium_asset, gear_pose, 'gear_medium', i, 0, 0)
            self.gear_medium_actor_ids_sim.append(actor_count)
            actor_count += 1

            gear_large_handle = self.gym.create_actor(env_ptr, gear_large_asset, gear_pose, 'gear_large', i, 0, 0)
            self.gear_large_actor_ids_sim.append(actor_count)
            actor_count += 1

            base_handle = self.gym.create_actor(env_ptr, base_asset, base_pose, 'base', i, 0, 0)
            self.base_actor_ids_sim.append(actor_count)
            actor_count += 1

            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, 'table', i, 0, 0)
            self.table_actor_ids_sim.append(actor_count)
            actor_count += 1

            link7_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_link7', gymapi.DOMAIN_ACTOR)
            hand_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand', gymapi.DOMAIN_ACTOR)
            left_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                  gymapi.DOMAIN_ACTOR)
            right_finger_id = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_rightfinger',
                                                                   gymapi.DOMAIN_ACTOR)
            self.shape_ids = [link7_id, hand_id, left_finger_id, right_finger_id]

            franka_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, franka_handle)
            for shape_id in self.shape_ids:
                franka_shape_props[shape_id].friction = self.cfg_base.env.franka_friction
                franka_shape_props[shape_id].rolling_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].torsion_friction = 0.0  # default = 0.0
                franka_shape_props[shape_id].restitution = 0.0  # default = 0.0
                franka_shape_props[shape_id].compliance = 0.0  # default = 0.0
                franka_shape_props[shape_id].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, franka_handle, franka_shape_props)

            gear_small_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, gear_small_handle)
            gear_small_shape_props[0].friction = self.cfg_env.env.gears_friction
            gear_small_shape_props[0].rolling_friction = 0.0  # default = 0.0
            gear_small_shape_props[0].torsion_friction = 0.0  # default = 0.0
            gear_small_shape_props[0].restitution = 0.0  # default = 0.0
            gear_small_shape_props[0].compliance = 0.0  # default = 0.0
            gear_small_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, gear_small_handle, gear_small_shape_props)

            gear_medium_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, gear_medium_handle)
            gear_medium_shape_props[0].friction = self.cfg_env.env.gears_friction
            gear_medium_shape_props[0].rolling_friction = 0.0  # default = 0.0
            gear_medium_shape_props[0].torsion_friction = 0.0  # default = 0.0
            gear_medium_shape_props[0].restitution = 0.0  # default = 0.0
            gear_medium_shape_props[0].compliance = 0.0  # default = 0.0
            gear_medium_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, gear_medium_handle, gear_medium_shape_props)

            gear_large_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, gear_large_handle)
            gear_large_shape_props[0].friction = self.cfg_env.env.gears_friction
            gear_large_shape_props[0].rolling_friction = 0.0  # default = 0.0
            gear_large_shape_props[0].torsion_friction = 0.0  # default = 0.0
            gear_large_shape_props[0].restitution = 0.0  # default = 0.0
            gear_large_shape_props[0].compliance = 0.0  # default = 0.0
            gear_large_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, gear_large_handle, gear_large_shape_props)

            base_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, base_handle)
            base_shape_props[0].friction = self.cfg_env.env.base_friction
            base_shape_props[0].rolling_friction = 0.0  # default = 0.0
            base_shape_props[0].torsion_friction = 0.0  # default = 0.0
            base_shape_props[0].restitution = 0.0  # default = 0.0
            base_shape_props[0].compliance = 0.0  # default = 0.0
            base_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, base_handle, base_shape_props)

            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            table_shape_props[0].friction = self.cfg_base.env.table_friction
            table_shape_props[0].rolling_friction = 0.0  # default = 0.0
            table_shape_props[0].torsion_friction = 0.0  # default = 0.0
            table_shape_props[0].restitution = 0.0  # default = 0.0
            table_shape_props[0].compliance = 0.0  # default = 0.0
            table_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            self.franka_num_dofs = self.gym.get_actor_dof_count(env_ptr, franka_handle)

            self.gym.enable_actor_dof_force_sensors(env_ptr, franka_handle)

            self.env_ptrs.append(env_ptr)
            self.franka_handles.append(franka_handle)
            self.gear_small_handles.append(gear_small_handle)
            self.gear_medium_handles.append(gear_medium_handle)
            self.gear_large_handles.append(gear_large_handle)
            self.base_handles.append(base_handle)
            self.table_handles.append(table_handle)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(self.franka_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.gear_small_actor_ids_sim = torch.tensor(self.gear_small_actor_ids_sim, dtype=torch.int32,
                                                     device=self.device)
        self.gear_medium_actor_ids_sim = torch.tensor(self.gear_medium_actor_ids_sim, dtype=torch.int32,
                                                      device=self.device)
        self.gear_large_actor_ids_sim = torch.tensor(self.gear_large_actor_ids_sim, dtype=torch.int32,
                                                     device=self.device)
        self.base_actor_ids_sim = torch.tensor(self.base_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.gear_small_actor_id_env = self.gym.find_actor_index(env_ptr, 'gear_small', gymapi.DOMAIN_ENV)
        self.gear_medium_actor_id_env = self.gym.find_actor_index(env_ptr, 'gear_medium', gymapi.DOMAIN_ENV)
        self.gear_large_actor_id_env = self.gym.find_actor_index(env_ptr, 'gear_large', gymapi.DOMAIN_ENV)
        self.base_actor_id_env = self.gym.find_actor_index(env_ptr, 'base', gymapi.DOMAIN_ENV)

        # For extracting body pos/quat, force, and Jacobian
        self.gear_small_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, gear_small_handle, 'gear_small',
                                                                           gymapi.DOMAIN_ENV)
        self.gear_mediums_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, gear_medium_handle, 'gear_small',
                                                                             gymapi.DOMAIN_ENV)
        self.gear_large_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, gear_large_handle, 'gear_small',
                                                                           gymapi.DOMAIN_ENV)
        self.base_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, base_handle, 'base', gymapi.DOMAIN_ENV)
        self.hand_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_hand',
                                                                     gymapi.DOMAIN_ENV)
        self.left_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle, 'panda_leftfinger',
                                                                            gymapi.DOMAIN_ENV)
        self.right_finger_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                             'panda_rightfinger', gymapi.DOMAIN_ENV)
        self.fingertip_centered_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, franka_handle,
                                                                                   'panda_fingertip_centered',
                                                                                   gymapi.DOMAIN_ENV)

    def _acquire_env_tensors(self):
        """Acquire and wrap tensors. Create views."""

        self.gear_small_pos = self.root_pos[:, self.gear_small_actor_id_env, 0:3]
        self.gear_small_quat = self.root_quat[:, self.gear_small_actor_id_env, 0:4]
        self.gear_small_linvel = self.root_linvel[:, self.gear_small_actor_id_env, 0:3]
        self.gear_small_angvel = self.root_angvel[:, self.gear_small_actor_id_env, 0:3]

        self.gear_medium_pos = self.root_pos[:, self.gear_medium_actor_id_env, 0:3]
        self.gear_medium_quat = self.root_quat[:, self.gear_medium_actor_id_env, 0:4]
        self.gear_medium_linvel = self.root_linvel[:, self.gear_medium_actor_id_env, 0:3]
        self.gear_medium_angvel = self.root_angvel[:, self.gear_medium_actor_id_env, 0:3]

        self.gear_large_pos = self.root_pos[:, self.gear_large_actor_id_env, 0:3]
        self.gear_large_quat = self.root_quat[:, self.gear_large_actor_id_env, 0:4]
        self.gear_large_linvel = self.root_linvel[:, self.gear_large_actor_id_env, 0:3]
        self.gear_large_angvel = self.root_angvel[:, self.gear_large_actor_id_env, 0:3]

        self.base_pos = self.root_pos[:, self.base_actor_id_env, 0:3]
        self.base_quat = self.root_quat[:, self.base_actor_id_env, 0:4]

        self.gear_small_com_pos = fc.translate_along_local_z(pos=self.gear_small_pos,
                                                             quat=self.gear_small_quat,
                                                             offset=self.asset_info_gears.gear_base_height + self.asset_info_gears.gear_height * 0.5,
                                                             device=self.device)
        self.gear_small_com_quat = self.gear_small_quat  # always equal
        self.gear_small_com_linvel = self.gear_small_linvel + torch.cross(self.gear_small_angvel,
                                                                          (self.gear_small_com_pos - self.gear_small_pos),
                                                                          dim=1)
        self.gear_small_com_angvel = self.gear_small_angvel  # always equal

        self.gear_medium_com_pos = fc.translate_along_local_z(pos=self.gear_medium_pos,
                                                              quat=self.gear_medium_quat,
                                                              offset=self.asset_info_gears.gear_base_height + self.asset_info_gears.gear_height * 0.5,
                                                              device=self.device)
        self.gear_medium_com_quat = self.gear_medium_quat  # always equal
        self.gear_medium_com_linvel = self.gear_medium_linvel + torch.cross(self.gear_medium_angvel,
                                                                            (self.gear_medium_com_pos - self.gear_medium_pos),
                                                                            dim=1)
        self.gear_medium_com_angvel = self.gear_medium_angvel  # always equal

        self.gear_large_com_pos = fc.translate_along_local_z(pos=self.gear_large_pos,
                                                             quat=self.gear_large_quat,
                                                             offset=self.asset_info_gears.gear_base_height + self.asset_info_gears.gear_height * 0.5,
                                                             device=self.device)
        self.gear_large_com_quat = self.gear_large_quat  # always equal
        self.gear_large_com_linvel = self.gear_large_linvel + torch.cross(self.gear_large_angvel,
                                                                          (self.gear_large_com_pos - self.gear_large_pos),
                                                                          dim=1)
        self.gear_large_com_angvel = self.gear_large_angvel  # always equal

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        self.gear_small_com_pos = fc.translate_along_local_z(pos=self.gear_small_pos,
                                                             quat=self.gear_small_quat,
                                                             offset=self.asset_info_gears.gear_base_height + self.asset_info_gears.gear_height * 0.5,
                                                             device=self.device)
        self.gear_small_com_linvel = self.gear_small_linvel + torch.cross(self.gear_small_angvel,
                                                                          (self.gear_small_com_pos - self.gear_small_pos),
                                                                          dim=1)

        self.gear_medium_com_pos = fc.translate_along_local_z(pos=self.gear_medium_pos,
                                                              quat=self.gear_medium_quat,
                                                              offset=self.asset_info_gears.gear_base_height + self.asset_info_gears.gear_height * 0.5,
                                                              device=self.device)
        self.gear_medium_com_linvel = self.gear_medium_linvel + torch.cross(self.gear_medium_angvel,
                                                                            (self.gear_medium_com_pos - self.gear_medium_pos),
                                                                            dim=1)

        self.gear_large_com_pos = fc.translate_along_local_z(pos=self.gear_large_pos,
                                                             quat=self.gear_large_quat,
                                                             offset=self.asset_info_gears.gear_base_height + self.asset_info_gears.gear_height * 0.5,
                                                             device=self.device)
        self.gear_large_com_linvel = self.gear_large_linvel + torch.cross(self.gear_large_angvel,
                                                                          (self.gear_large_com_pos - self.gear_large_pos),
                                                                          dim=1)
