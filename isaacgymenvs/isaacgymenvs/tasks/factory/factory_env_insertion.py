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

"""Factory: class for insertion env.

Inherits base class and abstract environment class. Inherited by insertion task class. Not directly executed.

Configuration defined in FactoryEnvInsertion.yaml. Asset info defined in factory_asset_info_insertion.yaml.
"""

import hydra
import numpy as np
import os
import torch

from isaacgym import gymapi
from isaacgymenvs.tasks.factory.factory_base import FactoryBase
from isaacgymenvs.tasks.factory.factory_schema_class_env import FactoryABCEnv
from isaacgymenvs.tasks.factory.factory_schema_config_env import FactorySchemaConfigEnv


class FactoryEnvInsertion(FactoryBase, FactoryABCEnv):

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

        config_path = 'task/FactoryEnvInsertion.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_env = hydra.compose(config_name=config_path)
        self.cfg_env = self.cfg_env['task']  # strip superfluous nesting

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_insertion.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_insertion = hydra.compose(config_name=asset_info_path)
        self.asset_info_insertion = self.asset_info_insertion['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

    def create_envs(self):
        """Set env options. Import assets. Create actors."""

        lower = gymapi.Vec3(-self.cfg_base.env.env_spacing, -self.cfg_base.env.env_spacing, 0.0)
        upper = gymapi.Vec3(self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing, self.cfg_base.env.env_spacing)
        num_per_row = int(np.sqrt(self.num_envs))

        self.print_sdf_warning()
        franka_asset, table_asset = self.import_franka_assets()
        plug_assets, socket_assets = self._import_env_assets()
        self._create_actors(lower, upper, num_per_row, franka_asset, plug_assets, socket_assets, table_asset)

    def _import_env_assets(self):
        """Set plug and socket asset options. Import assets."""

        urdf_root = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'assets', 'factory', 'urdf')

        plug_options = gymapi.AssetOptions()
        plug_options.flip_visual_attachments = False
        plug_options.fix_base_link = False
        plug_options.thickness = 0.0  # default = 0.02
        plug_options.armature = 0.0  # default = 0.0
        plug_options.use_physx_armature = True
        plug_options.linear_damping = 0.0  # default = 0.0
        plug_options.max_linear_velocity = 1000.0  # default = 1000.0
        plug_options.angular_damping = 0.0  # default = 0.5
        plug_options.max_angular_velocity = 64.0  # default = 64.0
        plug_options.disable_gravity = False
        plug_options.enable_gyroscopic_forces = True
        plug_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        plug_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            plug_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        socket_options = gymapi.AssetOptions()
        socket_options.flip_visual_attachments = False
        socket_options.fix_base_link = True
        socket_options.thickness = 0.0  # default = 0.02
        socket_options.armature = 0.0  # default = 0.0
        socket_options.use_physx_armature = True
        socket_options.linear_damping = 0.0  # default = 0.0
        socket_options.max_linear_velocity = 1000.0  # default = 1000.0
        socket_options.angular_damping = 0.0  # default = 0.5
        socket_options.max_angular_velocity = 64.0  # default = 64.0
        socket_options.disable_gravity = False
        socket_options.enable_gyroscopic_forces = True
        socket_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        socket_options.use_mesh_materials = False
        if self.cfg_base.mode.export_scene:
            socket_options.mesh_normal_mode = gymapi.COMPUTE_PER_FACE

        plug_assets = []
        socket_assets = []
        for subassembly in self.cfg_env.env.desired_subassemblies:
            components = list(self.asset_info_insertion[subassembly])
            plug_file = self.asset_info_insertion[subassembly][components[0]]['urdf_path'] + '.urdf'
            socket_file = self.asset_info_insertion[subassembly][components[1]]['urdf_path'] + '.urdf'
            plug_options.density = self.asset_info_insertion[subassembly][components[0]]['density']
            socket_options.density = self.asset_info_insertion[subassembly][components[1]]['density']
            plug_asset = self.gym.load_asset(self.sim, urdf_root, plug_file, plug_options)
            socket_asset = self.gym.load_asset(self.sim, urdf_root, socket_file, socket_options)
            plug_assets.append(plug_asset)
            socket_assets.append(socket_asset)

        return plug_assets, socket_assets

    def _create_actors(self, lower, upper, num_per_row, franka_asset, plug_assets, socket_assets, table_asset):
        """Set initial actor poses. Create actors. Set shape and DOF properties."""

        franka_pose = gymapi.Transform()
        franka_pose.p.x = self.cfg_base.env.franka_depth
        franka_pose.p.y = 0.0
        franka_pose.p.z = 0.0
        franka_pose.r = gymapi.Quat(0.0, 0.0, 1.0, 0.0)

        table_pose = gymapi.Transform()
        table_pose.p.x = 0.0
        table_pose.p.y = 0.0
        table_pose.p.z = self.cfg_base.env.table_height * 0.5
        table_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.env_ptrs = []
        self.franka_handles = []
        self.plug_handles = []
        self.socket_handles = []
        self.table_handles = []
        self.shape_ids = []
        self.franka_actor_ids_sim = []  # within-sim indices
        self.plug_actor_ids_sim = []  # within-sim indices
        self.socket_actor_ids_sim = []  # within-sim indices
        self.table_actor_ids_sim = []  # within-sim indices
        actor_count = 0

        for i in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.cfg_env.sim.disable_franka_collisions:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i + self.num_envs,
                                                      0, 0)
            else:
                franka_handle = self.gym.create_actor(env_ptr, franka_asset, franka_pose, 'franka', i, 0, 0)
            self.franka_actor_ids_sim.append(actor_count)
            actor_count += 1

            j = np.random.randint(0, len(self.cfg_env.env.desired_subassemblies))
            subassembly = self.cfg_env.env.desired_subassemblies[j]
            components = list(self.asset_info_insertion[subassembly])

            plug_pose = gymapi.Transform()
            plug_pose.p.x = 0.0
            plug_pose.p.y = self.cfg_env.env.plug_lateral_offset
            plug_pose.p.z = self.cfg_base.env.table_height
            plug_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            plug_handle = self.gym.create_actor(env_ptr, plug_assets[j], plug_pose, 'plug', i, 0, 0)
            self.plug_actor_ids_sim.append(actor_count)
            actor_count += 1

            socket_pose = gymapi.Transform()
            socket_pose.p.x = 0.0
            socket_pose.p.y = 0.0
            socket_pose.p.z = self.cfg_base.env.table_height
            socket_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
            socket_handle = self.gym.create_actor(env_ptr, socket_assets[j], socket_pose, 'socket', i, 0, 0)
            self.socket_actor_ids_sim.append(actor_count)
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

            plug_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, plug_handle)
            plug_shape_props[0].friction = self.asset_info_insertion[subassembly][components[0]]['friction']
            plug_shape_props[0].rolling_friction = 0.0  # default = 0.0
            plug_shape_props[0].torsion_friction = 0.0  # default = 0.0
            plug_shape_props[0].restitution = 0.0  # default = 0.0
            plug_shape_props[0].compliance = 0.0  # default = 0.0
            plug_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, plug_handle, plug_shape_props)

            socket_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, socket_handle)
            socket_shape_props[0].friction = self.asset_info_insertion[subassembly][components[1]]['friction']
            socket_shape_props[0].rolling_friction = 0.0  # default = 0.0
            socket_shape_props[0].torsion_friction = 0.0  # default = 0.0
            socket_shape_props[0].restitution = 0.0  # default = 0.0
            socket_shape_props[0].compliance = 0.0  # default = 0.0
            socket_shape_props[0].thickness = 0.0  # default = 0.0
            self.gym.set_actor_rigid_shape_properties(env_ptr, socket_handle, socket_shape_props)

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
            self.plug_handles.append(plug_handle)
            self.socket_handles.append(socket_handle)
            self.table_handles.append(table_handle)

        self.num_actors = int(actor_count / self.num_envs)  # per env
        self.num_bodies = self.gym.get_env_rigid_body_count(env_ptr)  # per env
        self.num_dofs = self.gym.get_env_dof_count(env_ptr)  # per env

        # For setting targets
        self.franka_actor_ids_sim = torch.tensor(self.franka_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.plug_actor_ids_sim = torch.tensor(self.plug_actor_ids_sim, dtype=torch.int32, device=self.device)
        self.socket_actor_ids_sim = torch.tensor(self.socket_actor_ids_sim, dtype=torch.int32, device=self.device)

        # For extracting root pos/quat
        self.plug_actor_id_env = self.gym.find_actor_index(env_ptr, 'plug', gymapi.DOMAIN_ENV)
        self.socket_actor_id_env = self.gym.find_actor_index(env_ptr, 'socket', gymapi.DOMAIN_ENV)

        # For extracting body pos/quat, force, and Jacobian
        self.plug_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, plug_handle, 'plug', gymapi.DOMAIN_ENV)
        self.socket_body_id_env = self.gym.find_actor_rigid_body_index(env_ptr, socket_handle, 'socket',
                                                                       gymapi.DOMAIN_ENV)
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

        self.plug_pos = self.root_pos[:, self.plug_actor_id_env, 0:3]
        self.plug_quat = self.root_quat[:, self.plug_actor_id_env, 0:4]
        self.plug_linvel = self.root_linvel[:, self.plug_actor_id_env, 0:3]
        self.plug_angvel = self.root_angvel[:, self.plug_actor_id_env, 0:3]

        self.socket_pos = self.root_pos[:, self.socket_actor_id_env, 0:3]
        self.socket_quat = self.root_quat[:, self.socket_actor_id_env, 0:4]

        # TODO: Define socket height and plug height params in asset info YAML.
        # self.plug_com_pos = self.translate_along_local_z(pos=self.plug_pos,
        #                                                  quat=self.plug_quat,
        #                                                  offset=self.socket_heights + self.plug_heights * 0.5,
        #                                                  device=self.device)
        self.plug_com_quat = self.plug_quat  # always equal
        # self.plug_com_linvel = self.plug_linvel + torch.cross(self.plug_angvel,
        #                                                       (self.plug_com_pos - self.plug_pos),
        #                                                       dim=1)
        self.plug_com_angvel = self.plug_angvel  # always equal

    def refresh_env_tensors(self):
        """Refresh tensors."""
        # NOTE: Tensor refresh functions should be called once per step, before setters.

        # TODO: Define socket height and plug height params in asset info YAML.
        # self.plug_com_pos = self.translate_along_local_z(pos=self.plug_pos,
        #                                                  quat=self.plug_quat,
        #                                                  offset=self.socket_heights + self.plug_heights * 0.5,
        #                                                  device=self.device)
        # self.plug_com_linvel = self.plug_linvel + torch.cross(self.plug_angvel,
        #                                                       (self.plug_com_pos - self.plug_pos),
        #                                                       dim=1)
