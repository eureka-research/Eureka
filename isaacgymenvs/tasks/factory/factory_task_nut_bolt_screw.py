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

"""Factory: Class for nut-bolt screw task.

Inherits nut-bolt environment class and abstract task class (not enforced). Can be executed with
python train.py task=FactoryTaskNutBoltScrew

Initial Franka/nut states are ideal for M16 nut-and-bolt.
In this example, initial state randomization is not applied; thus, policy should succeed almost instantly.
"""

import hydra
import math
import omegaconf
import os
import torch

from isaacgym import gymapi, gymtorch, torch_utils
import isaacgymenvs.tasks.factory.factory_control as fc
from isaacgymenvs.tasks.factory.factory_env_nut_bolt import FactoryEnvNutBolt
from isaacgymenvs.tasks.factory.factory_schema_class_task import FactoryABCTask
from isaacgymenvs.tasks.factory.factory_schema_config_task import FactorySchemaConfigTask


class FactoryTaskNutBoltScrew(FactoryEnvNutBolt, FactoryABCTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """Initialize instance variables. Initialize environment superclass."""

        super().__init__(cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render)

        self.cfg = cfg
        self._get_task_yaml_params()
        self._acquire_task_tensors()
        self.parse_controller_spec()

        if self.cfg_task.sim.disable_gravity:
            self.disable_gravity()

        if self.viewer != None:
            self._set_viewer_params()

    def _get_task_yaml_params(self):
        """Initialize instance variables from YAML files."""

        cs = hydra.core.config_store.ConfigStore.instance()
        cs.store(name='factory_schema_config_task', node=FactorySchemaConfigTask)

        self.cfg_task = omegaconf.OmegaConf.create(self.cfg)
        self.max_episode_length = self.cfg_task.rl.max_episode_length  # required instance var for VecTask

        asset_info_path = '../../assets/factory/yaml/factory_asset_info_nut_bolt.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.asset_info_nut_bolt = hydra.compose(config_name=asset_info_path)
        self.asset_info_nut_bolt = self.asset_info_nut_bolt['']['']['']['']['']['']['assets']['factory']['yaml']  # strip superfluous nesting

        ppo_path = 'train/FactoryTaskNutBoltScrewPPO.yaml'  # relative to Gym's Hydra search path (cfg dir)
        self.cfg_ppo = hydra.compose(config_name=ppo_path)
        self.cfg_ppo = self.cfg_ppo['train']  # strip superfluous nesting

    def _acquire_task_tensors(self):
        """Acquire tensors."""

        target_heights = self.cfg_base.env.table_height + self.bolt_head_heights + self.nut_heights * 0.5
        self.target_pos = target_heights * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat((self.num_envs, 1))

    def _refresh_task_tensors(self):
        """Refresh tensors."""

        self.fingerpad_midpoint_pos = fc.translate_along_local_z(pos=self.finger_midpoint_pos,
                                                                 quat=self.hand_quat,
                                                                 offset=self.asset_info_franka_table.franka_finger_length - self.asset_info_franka_table.franka_fingerpad_length * 0.5,
                                                                 device=self.device)
        self.finger_nut_keypoint_dist = self._get_keypoint_dist(body='finger_nut')
        self.nut_keypoint_dist = self._get_keypoint_dist(body='nut')
        self.nut_dist_to_target = torch.norm(self.target_pos - self.nut_com_pos, p=2,
                                             dim=-1)  # distance between nut COM and target
        self.nut_dist_to_fingerpads = torch.norm(self.fingerpad_midpoint_pos - self.nut_com_pos, p=2,
                                                 dim=-1)  # distance between nut COM and midpoint between centers of fingerpads

    def pre_physics_step(self, actions):
        """Reset environments. Apply actions from policy. Simulation step called after this method."""

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.actions = actions.clone().to(self.device)  # shape = (num_envs, num_actions); values = [-1, 1]

        self._apply_actions_as_ctrl_targets(actions=self.actions,
                                            ctrl_target_gripper_dof_pos=0.0,
                                            do_scale=True)

    def post_physics_step(self):
        """Step buffers. Refresh tensors. Compute observations and reward. Reset environments."""

        self.progress_buf[:] += 1

        self.refresh_base_tensors()
        self.refresh_env_tensors()
        self._refresh_task_tensors()
        self.compute_observations()
        self.compute_reward()

    def compute_observations(self):
        """Compute observations."""

        # Shallow copies of tensors
        obs_tensors = [self.fingertip_midpoint_pos,
                       self.fingertip_midpoint_quat,
                       self.fingertip_midpoint_linvel,
                       self.fingertip_midpoint_angvel,
                       self.nut_com_pos,
                       self.nut_com_quat,
                       self.nut_com_linvel,
                       self.nut_com_angvel]

        if self.cfg_task.rl.add_obs_finger_force:
            obs_tensors += [self.left_finger_force, self.right_finger_force]

        obs_tensors = torch.cat(obs_tensors, dim=-1)
        self.obs_buf[:, :obs_tensors.shape[-1]] = obs_tensors  # shape = (num_envs, num_observations)

        return self.obs_buf

    def compute_reward(self):
        """Detect successes and failures. Update reward and reset buffers."""

        # Get successful and failed envs at current timestep
        curr_successes = self._get_curr_successes()
        curr_failures = self._get_curr_failures(curr_successes)

        self._update_reset_buf(curr_successes, curr_failures)
        self._update_rew_buf(curr_successes)

    def _update_reset_buf(self, curr_successes, curr_failures):
        """Assign environments for reset if successful or failed."""

        self.reset_buf[:] = torch.logical_or(curr_successes, curr_failures)

    def _update_rew_buf(self, curr_successes):
        """Compute reward at current timestep."""

        keypoint_reward = -(self.nut_keypoint_dist + self.finger_nut_keypoint_dist)
        action_penalty = torch.norm(self.actions, p=2, dim=-1)

        self.rew_buf[:] = keypoint_reward * self.cfg_task.rl.keypoint_reward_scale \
                          - action_penalty * self.cfg_task.rl.action_penalty_scale \
                          + curr_successes * self.cfg_task.rl.success_bonus

    def reset_idx(self, env_ids):
        """Reset specified environments. Zero buffers."""

        self._reset_franka(env_ids)
        self._reset_object(env_ids)

        self._reset_buffers(env_ids)

    def _reset_franka(self, env_ids):
        """Reset DOF states and DOF targets of Franka."""

        self.dof_pos[env_ids] = torch.cat((torch.tensor(self.cfg_task.randomize.franka_arm_initial_dof_pos,
                                                        device=self.device).repeat((len(env_ids), 1)),
                                           (self.nut_widths_max[env_ids] * 0.5) * 1.1,  # buffer on gripper DOF pos to prevent initial contact
                                           (self.nut_widths_max[env_ids] * 0.5) * 1.1),  # buffer on gripper DOF pos to prevent initial contact
                                          dim=-1)  # shape = (num_envs, num_dofs)
        self.dof_vel[env_ids] = 0.0  # shape = (num_envs, num_dofs)
        self.ctrl_target_dof_pos[env_ids] = self.dof_pos[env_ids]

        multi_env_ids_int32 = self.franka_actor_ids_sim[env_ids].flatten()
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

    def _reset_object(self, env_ids):
        """Reset root state of nut."""

        # shape of root_pos = (num_envs, num_actors, 3)
        # shape of root_quat = (num_envs, num_actors, 4)
        # shape of root_linvel = (num_envs, num_actors, 3)
        # shape of root_angvel = (num_envs, num_actors, 3)

        nut_pos = self.cfg_base.env.table_height + self.bolt_shank_lengths[env_ids]
        self.root_pos[env_ids, self.nut_actor_id_env] = \
            nut_pos * torch.tensor([0.0, 0.0, 1.0], device=self.device).repeat(len(env_ids), 1)
        
        nut_rot = self.cfg_task.randomize.nut_rot_initial * torch.ones((len(env_ids), 1), device=self.device) * math.pi / 180.0
        self.root_quat[env_ids, self.nut_actor_id_env] = torch.cat((torch.zeros((len(env_ids), 1), device=self.device),
                                                                    torch.zeros((len(env_ids), 1), device=self.device),
                                                                    torch.sin(nut_rot * 0.5),
                                                                    torch.cos(nut_rot * 0.5)),
                                                                   dim=-1)

        self.root_linvel[env_ids, self.nut_actor_id_env] = 0.0
        self.root_angvel[env_ids, self.nut_actor_id_env] = 0.0

        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state),
                                                     gymtorch.unwrap_tensor(self.nut_actor_ids_sim),
                                                     len(self.nut_actor_ids_sim))

    def _reset_buffers(self, env_ids):
        """Reset buffers."""

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0

    def _set_viewer_params(self):
        """Set viewer parameters."""

        cam_pos = gymapi.Vec3(-1.0, -1.0, 1.0)
        cam_target = gymapi.Vec3(0.0, 0.0, 0.5)
        self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

    def _apply_actions_as_ctrl_targets(self, actions, ctrl_target_gripper_dof_pos, do_scale):
        """Apply actions from policy as position/rotation targets or force/torque targets."""

        # Interpret actions as target pos displacements and set pos target
        pos_actions = actions[:, 0:3]
        if do_scale:
            pos_actions = pos_actions @ torch.diag(torch.tensor(self.cfg_task.rl.pos_action_scale, device=self.device))
        self.ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions

        # Interpret actions as target rot (axis-angle) displacements
        rot_actions = actions[:, 3:6]
        if self.cfg_task.rl.unidirectional_rot:
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
        if do_scale:
            rot_actions = rot_actions @ torch.diag(torch.tensor(self.cfg_task.rl.rot_action_scale, device=self.device))

        # Convert to quat and set rot target
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / angle.unsqueeze(-1)
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        if self.cfg_task.rl.clamp_rot:
            rot_actions_quat = torch.where(angle.unsqueeze(-1).repeat(1, 4) > self.cfg_task.rl.clamp_rot_thresh,
                                           rot_actions_quat,
                                           torch.tensor([0.0, 0.0, 0.0, 1.0], device=self.device).repeat(self.num_envs,
                                                                                                         1))
        self.ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)

        if self.cfg_ctrl['do_force_ctrl']:
            # Interpret actions as target forces and target torques
            force_actions = actions[:, 6:9]
            if self.cfg_task.rl.unidirectional_force:
                force_actions[:, 2] = -(force_actions[:, 2] + 1.0) * 0.5  # [-1, 0]
            if do_scale:
                force_actions = force_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.force_action_scale, device=self.device))

            torque_actions = actions[:, 9:12]
            if do_scale:
                torque_actions = torque_actions @ torch.diag(
                    torch.tensor(self.cfg_task.rl.torque_action_scale, device=self.device))

            self.ctrl_target_fingertip_contact_wrench = torch.cat((force_actions, torque_actions), dim=-1)

        self.ctrl_target_gripper_dof_pos = ctrl_target_gripper_dof_pos

        self.generate_ctrl_signals()

    def _get_keypoint_dist(self, body):
        """Get keypoint distances."""

        axis_length = self.asset_info_franka_table.franka_hand_length + self.asset_info_franka_table.franka_finger_length

        if body == 'finger' or body == 'nut':
            # Keypoint distance between finger/nut and target
            if body == 'finger':
                self.keypoint1 = self.fingertip_midpoint_pos
                self.keypoint2 = fc.translate_along_local_z(pos=self.keypoint1,
                                                            quat=self.fingertip_midpoint_quat,
                                                            offset=-axis_length,
                                                            device=self.device)

            elif body == 'nut':
                self.keypoint1 = self.nut_com_pos
                self.keypoint2 = fc.translate_along_local_z(pos=self.nut_com_pos,
                                                            quat=self.nut_com_quat,
                                                            offset=axis_length,
                                                            device=self.device)

            self.keypoint1_targ = self.target_pos
            self.keypoint2_targ = self.keypoint1_targ + torch.tensor([0.0, 0.0, axis_length], device=self.device)

        elif body == 'finger_nut':
            # Keypoint distance between finger and nut
            self.keypoint1 = self.fingerpad_midpoint_pos
            self.keypoint2 = fc.translate_along_local_z(pos=self.keypoint1,
                                                        quat=self.fingertip_midpoint_quat,
                                                        offset=-axis_length,
                                                        device=self.device)

            self.keypoint1_targ = self.nut_com_pos
            self.keypoint2_targ = fc.translate_along_local_z(pos=self.nut_com_pos,
                                                             quat=self.nut_com_quat,
                                                             offset=axis_length,
                                                             device=self.device)

        self.keypoint3 = self.keypoint1 + (self.keypoint2 - self.keypoint1) * 1.0 / 3.0
        self.keypoint4 = self.keypoint1 + (self.keypoint2 - self.keypoint1) * 2.0 / 3.0
        self.keypoint3_targ = self.keypoint1_targ + (self.keypoint2_targ - self.keypoint1_targ) * 1.0 / 3.0
        self.keypoint4_targ = self.keypoint1_targ + (self.keypoint2_targ - self.keypoint1_targ) * 2.0 / 3.0
        keypoint_dist = torch.norm(self.keypoint1_targ - self.keypoint1, p=2, dim=-1) \
                        + torch.norm(self.keypoint2_targ - self.keypoint2, p=2, dim=-1) \
                        + torch.norm(self.keypoint3_targ - self.keypoint3, p=2, dim=-1) \
                        + torch.norm(self.keypoint4_targ - self.keypoint4, p=2, dim=-1)

        return keypoint_dist

    def _get_curr_successes(self):
        """Get success mask at current timestep."""

        curr_successes = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # If nut is close enough to target pos
        is_close = torch.where(self.nut_dist_to_target < self.thread_pitches.squeeze(-1),
                               torch.ones_like(curr_successes),
                               torch.zeros_like(curr_successes))

        curr_successes = torch.logical_or(curr_successes, is_close)

        return curr_successes

    def _get_curr_failures(self, curr_successes):
        """Get failure mask at current timestep."""

        curr_failures = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

        # If max episode length has been reached
        self.is_expired = torch.where(self.progress_buf[:] >= self.cfg_task.rl.max_episode_length,
                                      torch.ones_like(curr_failures),
                                      curr_failures)

        # If nut is too far from target pos
        self.is_far = torch.where(self.nut_dist_to_target > self.cfg_task.rl.far_error_thresh,
                                  torch.ones_like(curr_failures),
                                  curr_failures)

        # If nut has slipped (distance-based definition)
        self.is_slipped = \
            torch.where(
                self.nut_dist_to_fingerpads > self.asset_info_franka_table.franka_fingerpad_length * 0.5 + self.nut_heights.squeeze(-1) * 0.5,
                torch.ones_like(curr_failures),
                curr_failures)
        self.is_slipped = torch.logical_and(self.is_slipped, torch.logical_not(curr_successes))  # ignore slip if successful

        # If nut has fallen (i.e., if nut XY pos has drifted from center of bolt and nut Z pos has drifted below top of bolt)
        self.is_fallen = torch.logical_and(
            torch.norm(self.nut_com_pos[:, 0:2], p=2, dim=-1) > self.bolt_widths.squeeze(-1) * 0.5,
            self.nut_com_pos[:, 2] < self.cfg_base.env.table_height + self.bolt_head_heights.squeeze(
                -1) + self.bolt_shank_lengths.squeeze(-1) + self.nut_heights.squeeze(-1) * 0.5)

        curr_failures = torch.logical_or(curr_failures, self.is_expired)
        curr_failures = torch.logical_or(curr_failures, self.is_far)
        curr_failures = torch.logical_or(curr_failures, self.is_slipped)
        curr_failures = torch.logical_or(curr_failures, self.is_fallen)

        return curr_failures
