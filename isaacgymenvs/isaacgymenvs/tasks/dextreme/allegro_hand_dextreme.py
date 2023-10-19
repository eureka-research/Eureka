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
import math 
import os
from typing import Tuple, List 

import itertools
from itertools import permutations
from tkinter import W
from typing import Tuple, Dict, List, Set


import numpy as np

import torch

from isaacgym import gymapi
from isaacgym import gymtorch

from isaacgym.torch_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp  

from torch import Tensor

from isaacgymenvs.tasks.dextreme.adr_vec_task import ADRVecTask
from isaacgymenvs.utils.torch_jit_utils import quaternion_to_matrix, matrix_to_quaternion

from isaacgymenvs.utils.rna_util import RandomNetworkAdversary


class AllegroHandDextreme(ADRVecTask):

    dict_obs_cls = True

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        '''
        obligatory constructor to fill-in class variables and setting
        up the simulation.

        self._read_cfg() is about initialising class variables from a 
                         config file.
        
        self._init_pre_sim_buffers() initialises particular tensors 
                         that are useful in storing various states 
                         randomised or otherwise 

        self._init_post_sim_buffers() initialises the root tensors and
                         other auxiliary variables that can be provided
                         as input to the controller or the value function     

        '''

        self.cfg = cfg

        # Read the task config file and store all the relevant variables in the class
        self._read_cfg()

        self.fingertips = [s+"_link_3" for s in ["index", "middle", "ring", "thumb"]]
        self.num_fingertips = len(self.fingertips)
        num_dofs = 16
        
        self.num_obs_dict = self.get_num_obs_dict(num_dofs)

        self.cfg["env"]["obsDims"] = {} 

        for o in self.num_obs_dict.keys():
            if o not in self.num_obs_dict:
                raise Exception(f"Unknown type of observation {o}!")
            self.cfg["env"]["obsDims"][o] = (self.num_obs_dict[o],)

        self.up_axis = 'z'

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        self.cfg["env"]["numActions"] = 16

        self.sim_device = sim_device

        rl_device = self.cfg.get("rl_device", "cuda:0")

        self._init_pre_sim_buffers()
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, use_dict_obs=True)
        self._init_post_sim_buffers()

        reward_keys = ['dist_rew', 'rot_rew', 'action_penalty', 'action_delta_penalty',
                       'velocity_penalty', 'reach_goal_rew', 'fall_rew', 'timeout_rew']
        self.rewards_episode = {key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device) for key in reward_keys}

        if self.use_adr:
            self.apply_reset_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device) 


        if self.print_success_stat:                        
            self.last_success_step = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.success_time = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.last_ep_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.total_num_resets = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            self.successes_count = torch.zeros(self.max_consecutive_successes + 1, dtype=torch.float, device=self.device)
            from tensorboardX import SummaryWriter
            self.eval_summary_dir = './eval_summaries'
            # remove the old directory if it exists
            if os.path.exists(self.eval_summary_dir):
                import shutil
                shutil.rmtree(self.eval_summary_dir)
            self.eval_summaries = SummaryWriter(self.eval_summary_dir, flush_secs=3)

    def get_env_state(self):

        env_dict=dict(act_moving_average=self.act_moving_average)

        if self.use_adr:
            env_dict = dict(**env_dict, **super().get_env_state())

        return env_dict

    def get_save_tensors(self):
        
        if hasattr(self, 'actions'):
            actions = self.actions
        else:
            actions = torch.zeros((self.num_envs, self.cfg["env"]["numActions"])).to(self.device)

        # scale is [-1, 1] -> [low, upper]
        # unscale is [low, upper] -> [-1, 1]
        # self.actions are in [-1, 1] as they are raw 
        # actions returned by the policy 

        return {
            # 'observations': self.obs_buf,
            'actions': actions,
            'cube_state': self.root_state_tensor[self.object_indices],
            'goal_state': self.goal_states,
            'joint_positions': self.dof_pos,
            'joint_velocities': self.dof_vel,
            'root_state': self.root_state_tensor[self.hand_indices],
        }

    def save_step(self):
        self.capture.append_experience(self.get_save_tensors())

    def get_num_obs_dict(self, num_dofs):
    
        # This is what we use for ADR 
        num_obs = {
            "dof_pos": num_dofs,
            "dof_pos_randomized": num_dofs,
            "dof_vel": num_dofs,
            "dof_force": num_dofs, # generalised forces

            "object_vels": 6,

            "last_actions": num_dofs,
            "cube_random_params": 3,
            "hand_random_params": 1,
            "gravity_vec": 3,
            "ft_states": 13 * self.num_fingertips, # (pos, quat, linvel, angvel) per fingertip
            "ft_force_torques": 6 * self.num_fingertips, # wrenches
            "rb_forces": 3, # random forces being applied to the cube

            "rot_dist": 2,

            "stochastic_delay_params": 4, # cube obs + action delay prob, action fixed latency, pose refresh rate
            "affine_params": 16*2 + 7*2 + 16*2,

            "object_pose": 7,
            "goal_pose": 7,
            "goal_relative_rot": 4,
            "object_pose_cam_randomized": 7,
            "goal_relative_rot_cam_randomized": 4,
        }
        
        return num_obs

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)


    def _create_envs(self, num_envs, spacing, num_per_row):

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../assets')
        hand_asset_file = "urdf/kuka_allegro_description/allegro.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            asset_file = self.cfg["env"]["asset"].get("assetFileName", hand_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]

        # load allegro hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = False
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True

        # The control interface i.e. we will be sending target positions to the robot
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        hand_asset = self.gym.load_asset(self.sim, asset_root, hand_asset_file, asset_options)

        self.num_hand_bodies = self.gym.get_asset_rigid_body_count(hand_asset)
        self.num_hand_shapes = self.gym.get_asset_rigid_shape_count(hand_asset)
        self.num_hand_dofs = self.gym.get_asset_dof_count(hand_asset)
        print("Num dofs: ", self.num_hand_dofs)
        self.num_hand_actuators = self.num_hand_dofs 

        self.actuated_dof_indices = [i for i in range(self.num_hand_dofs)]

        # set allegro_hand dof properties
        hand_dof_props = self.gym.get_asset_dof_properties(hand_asset)

        self.hand_dof_lower_limits = []
        self.hand_dof_upper_limits = []
        self.hand_dof_default_pos = []
        self.hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(hand_asset, name) for name in self.fingertips]

        # create fingertip force sensors
        sensor_pose = gymapi.Transform()
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(hand_asset, ft_handle, sensor_pose)

        for i in range(self.num_hand_dofs):
            self.hand_dof_lower_limits.append(hand_dof_props['lower'][i])
            self.hand_dof_upper_limits.append(hand_dof_props['upper'][i])
            self.hand_dof_default_pos.append(0.0)
            self.hand_dof_default_vel.append(0.0)

            hand_dof_props['effort'][i] = self.max_effort
            hand_dof_props['stiffness'][i] = 2
            hand_dof_props['damping'][i] = 0.1
            hand_dof_props['friction'][i] = 0.01
            hand_dof_props['armature'][i] = 0.002

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.hand_dof_lower_limits = to_torch(self.hand_dof_lower_limits, device=self.device)
        self.hand_dof_upper_limits = to_torch(self.hand_dof_upper_limits, device=self.device)
        self.hand_dof_default_pos = to_torch(self.hand_dof_default_pos, device=self.device)
        self.hand_dof_default_vel = to_torch(self.hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)

        hand_start_pose = gymapi.Transform()
        hand_start_pose.p = gymapi.Vec3(*get_axis_params(0.5, self.up_axis_idx))
        hand_start_pose.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.pi) * \
                            gymapi.Quat.from_axis_angle(gymapi.Vec3(1, 0, 0), 0.47 * np.pi) * \
                            gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), 0.25 * np.pi)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3()
        object_start_pose.p.x = hand_start_pose.p.x
        pose_dy, pose_dz = self.start_object_pose_dy, self.start_object_pose_dz

        object_start_pose.p.y = hand_start_pose.p.y + pose_dy
        object_start_pose.p.z = hand_start_pose.p.z + pose_dz

        self.goal_displacement = gymapi.Vec3(-0.2, -0.06, 0.12)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.y -= 0.02
        goal_start_pose.p.z -= 0.04

        # compute aggregate size
        max_agg_bodies = self.num_hand_bodies + 2
        max_agg_shapes = self.num_hand_shapes + 2

        self.allegro_hands = []
        self.object_handles = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(hand_asset, name) for name in self.fingertips]

        hand_rb_count = self.gym.get_asset_rigid_body_count(hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(hand_rb_count, hand_rb_count + object_rb_count))

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            hand_actor = self.gym.create_actor(env_ptr, hand_asset, hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append([hand_start_pose.p.x, hand_start_pose.p.y, hand_start_pose.p.z,
                                           hand_start_pose.r.x, hand_start_pose.r.y, hand_start_pose.r.z, hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, hand_actor, hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            self.gym.enable_actor_dof_force_sensors(env_ptr, hand_actor)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)


            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.allegro_hands.append(hand_actor)
            self.object_handles.append(object_handle)



        self.palm_link_handle =  self.gym.find_actor_rigid_body_handle(env_ptr, hand_actor, "palm_link"),


        object_rb_props = self.gym.get_actor_rigid_body_properties(env_ptr, object_handle)
        self.object_rb_masses = [prop.mass for prop in object_rb_props]

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.object_rb_handles = to_torch(self.object_rb_handles, dtype=torch.long, device=self.device)
        self.object_rb_masses = to_torch(self.object_rb_masses, dtype=torch.float, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)


        # Random Network Adversary 
        # As mentioned in OpenAI et al. 2019 (Appendix B.3) https://arxiv.org/abs/1910.07113
        # and DeXtreme, 2022 (Section 2.6.2) https://arxiv.org/abs/2210.13702
        if self.enable_rna:

            softmax_bins = 32 
            num_dofs = len(self.hand_dof_lower_limits)
            self.discretised_dofs = torch.zeros((num_dofs, softmax_bins)).to(self.device)

            # Discretising the joing angles into 32 bins
            for i in range(0, len(self.hand_dof_lower_limits)):
                self.discretised_dofs[i] = torch.linspace(self.hand_dof_lower_limits[i], 
                                                          self.hand_dof_upper_limits[i], steps=softmax_bins).to(self.device)

            # input is the joint angles and cube pose (pos: 3 + quat: 4), therefore a total of 16+7 dimensions
            self.rna_network = RandomNetworkAdversary(num_envs=self.num_envs, in_dims=num_dofs+7, \
                out_dims=num_dofs, softmax_bins=softmax_bins, device=self.device)




        # Random cube observations. Need this tensor for Random Cube Pose Injection 
        self.random_cube_poses = torch.zeros(self.num_envs, 7, device=self.device)

    def compute_reward(self, actions):

        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], \
        self.hold_count_buf[:], self.successes[:], self.consecutive_successes[:], \
        dist_rew, rot_rew, action_penalty, action_delta_penalty, velocity_penalty, reach_goal_rew, fall_rew, timeout_rew = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.hold_count_buf, self.cur_targets, self.prev_targets,
            self.dof_vel, self.successes, self.consecutive_successes, self.max_episode_length,
            self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.dist_reward_scale, self.rot_reward_scale, self.rot_eps,
            self.actions, self.action_penalty_scale, self.action_delta_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, self.num_success_hold_steps
        )

        # update best rotation distance in the current episode
        self.best_rotation_dist = torch.minimum(self.best_rotation_dist, self.curr_rotation_dist)
        self.extras['consecutive_successes'] = self.consecutive_successes.mean()
        self.extras['true_objective'] = self.successes

        episode_cumulative = dict()
        episode_cumulative['dist_rew'] = dist_rew
        episode_cumulative['rot_rew'] = rot_rew
        episode_cumulative['action_penalty'] = action_penalty
        episode_cumulative['action_delta_penalty'] = action_delta_penalty
        episode_cumulative['velocity_penalty'] = velocity_penalty
        episode_cumulative['reach_goal_rew'] = reach_goal_rew
        episode_cumulative['fall_rew'] = fall_rew
        episode_cumulative['timeout_rew'] = timeout_rew
        self.extras['episode_cumulative'] = episode_cumulative

        if self.print_success_stat:
            is_success = self.reset_goal_buf.to(torch.bool)

            frame_ = torch.empty_like(self.last_success_step).fill_(self.frame)
            self.success_time = torch.where(is_success, frame_ - self.last_success_step, self.success_time)
            self.last_success_step = torch.where(is_success, frame_, self.last_success_step)
            mask_ = self.success_time > 0
            if any(mask_):
                avg_time_mean = ((self.success_time * mask_).sum(dim=0) / mask_.sum(dim=0)).item()
            else:
                avg_time_mean = math.nan
            
            envs_reset = self.reset_buf 
            if self.use_adr:
                envs_reset = self.reset_buf & ~self.apply_reset_buf
            
            self.total_resets = self.total_resets + envs_reset.sum() 
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * envs_reset).sum() 

            self.total_num_resets += envs_reset

            self.last_ep_successes = torch.where(envs_reset > 0, self.successes, self.last_ep_successes)
            reset_ids = envs_reset.nonzero().squeeze()
            last_successes = self.successes[reset_ids].long()
            self.successes_count[last_successes] += 1

            if self.frame % 100 == 0:
                # The direct average shows the overall result more quickly, but slightly undershoots long term
                # policy performance.
                print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
                if self.total_resets > 0:
                    print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))
                print(f"Max num successes: {self.successes.max().item()}")
                print(f"Average consecutive successes: {self.consecutive_successes.mean().item():.2f}")
                print(f"Total num resets: {self.total_num_resets.sum().item()} --> {self.total_num_resets}")
                print(f"Reset percentage: {(self.total_num_resets > 0).sum() / self.num_envs:.2%}")

                print(f"Last ep successes: {self.last_ep_successes.mean().item():.2f} {self.last_ep_successes}")

                self.eval_summaries.add_scalar("consecutive_successes", self.consecutive_successes.mean().item(), self.frame)
                self.eval_summaries.add_scalar("last_ep_successes", self.last_ep_successes.mean().item(), self.frame)
                self.eval_summaries.add_scalar("reset_stats/reset_percentage", (self.total_num_resets > 0).sum() / self.num_envs, self.frame)
                self.eval_summaries.add_scalar("reset_stats/min_num_resets", self.total_num_resets.min().item(), self.frame)

                self.eval_summaries.add_scalar("policy_speed/avg_success_time_frames", avg_time_mean, self.frame)
                frame_time = self.control_freq_inv * self.dt
                self.eval_summaries.add_scalar("policy_speed/avg_success_time_seconds", avg_time_mean * frame_time, self.frame)
                self.eval_summaries.add_scalar("policy_speed/avg_success_per_minute", 60.0 / (avg_time_mean * frame_time), self.frame)
                print(f"Policy speed (successes per minute): {60.0 / (avg_time_mean * frame_time):.2f}")

                dof_delta = self.dof_delta.abs()
                print(f"Max dof deltas: {dof_delta.max(dim=0).values}, max across dofs: {self.dof_delta.abs().max().item():.2f}, mean: {self.dof_delta.abs().mean().item():.2f}")
                print(f"Max dof delta radians per sec: {dof_delta.max().item() / frame_time:.2f}, mean: {dof_delta.mean().item() / frame_time:.2f}")

                # create a matplotlib bar chart of the self.successes_count
                import matplotlib.pyplot as plt
                plt.bar(list(range(self.max_consecutive_successes + 1)), self.successes_count.cpu().numpy())
                plt.title("Successes histogram")
                plt.xlabel("Successes")
                plt.ylabel("Frequency")
                plt.savefig(f"{self.eval_summary_dir}/successes_histogram.png")
                plt.clf()



    def compute_poses_wrt_wrist(self, object_pose, palm_link_pose, goal_pose=None):

        object_pos = object_pose[:, 0:3]
        object_rot = object_pose[:, 3:7]

        palm_link_pos = palm_link_pose[:, 0:3]
        palm_link_quat_xyzw = palm_link_pose[:, 3:7]

        palm_link_quat_wxyz = palm_link_quat_xyzw[:, [3, 0, 1, 2]]   
        R_W_P = quaternion_to_matrix(palm_link_quat_wxyz)

        T_W_P = torch.eye(4).repeat(R_W_P.shape[0], 1, 1).to(R_W_P.device)
        T_W_P[:, 0:3, 0:3] = R_W_P
        T_W_P[:, 0:3, 3] = palm_link_pos

        object_quat_xyzw = object_rot
        object_quat_wxyz = object_quat_xyzw[:, [3, 0, 1, 2]]
        R_W_O = quaternion_to_matrix(object_quat_wxyz)

        T_W_O = torch.eye(4).repeat(R_W_O.shape[0], 1, 1).to(R_W_O.device)
        T_W_O[:, 0:3, 0:3] = R_W_O
        T_W_O[:, 0:3, 3] = object_pos

        relative_pose = torch.matmul(torch.inverse(T_W_P), T_W_O)

        relative_translation = relative_pose[:, 0:3, 3]
        relative_quat_wxyz = matrix_to_quaternion(relative_pose[:, 0:3, 0:3])

        relative_quat_xyzw = relative_quat_wxyz[:, [1, 2, 3, 0]]

        object_pos_wrt_wrist = relative_translation
        object_quat_wrt_wrist = relative_quat_xyzw

        object_pose_wrt_wrist = torch.cat((object_pos_wrt_wrist, object_quat_wrt_wrist), axis=-1)

        if goal_pose == None:
            return object_pose_wrt_wrist

        goal_pos = goal_pose[:, 0:3]
        goal_quat_xyzw = goal_pose[:, 3:7]
        goal_quat_wxyz = goal_quat_xyzw[:, [3, 0, 1, 2]]

        R_W_G = quaternion_to_matrix(goal_quat_wxyz)
        T_W_G = torch.eye(4).repeat(R_W_G.shape[0], 1, 1).to(R_W_G.device)

        T_W_G[:, 0:3, 0:3] = R_W_G
        T_W_G[:, 0:3, 3] = goal_pos

        relative_goal_pose = torch.matmul(torch.inverse(T_W_P), T_W_G)
        relative_goal_translation = relative_goal_pose[:, 0:3, 3]
        relative_goal_quat_wxyz = matrix_to_quaternion(relative_goal_pose[:, 0:3, 0:3])
        relative_goal_quat_xyzw = relative_goal_quat_wxyz[:, [1, 2, 3, 0]]

        goal_pose_wrt_wrist = torch.cat((relative_goal_translation, relative_goal_quat_xyzw), axis=-1)

        return object_pose_wrt_wrist, goal_pose_wrt_wrist


    def convert_pos_quat_to_mat(self, obj_pose_pos_quat):

        pos = obj_pose_pos_quat[:, 0:3]
        quat_xyzw = obj_pose_pos_quat[:, 3:7]

        quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]   
        R = quaternion_to_matrix(quat_wxyz)

        T = torch.eye(4).repeat(R.shape[0], 1, 1).to(R.device)
        T[:, 0:3, 0:3] = R
        T[:, 0:3, 3] = pos

        return T    

      
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]

        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        # Need to update the pose of the cube so that it is represented wrt wrist 
        self.palm_link_pose = self.rigid_body_states[:, self.palm_link_handle, 0:7].view(-1, 7)

        self.object_pose_wrt_wrist, self.goal_pose_wrt_wrist = self.compute_poses_wrt_wrist(self.object_pose,
                                                                                            self.palm_link_pose,
                                                                                            self.goal_pose)

        self.goal_wrt_wrist_rot = self.goal_pose_wrt_wrist[:, 3:7]
        
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        if not self.use_adr and self.randomize:
            update_freq = torch.remainder(self.frame + self.cube_pose_refresh_offset, self.cube_pose_refresh_rates) == 0

            self.obs_object_pose_freq[update_freq] = self.object_pose_wrt_wrist[update_freq]

            # simulate adding delay
            update_delay = torch.randn(self.num_envs, device=self.device) > self.cube_obs_delay_prob
            self.obs_object_pose[update_delay] = self.obs_object_pose_freq[update_delay]
            
        
        # increment the frame counter both for manual DR and ADR
        self.frame += 1
            
        cube_scale = self.cube_random_params[:, 0]
        cube_scale = cube_scale.reshape(-1, 1)


        # unscale is [low, upper] -> [-1, 1]
 
        self.obs_dict["dof_pos"][:] = unscale(self.dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits)
        self.obs_dict["dof_vel"][:] = self.dof_vel
        self.obs_dict["dof_force"][:] = self.force_torque_obs_scale * self.dof_force_tensor

        self.obs_dict["object_pose"][:] = self.object_pose_wrt_wrist

        self.obs_dict["object_vels"][:, 0:3] = self.object_linvel
        self.obs_dict["object_vels"][:, 3:6] = self.vel_obs_scale * self.object_angvel

        self.obs_dict["goal_pose"][:] = self.goal_pose_wrt_wrist
        self.obs_dict["goal_relative_rot"][:] = quat_mul(self.object_pose_wrt_wrist[:, 3:7], quat_conjugate(self.goal_wrt_wrist_rot))

        # This is only needed for manul DR experiments
        if not self.use_adr:

            self.obs_dict["object_pose_cam"][:] = self.obs_object_pose
            self.obs_dict["goal_relative_rot_cam"][:] = quat_mul(self.obs_object_pose[:, 3:7], quat_conjugate(self.goal_wrt_wrist_rot))
  
        self.obs_dict["ft_states"][:] = self.fingertip_state.reshape(self.num_envs, 13 * self.num_fingertips)
        self.obs_dict["ft_force_torques"][:] = self.force_torque_obs_scale * self.vec_sensor_tensor # wrenches
        self.obs_dict["rb_forces"] = self.rb_forces[:, self.object_rb_handles, :].view(-1, 3)

        self.obs_dict["last_actions"][:] = self.actions

        if self.randomize:

            self.obs_dict["cube_random_params"][:] = self.cube_random_params
            self.obs_dict["hand_random_params"][:] = self.hand_random_params
            self.obs_dict["gravity_vec"][:] = self.gravity_vec
        
        quat_diff = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))
        self.curr_rotation_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
        self.best_rotation_dist = torch.where(self.best_rotation_dist < 0.0, self.curr_rotation_dist, self.best_rotation_dist)

        # add rotation distances to the observations so that critic could predict the rewards better
        self.obs_dict["rot_dist"][:, 0] = self.curr_rotation_dist
        self.obs_dict["rot_dist"][:, 1] = self.best_rotation_dist


    def get_random_quat(self, env_ids):

        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
        # https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py#L261

        uvw = torch_rand_float(0, 1.0, (len(env_ids), 3), device=self.device)
        q_w = torch.sqrt(1.0 - uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 1]))
        q_x = torch.sqrt(1.0 - uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 1]))
        q_y = torch.sqrt(uvw[:, 0]) * (torch.sin(2 * np.pi * uvw[:, 2]))
        q_z = torch.sqrt(uvw[:, 0]) * (torch.cos(2 * np.pi * uvw[:, 2]))
        new_rot = torch.cat((q_x.unsqueeze(-1), q_y.unsqueeze(-1), q_z.unsqueeze(-1), q_w.unsqueeze(-1)), dim=-1)
        
        return new_rot

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        if self.apply_random_quat:

            new_rot = self.get_random_quat(env_ids)

        else:
            new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

        # change back to non-initialized state
        self.best_rotation_dist[env_ids] = -1


    def get_relative_rot(self, obj_rot, goal_rot):
        return quat_mul(obj_rot, quat_conjugate(goal_rot))

    def get_random_cube_observation(self, current_cube_pose):
        '''
        This function replaces cube pose in some environments 
        with a random cube pose to simulate noisy perception 
        estimates in the real world.

        It is also called random cube pose injection.
        '''

        env_ids = np.arange(0, self.num_envs)

        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 5), device=self.device)

        if self.apply_random_quat:
            new_object_rot = self.get_random_quat(env_ids)
        else:
            new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], 
                                                self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])


        self.random_cube_poses[:, 0:2] = self.object_init_state[env_ids, 0:2] +\
            0.5 * rand_floats[:, 0:2]
        
        self.random_cube_poses[:, 2] = self.object_init_state[env_ids, 2] + \
            0.5 * rand_floats[:, 2]

        self.random_cube_poses[:, 3:7] = new_object_rot

        random_cube_pose_mask = torch.rand(len(env_ids), 1, device=self.device) < self.random_cube_pose_prob

        current_cube_pose = current_cube_pose * ~random_cube_pose_mask + self.random_cube_poses * random_cube_pose_mask

        return current_cube_pose



    def reset_idx(self, env_ids, goal_env_ids):

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset rigid body forces
        self.rb_forces[env_ids, :, :] = 0.0

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise_z * rand_floats[:, self.up_axis_idx]

        if self.apply_random_quat:
            new_object_rot = self.get_random_quat(env_ids)
        else:
            new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        
        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset random force probabilities
        self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                                    * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset allegro hand
        delta_max = self.hand_dof_upper_limits - self.hand_dof_default_pos
        delta_min = self.hand_dof_lower_limits - self.hand_dof_default_pos
        rand_floats_dof_pos = (rand_floats[:, 5:5+self.num_hand_dofs] + 1) / 2
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats_dof_pos

        pos = self.hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.dof_pos[env_ids, :] = pos
        self.dof_vel[env_ids, :] = self.hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_hand_dofs:5+self.num_hand_dofs*2]
        
        self.prev_targets[env_ids, :self.num_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_hand_dofs] = pos
        self.prev_prev_targets[env_ids, :self.num_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))



        # Need to update the pose of the cube so that it is represented wrt wrist 
        self.palm_link_pose = self.rigid_body_states[:, self.palm_link_handle, 0:7].view(-1, 7)

        self.object_pose_wrt_wrist  = self.compute_poses_wrt_wrist(self.object_pose,
                                                                    self.palm_link_pose)

        # object pose is represented with respect to the wrist 
        self.obs_object_pose[env_ids] = self.object_pose_wrt_wrist[env_ids].clone()
        self.obs_object_pose_freq[env_ids] = self.object_pose_wrt_wrist[env_ids].clone()

        if self.use_adr and len(env_ids) == self.num_envs:
            self.progress_buf = torch.randint(0, self.max_episode_length, size=(self.num_envs,), dtype=torch.long, device=self.device)
        else:
            self.progress_buf[env_ids] = 0

        self.reset_buf[env_ids] = 0
        if self.use_adr:
            self.apply_reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        self.best_rotation_dist[env_ids] = -1
        self.hold_count_buf[env_ids] = 0

    def get_rna_alpha(self):
        """Function to get RNA alpha value."""
        raise NotImplementedError

    def get_random_network_adversary_action(self, canonical_action):

        if self.enable_rna:

            if self.last_step > 0 and self.last_step % self.random_adversary_weight_sample_freq == 0:
                self.rna_network._refresh()

            rand_action_softmax = self.rna_network(torch.cat([self.dof_pos, self.object_pose_wrt_wrist], axis=-1))
            rand_action_inds    = torch.argmax(rand_action_softmax, axis=-1)

            rand_action_inds  = torch.permute(rand_action_inds, (1, 0))
            rand_perturbation = torch.gather(self.discretised_dofs, 1, rand_action_inds)
            rand_perturbation = torch.permute(rand_perturbation, (1, 0))

            # unscale it first (normalise it to [-1, 1])
            rand_perturbation = unscale(rand_perturbation, 
                                        self.hand_dof_lower_limits[self.actuated_dof_indices],
                                        self.hand_dof_upper_limits[self.actuated_dof_indices])

            if not self.use_adr:
                action_perturb_mask = torch.rand(self.num_envs, 1, device=self.device) < self.action_perturb_prob                                        
                rand_perturbation = ~action_perturb_mask * canonical_action + action_perturb_mask * rand_perturbation

            rna_alpha = self.get_rna_alpha()

            rand_perturbation = rna_alpha * rand_perturbation + (1 - rna_alpha) * canonical_action

            return rand_perturbation

        else:
            return canonical_action

    def update_action_moving_average(self):

        # scheduling action moving average 

        if self.last_step > 0 and self.last_step % self.act_moving_average_scheduled_freq == 0:

            sched_scaling = 1.0 / self.act_moving_average_scheduled_steps * min(self.last_step, self.act_moving_average_scheduled_steps)
            self.act_moving_average = self.act_moving_average_upper + (self.act_moving_average_lower - self.act_moving_average_upper) * \
                                        sched_scaling
            
            print('action moving average: {}'.format(self.act_moving_average))
            print('last_step: {}'.format(self.last_step), ' scheduled steps: {}'.format(self.act_moving_average_scheduled_steps))

            self.extras['annealing/action_moving_average_scalar'] = self.act_moving_average


    def pre_physics_step(self, actions):

        # Anneal action moving average 
        self.update_action_moving_average()
       
        env_ids_reset = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        if self.randomize and not self.use_adr:
            self.apply_randomizations(dr_params=self.randomization_params, randomisation_callback=self.randomisation_callback)

        elif self.randomize and self.use_adr:
                       
            # NB - when we are daing ADR, we must calculate the ADR or new DR vals one step BEFORE applying randomisations
            # this is because reset needs to be applied on the next step for it to take effect
            env_mask_randomize = (self.reset_buf & ~self.apply_reset_buf).bool()
            env_ids_reset = self.apply_reset_buf.nonzero(as_tuple=False).flatten()            
            if len(env_mask_randomize.nonzero(as_tuple=False).flatten()) > 0:
                self.apply_randomizations(dr_params=self.randomization_params,
                                         randomize_buf=env_mask_randomize,
                                         adr_objective=self.successes,
                                         randomisation_callback=self.randomisation_callback)

                self.apply_reset_buf[env_mask_randomize] = 1

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids_reset) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)

        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids_reset) > 0:
            self.reset_idx(env_ids_reset, goal_env_ids)

        self.apply_actions(actions)
        self.apply_random_forces()

    def apply_action_noise_latency(self):
        return self.actions 


    def apply_actions(self, actions):

        self.actions = actions.clone().to(self.device)

        refreshed = self.progress_buf == 0
        self.prev_actions_queue[refreshed] = unscale(self.dof_pos[refreshed], self.hand_dof_lower_limits, 
                                                     self.hand_dof_upper_limits).view(-1, 1, self.num_actions)
        
        # Needed for the first step and every refresh 
        # you don't want to mix with zeros
        self.prev_actions[refreshed] = unscale(self.dof_pos[refreshed], self.hand_dof_lower_limits, 
                                               self.hand_dof_upper_limits).view(-1, self.num_actions)
        
        # update the actions queue
        self.prev_actions_queue[:, 1:] = self.prev_actions_queue[:, :-1].detach()
        self.prev_actions_queue[:, 0, :] = self.actions

        # apply action delay         
        actions_delayed = self.apply_action_noise_latency()

        # apply random network adversary 
        actions_delayed = self.get_random_network_adversary_action(actions_delayed)

        if self.use_relative_control:

            targets = self.prev_targets[:, self.actuated_dof_indices] + self.hand_dof_speed_scale * self.dt * actions_delayed
            self.cur_targets[:, self.actuated_dof_indices]  = targets 

        elif self.use_capped_dof_control:

            # This is capping the maximum dof velocity
            targets = scale(actions_delayed, self.hand_dof_lower_limits[self.actuated_dof_indices], 
                            self.hand_dof_upper_limits[self.actuated_dof_indices])

            delta = targets[:, self.actuated_dof_indices] - self.prev_targets[:, self.actuated_dof_indices]
            
            max_dof_delta = self.max_dof_radians_per_second * self.dt * self.control_freq_inv
            
            delta = torch.clamp_(delta, -max_dof_delta, max_dof_delta)

            self.cur_targets[:, self.actuated_dof_indices] = self.prev_targets[:, self.actuated_dof_indices] + delta


        else:

            self.cur_targets[:, self.actuated_dof_indices] = scale(actions_delayed,
                                                                   self.hand_dof_lower_limits[self.actuated_dof_indices],
                                                                   self.hand_dof_upper_limits[self.actuated_dof_indices])

        self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + \
                                                            (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]


        self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.hand_dof_lower_limits[self.actuated_dof_indices], self.hand_dof_upper_limits[self.actuated_dof_indices])

        self.dof_delta = self.cur_targets[:, self.actuated_dof_indices] - self.prev_targets[:, self.actuated_dof_indices]

        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))

        self.prev_actions[:] = self.actions.clone()

    def apply_random_forces(self):
        """Applies random forces to the object.
        Forces are applied as in https://arxiv.org/abs/1808.00177
        """

        if self.force_scale > 0.0:
            self.rb_forces *= torch.pow(self.force_decay, self.dt / self.force_decay_interval)

            # apply new forces
            force_indices = (torch.rand(self.num_envs, device=self.device) < self.random_force_prob).nonzero()
            self.rb_forces[force_indices, self.object_rb_handles, :] = torch.randn(
                self.rb_forces[force_indices, self.object_rb_handles, :].shape, device=self.device) * self.object_rb_masses * self.force_scale

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.rb_forces), None, gymapi.LOCAL_SPACE)
        

    def post_physics_step(self):

        self.progress_buf += 1

        # This is for manual DR so ADR has to be OFF
        if self.randomize and not self.use_adr:
            # This buffer is needed for manual DR randomisation
            self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        # update the previous targets 
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # save and viz dr params changing on the fly 
        self.track_dr_params()
           
        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])
        

    def track_dr_params(self):
        '''
        Track the parameters you wish to here
        '''
        pass 


    def _read_cfg(self):

        '''
        reads various variables from the config file
        '''

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.action_delta_penalty_scale = self.cfg["env"]["actionDeltaPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations

        if "max_effort" in self.cfg["env"]:
             self.max_effort = self.cfg["env"]["max_effort"]
        else:
            self.max_effort = 0.35

        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_position_noise_z = self.cfg["env"]["resetPositionNoiseZ"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        self.start_object_pose_dy = self.cfg["env"]["startObjectPoseDY"]
        self.start_object_pose_dz = self.cfg["env"]["startObjectPoseDZ"]

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]

        self.use_capped_dof_control = self.cfg["env"]["use_capped_dof_control"]
        self.max_dof_radians_per_second = self.cfg["env"]["max_dof_radians_per_second"]

        self.num_success_hold_steps = self.cfg["env"].get("num_success_hold_steps", 1)
        
        # Moving average related 

        self.act_moving_average_range = self.cfg["env"]["actionsMovingAverage"]["range"]
        self.act_moving_average_scheduled_steps = self.cfg["env"]["actionsMovingAverage"]["schedule_steps"]
        self.act_moving_average_scheduled_freq = self.cfg["env"]["actionsMovingAverage"]["schedule_freq"]
        
        self.act_moving_average_lower = self.act_moving_average_range[0]
        self.act_moving_average_upper = self.act_moving_average_range[1]

        self.act_moving_average = self.act_moving_average_upper

        # Random cube observation 

        has_random_cube_obs = 'random_cube_observation' in self.cfg["env"]
        if has_random_cube_obs:
            self.enable_random_obs = self.cfg["env"]["random_cube_observation"]["enable"]
            self.random_cube_pose_prob = self.cfg["env"]["random_cube_observation"]["prob"]
        else:
            self.enable_random_obs = False

        # We have two ways to sample quaternions where one of the samplings is biased
        # If this flag is enabled, the sampling will be UNBIASED
        self.apply_random_quat = self.cfg['env'].get("apply_random_quat", True)

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.eval_stats_name = self.cfg["env"].get("evalStatsName", '')
        self.num_eval_frames = self.cfg["env"].get("numEvalFrames", None)
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.cube_obs_delay_prob = self.cfg["env"].get("cubeObsDelayProb", 0.0)

        # Action delay 

        self.action_delay_prob_max = self.cfg["env"]["actionDelayProbMax"]
        self.action_latency_max = self.cfg["env"]["actionLatencyMax"]
        self.action_latency_scheduled_steps = self.cfg["env"]["actionLatencyScheduledSteps"]

        self.frame = 0
        self.max_skip_obs = self.cfg["env"].get("maxObjectSkipObs", 1)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg"]

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",

            # "block": "urdf/objects/cube_multicolor_sdf.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])

        # Random Network Adversary 
        self.enable_rna = "random_network_adversary" in self.cfg["env"] and self.cfg["env"]["random_network_adversary"]["enable"]

        if self.enable_rna:
            if "prob" in self.cfg["env"]["random_network_adversary"]:
                self.action_perturb_prob = self.cfg["env"]["random_network_adversary"]["prob"]
            
            # how often we want to resample the weights of the random neural network
            self.random_adversary_weight_sample_freq = self.cfg["env"]["random_network_adversary"]["weight_sample_freq"]

    def _init_pre_sim_buffers(self):

        """Initialise buffers that must be initialised before sim startup."""
        
        # 0 - scale, 1 - mass, 2 - friction 
        self.cube_random_params = torch.zeros((self.cfg["env"]["numEnvs"], 3), dtype=torch.float, device=self.sim_device)
        # 0 - scale
        self.hand_random_params = torch.zeros((self.cfg["env"]["numEnvs"], 1), dtype=torch.float, device=self.sim_device)

        self.gravity_vec = torch.zeros((self.cfg["env"]["numEnvs"], 3), dtype=torch.float, device=self.sim_device)

    def _init_post_sim_buffers(self):
        """Initialise buffers that must be initialised after sim startup."""

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_hand_dofs)

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.hand_default_dof_pos = torch.zeros(self.num_hand_dofs, dtype=torch.float, device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_hand_dofs]
        self.dof_pos = self.dof_state[..., 0]
        self.dof_vel = self.dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.hold_count_buf = self.progress_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        # object apply random forces parameters
        self.force_decay = to_torch(self.force_decay, dtype=torch.float, device=self.device)
        self.force_prob_range = to_torch(self.force_prob_range, dtype=torch.float, device=self.device)
        self.random_force_prob = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
                                           * torch.rand(self.num_envs, device=self.device) + torch.log(self.force_prob_range[1]))

        self.rb_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float, device=self.device)

        # object observations parameters

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]
        
        # buffer storing object poses which are only refreshed every n steps
        self.obs_object_pose_freq = self.object_pose.clone()
        # buffer storing object poses with added delay which are only refreshed every n steps
        self.obs_object_pose = self.object_pose.clone()

        self.current_object_pose = self.object_pose.clone()

        self.object_pose_wrt_wrist = torch.zeros_like(self.object_pose)
        self.object_pose_wrt_wrist[:, 6] = 1.0

        self.prev_object_pose = self.object_pose.clone()
        
        # inverse refresh rate for each environment
        self.cube_pose_refresh_rates = torch.randint(1, self.max_skip_obs+1, size=(self.num_envs,), device=self.device)
        # offset so not all the environments have it each time
        self.cube_pose_refresh_offset = torch.randint(0, self.max_skip_obs, size=(self.num_envs,), device=self.device)
        
        self.prev_actions = torch.zeros(self.num_envs, self.num_actions, dtype=torch.float, device=self.device)

        # Related to action delay 

        self.prev_actions_queue = torch.zeros(self.cfg["env"]["numEnvs"], \
            self.action_latency_max+1, self.cfg["env"]["numActions"], dtype=torch.float, device=self.sim_device)

        # We have action latency MIN and MAX (declared in _read_cfg() function reading from a config file)
        self.action_latency_min = 1
        self.action_latency = torch.randint(0, self.action_latency_min + 1, \
            size=(self.cfg["env"]["numEnvs"],), dtype=torch.long, device=self.device)

        # tensors for rotation approach reward (-1 stands for not initialized)
        self.curr_rotation_dist = None
        self.best_rotation_dist = -torch.ones(self.num_envs, dtype=torch.float, device=self.device)

        self.unique_cube_rotations = torch.tensor(unique_cube_rotations_3d(), dtype=torch.float, device=self.device)
        self.unique_cube_rotations = matrix_to_quaternion(self.unique_cube_rotations)
        self.num_unique_cube_rotations = self.unique_cube_rotations.shape[0]

    def randomisation_callback(self, param_name, param_val, env_id=None, actor=None):
        if param_name == "gravity":
            self.gravity_vec[:, 0] = param_val.x
            self.gravity_vec[:, 1] = param_val.y
            self.gravity_vec[:, 2] = param_val.z
        elif param_name == "scale" and actor == "object":
            self.cube_random_params[env_id, 0] = param_val.mean()
        elif param_name == "mass" and actor == "object":
            self.cube_random_params[env_id, 1] = np.mean(param_val)
        elif param_name == "friction" and actor == "object":
            self.cube_random_params[env_id, 2] = np.mean(param_val)
        elif param_name == "scale" and actor == "hand":
            self.hand_random_params[env_id, 0] = param_val.mean() 



class AllegroHandDextremeADR(AllegroHandDextreme):

    def _init_pre_sim_buffers(self):
        super()._init_pre_sim_buffers()
        """Initialise buffers that must be initialised before sim startup."""

        self.cube_pose_refresh_rate = torch.zeros(self.cfg["env"]["numEnvs"], device=self.sim_device, dtype=torch.long)
        # offset so not all the environments have it each time
        self.cube_pose_refresh_offset = torch.zeros(self.cfg["env"]["numEnvs"], device=self.sim_device, dtype=torch.long)
        
        # stores previous actions
        self.prev_actions_queue = torch.zeros(self.cfg["env"]["numEnvs"], self.action_latency_max + 1, self.cfg["env"]["numActions"], dtype=torch.float, device=self.sim_device)
        
        # tensors to store random affine transforms
        self.affine_actions_scaling = torch.ones(self.cfg["env"]["numEnvs"], self.cfg["env"]["numActions"], dtype=torch.float, device=self.sim_device)
        self.affine_actions_additive = torch.zeros(self.cfg["env"]["numEnvs"], self.cfg["env"]["numActions"], dtype=torch.float, device=self.sim_device)

        self.affine_cube_pose_scaling = torch.ones(self.cfg["env"]["numEnvs"], 7, dtype=torch.float, device=self.sim_device)
        self.affine_cube_pose_additive = torch.zeros(self.cfg["env"]["numEnvs"], 7, dtype=torch.float, device=self.sim_device)

        self.affine_dof_pos_scaling = torch.ones(self.cfg["env"]["numEnvs"], 16, dtype=torch.float, device=self.sim_device)
        self.affine_dof_pos_additive = torch.zeros(self.cfg["env"]["numEnvs"], 16, dtype=torch.float, device=self.sim_device)

        self.action_latency = torch.zeros(self.cfg["env"]["numEnvs"], dtype=torch.long, device=self.sim_device)


    def sample_discrete_adr(self, param_name, env_ids):
        """Samples a discrete value from ADR continuous distribution.
        Eg, given a parameter with uniform sampling range
        [0, 0.4]
        Will sample 0 with 40% probability and 1 with 60% probability.
        """
        adr_value = self.get_adr_tensor(param_name, env_ids=env_ids)
        continuous_fuzzed = adr_value + (- (torch.rand_like(adr_value) - 0.5))
        return continuous_fuzzed.round().long()

    def sample_gaussian_adr(self, param_name, env_ids, trailing_dim=1):
        adr_value = self.get_adr_tensor(param_name, env_ids=env_ids).view(-1, 1)
        nonlinearity = torch.exp(torch.pow(adr_value, 2.)) - 1.
        stdev = torch.where(adr_value > 0, nonlinearity, torch.zeros_like(adr_value))
        return torch.randn(len(env_ids), trailing_dim, device=self.device, dtype=torch.float) * stdev

    def get_rna_alpha(self):
        return self.get_adr_tensor('rna_alpha').view(-1, 1)

    def apply_randomizations(self, dr_params, randomize_buf, adr_objective=None, randomisation_callback=None):

        super().apply_randomizations(dr_params, randomize_buf, adr_objective, randomisation_callback=self.randomisation_callback)

        randomize_env_ids = randomize_buf.nonzero(as_tuple=False).squeeze(-1)

        self.action_latency[randomize_env_ids] = self.sample_discrete_adr("action_latency", randomize_env_ids)

        self.cube_pose_refresh_rate[randomize_env_ids] = self.sample_discrete_adr("cube_pose_refresh_rate", randomize_env_ids)

        # Nb - code is to generate uniform from 1 to max_skip_obs (inclusive), but cant use 
        # torch.uniform as it doesn't support a different max/min value on each
        self.cube_pose_refresh_offset[randomize_buf] = \
            (torch.rand(randomize_env_ids.shape, device=self.device, dtype=torch.float) \
                * (self.cube_pose_refresh_rate[randomize_env_ids].view(-1).float()) - 0.5).round().long() # offset range shifted back by one
        
        self.affine_actions_scaling[randomize_env_ids] = 1. + self.sample_gaussian_adr("affine_action_scaling", randomize_env_ids, trailing_dim=self.num_actions)
        self.affine_actions_additive[randomize_env_ids] = self.sample_gaussian_adr("affine_action_additive", randomize_env_ids, trailing_dim=self.num_actions)

        self.affine_cube_pose_scaling[randomize_env_ids] = 1. + self.sample_gaussian_adr("affine_cube_pose_scaling", randomize_env_ids, trailing_dim=7)
        self.affine_cube_pose_additive[randomize_env_ids] = self.sample_gaussian_adr("affine_cube_pose_additive", randomize_env_ids, trailing_dim=7)

        self.affine_dof_pos_scaling[randomize_env_ids] = 1. + self.sample_gaussian_adr("affine_dof_pos_scaling", randomize_env_ids, trailing_dim=16)
        self.affine_dof_pos_additive[randomize_env_ids] = self.sample_gaussian_adr("affine_dof_pos_additive", randomize_env_ids, trailing_dim=16)


    def create_sim(self):

        super().create_sim()

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize and self.use_adr:
            adr_objective = torch.zeros(self.num_envs, dtype=float, device=self.device) if self.use_adr else None
            apply_rand_ones = torch.ones(self.num_envs, dtype=bool, device=self.device)
            self.apply_randomizations(self.randomization_params, apply_rand_ones, adr_objective=adr_objective, 
                                    randomisation_callback=self.randomisation_callback)


    def apply_action_noise_latency(self):
        
        action_delay_mask = (torch.rand(self.num_envs, device=self.device) < self.get_adr_tensor("action_delay_prob")).view(-1, 1)            

        actions = \
                self.prev_actions_queue[torch.arange(self.prev_actions_queue.shape[0]), self.action_latency] * ~action_delay_mask \
                + self.prev_actions * action_delay_mask
        
        white_noise = self.sample_gaussian_adr("affine_action_white", self.all_env_ids, trailing_dim=self.num_actions)
        actions = self.affine_actions_scaling * actions + self.affine_actions_additive + white_noise
                    
        return actions

    def compute_observations(self):
        super().compute_observations()

        update_freq = torch.remainder(self.frame + self.cube_pose_refresh_offset, self.cube_pose_refresh_rate) == 0

        # get white noise
        white_noise_pose = self.sample_gaussian_adr("affine_cube_pose_white", self.all_env_ids, trailing_dim=7)
        # compute noisy object pose as a stochatsic affine transform of actual
        noisy_object_pose = self.get_random_cube_observation(
            self.affine_cube_pose_scaling * self.object_pose_wrt_wrist + self.affine_cube_pose_additive + white_noise_pose
        )

        self.obs_object_pose_freq[update_freq] = noisy_object_pose[update_freq]

        # simulate adding delay
        cube_obs_delay_prob = self.get_adr_tensor("cube_obs_delay_prob", self.all_env_ids).view(self.num_envs,)
        update_delay = torch.rand(self.num_envs, device=self.device) < cube_obs_delay_prob
        # update environments that are NOT delayed
        self.obs_object_pose[~update_delay] = self.obs_object_pose_freq[~update_delay]

        white_noise_dof_pos = self.sample_gaussian_adr("affine_dof_pos_white", self.all_env_ids, trailing_dim=16)
        self.dof_pos_randomized = self.affine_dof_pos_scaling * self.dof_pos + self.affine_dof_pos_additive + white_noise_dof_pos

        cube_scale = self.cube_random_params[:, 0]
        cube_scale = cube_scale.reshape(-1, 1)

        self.obs_dict["dof_pos_randomized"][:] = unscale(self.dof_pos_randomized, self.hand_dof_lower_limits, self.hand_dof_upper_limits)

        self.obs_dict["object_pose_cam_randomized"][:] = self.obs_object_pose
        self.obs_dict["goal_relative_rot_cam_randomized"][:] = quat_mul(self.obs_object_pose[:, 3:7], quat_conjugate(self.goal_wrt_wrist_rot))
        
        self.obs_dict["stochastic_delay_params"][:] = torch.stack([
                self.get_adr_tensor("cube_obs_delay_prob"),
                self.cube_pose_refresh_rate.float() / 6.0,
                self.get_adr_tensor("action_delay_prob"),
                self.action_latency.float() / 60.0,
            ], dim=1)

        self.obs_dict["affine_params"][:] = torch.cat([
                self.affine_actions_scaling,
                self.affine_actions_additive,
                self.affine_cube_pose_scaling,
                self.affine_cube_pose_additive,
                self.affine_dof_pos_scaling,
                self.affine_dof_pos_additive
            ],
        dim=-1)

    def _read_cfg(self):
        super()._read_cfg()

        self.vel_obs_scale = 1.0  # scale factor of velocity based observations
        self.force_torque_obs_scale = 1.0  # scale factor of velocity based observations

        return


class AllegroHandDextremeManualDR(AllegroHandDextreme):

    def _init_post_sim_buffers(self):
        super()._init_post_sim_buffers()

        # We could potentially update this regularly
        self.action_delay_prob = self.action_delay_prob_max * \
            torch.rand(self.cfg["env"]["numEnvs"], dtype=torch.float, device=self.device)
        
        # inverse refresh rate for each environment
        self.cube_pose_refresh_rate = torch.randint(1, self.max_skip_obs+1, size=(self.num_envs,), device=self.device)
        # offset so not all the environments have it each time
        self.cube_pose_refresh_offset = torch.randint(0, self.max_skip_obs, size=(self.num_envs,), device=self.device)

        
    def get_num_obs_dict(self, num_dofs=16):

        return {"dof_pos": num_dofs,
            "dof_vel": num_dofs,
            "dof_force": num_dofs, # generalised forces
            
            "object_pose": 7,            
            "object_vels": 6,

            "goal_pose": 7,            
            "goal_relative_rot": 4,
            
            "object_pose_cam": 7,

            "goal_relative_rot_cam": 4,

            "last_actions": num_dofs,
            "cube_random_params": 3,
            "hand_random_params": 1,
            "gravity_vec": 3,

            "rot_dist": 2,
            "ft_states": 13 * self.num_fingertips, # (pos, quat, linvel, angvel) per fingertip
            "ft_force_torques": 6 * self.num_fingertips, # wrenches
            }


    def get_rna_alpha(self):
        if self.randomize:
            return torch.rand(self.num_envs, 1, device=self.device)
        else:
            return torch.zeros(self.num_envs, 1, device=self.device)

    def create_sim(self):

        super().create_sim()

        # If randomizing, apply once immediately on startup before the fist sim step
        # ADR has its own create_sim and randomisation is called there with appropriate 
        # inputs
        if self.randomize and not self.use_adr:
            self.apply_randomizations(self.randomization_params, randomisation_callback=self.randomisation_callback)



    def apply_randomizations(self, dr_params, randomize_buf=None, adr_objective=None, randomisation_callback=None):

        super().apply_randomizations(dr_params, randomize_buf=None, adr_objective=None, randomisation_callback=self.randomisation_callback)
        
    def apply_action_noise_latency(self):

        # anneal action latency 
        if self.randomize:

            self.cur_action_latency = 1.0 / self.action_latency_scheduled_steps \
                * min(self.last_step, self.action_latency_scheduled_steps)

            self.cur_action_latency = min(max(int(self.cur_action_latency), self.action_latency_min), self.action_latency_max)

            self.extras['annealing/cur_action_latency_max'] = self.cur_action_latency

            self.action_latency = torch.randint(0, self.cur_action_latency + 1, \
                size=(self.cfg["env"]["numEnvs"],), dtype=torch.long, device=self.device)

        # probability of not updating the action this step (on top of the delay)
        action_delay_mask = (torch.rand(self.num_envs, device=self.device) > self.action_delay_prob).view(-1, 1)

        actions_delayed = \
            self.prev_actions_queue[torch.arange(self.prev_actions_queue.shape[0]), self.action_latency] * action_delay_mask \
            + self.prev_actions * ~action_delay_mask
        
        return actions_delayed


    def compute_observations(self):
        super().compute_observations()
        
            
#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, hold_count_buf, cur_targets, prev_targets, hand_dof_vel, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float, action_delta_penalty_scale: float, #max_velocity: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, num_success_hold_steps: int
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    # Distance from the hand to the object
    goal_dist = torch.norm(object_pos - target_pos, p=2, dim=-1)

    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    dist_rew = goal_dist * dist_reward_scale
    rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty =  action_penalty_scale * torch.sum(actions ** 2, dim=-1)
    action_delta_penalty = action_delta_penalty_scale * torch.sum((cur_targets - prev_targets) ** 2, dim=-1)
  
    max_velocity = 5.0 #rad/s
    vel_tolerance = 1.0
    velocity_penalty_coef = -0.05

    # todo add actions regularization

    velocity_penalty = velocity_penalty_coef * torch.sum((hand_dof_vel/(max_velocity - vel_tolerance)) ** 2, dim=-1)

    # Find out which envs hit the goal and update successes count
    goal_reached = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.ones_like(reset_goal_buf), reset_goal_buf)
    hold_count_buf = torch.where(goal_reached, hold_count_buf + 1, torch.zeros_like(goal_reached))

    goal_resets = torch.where(hold_count_buf > num_success_hold_steps, torch.ones_like(reset_goal_buf), reset_goal_buf)
    successes = successes + goal_resets

    # Success bonus: orientation is within `success_tolerance` of goal orientation
    reach_goal_rew = (goal_resets == 1) * reach_goal_bonus

    # Fall penalty: distance to the goal is larger than a threashold
    fall_rew = (goal_dist >= fall_dist) * fall_penalty

    # Check env termination conditions, including maximum success number
    resets = torch.where(goal_dist >= fall_dist, torch.ones_like(reset_buf), reset_buf)
    if max_consecutive_successes > 0:
        # Reset progress buffer on goal envs if max_consecutive_successes > 0
        progress_buf = torch.where(torch.abs(rot_dist) <= success_tolerance, torch.zeros_like(progress_buf), progress_buf)
        resets = torch.where(successes >= max_consecutive_successes, torch.ones_like(resets), resets)

    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    # Apply penalty for not reaching the goal
    timeout_rew = timed_out * 0.5 * fall_penalty

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    reward = dist_rew + rot_rew + action_penalty + action_delta_penalty + velocity_penalty + reach_goal_rew + fall_rew + timeout_rew

    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor*finished_cons_successes/num_resets + (1.0 - av_factor)*consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, hold_count_buf, successes, cons_successes, \
        dist_rew, rot_rew, action_penalty, action_delta_penalty, velocity_penalty, reach_goal_rew, fall_rew, timeout_rew  # return individual rewards for visualization


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))



def unique_cube_rotations_3d() -> List[np.ndarray]:
    """
    Returns the list of all possible 90-degree cube rotations in 3D.
    Based on https://stackoverflow.com/a/70413438/1645784
    """

    all_rotations = []
    for x, y, z in permutations([0, 1, 2]):
        for sx, sy, sz in itertools.product([-1, 1], repeat=3):
            rotation_matrix = np.zeros((3, 3))
            rotation_matrix[0, x] = sx
            rotation_matrix[1, y] = sy
            rotation_matrix[2, z] = sz
            if np.linalg.det(rotation_matrix) == 1:
                all_rotations.append(rotation_matrix)

    return all_rotations