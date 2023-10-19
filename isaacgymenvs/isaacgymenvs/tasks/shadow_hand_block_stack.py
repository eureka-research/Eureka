# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from unittest import TextTestRunner
from matplotlib.pyplot import axis
from PIL import Image as Im

import numpy as np
import os
import random
import torch

# from bidexhands.utils.torch_jit_utils import *
# from bidexhands.tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask


class ShadowHandBlockStack(VecTask):
    """
    This class corresponds to the Block Stack task. This environment involves dual hands and two blocks, and we need to stack the block
    as a tower.

    Args:
        cfg (dict): The configuration file of the environment, which is the parameter defined in the
            dexteroushandenvs/cfg folder

        sim_params (isaacgym._bindings.linux-x86_64.gym_37.SimParams): Isaacgym simulation parameters 
            which contains the parameter settings of the isaacgym physics engine. Also defined in the 
            dexteroushandenvs/cfg folder

        physics_engine (isaacgym._bindings.linux-x86_64.gym_37.SimType): Isaacgym simulation backend
            type, which only contains two members: PhysX and Flex. Our environment use the PhysX backend

        device_type (str): Specify the computing device for isaacgym simulation calculation, there are 
            two options: 'cuda' and 'cpu'. The default is 'cuda'

        device_id (int): Specifies the number of the computing device used when simulating. It is only 
            useful when device_type is cuda. For example, when device_id is 1, the device used 
            is 'cuda:1'

        headless (bool): Specifies whether to visualize during training

        agent_index (list): Specifies how to divide the agents of the hands, useful only when using a 
            multi-agent algorithm. It contains two lists, representing the left hand and the right hand. 
            Each list has six numbers from 0 to 5, representing the palm, middle finger, ring finger, 
            tail finger, index finger, and thumb. Each part can be combined arbitrarily, and if placed 
            in the same list, it means that it is divided into the same agent. The default setting is
            [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], which means that the two whole hands are 
            regarded as one agent respectively.

        is_multi_agent (bool): Specifies whether it is a multi-agent environment
    """

    # def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless, agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        # self.sim_params = sim_params
        # self.physics_engine = physics_engine
        # self.agent_index = agent_index
        self.agent_index = [[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]]
        self.is_multi_agent = False

        # Domain randomization configuration
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        # Reward configuration
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]

        # Scale factor of velocity based observations
        self.vel_obs_scale = 0.2
        # Scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0 

        # The noise of the initial state each time the environment is reset
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]

        # The configuration of how to control the ShadowHand (Action in RL)
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        # Whether to enable debug mode during visualization
        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        # Success and goal configuration
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        # Scale factor of transition and orientation when the base of Shadowhand is not fixed
        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        # The inverser number of the control frequency
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        # Specifies the object to be manipulated, which can be an object in the sapien dataset. 
        # Only useful in certain environments
        self.object_type = self.cfg["env"]["objectType"]
        self.ignore_z = (self.object_type == "pen")

        # Specify the path of the asset
        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "pot": "mjcf/pen/mobility.urdf"
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "openai", "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["point_cloud", "full_state"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [point_cloud, full_state]")

        print("Obs type:", self.obs_type)
        # Specify the number of action and observation in the environment
        self.num_point_cloud_feature_dim = 768
        self.num_obs_dict = {
            "point_cloud": 428 + self.num_point_cloud_feature_dim * 3,
            "point_cloud_for_distill": 428 + self.num_point_cloud_feature_dim * 3,
            "full_state": 428
        }
        self.num_hand_obs = 72 + 95 + 26 + 6
        self.up_axis = 'z'

        # The names of the five fingertips of Shadowhand, which are used to obtain force information later
        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal", "robot0:thdistal"]
        self.a_fingertips = ["robot1:ffdistal", "robot1:mfdistal", "robot1:rfdistal", "robot1:lfdistal", "robot1:thdistal"]
        self.num_fingertips = len(self.fingertips) * 2

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 211

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        if self.is_multi_agent:
            self.num_agents = 2
            self.cfg["env"]["numActions"] = 26
            
        else:
            self.num_agents = 1
            self.cfg["env"]["numActions"] = 52

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # self.cfg["device_type"] = device_type
        # self.cfg["device_id"] = device_id
        # self.cfg["headless"] = headless

        if self.obs_type in ["point_cloud"]:
            from PIL import Image as Im
            from bidexhands.utils import o3dviewer
            # from pointnet2_ops import pointnet2_utils

        self.camera_debug = self.cfg["env"].get("cameraDebug", False)
        self.point_cloud_debug = self.cfg["env"].get("pointCloudDebug", False)

        # super().__init__(cfg=self.cfg)

        # Viewer settings, including the camera's initial position and viewing direction
        if self.viewer != None:
            # cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            # cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            cam_pos = gymapi.Vec3(-1.0, 5.0, 2.0)
            cam_target = gymapi.Vec3(1., -1., -2.)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_shadow_hand_dofs * 2 + self.num_object_dofs * 2)
        self.dof_force_tensor = self.dof_force_tensor[:, :48]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        # self.shadow_hand_default_dof_pos = to_torch([0.0, 0.0, -0,  -0,  -0,  -0, -0, -0,
        #                                     -0,  -0, -0,  -0,  -0,  -0, -0, -0,
        #                                     -0,  -0, -0,  -1.04,  1.2,  0., 0, -1.57], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]

        self.shadow_hand_another_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2]
        self.shadow_hand_another_dof_pos = self.shadow_hand_another_dof_state[..., 0]
        self.shadow_hand_another_dof_vel = self.shadow_hand_another_dof_state[..., 1]

        self.object_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, self.num_shadow_hand_dofs*2:self.num_shadow_hand_dofs*2 + self.num_object_dofs]
        self.object_dof_pos = self.object_dof_state[..., 0]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone() 

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

        self.total_successes = 0
        self.total_resets = 0

    def create_sim(self):
        """
        Allocates which device will simulate and which device will render the scene. Defines the simulation type to be used
        """
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 if self.up_axis == 'z' else 1 # index of up axis: Y=1, Z=2
        # self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        """
        Adds ground plane to simulation
        """

        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        """
        Create multiple parallel isaacgym environments

        Args:
            num_envs (int): The total number of environment 

            spacing (float): Specifies half the side length of the square area occupied by each environment

            num_per_row (int): Specify how many environments in a row
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        # asset_root = "../../assets"
        asset_root = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets'))
        shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        shadow_hand_another_asset_file = "mjcf/open_ai_assets/hand/shadow_hand1.xml"
        table_texture_files = asset_root + "/" + "textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        if "asset" in self.cfg["env"]:
            # asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]

        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = False
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)
        shadow_hand_another_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_another_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        a_relevant_tendons = ["robot1:T_FFJ1c", "robot1:T_MFJ1c", "robot1:T_RFJ1c", "robot1:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)
        a_tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_another_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping
            for rt in a_relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_another_asset, i) == rt:
                    a_tendon_props[i].limit_stiffness = limit_stiffness
                    a_tendon_props[i].damping = t_damping
        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        self.gym.set_asset_tendon_properties(shadow_hand_another_asset, a_tendon_props)
        
        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]

        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        shadow_hand_another_dof_props = self.gym.get_asset_dof_properties(shadow_hand_another_asset)

        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()

        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.density = 100
        object_asset_options.fix_base_link = False
        # object_asset_options.collapse_fixed_joints = True
        # object_asset_options.disable_gravity = True
        object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        block_asset_file = "urdf/objects/cube_multicolor1.urdf"
        block_asset = self.gym.load_asset(self.sim, asset_root, block_asset_file, object_asset_options)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

        # set object dof properties
        self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
        object_dof_props = self.gym.get_asset_dof_properties(object_asset)

        self.object_dof_lower_limits = []
        self.object_dof_upper_limits = []

        for i in range(self.num_object_dofs):
            self.object_dof_lower_limits.append(object_dof_props['lower'][i])
            self.object_dof_upper_limits.append(object_dof_props['upper'][i])

        self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
        self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)

        # create table asset
        table_dims = gymapi.Vec3(0.5, 1.0, 0.6)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.6, 0.7)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(3.14159, 0, 3.14)

        shadow_another_hand_start_pose = gymapi.Transform()
        shadow_another_hand_start_pose.p = gymapi.Vec3(0.0, -0.6, 0.7)
        shadow_another_hand_start_pose.r = gymapi.Quat().from_euler_zyx(3.14159, 0, 0)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(0.0, 0.1, 0.625)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 1.57, 0)
        pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0

        block_start_pose = gymapi.Transform()
        block_start_pose.p = gymapi.Vec3(0.0, -0.1, 0.625)
        block_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 1.57, 0)
        # object_start_pose.p.x = shadow_hand_start_pose.p.x + pose_dx
        # object_start_pose.p.y = shadow_hand_start_pose.p.y + pose_dy
        # object_start_pose.p.z = shadow_hand_start_pose.p.z + pose_dz

        if self.object_type == "pen":
            object_start_pose.p.z = shadow_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 10)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.0

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies * 2 + 3 * self.num_object_bodies + 1
        max_agg_shapes = self.num_shadow_hand_shapes * 2 + 3 * self.num_object_shapes + 1

        self.shadow_hands = []
        self.envs = []

        self.object_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.another_hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []
        self.block_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        self.fingertip_another_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_another_asset, name) for name in self.a_fingertips]

        # create fingertip force sensors, if needed
        sensor_pose = gymapi.Transform()
        for ft_handle in self.fingertip_handles:
            self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)
        for ft_a_handle in self.fingertip_another_handles:
            self.gym.create_asset_force_sensor(shadow_hand_another_asset, ft_a_handle, sensor_pose)
        
        if self.obs_type in ["point_cloud"]:
            self.cameras = []
            self.camera_tensors = []
            self.camera_view_matrixs = []
            self.camera_proj_matrixs = []

            self.camera_props = gymapi.CameraProperties()
            self.camera_props.width = 256
            self.camera_props.height = 256
            self.camera_props.enable_tensors = True

            self.env_origin = torch.zeros((self.num_envs, 3), device=self.device, dtype=torch.float)
            self.pointCloudDownsampleNum = 768
            self.camera_u = torch.arange(0, self.camera_props.width, device=self.device)
            self.camera_v = torch.arange(0, self.camera_props.height, device=self.device)

            self.camera_v2, self.camera_u2 = torch.meshgrid(self.camera_v, self.camera_u, indexing='ij')

            if self.point_cloud_debug:
                import open3d as o3d
                from bidexhands.utils.o3dviewer import PointcloudVisualizer
                self.pointCloudVisualizer = PointcloudVisualizer()
                self.pointCloudVisualizerInitialized = False
                self.o3d_pc = o3d.geometry.PointCloud()
            else:
                self.pointCloudVisualizer = None

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)


            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, 0, 0)
            shadow_hand_another_actor = self.gym.create_actor(env_ptr, shadow_hand_another_asset, shadow_another_hand_start_pose, "another_hand", i, 0, 0)
            
            self.hand_start_states.append([shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                                           shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z, shadow_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_another_actor, shadow_hand_another_dof_props)
            another_hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_another_actor, gymapi.DOMAIN_SIM)
            self.another_hand_indices.append(another_hand_idx)            

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_actor)
            hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
            
            for n in self.agent_index[0]:
                colorx = random.uniform(0, 1)
                colory = random.uniform(0, 1)
                colorz = random.uniform(0, 1)
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(colorx, colory, colorz))
            for n in self.agent_index[1]:                
                colorx = random.uniform(0, 1)
                colory = random.uniform(0, 1)
                colorz = random.uniform(0, 1)
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        self.gym.set_rigid_body_color(env_ptr, shadow_hand_another_actor, o, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(colorx, colory, colorz))
                # gym.set_rigid_body_texture(env, actor_handles[-1], n, gymapi.MESH_VISUAL,
                #                            loaded_texture_handle_list[random.randint(0, len(loaded_texture_handle_list)-1)])

            # create fingertip force-torque sensors
            self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)
            self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_another_actor)
            
            # add object
            object_handle = self.gym.create_actor(env_ptr, block_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            # self.gym.set_actor_scale(env_ptr, object_handle, 0.3)

            block_handle = self.gym.create_actor(env_ptr, block_asset, block_start_pose, "block", i, 0, 0)
            block_idx = self.gym.get_actor_index(env_ptr, block_handle, gymapi.DOMAIN_SIM)
            self.block_indices.append(block_idx)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            # self.gym.set_actor_scale(env_ptr, goal_handle, 0.3)

            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            block_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, block_handle)
            object_shape_props[0].friction = 0.25
            block_shape_props[0].friction = 0.25
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, block_handle, block_shape_props)

            hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, shadow_hand_actor)
            another_hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, shadow_hand_another_actor)
            for object_shape_prop in hand_shape_props:
                object_shape_prop.friction = 0.5
            for object_shape_prop in another_hand_shape_props:
                object_shape_prop.friction = 0.5
            self.gym.set_actor_rigid_shape_properties(env_ptr, shadow_hand_actor, hand_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, shadow_hand_another_actor, another_hand_shape_props)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)
            
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            for object_shape_prop in table_shape_props:
                object_shape_prop.friction = 0.2
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.obs_type in ["point_cloud"]:
                camera_handle = self.gym.create_camera_sensor(env_ptr, self.camera_props)
                self.gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(0.25, 0, 1.0), gymapi.Vec3(-0.25, 0, -0.5))
                camera_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_DEPTH)
                torch_cam_tensor = gymtorch.wrap_tensor(camera_tensor)
                cam_vinv = torch.inverse((torch.tensor(self.gym.get_camera_view_matrix(self.sim, env_ptr, camera_handle)))).to(self.device)
                cam_proj = torch.tensor(self.gym.get_camera_proj_matrix(self.sim, env_ptr, camera_handle), device=self.device)

                origin = self.gym.get_env_origin(env_ptr)
                self.env_origin[i][0] = origin.x
                self.env_origin[i][1] = origin.y
                self.env_origin[i][2] = origin.z
                self.camera_tensors.append(torch_cam_tensor)
                self.camera_view_matrixs.append(cam_vinv)
                self.camera_proj_matrixs.append(cam_proj)
                self.cameras.append(camera_handle)
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.object_init_state.clone()
        # self.goal_pose = self.goal_states[:, 0:7]
        # self.goal_pos = self.goal_states[:, 0:3]
        # self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.fingertip_another_handles = to_torch(self.fingertip_another_handles, dtype=torch.long, device=self.device)

        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.another_hand_indices = to_torch(self.another_hand_indices, dtype=torch.long, device=self.device)

        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)
        self.block_indices = to_torch(self.block_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        """
        Compute the reward of all environment. The core function is compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.block_right_handle_pos, self.block_left_handle_pos, 
            self.left_hand_pos, self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos, 
            self.left_hand_ff_pos, self.left_hand_mf_pos, self.left_hand_rf_pos, self.left_hand_lf_pos, self.left_hand_th_pos, 
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )
        , which we will introduce in detail there

        Args:
            actions (tensor): Actions of agents in the all environment 
        """
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.rew_buf, self.reset_buf, self.reset_goal_buf, self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot, self.goal_pos, self.goal_rot, self.block_right_handle_pos, self.block_left_handle_pos, 
            self.left_hand_pos, self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos, self.right_hand_lf_pos, self.right_hand_th_pos, 
            self.left_hand_ff_pos, self.left_hand_mf_pos, self.left_hand_rf_pos, self.left_hand_lf_pos, self.left_hand_th_pos, 
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, (self.object_type == "pen")
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_observations(self):
        """
        Compute the observations of all environment. The core function is self.compute_full_state(True), 
        which we will introduce in detail there

        """
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)

        if self.obs_type in ["point_cloud"]:
            self.gym.render_all_camera_sensors(self.sim)
            self.gym.start_access_image_tensors(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.block_pose = self.root_state_tensor[self.block_indices, 0:7]
        self.block_pos = self.root_state_tensor[self.block_indices, 0:3]
        self.block_rot = self.root_state_tensor[self.block_indices, 3:7]
        self.block_linvel = self.root_state_tensor[self.block_indices, 7:10]
        self.block_angvel = self.root_state_tensor[self.block_indices, 10:13]

        self.block_right_handle_pos = self.rigid_body_states[:, 26 * 2, 0:3]
        self.block_right_handle_rot = self.rigid_body_states[:, 26 * 2, 3:7]
        self.block_right_handle_pos = self.block_right_handle_pos + quat_apply(self.block_right_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.)
        self.block_right_handle_pos = self.block_right_handle_pos + quat_apply(self.block_right_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.0)
        self.block_right_handle_pos = self.block_right_handle_pos + quat_apply(self.block_right_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.0)

        self.block_left_handle_pos = self.rigid_body_states[:, 26 * 2 + 1, 0:3]
        self.block_left_handle_rot = self.rigid_body_states[:, 26 * 2 + 1, 3:7]
        self.block_left_handle_pos = self.block_left_handle_pos + quat_apply(self.block_left_handle_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * 0.0)
        self.block_left_handle_pos = self.block_left_handle_pos + quat_apply(self.block_left_handle_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.0)
        self.block_left_handle_pos = self.block_left_handle_pos + quat_apply(self.block_left_handle_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.0)

        self.left_hand_pos = self.rigid_body_states[:, 3 + 26, 0:3]
        self.left_hand_rot = self.rigid_body_states[:, 3 + 26, 3:7]
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.left_hand_pos = self.left_hand_pos + quat_apply(self.left_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        self.right_hand_pos = self.rigid_body_states[:, 3, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, 3, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot, to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        self.right_hand_ff_pos = self.rigid_body_states[:, 7, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, 7, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_mf_pos = self.rigid_body_states[:, 11, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, 11, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_rf_pos = self.rigid_body_states[:, 15, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, 15, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_lf_pos = self.rigid_body_states[:, 20, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, 20, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.right_hand_th_pos = self.rigid_body_states[:, 25, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, 25, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.left_hand_ff_pos = self.rigid_body_states[:, 7 + 26, 0:3]
        self.left_hand_ff_rot = self.rigid_body_states[:, 7 + 26, 3:7]
        self.left_hand_ff_pos = self.left_hand_ff_pos + quat_apply(self.left_hand_ff_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_mf_pos = self.rigid_body_states[:, 11 + 26, 0:3]
        self.left_hand_mf_rot = self.rigid_body_states[:, 11 + 26, 3:7]
        self.left_hand_mf_pos = self.left_hand_mf_pos + quat_apply(self.left_hand_mf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_rf_pos = self.rigid_body_states[:, 15 + 26, 0:3]
        self.left_hand_rf_rot = self.rigid_body_states[:, 15 + 26, 3:7]
        self.left_hand_rf_pos = self.left_hand_rf_pos + quat_apply(self.left_hand_rf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_lf_pos = self.rigid_body_states[:, 20 + 26, 0:3]
        self.left_hand_lf_rot = self.rigid_body_states[:, 20 + 26, 3:7]
        self.left_hand_lf_pos = self.left_hand_lf_pos + quat_apply(self.left_hand_lf_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        self.left_hand_th_pos = self.rigid_body_states[:, 25 + 26, 0:3]
        self.left_hand_th_rot = self.rigid_body_states[:, 25 + 26, 3:7]
        self.left_hand_th_pos = self.left_hand_th_pos + quat_apply(self.left_hand_th_rot, to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]
        self.fingertip_another_state = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:13]
        self.fingertip_another_pos = self.rigid_body_states[:, self.fingertip_another_handles][:, :, 0:3]

        if self.obs_type == "full_state":
            self.compute_full_state()
        elif self.obs_type == "point_cloud":
            self.compute_point_cloud_observation()
        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        """
        Compute the observations of all environment. The observation is composed of three parts: 
        the state values of the left and right hands, and the information of objects and target. 
        The state values of the left and right hands were the same for each task, including hand 
        joint and finger positions, velocity, and force information. The detail 428-dimensional 
        observational space as shown in below:

        Index       Description
        0 - 23	    right shadow hand dof position
        24 - 47	    right shadow hand dof velocity
        48 - 71	    right shadow hand dof force
        72 - 136	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        137 - 166	right shadow hand fingertip force, torque (5 x 6)
        167 - 169	right shadow hand base position
        170 - 172	right shadow hand base rotation
        173 - 198	right shadow hand actions
        199 - 222	left shadow hand dof position
        223 - 246	left shadow hand dof velocity
        247 - 270	left shadow hand dof force
        271 - 335	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        336 - 365	left shadow hand fingertip force, torque (5 x 6)
        366 - 368	left shadow hand base position
        369 - 371	left shadow hand base rotation
        372 - 397	left shadow hand actions
        398 - 404	object pose
        405 - 407	object linear velocity
        408 - 410	object angle velocity
        411 - 417	goal pose
        418 - 421	goal rot - object rot
        422 - 424	block1 position
        425 - 427	block2 position
        """

        num_ft_states = 13 * int(self.num_fingertips / 2)  # 65
        num_ft_force_torques = 6 * int(self.num_fingertips / 2)  # 30

        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

        fingertip_obs_start = 72  # 168 = 157 + 11
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
        
        hand_pose_start = fingertip_obs_start + 95
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + 26] = self.actions[:, :26]

        # another_hand
        another_hand_start = action_obs_start + 26
        self.obs_buf[:, another_hand_start:self.num_shadow_hand_dofs + another_hand_start] = unscale(self.shadow_hand_another_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs + another_hand_start:2*self.num_shadow_hand_dofs + another_hand_start] = self.vel_obs_scale * self.shadow_hand_another_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs + another_hand_start:3*self.num_shadow_hand_dofs + another_hand_start] = self.force_torque_obs_scale * self.dof_force_tensor[:, 24:48]

        fingertip_another_obs_start = another_hand_start + 72
        self.obs_buf[:, fingertip_another_obs_start:fingertip_another_obs_start + num_ft_states] = self.fingertip_another_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_another_obs_start + num_ft_states:fingertip_another_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, 30:]

        hand_another_pose_start = fingertip_another_obs_start + 95
        self.obs_buf[:, hand_another_pose_start:hand_another_pose_start + 3] = self.left_hand_pos
        self.obs_buf[:, hand_another_pose_start+3:hand_another_pose_start+4] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+4:hand_another_pose_start+5] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+5:hand_another_pose_start+6] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[2].unsqueeze(-1)

        action_another_obs_start = hand_another_pose_start + 6
        self.obs_buf[:, action_another_obs_start:action_another_obs_start + 26] = self.actions[:, 26:]

        obj_obs_start = action_another_obs_start + 26  # 144
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 13 + 7] = self.block_pose
        self.obs_buf[:, obj_obs_start + 13 + 7:obj_obs_start + 13 + 10] = self.block_linvel
        self.obs_buf[:, obj_obs_start + 13 + 10:obj_obs_start + 13 + 13] = self.vel_obs_scale * self.block_angvel

        # self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.block_right_handle_pos
        # self.obs_buf[:, obj_obs_start + 16:obj_obs_start + 19] = self.block_left_handle_pos
        # goal_obs_start = obj_obs_start + 13  # 157 = 144 + 13
        # self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
        # self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

    def compute_point_cloud_observation(self, collect_demonstration=False):
        """
        Compute the observations of all environment. The observation is composed of three parts: 
        the state values of the left and right hands, and the information of objects and target. 
        The state values of the left and right hands were the same for each task, including hand 
        joint and finger positions, velocity, and force information. The detail 428-dimensional 
        observational space as shown in below:

        Index       Description
        0 - 23	    right shadow hand dof position
        24 - 47	    right shadow hand dof velocity
        48 - 71	    right shadow hand dof force
        72 - 136	right shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        137 - 166	right shadow hand fingertip force, torque (5 x 6)
        167 - 169	right shadow hand base position
        170 - 172	right shadow hand base rotation
        173 - 198	right shadow hand actions
        199 - 222	left shadow hand dof position
        223 - 246	left shadow hand dof velocity
        247 - 270	left shadow hand dof force
        271 - 335	left shadow hand fingertip pose, linear velocity, angle velocity (5 x 13)
        336 - 365	left shadow hand fingertip force, torque (5 x 6)
        366 - 368	left shadow hand base position
        369 - 371	left shadow hand base rotation
        372 - 397	left shadow hand actions
        398 - 404	object pose
        405 - 407	object linear velocity
        408 - 410	object angle velocity
        411 - 417	goal pose
        418 - 421	goal rot - object rot
        422 - 424	block1 position
        425 - 427	block2 position
        """

        num_ft_states = 13 * int(self.num_fingertips / 2)  # 65
        num_ft_force_torques = 6 * int(self.num_fingertips / 2)  # 30

        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :24]

        fingertip_obs_start = 72  # 168 = 157 + 11
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]
        
        hand_pose_start = fingertip_obs_start + 95
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
        self.obs_buf[:, hand_pose_start+3:hand_pose_start+4] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+4:hand_pose_start+5] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start+5:hand_pose_start+6] = get_euler_xyz(self.hand_orientations[self.hand_indices, :])[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        self.obs_buf[:, action_obs_start:action_obs_start + 26] = self.actions[:, :26]

        # another_hand
        another_hand_start = action_obs_start + 26
        self.obs_buf[:, another_hand_start:self.num_shadow_hand_dofs + another_hand_start] = unscale(self.shadow_hand_another_dof_pos,
                                                            self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, self.num_shadow_hand_dofs + another_hand_start:2*self.num_shadow_hand_dofs + another_hand_start] = self.vel_obs_scale * self.shadow_hand_another_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs + another_hand_start:3*self.num_shadow_hand_dofs + another_hand_start] = self.force_torque_obs_scale * self.dof_force_tensor[:, 24:48]

        fingertip_another_obs_start = another_hand_start + 72
        self.obs_buf[:, fingertip_another_obs_start:fingertip_another_obs_start + num_ft_states] = self.fingertip_another_state.reshape(self.num_envs, num_ft_states)
        self.obs_buf[:, fingertip_another_obs_start + num_ft_states:fingertip_another_obs_start + num_ft_states +
                    num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, 30:]

        hand_another_pose_start = fingertip_another_obs_start + 95
        self.obs_buf[:, hand_another_pose_start:hand_another_pose_start + 3] = self.left_hand_pos
        self.obs_buf[:, hand_another_pose_start+3:hand_another_pose_start+4] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[0].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+4:hand_another_pose_start+5] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[1].unsqueeze(-1)
        self.obs_buf[:, hand_another_pose_start+5:hand_another_pose_start+6] = get_euler_xyz(self.hand_orientations[self.another_hand_indices, :])[2].unsqueeze(-1)

        action_another_obs_start = hand_another_pose_start + 6
        self.obs_buf[:, action_another_obs_start:action_another_obs_start + 26] = self.actions[:, 26:]

        obj_obs_start = action_another_obs_start + 26  # 144
        self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 13 + 7] = self.block_pose
        self.obs_buf[:, obj_obs_start + 13 + 7:obj_obs_start + 13 + 10] = self.block_linvel
        self.obs_buf[:, obj_obs_start + 13 + 10:obj_obs_start + 13 + 13] = self.vel_obs_scale * self.block_angvel
        point_clouds = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)

        if self.camera_debug:
            import matplotlib.pyplot as plt
            self.camera_rgba_debug_fig = plt.figure("CAMERA_RGBD_DEBUG")
            camera_rgba_image = self.camera_visulization(is_depth_image=False)
            plt.imshow(camera_rgba_image)
            plt.pause(1e-9)

        for i in range(self.num_envs):
            # Here is an example. In practice, it's better not to convert tensor from GPU to CPU
            points = depth_image_to_point_cloud_GPU(self.camera_tensors[i], self.camera_view_matrixs[i], self.camera_proj_matrixs[i], self.camera_u2, self.camera_v2, self.camera_props.width, self.camera_props.height, 10, self.device)
            
            if points.shape[0] > 0:
                selected_points = self.sample_points(points, sample_num=self.pointCloudDownsampleNum, sample_mathed='random')
            else:
                selected_points = torch.zeros((self.num_envs, self.pointCloudDownsampleNum, 3), device=self.device)
            
            point_clouds[i] = selected_points

        if self.pointCloudVisualizer != None :
            import open3d as o3d
            points = point_clouds[0, :, :3].cpu().numpy()
            # colors = plt.get_cmap()(point_clouds[0, :, 3].cpu().numpy())
            self.o3d_pc.points = o3d.utility.Vector3dVector(points)
            # self.o3d_pc.colors = o3d.utility.Vector3dVector(colors[..., :3])

            if self.pointCloudVisualizerInitialized == False :
                self.pointCloudVisualizer.add_geometry(self.o3d_pc)
                self.pointCloudVisualizerInitialized = True
            else :
                self.pointCloudVisualizer.update(self.o3d_pc)

        self.gym.end_access_image_tensors(self.sim)
        point_clouds -= self.env_origin.view(self.num_envs, 1, 3)

        point_clouds_start = obj_obs_start + 13 + 13
        self.obs_buf[:, point_clouds_start:point_clouds_start + self.pointCloudDownsampleNum * 3].copy_(point_clouds.view(self.num_envs, self.pointCloudDownsampleNum * 3))

    def reset_target_pose(self, env_ids, apply_reset=False):
        """
        Reset and randomize the goal pose

        Args:
            env_ids (tensor): The index of the environment that needs to reset goal pose

            apply_reset (bool): Whether to reset the goal directly here, usually used 
                when the same task wants to complete multiple goals

        """
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        # self.goal_states[env_ids, 1] += 0.25
        # self.goal_states[env_ids, 2] -= 10.0

        # self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def reset_idx(self, env_ids, goal_env_ids):
        """
        Reset and randomize the environment

        Args:
            env_ids (tensor): The index of the environment that needs to reset

            goal_env_ids (tensor): The index of the environment that only goals need reset

        """
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        if self.object_type == "pen":
            rand_angle_y = torch.tensor(0.3)
            new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
                                                    self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        # self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,
        #                                              gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5+self.num_shadow_hand_dofs]

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta

        self.shadow_hand_dof_pos[env_ids, :] = pos
        self.shadow_hand_another_dof_pos[env_ids, :] = pos

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]   

        self.shadow_hand_another_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_shadow_hand_dofs:5+self.num_shadow_hand_dofs*2]

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos

        self.prev_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2] = pos
        self.cur_targets[env_ids, self.num_shadow_hand_dofs:self.num_shadow_hand_dofs*2] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        another_hand_indices = self.another_hand_indices[env_ids].to(torch.int32)

        all_hand_indices = torch.unique(torch.cat([hand_indices,
                                                 another_hand_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))  

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices[env_ids],
                                              self.table_indices[env_ids],
                                              self.block_indices[env_ids]]).to(torch.int32))

        self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]
        self.hand_linvels[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 7:10]
        self.hand_angvels[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 10:13]

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
                                              
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0


    def pre_physics_step(self, actions):
        """
        The pre-processing of the physics step. Determine whether the reset environment is needed, 
        and calculate the next movement of Shadowhand through the given action. The 52-dimensional 
        action space as shown in below:
        
        Index   Description
        0 - 19 	right shadow hand actuated joint
        20 - 22	right shadow hand base translation
        23 - 25	right shadow hand base rotation
        26 - 45	left shadow hand actuated joint
        46 - 48	left shadow hand base translation
        49 - 51	left shadow hand base rotation

        Args:
            actions (tensor): Actions of agents in the all environment 
        """
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset_idx()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 6:26],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

            self.cur_targets[:, self.actuated_dof_indices + 24] = scale(self.actions[:, 32:52],
                                                                   self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices + 24] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices + 24] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices + 24] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices + 24],
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            # self.cur_targets[:, 49] = scale(self.actions[:, 0],
            #                                 self.object_dof_lower_limits[1], self.object_dof_upper_limits[1])
            # angle_offsets = self.actions[:, 26:32] * self.dt * self.orientation_scale

            self.apply_forces[:, 1, :] = actions[:, 0:3] * self.dt * self.transition_scale * 100000
            self.apply_forces[:, 1 + 26, :] = actions[:, 26:29] * self.dt * self.transition_scale * 100000
            self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000
            self.apply_torque[:, 1 + 26, :] = self.actions[:, 29:32] * self.dt * self.orientation_scale * 1000   

            self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces), gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]
        self.prev_targets[:, self.actuated_dof_indices + 24] = self.cur_targets[:, self.actuated_dof_indices + 24]

        # self.prev_targets[:, 49] = self.cur_targets[:, 49]
        # self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        all_hand_indices = torch.unique(torch.cat([self.hand_indices,
                                            self.another_hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                gymtorch.unwrap_tensor(self.prev_targets),
                                                gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))  

    def post_physics_step(self):
        """
        The post-processing of the physics step. Compute the observation and reward, and visualize auxiliary 
        lines for debug when needed
        
        """
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.block_right_handle_pos[i], self.block_right_handle_rot[i])
                self.add_debug_lines(self.envs[i], self.block_left_handle_pos[i], self.block_left_handle_rot[i])

                # self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

                # self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.right_hand_th_rot[i])


    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])

    def rand_row(self, tensor, dim_needed):  
        row_total = tensor.shape[0]
        return tensor[torch.randint(low=0, high=row_total, size=(dim_needed,)),:]

    def sample_points(self, points, sample_num=1000, sample_mathed='furthest'):
        eff_points = points[points[:, 2]>0.04]
        if eff_points.shape[0] < sample_num :
            eff_points = points
        if sample_mathed == 'random':
            sampled_points = self.rand_row(eff_points, sample_num)
        elif sample_mathed == 'furthest':
            sampled_points_id = pointnet2_utils.furthest_point_sample(eff_points.reshape(1, *eff_points.shape), sample_num)
            sampled_points = eff_points.index_select(0, sampled_points_id[0].long())
        return sampled_points

    def camera_visulization(self, is_depth_image=False):
        if is_depth_image:
            camera_depth_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_DEPTH)
            torch_depth_tensor = gymtorch.wrap_tensor(camera_depth_tensor)
            torch_depth_tensor = torch.clamp(torch_depth_tensor, -1, 1)
            torch_depth_tensor = scale(torch_depth_tensor, to_torch([0], dtype=torch.float, device=self.device),
                                                         to_torch([256], dtype=torch.float, device=self.device))
            camera_image = torch_depth_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        else:
            camera_rgba_tensor = self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[0], self.cameras[0], gymapi.IMAGE_COLOR)
            torch_rgba_tensor = gymtorch.wrap_tensor(camera_rgba_tensor)
            camera_image = torch_rgba_tensor.cpu().numpy()
            camera_image = Im.fromarray(camera_image)
        
        return camera_image

@torch.jit.script
def depth_image_to_point_cloud_GPU(camera_tensor, camera_view_matrix_inv, camera_proj_matrix, u, v, width:float, height:float, depth_bar:float, device:torch.device):
    # time1 = time.time()
    depth_buffer = camera_tensor.to(device)

    # Get the camera view matrix and invert it to transform points from camera to world space
    vinv = camera_view_matrix_inv

    # Get the camera projection matrix and get the necessary scaling
    # coefficients for deprojection
    
    proj = camera_proj_matrix
    fu = 2/proj[0, 0]
    fv = 2/proj[1, 1]

    centerU = width/2
    centerV = height/2

    Z = depth_buffer
    X = -(u-centerU)/width * Z * fu
    Y = (v-centerV)/height * Z * fv

    Z = Z.view(-1)
    valid = Z > -depth_bar
    X = X.view(-1)
    Y = Y.view(-1)

    position = torch.vstack((X, Y, Z, torch.ones(len(X), device=device)))[:, valid]
    position = position.permute(1, 0)
    position = position@vinv

    points = position[:, 0:3]

    return points

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

#####################################################################
###=========================jit functions=========================###
#####################################################################

@torch.jit.script
def compute_hand_reward(
    rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
    max_episode_length: float, object_pos, object_rot, target_pos, target_rot, block_right_handle_pos, block_left_handle_pos,
    left_hand_pos, right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
    left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos,
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, ignore_z_rot: bool
):
    """
    Compute the reward of all environment.

    Args:
        rew_buf (tensor): The reward buffer of all environments at this time

        reset_buf (tensor): The reset buffer of all environments at this time

        reset_goal_buf (tensor): The only-goal reset buffer of all environments at this time

        progress_buf (tensor): The porgress buffer of all environments at this time

        successes (tensor): The successes buffer of all environments at this time

        consecutive_successes (tensor): The consecutive successes buffer of all environments at this time

        max_episode_length (float): The max episode length in this environment

        object_pos (tensor): The position of the object

        object_rot (tensor): The rotation of the object

        target_pos (tensor): The position of the target

        target_rot (tensor): The rotate of the target

        block_right_handle_pos (tensor): The position of the right block

        block_left_handle_pos (tensor): The position of the left block

        left_hand_pos, right_hand_pos (tensor): The position of the bimanual hands
        
        right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos (tensor): The position of the five fingers 
            of the right hand

        left_hand_ff_pos, left_hand_mf_pos, left_hand_rf_pos, left_hand_lf_pos, left_hand_th_pos (tensor): The position of the five fingers 
            of the left hand

        dist_reward_scale (float): The scale of the distance reward

        rot_reward_scale (float): The scale of the rotation reward

        rot_eps (float): The epsilon of the rotation calculate

        actions (tensor): The action buffer of all environments at this time

        action_penalty_scale (float): The scale of the action penalty reward

        success_tolerance (float): The tolerance of the success determined

        reach_goal_bonus (float): The reward given when the object reaches the goal

        fall_dist (float): When the object is far from the Shadowhand, it is judged as falling

        fall_penalty (float): The reward given when the object is fell

        max_consecutive_successes (float): The maximum of the consecutive successes

        av_factor (float): The average factor for calculate the consecutive successes

        ignore_z_rot (bool): Is it necessary to ignore the rot of the z-axis, which is usually used 
            for some specific objects (e.g. pen)
    """
    
    # Distance from the hand to the object
    stack_pos1 = target_pos.clone()
    stack_pos2 = target_pos.clone()

    stack_pos1[:, 1] -= 0.1
    stack_pos2[:, 1] -= 0.1
    # stack_pos1[:, 2] += 0.025
    stack_pos1[:, 2] += 0.05

    goal_dist1 = torch.norm(stack_pos1 - block_left_handle_pos, p=2, dim=-1)
    goal_dist2 = torch.norm(stack_pos2 - block_right_handle_pos, p=2, dim=-1)
    # goal_dist = target_pos[:, 2] - object_pos[:, 2]

    right_hand_dist = torch.norm(block_right_handle_pos - right_hand_pos, p=2, dim=-1)
    left_hand_dist = torch.norm(block_left_handle_pos - left_hand_pos, p=2, dim=-1)

    right_hand_finger_dist = (torch.norm(block_right_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(block_right_handle_pos - right_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(block_right_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(block_right_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(block_right_handle_pos - right_hand_th_pos, p=2, dim=-1))
    left_hand_finger_dist = (torch.norm(block_left_handle_pos - left_hand_ff_pos, p=2, dim=-1) + torch.norm(block_left_handle_pos - left_hand_mf_pos, p=2, dim=-1)
                            + torch.norm(block_left_handle_pos - left_hand_rf_pos, p=2, dim=-1) + torch.norm(block_left_handle_pos - left_hand_lf_pos, p=2, dim=-1) 
                            + torch.norm(block_left_handle_pos - left_hand_th_pos, p=2, dim=-1))
    # Orientation alignment for the cube in hand and goal cube
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))

    right_hand_dist_rew = right_hand_finger_dist
    left_hand_dist_rew = left_hand_finger_dist

    # rot_rew = 1.0/(torch.abs(rot_dist) + rot_eps) * rot_reward_scale

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # Total reward is: position distance + orientation alignment + action regularization + success bonus + fall penalty
    # reward = torch.exp(-0.05*(up_rew * dist_reward_scale)) + torch.exp(-0.05*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.05*(left_hand_dist_rew * dist_reward_scale))
    up_rew = torch.zeros_like(right_hand_dist_rew)
    up_rew = torch.where(right_hand_finger_dist < 0.5,
                    torch.where(left_hand_finger_dist < 0.5,
                        (0.24 - goal_dist1 - goal_dist2) * 2, up_rew), up_rew)

    # up_rew = torch.where(right_hand_finger_dist < 0.75,
    #                 torch.where(left_hand_finger_dist < 0.75,
    #                     torch.where(goal_dist2 < 0.05,
    #                         (0.24 - goal_dist1 - goal_dist2) * 10, up_rew), up_rew), up_rew)

    # up_rew =  torch.where(right_hand_finger_dist <= 0.3, torch.norm(bottle_cap_up - bottle_pos, p=2, dim=-1) * 30, up_rew)
    stack_rew = torch.zeros_like(right_hand_dist_rew)
    stack_rew = torch.where(goal_dist2 < 0.07,
                    torch.where(goal_dist1 < 0.07,
                        (0.05-torch.abs(stack_pos1[:, 2] - block_left_handle_pos[:, 2])) * 50 ,stack_rew),stack_rew)

    # reward = torch.exp(-0.1*(right_hand_dist_rew * dist_reward_scale)) + torch.exp(-0.1*(left_hand_dist_rew * dist_reward_scale))
    reward = 1.5 - right_hand_dist_rew - left_hand_dist_rew + up_rew + stack_rew

    resets = torch.where(right_hand_dist_rew <= 0, torch.ones_like(reset_buf), reset_buf)
    resets = torch.where(right_hand_finger_dist >= 0.75, torch.ones_like(resets), resets)
    resets = torch.where(left_hand_finger_dist >= 0.75, torch.ones_like(resets), resets)

    # print(goal_dist1[0])
    # print(goal_dist2[0])
    # print("right_hand_dist_rew: ", right_hand_dist_rew[0])
    # print("left_hand_dist_rew: ", left_hand_dist_rew[0])
    # print("up_rew: ", up_rew[0])
    # print("block_rew: ", stack_rew[0])

    # Find out which envs hit the goal and update successes count
    successes = torch.where(successes == 0, 
                    torch.where(stack_rew > 1, torch.ones_like(successes), successes), successes)

    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = torch.zeros_like(resets)

    cons_successes = torch.where(resets > 0, successes * resets, consecutive_successes)
    # reward = successes 

    return reward, resets, goal_resets, progress_buf, successes, cons_successes