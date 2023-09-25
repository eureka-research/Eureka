
import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from collections import OrderedDict

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

from isaacgymenvs.utils.torch_jit_utils import *
from isaacgymenvs.tasks.base.vec_task import VecTask

from types import SimpleNamespace
from collections import deque
from typing import Deque, Dict, Tuple, Union


import enum
import numpy as np



class TrifingerGPTDimensions(enum.Enum):
    PoseDim = 7,
    VelocityDim = 6
    StateDim = 13
    WrenchDim = 6
    NumFingers = 3
    JointPositionDim = 9
    JointVelocityDim = 9
    JointTorqueDim = 9
    GeneralizedCoordinatesDim = JointPositionDim
    GeneralizedVelocityDim = JointVelocityDim
    ObjectPoseDim = 7
    ObjectVelocityDim = 6



ARENA_RADIUS = 0.195


class CuboidalObject:
    radius_3d: float
    max_com_distance_to_center: float
    min_height: float
    max_height = 0.1

    NumKeypoints = 8
    ObjectPositionDim = 3
    KeypointsCoordsDim = NumKeypoints * ObjectPositionDim

    def __init__(self, size: Union[float, Tuple[float, float, float]]):
        if isinstance(size, float):
            self._size = (size, size, size)
        else:
            self._size = size
        self.__compute()


    @property
    def size(self) -> Tuple[float, float, float]:
        return self._size


    @size.setter
    def size(self, size: Union[float, Tuple[float, float, float]]):
        if isinstance(size, float):
            self._size = (size, size, size)
        else:
            self._size = size
        self.__compute()


    def __compute(self):
        max_len = max(self._size)
        self.radius_3d = max_len * np.sqrt(3) / 2
        self.max_com_distance_to_center = ARENA_RADIUS - self.radius_3d
        self.min_height = self._size[2] / 2


class TrifingerGPT(VecTask):

    _trifinger_assets_dir = os.path.join(project_dir, "../", "assets", "trifinger")
    _robot_urdf_file = "robot_properties_fingers/urdf/pro/trifingerpro.urdf"
    _table_urdf_file = "robot_properties_fingers/urdf/table_without_border.urdf"
    _boundary_urdf_file = "robot_properties_fingers/urdf/high_table_boundary.urdf"
    _object_urdf_file = "objects/urdf/cube_multicolor_rrc.urdf"

    _object_dims = CuboidalObject(0.065)
    _dims = TrifingerGPTDimensions
    _max_torque_Nm = 0.36
    _max_velocity_radps = 10

    _state_history_len = 2

    _object_goal_poses_buf: torch.Tensor
    _dof_state: torch.Tensor
    _rigid_body_state: torch.Tensor
    _actors_root_state: torch.Tensor
    _ft_sensors_values: torch.Tensor
    _dof_position: torch.Tensor
    _dof_velocity: torch.Tensor
    _dof_torque: torch.Tensor
    _fingertips_frames_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    _object_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    _last_action: torch.Tensor
    _successes: torch.Tensor
    _consecutive_successes: float

    _robot_limits: dict = {
        "joint_position": SimpleNamespace(
            low=np.array([-0.33, 0.0, -2.7] * _dims.NumFingers.value, dtype=np.float32),
            high=np.array([1.0, 1.57, 0.0] * _dims.NumFingers.value, dtype=np.float32),
            default=np.array([0.0, 0.9, -2.0] * _dims.NumFingers.value, dtype=np.float32),
        ),
        "joint_velocity": SimpleNamespace(
            low=np.full(_dims.JointVelocityDim.value, -_max_velocity_radps, dtype=np.float32),
            high=np.full(_dims.JointVelocityDim.value, _max_velocity_radps, dtype=np.float32),
            default=np.zeros(_dims.JointVelocityDim.value, dtype=np.float32),
        ),
        "joint_torque": SimpleNamespace(
            low=np.full(_dims.JointTorqueDim.value, -_max_torque_Nm, dtype=np.float32),
            high=np.full(_dims.JointTorqueDim.value, _max_torque_Nm, dtype=np.float32),
            default=np.zeros(_dims.JointTorqueDim.value, dtype=np.float32),
        ),
        "fingertip_position": SimpleNamespace(
            low=np.array([-0.4, -0.4, 0], dtype=np.float32),
            high=np.array([0.4, 0.4, 0.5], dtype=np.float32),
        ),
        "fingertip_orientation": SimpleNamespace(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
        ),
        "fingertip_velocity": SimpleNamespace(
            low=np.full(_dims.VelocityDim.value, -0.2, dtype=np.float32),
            high=np.full(_dims.VelocityDim.value, 0.2, dtype=np.float32),
        ),
        "fingertip_wrench": SimpleNamespace(
            low=np.full(_dims.WrenchDim.value, -1.0, dtype=np.float32),
            high=np.full(_dims.WrenchDim.value, 1.0, dtype=np.float32),
        ),
        "joint_stiffness": SimpleNamespace(
            low=np.array([1.0, 1.0, 1.0] * _dims.NumFingers.value, dtype=np.float32),
            high=np.array([50.0, 50.0, 50.0] * _dims.NumFingers.value, dtype=np.float32),
        ),
        "joint_damping": SimpleNamespace(
            low=np.array([0.01, 0.03, 0.0001] * _dims.NumFingers.value, dtype=np.float32),
            high=np.array([1.0, 3.0, 0.01] * _dims.NumFingers.value, dtype=np.float32),
        ),
    }
    _object_limits: dict = {
        "position": SimpleNamespace(
            low=np.array([-0.3, -0.3, 0], dtype=np.float32),
            high=np.array([0.3, 0.3, 0.3], dtype=np.float32),
            default=np.array([0, 0, _object_dims.min_height], dtype=np.float32)
        ),
        "position_delta": SimpleNamespace(
            low=np.array([-0.6, -0.6, 0], dtype=np.float32),
            high=np.array([0.6, 0.6, 0.3], dtype=np.float32),
            default=np.array([0, 0, 0], dtype=np.float32)
        ),
        "orientation": SimpleNamespace(
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
            default=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        ),
        "velocity": SimpleNamespace(
            low=np.full(_dims.VelocityDim.value, -0.5, dtype=np.float32),
            high=np.full(_dims.VelocityDim.value, 0.5, dtype=np.float32),
            default=np.zeros(_dims.VelocityDim.value, dtype=np.float32)
        ),
        "scale": SimpleNamespace(
            low=np.full(1, 0.0, dtype=np.float32),
            high=np.full(1, 1.0, dtype=np.float32),
        ),
    }
    _robot_dof_gains = {
        "stiffness": [10.0, 10.0, 10.0] * _dims.NumFingers.value,
        "damping": [0.1, 0.3, 0.001] * _dims.NumFingers.value,
        "safety_damping": [0.08, 0.08, 0.04] * _dims.NumFingers.value
    }
    action_dim = _dims.JointTorqueDim.value

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        self.obs_spec = {
            "robot_q": self._dims.GeneralizedCoordinatesDim.value,
            "robot_u": self._dims.GeneralizedVelocityDim.value,
            "object_q": self._dims.ObjectPoseDim.value,
            "object_q_des": self._dims.ObjectPoseDim.value,
            "command": self.action_dim
        }
        if self.cfg["env"]["asymmetric_obs"]:
            self.state_spec = {
                **self.obs_spec,
                "object_u": self._dims.ObjectVelocityDim.value,
                "fingertip_state": self._dims.NumFingers.value * self._dims.StateDim.value,
                "robot_a": self._dims.GeneralizedVelocityDim.value,
                "fingertip_wrench": self._dims.NumFingers.value * self._dims.WrenchDim.value,
            }
        else:
            self.state_spec = self.obs_spec

        self.action_spec = {
            "command":  self.action_dim
        }

        self.cfg["env"]["numObservations"] = sum(self.obs_spec.values())
        self.cfg["env"]["numStates"] = sum(self.state_spec.values())
        self.cfg["env"]["numActions"] = sum(self.action_spec.values())
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]


        prim_names = ["robot", "table", "boundary", "object", "goal_object"]
        self.gym_assets = dict.fromkeys(prim_names)
        self.gym_indices = dict.fromkeys(prim_names)
        fingertips_frames = ["finger_tip_link_0", "finger_tip_link_120", "finger_tip_link_240"]
        self._fingertips_handles = OrderedDict.fromkeys(fingertips_frames, None)
        robot_dof_names = list()
        for finger_pos in ['0', '120', '240']:
            robot_dof_names += [f'finger_base_to_upper_joint_{finger_pos}',
                                f'finger_upper_to_middle_joint_{finger_pos}',
                                f'finger_middle_to_lower_joint_{finger_pos}']
        self._robot_dof_indices = OrderedDict.fromkeys(robot_dof_names, None)

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(0.7, 0.0, 0.7)
            cam_target = gymapi.Vec3(0.0, 0.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)


        for limit_name in self._robot_limits:
            limit_dict = self._robot_limits[limit_name].__dict__
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)
        for limit_name in self._object_limits:
            limit_dict = self._object_limits[limit_name].__dict__
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)
        for gain_name, value in self._robot_dof_gains.items():
            self._robot_dof_gains[gain_name] = torch.tensor(value, dtype=torch.float, device=self.device)

        self._object_goal_poses_buf = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
        if self.cfg["env"]["enable_ft_sensors"] or self.cfg["env"]["asymmetric_obs"]:
            num_ft_dims = self._dims.NumFingers.value * self._dims.WrenchDim.value

            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self._ft_sensors_values = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, num_ft_dims)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self._dof_torque = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self._dims.JointTorqueDim.value)

        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2)
        self._dof_position = self._dof_state[..., 0]
        self._dof_velocity = self._dof_state[..., 1]
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self._actors_root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        action_dim = sum(self.action_spec.values())
        self._last_action = torch.zeros(self.num_envs, action_dim, dtype=torch.float, device=self.device)
        fingertip_handles_indices = list(self._fingertips_handles.values())
        object_indices = self.gym_indices["object"]
        curr_history_length = 0
        while curr_history_length < self._state_history_len:
            print(self._rigid_body_state.shape)
            self._fingertips_frames_state_history.append(self._rigid_body_state[:, fingertip_handles_indices])
            self._object_state_history.append(self._actors_root_state[object_indices])
            curr_history_length += 1

        self._observations_scale = SimpleNamespace(low=None, high=None)
        self._states_scale = SimpleNamespace(low=None, high=None)
        self._action_scale = SimpleNamespace(low=None, high=None)

        self._successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._successes_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._successes_quat = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)

        self.__configure_mdp_spaces()

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_scene_assets()
        self._create_envs(self.num_envs, self.cfg["env"]["envSpacing"], int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.distance = 0.013
        plane_params.static_friction = 1.0
        plane_params.dynamic_friction = 1.0
        self.gym.add_ground(self.sim, plane_params)

    def _create_scene_assets(self):
        self.gym_assets["robot"] = self.__define_robot_asset()
        self.gym_assets["table"] = self.__define_table_asset()
        self.gym_assets["boundary"] = self.__define_boundary_asset()
        self.gym_assets["object"] = self.__define_object_asset()
        self.gym_assets["goal_object"] = self.__define_goal_object_asset()
        print("TrifingerGPT Robot Asset: ")
        print(f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self.gym_assets["robot"])}')
        print(f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self.gym_assets["robot"])}')
        print(f'\t Number of dofs: {self.gym.get_asset_dof_count(self.gym_assets["robot"])}')
        print(f'\t Number of actuated dofs: {self._dims.JointTorqueDim.value}')
        print("TrifingerGPT Table Asset: ")
        print(f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self.gym_assets["table"])}')
        print(f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self.gym_assets["table"])}')
        print("TrifingerGPT Boundary Asset: ")
        print(f'\t Number of bodies: {self.gym.get_asset_rigid_body_count(self.gym_assets["boundary"])}')
        print(f'\t Number of shapes: {self.gym.get_asset_rigid_shape_count(self.gym_assets["boundary"])}')

    def _create_envs(self, num_envs, spacing, num_per_row):
        robot_dof_props = self.gym.get_asset_dof_properties(self.gym_assets["robot"])
        for k, dof_index in enumerate(self._robot_dof_indices.values()):
            robot_dof_props['driveMode'][dof_index] = gymapi.DOF_MODE_EFFORT
            robot_dof_props['stiffness'][dof_index] = 0.0
            robot_dof_props['damping'][dof_index] = 0.0
            robot_dof_props['effort'][dof_index] = self._max_torque_Nm
            robot_dof_props['velocity'][dof_index] = self._max_velocity_radps
            robot_dof_props['lower'][dof_index] = float(self._robot_limits["joint_position"].low[k])
            robot_dof_props['upper'][dof_index] = float(self._robot_limits["joint_position"].high[k])

        self.envs = []

        env_lower_bound = gymapi.Vec3(-self.cfg["env"]["envSpacing"], -self.cfg["env"]["envSpacing"], 0.0)
        env_upper_bound = gymapi.Vec3(self.cfg["env"]["envSpacing"], self.cfg["env"]["envSpacing"], self.cfg["env"]["envSpacing"])
        num_envs_per_row = int(np.sqrt(self.num_envs))
        for asset_name in self.gym_indices.keys():
            self.gym_indices[asset_name] = list()
        max_agg_bodies = 0
        max_agg_shapes = 0
        for asset in self.gym_assets.values():
            max_agg_bodies += self.gym.get_asset_rigid_body_count(asset)
            max_agg_shapes += self.gym.get_asset_rigid_shape_count(asset)
        for env_index in range(self.num_envs):
            env_ptr = self.gym.create_env(self.sim, env_lower_bound, env_upper_bound, num_envs_per_row)
            if self.cfg["env"]["aggregate_mode"]:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            trifinger_actor = self.gym.create_actor(env_ptr, self.gym_assets["robot"], gymapi.Transform(),
                                                     "robot", env_index, 0, 0)
            trifinger_idx = self.gym.get_actor_index(env_ptr, trifinger_actor, gymapi.DOMAIN_SIM)

            table_handle = self.gym.create_actor(env_ptr, self.gym_assets["table"], gymapi.Transform(),
                                                  "table", env_index, 1, 0)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)

            boundary_handle = self.gym.create_actor(env_ptr, self.gym_assets["boundary"], gymapi.Transform(),
                                                     "boundary", env_index, 1, 0)
            boundary_idx = self.gym.get_actor_index(env_ptr, boundary_handle, gymapi.DOMAIN_SIM)

            object_handle = self.gym.create_actor(env_ptr, self.gym_assets["object"], gymapi.Transform(),
                                                   "object", env_index, 0, 0)
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            goal_handle = self.gym.create_actor(env_ptr, self.gym_assets["goal_object"], gymapi.Transform(),
                                                 "goal_object", env_index + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.gym.set_actor_dof_properties(env_ptr, trifinger_actor, robot_dof_props)
            stage_color = gymapi.Vec3(0.73, 0.68, 0.72)
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, stage_color)
            self.gym.set_rigid_body_color(env_ptr, boundary_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, stage_color)
            if self.cfg["env"]["aggregate_mode"]:
                self.gym.end_aggregate(env_ptr)
            self.envs.append(env_ptr)
            self.gym_indices["robot"].append(trifinger_idx)
            self.gym_indices["table"].append(table_idx)
            self.gym_indices["boundary"].append(boundary_idx)
            self.gym_indices["object"].append(object_idx)
            self.gym_indices["goal_object"].append(goal_object_idx)
        for asset_name, asset_indices in self.gym_indices.items():
            self.gym_indices[asset_name] = torch.tensor(asset_indices, dtype=torch.long, device=self.device)

    def __configure_mdp_spaces(self):
        if self.cfg["env"]["command_mode"] == "position":
            self._action_scale.low = self._robot_limits["joint_position"].low
            self._action_scale.high = self._robot_limits["joint_position"].high
        elif self.cfg["env"]["command_mode"] == "torque":
            self._action_scale.low = self._robot_limits["joint_torque"].low
            self._action_scale.high = self._robot_limits["joint_torque"].high
        else:
            msg = f"Invalid command mode. Input: {self.cfg['env']['command_mode']} not in ['torque', 'position']."
            raise ValueError(msg)

        if self.cfg["env"]["normalize_action"]:
            obs_action_scale = SimpleNamespace(
                low=torch.full((self.action_dim,), -1, dtype=torch.float, device=self.device),
                high=torch.full((self.action_dim,), 1, dtype=torch.float, device=self.device)
            )
        else:
            obs_action_scale = self._action_scale

        object_obs_low = torch.cat([
                                       self._object_limits["position"].low,
                                       self._object_limits["orientation"].low,
                                   ]*2)
        object_obs_high = torch.cat([
                                        self._object_limits["position"].high,
                                        self._object_limits["orientation"].high,
                                    ]*2)

        self._observations_scale.low = torch.cat([
            self._robot_limits["joint_position"].low,
            self._robot_limits["joint_velocity"].low,
            object_obs_low,
            obs_action_scale.low
        ])
        self._observations_scale.high = torch.cat([
            self._robot_limits["joint_position"].high,
            self._robot_limits["joint_velocity"].high,
            object_obs_high,
            obs_action_scale.high
        ])
        if self.cfg["env"]["asymmetric_obs"]:
            fingertip_state_scale = SimpleNamespace(
                low=torch.cat([
                    self._robot_limits["fingertip_position"].low,
                    self._robot_limits["fingertip_orientation"].low,
                    self._robot_limits["fingertip_velocity"].low,
                ]),
                high=torch.cat([
                    self._robot_limits["fingertip_position"].high,
                    self._robot_limits["fingertip_orientation"].high,
                    self._robot_limits["fingertip_velocity"].high,
                ])
            )
            states_low = [
                self._observations_scale.low,
                self._object_limits["velocity"].low,
                fingertip_state_scale.low.repeat(self._dims.NumFingers.value),
                self._robot_limits["joint_torque"].low,
                self._robot_limits["fingertip_wrench"].low.repeat(self._dims.NumFingers.value),
            ]
            states_high = [
                self._observations_scale.high,
                self._object_limits["velocity"].high,
                fingertip_state_scale.high.repeat(self._dims.NumFingers.value),
                self._robot_limits["joint_torque"].high,
                self._robot_limits["fingertip_wrench"].high.repeat(self._dims.NumFingers.value),
            ]
            self._states_scale.low = torch.cat(states_low)
            self._states_scale.high = torch.cat(states_high)
        state_dim = sum(self.state_spec.values())
        obs_dim = sum(self.obs_spec.values())
        action_dim = sum(self.action_spec.values())
        if self._observations_scale.low.shape[0] != obs_dim or self._observations_scale.high.shape[0] != obs_dim:
            msg = f"Observation scaling dimensions mismatch. " \
                  f"\tLow: {self._observations_scale.low.shape[0]}, " \
                  f"\tHigh: {self._observations_scale.high.shape[0]}, " \
                  f"\tExpected: {obs_dim}."
            raise AssertionError(msg)
        if self.cfg["env"]["asymmetric_obs"] \
                and (self._states_scale.low.shape[0] != state_dim or self._states_scale.high.shape[0] != state_dim):
            msg = f"States scaling dimensions mismatch. " \
                  f"\tLow: {self._states_scale.low.shape[0]}, " \
                  f"\tHigh: {self._states_scale.high.shape[0]}, " \
                  f"\tExpected: {state_dim}."
            raise AssertionError(msg)
        if self._action_scale.low.shape[0] != action_dim or self._action_scale.high.shape[0] != action_dim:
            msg = f"Actions scaling dimensions mismatch. " \
                  f"\tLow: {self._action_scale.low.shape[0]}, " \
                  f"\tHigh: {self._action_scale.high.shape[0]}, " \
                  f"\tExpected: {action_dim}."
            raise AssertionError(msg)
        print(f'MDP Raw observation bounds\n'
                   f'\tLow: {self._observations_scale.low}\n'
                   f'\tHigh: {self._observations_scale.high}')
        print(f'MDP Raw state bounds\n'
                   f'\tLow: {self._states_scale.low}\n'
                   f'\tHigh: {self._states_scale.high}')
        print(f'MDP Raw action bounds\n'
                   f'\tLow: {self._action_scale.low}\n'
                   f'\tHigh: {self._action_scale.high}')

    def compute_reward(self, actions):
        self.rew_buf[:], self.rew_dict = compute_reward(self.object_state, self.object_goal_poses) 
        self.extras['gpt_reward'] = self.rew_buf.mean() 
        for rew_state in self.rew_dict: self.extras[rew_state] = self.rew_dict[rew_state].mean()
        self.rew_buf[:] = 0.
        self.reset_buf[:] = 0.

        self.gt_rew_buf, self.reset_buf[:], log_dict, self.consecutive_successes[:] = compute_success(
            self.obs_buf,
            self.reset_buf,
            self.consecutive_successes,
            self.progress_buf,
            self.max_episode_length,
            self.cfg["sim"]["dt"],
            self.cfg["env"]["reward_terms"]["finger_move_penalty"]["weight"],
            self.cfg["env"]["reward_terms"]["finger_reach_object_rate"]["weight"],
            self.cfg["env"]["reward_terms"]["object_dist"]["weight"],
            self.cfg["env"]["reward_terms"]["object_rot"]["weight"],
            self.env_steps_count,
            self._object_goal_poses_buf,
            self._object_state_history[0],
            self._object_state_history[1],
            self._fingertips_frames_state_history[0],
            self._fingertips_frames_state_history[1],
            self.cfg["env"]["reward_terms"]["keypoints_dist"]["activate"]
        )
        self.extras['gt_reward'] = self.gt_rew_buf.mean()
        self.extras.update({"env/rewards/"+k: v.mean() for k, v in log_dict.items()})

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.cfg["env"]["enable_ft_sensors"] or self.cfg["env"]["asymmetric_obs"]:
            self.gym.refresh_dof_force_tensor(self.sim)
            self.gym.refresh_force_sensor_tensor(self.sim)
            joint_torques = self._dof_torque
            tip_wrenches = self._ft_sensors_values

        else:
            joint_torques = torch.zeros(self.num_envs, self._dims.JointTorqueDim.value, dtype=torch.float32, device=self.device)
            tip_wrenches = torch.zeros(self.num_envs, self._dims.NumFingers.value * self._dims.WrenchDim.value, dtype=torch.float32, device=self.device)

        fingertip_handles_indices = list(self._fingertips_handles.values())
        object_indices = self.gym_indices["object"]
        self._fingertips_frames_state_history.appendleft(self._rigid_body_state[:, fingertip_handles_indices])
        self._object_state_history.appendleft(self._actors_root_state[object_indices])
        self.object_state = self._object_state_history[0]
        self.object_goal_poses = self._object_goal_poses_buf

        self.obs_buf[:], self.states_buf[:] = compute_trifinger_observations_states(
            self.cfg["env"]["asymmetric_obs"],
            self._dof_position,
            self._dof_velocity,
            self._object_state_history[0],
            self._object_goal_poses_buf,
            self.actions,
            self._fingertips_frames_state_history[0],
            joint_torques,
            tip_wrenches,
        )

        if self.cfg["env"]["normalize_obs"]:
            self.obs_buf = scale_transform(
                self.obs_buf,
                lower=self._observations_scale.low,
                upper=self._observations_scale.high
            )

    def reset_idx(self, env_ids):

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0
        self._successes[env_ids] = 0
        self._successes_pos[env_ids] = 0
        self._successes_quat[env_ids] = 0
        robot_initial_state_config = self.cfg["env"]["reset_distribution"]["robot_initial_state"]
        self._sample_robot_state(
            env_ids,
            distribution=robot_initial_state_config["type"],
            dof_pos_stddev=robot_initial_state_config["dof_pos_stddev"],
            dof_vel_stddev=robot_initial_state_config["dof_vel_stddev"]
        )
        object_initial_state_config = self.cfg["env"]["reset_distribution"]["object_initial_state"]
        self._sample_object_poses(
            env_ids,
            distribution=object_initial_state_config["type"],
        )
        self._sample_object_goal_poses(
            env_ids,
            difficulty=self.cfg["env"]["task_difficulty"]
        )
        robot_indices = self.gym_indices["robot"][env_ids].to(torch.int32)
        object_indices = self.gym_indices["object"][env_ids].to(torch.int32)
        goal_object_indices = self.gym_indices["goal_object"][env_ids].to(torch.int32)
        all_indices = torch.unique(torch.cat([robot_indices, object_indices, goal_object_indices]))
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._dof_state),
                                               gymtorch.unwrap_tensor(robot_indices), len(robot_indices))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._actors_root_state),
                                                      gymtorch.unwrap_tensor(all_indices), len(all_indices))

    def _sample_robot_state(self, instances: torch.Tensor, distribution: str = 'default',
                             dof_pos_stddev: float = 0.0, dof_vel_stddev: float = 0.0):
        num_samples = instances.size()[0]
        if distribution == "none":
            return
        elif distribution == "default":
            self._dof_position[instances] = self._robot_limits["joint_position"].default
            self._dof_velocity[instances] = self._robot_limits["joint_velocity"].default
        elif distribution == "random":
            dof_state_dim = self._dims.JointPositionDim.value + self._dims.JointVelocityDim.value
            dof_state_noise = 2 * torch.rand((num_samples, dof_state_dim,), dtype=torch.float,
                                             device=self.device) - 1
            self._dof_position[instances] = self._robot_limits["joint_position"].default
            self._dof_velocity[instances] = self._robot_limits["joint_velocity"].default
            start_offset = 0
            end_offset = self._dims.JointPositionDim.value
            self._dof_position[instances] += dof_pos_stddev * dof_state_noise[:, start_offset:end_offset]
            start_offset = end_offset
            end_offset += self._dims.JointVelocityDim.value
            self._dof_velocity[instances] += dof_vel_stddev * dof_state_noise[:, start_offset:end_offset]
        else:
            msg = f"Invalid robot initial state distribution. Input: {distribution} not in [`default`, `random`]."
            raise ValueError(msg)
        for idx in range(1, self._state_history_len):
            self._fingertips_frames_state_history[idx][instances] = 0.0

    def _sample_object_poses(self, instances: torch.Tensor, distribution: str):
        num_samples = instances.size()[0]
        if distribution == "none":
            return
        elif distribution == "default":
            pos_x, pos_y, pos_z = self._object_limits["position"].default
            orientation = self._object_limits["orientation"].default
        elif distribution == "random":
            pos_x, pos_y = random_xy(num_samples, self._object_dims.max_com_distance_to_center, self.device)
            pos_z = self._object_dims.size[2] / 2 + 0.0015
            orientation = random_yaw_orientation(num_samples, self.device)
        else:
            msg = f"Invalid object initial state distribution. Input: {distribution} " \
                  "not in [`default`, `random`, `none`]."
            raise ValueError(msg)
        object_indices = self.gym_indices["object"][instances]
        self._object_state_history[0][instances, 0] = pos_x
        self._object_state_history[0][instances, 1] = pos_y
        self._object_state_history[0][instances, 2] = pos_z
        self._object_state_history[0][instances, 3:7] = orientation
        self._object_state_history[0][instances, 7:13] = 0
        for idx in range(1, self._state_history_len):
            self._object_state_history[idx][instances] = 0.0
        self._actors_root_state[object_indices] = self._object_state_history[0][instances]

    def _sample_object_goal_poses(self, instances: torch.Tensor, difficulty: int):
        num_samples = instances.size()[0]
        if difficulty == -1:
            pos_x, pos_y = random_xy(num_samples, self._object_dims.max_com_distance_to_center, self.device)
            pos_z = self._object_dims.size[2] / 2
            orientation = random_yaw_orientation(num_samples, self.device)
        elif difficulty == 1:
            pos_x, pos_y = random_xy(num_samples, self._object_dims.max_com_distance_to_center, self.device)
            pos_z = self._object_dims.size[2] / 2
            orientation = default_orientation(num_samples, self.device)
        elif difficulty == 2:
            pos_x, pos_y = 0.0, 0.0
            pos_z = self._object_dims.min_height + 0.05
            orientation = default_orientation(num_samples, self.device)
        elif difficulty == 3:
            pos_x, pos_y = random_xy(num_samples, self._object_dims.max_com_distance_to_center, self.device)
            pos_z = random_z(num_samples, self._object_dims.min_height, self._object_dims.max_height, self.device)
            orientation = default_orientation(num_samples, self.device)
        elif difficulty == 4:
            max_goal_radius = self._object_dims.max_com_distance_to_center
            max_height = self._object_dims.max_height
            orientation = random_orientation(num_samples, self.device)

            pos_x, pos_y = random_xy(num_samples, max_goal_radius, self.device)
            pos_z = random_z(num_samples, self._object_dims.radius_3d, max_height, self.device)
        else:
            msg = f"Invalid difficulty index for task: {difficulty}."
            raise ValueError(msg)

        goal_object_indices = self.gym_indices["goal_object"][instances]
        self._object_goal_poses_buf[instances, 0] = pos_x
        self._object_goal_poses_buf[instances, 1] = pos_y
        self._object_goal_poses_buf[instances, 2] = pos_z
        self._object_goal_poses_buf[instances, 3:7] = orientation
        self._actors_root_state[goal_object_indices, 0:7] = self._object_goal_poses_buf[instances]

    def pre_physics_step(self, actions):

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        if len(env_ids) > 0:
            self.reset_idx(env_ids)
            self.gym.simulate(self.sim)

        self.actions = actions.clone().to(self.device)

        if self.cfg["env"]["normalize_action"]:
            action_transformed = unscale_transform(
                self.actions,
                lower=self._action_scale.low,
                upper=self._action_scale.high
            )
        else:
            action_transformed = self.actions

        if self.cfg["env"]["command_mode"] == 'torque':
            computed_torque = action_transformed
        elif self.cfg["env"]["command_mode"] == 'position':
            desired_dof_position = action_transformed
            computed_torque = self._robot_dof_gains["stiffness"] * (desired_dof_position - self._dof_position)
            computed_torque -= self._robot_dof_gains["damping"] * self._dof_velocity
        else:
            msg = f"Invalid command mode. Input: {self.cfg['env']['command_mode']} not in ['torque', 'position']."
            raise ValueError(msg)
        applied_torque = saturate(
            computed_torque,
            lower=self._robot_limits["joint_torque"].low,
            upper=self._robot_limits["joint_torque"].high
        )
        if self.cfg["env"]["apply_safety_damping"]:
            applied_torque -= self._robot_dof_gains["safety_damping"] * self._dof_velocity
            applied_torque = saturate(
                applied_torque,
                lower=self._robot_limits["joint_torque"].low,
                upper=self._robot_limits["joint_torque"].high
            )
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(applied_torque))

    def post_physics_step(self):

        self._step_info = {}

        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        self._check_termination()
        self.extras['consecutive_successes'] = self._successes.float().mean()

        if torch.sum(self.reset_buf) > 0:
            self._step_info['consecutive_successes'] = np.mean(self._successes.float().cpu().numpy())
            self._step_info['consecutive_successes_pos'] = np.mean(self._successes_pos.float().cpu().numpy())
            self._step_info['consecutive_successes_quat'] = np.mean(self._successes_quat.float().cpu().numpy())

    def _check_termination(self):
        termination_config = self.cfg["env"]["termination_conditions"]
        object_goal_position_dist = torch.norm(
            self._object_goal_poses_buf[:, 0:3] - self._object_state_history[0][:, 0:3],
            p=2, dim=-1
        )
        goal_position_reset = torch.le(object_goal_position_dist,
                                       termination_config["success"]["position_tolerance"])
        self._step_info['env/current_position_goal/per_env'] = np.mean(goal_position_reset.float().cpu().numpy())
        object_goal_orientation_dist = quat_diff_rad(self._object_state_history[0][:, 3:7],
                                                     self._object_goal_poses_buf[:, 3:7])
        goal_orientation_reset = torch.le(object_goal_orientation_dist,
                                          termination_config["success"]["orientation_tolerance"])
        self._step_info['env/current_orientation_goal/per_env'] =  np.mean(goal_orientation_reset.float().cpu().numpy())

        if self.cfg["env"]['task_difficulty'] < 4:
            task_completion_reset = goal_position_reset
        elif self.cfg["env"]['task_difficulty'] == 4:
            task_completion_reset = torch.logical_and(goal_position_reset, goal_orientation_reset)
        else:
            task_completion_reset = goal_orientation_reset
        self._successes = task_completion_reset
        self._successes_pos = goal_position_reset
        self._successes_quat = goal_orientation_reset



    def __define_robot_asset(self):
        robot_asset_options = gymapi.AssetOptions()
        robot_asset_options.flip_visual_attachments = False
        robot_asset_options.fix_base_link = True
        robot_asset_options.collapse_fixed_joints = False
        robot_asset_options.disable_gravity = False
        robot_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        robot_asset_options.thickness = 0.001
        robot_asset_options.angular_damping = 0.01

        robot_asset_options.vhacd_enabled = True
        robot_asset_options.vhacd_params = gymapi.VhacdParams()
        robot_asset_options.vhacd_params.resolution = 100000
        robot_asset_options.vhacd_params.concavity = 0.0025
        robot_asset_options.vhacd_params.alpha = 0.04
        robot_asset_options.vhacd_params.beta = 1.0
        robot_asset_options.vhacd_params.convex_hull_downsampling = 4
        robot_asset_options.vhacd_params.max_num_vertices_per_ch = 256

        if self.physics_engine == gymapi.SIM_PHYSX:
            robot_asset_options.use_physx_armature = True
        trifinger_asset = self.gym.load_asset(self.sim, self._trifinger_assets_dir,
                                               self._robot_urdf_file, robot_asset_options)
        trifinger_props = self.gym.get_asset_rigid_shape_properties(trifinger_asset)
        for p in trifinger_props:
            p.friction = 1.0
            p.torsion_friction = 1.0
            p.restitution = 0.8
        self.gym.set_asset_rigid_shape_properties(trifinger_asset, trifinger_props)
        for frame_name in self._fingertips_handles.keys():
            self._fingertips_handles[frame_name] = self.gym.find_asset_rigid_body_index(trifinger_asset,
                                                                                         frame_name)
            if self._fingertips_handles[frame_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid handle received for frame: `{frame_name}`."
                print(msg)

        if self.cfg["env"]["enable_ft_sensors"] or self.cfg["env"]["asymmetric_obs"]:
            sensor_pose = gymapi.Transform()
            for fingertip_handle in self._fingertips_handles.values():
                self.gym.create_asset_force_sensor(trifinger_asset, fingertip_handle, sensor_pose)
        for dof_name in self._robot_dof_indices.keys():
            self._robot_dof_indices[dof_name] = self.gym.find_asset_dof_index(trifinger_asset, dof_name)
            if self._robot_dof_indices[dof_name] == gymapi.INVALID_HANDLE:
                msg = f"Invalid index received for DOF: `{dof_name}`."
                print(msg)
        return trifinger_asset

    def __define_table_asset(self):
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.disable_gravity = True
        table_asset_options.fix_base_link = True
        table_asset_options.thickness = 0.001

        table_asset = self.gym.load_asset(self.sim, self._trifinger_assets_dir,
                                           self._table_urdf_file, table_asset_options)
        table_props = self.gym.get_asset_rigid_shape_properties(table_asset)
        for p in table_props:
            p.friction = 0.1
            p.torsion_friction = 0.1
        self.gym.set_asset_rigid_shape_properties(table_asset, table_props)
        return table_asset

    def __define_boundary_asset(self):
        boundary_asset_options = gymapi.AssetOptions()
        boundary_asset_options.disable_gravity = True
        boundary_asset_options.fix_base_link = True
        boundary_asset_options.thickness = 0.001

        boundary_asset_options.vhacd_enabled = True
        boundary_asset_options.vhacd_params = gymapi.VhacdParams()
        boundary_asset_options.vhacd_params.resolution = 100000
        boundary_asset_options.vhacd_params.concavity = 0.0
        boundary_asset_options.vhacd_params.alpha = 0.04
        boundary_asset_options.vhacd_params.beta = 1.0
        boundary_asset_options.vhacd_params.max_num_vertices_per_ch = 1024

        boundary_asset = self.gym.load_asset(self.sim, self._trifinger_assets_dir,
                                              self._boundary_urdf_file, boundary_asset_options)
        boundary_props = self.gym.get_asset_rigid_shape_properties(boundary_asset)

        self.gym.set_asset_rigid_shape_properties(boundary_asset, boundary_props)
        return boundary_asset

    def __define_object_asset(self):
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.thickness = 0.001
        object_asset_options.flip_visual_attachments = True
        object_asset = self.gym.load_asset(self.sim, self._trifinger_assets_dir,
                                            self._object_urdf_file, object_asset_options)
        object_props = self.gym.get_asset_rigid_shape_properties(object_asset)
        for p in object_props:
            p.friction = 1.0
            p.torsion_friction = 0.001
            p.restitution = 0.0
        self.gym.set_asset_rigid_shape_properties(object_asset, object_props)
        return object_asset

    def __define_goal_object_asset(self):
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = True
        object_asset_options.fix_base_link = True
        object_asset_options.thickness = 0.001
        object_asset_options.flip_visual_attachments = True
        goal_object_asset = self.gym.load_asset(self.sim, self._trifinger_assets_dir,
                                                 self._object_urdf_file, object_asset_options)
        return goal_object_asset

    @property
    def env_steps_count(self) -> int:
        return self.gym.get_frame_count(self.sim) * self.num_envs

@torch.jit.script
def lgsk_kernel(x: torch.Tensor, scale: float = 50.0, eps:float=2) -> torch.Tensor:
    scaled = x * scale
    return 1.0 / (scaled.exp() + eps + (-scaled).exp())

@torch.jit.script
def gen_keypoints(pose: torch.Tensor, num_keypoints: int = 8, size: Tuple[float, float, float] = (0.065, 0.065, 0.065)):

    num_envs = pose.shape[0]

    keypoints_buf = torch.ones(num_envs, num_keypoints, 3, dtype=torch.float32, device=pose.device)

    for i in range(num_keypoints):
        n = [((i >> k) & 1) == 0 for k in range(3)]
        corner_loc = [(1 if n[k] else -1) * s / 2 for k, s in enumerate(size)],
        corner = torch.tensor(corner_loc, dtype=torch.float32, device=pose.device) * keypoints_buf[:, i, :]
        keypoints_buf[:, i, :] = local_to_world_space(corner, pose)
    return keypoints_buf

@torch.jit.script
def compute_trifinger_observations_states(
        asymmetric_obs: bool,
        dof_position: torch.Tensor,
        dof_velocity: torch.Tensor,
        object_state: torch.Tensor,
        object_goal_poses: torch.Tensor,
        actions: torch.Tensor,
        fingertip_state: torch.Tensor,
        joint_torques: torch.Tensor,
        tip_wrenches: torch.Tensor
):

    num_envs = dof_position.shape[0]

    obs_buf = torch.cat([
        dof_position,
        dof_velocity,
        object_state[:, 0:7], # pose
        object_goal_poses,
        actions
    ], dim=-1)

    if asymmetric_obs:
        states_buf = torch.cat([
            obs_buf,
            object_state[:, 7:13], # linear / angular velocity
            fingertip_state.reshape(num_envs, -1),
            joint_torques,
            tip_wrenches
        ], dim=-1)
    else:
        states_buf = obs_buf

    return obs_buf, states_buf


@torch.jit.script
def random_xy(num: int, max_com_distance_to_center: float, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    radius = torch.sqrt(torch.rand(num, dtype=torch.float, device=device))
    radius *= max_com_distance_to_center
    theta = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)
    x = radius * torch.cos(theta)
    y = radius * torch.sin(theta)

    return x, y


@torch.jit.script
def random_z(num: int, min_height: float, max_height: float, device: str) -> torch.Tensor:
    z = torch.rand(num, dtype=torch.float, device=device)
    z = (max_height - min_height) * z + min_height

    return z


@torch.jit.script
def default_orientation(num: int, device: str) -> torch.Tensor:
    quat = torch.zeros((num, 4,), dtype=torch.float, device=device)
    quat[..., -1] = 1.0

    return quat


@torch.jit.script
def random_orientation(num: int, device: str) -> torch.Tensor:
    quat = torch.randn((num, 4,), dtype=torch.float, device=device)
    quat = torch.nn.functional.normalize(quat, p=2., dim=-1, eps=1e-12)

    return quat

@torch.jit.script
def random_orientation_within_angle(num: int, device:str, base: torch.Tensor, max_angle: float):
    quat = torch.zeros((num, 4,), dtype=torch.float, device=device)

    rand = torch.rand((num, 3), dtype=torch.float, device=device)

    c = torch.cos(rand[:, 0]*max_angle)
    n = torch.sqrt((1.-c)/2.)

    quat[:, 3] = torch.sqrt((1+c)/2.)
    quat[:, 2] = (rand[:, 1]*2.-1.) * n
    quat[:, 0] = (torch.sqrt(1-quat[:, 2]**2.) * torch.cos(2*np.pi*rand[:, 2])) * n
    quat[:, 1] = (torch.sqrt(1-quat[:, 2]**2.) * torch.sin(2*np.pi*rand[:, 2])) * n

    quat = torch.nn.functional.normalize(quat, p=2., dim=-1, eps=1e-12)

    return quat_mul(quat, base)


@torch.jit.script
def random_angular_vel(num: int, device: str, magnitude_stdev: float) -> torch.Tensor:
    axis = torch.randn((num, 3,), dtype=torch.float, device=device)
    axis /= torch.norm(axis, p=2, dim=-1).view(-1, 1)
    magnitude = torch.randn((num, 1,), dtype=torch.float, device=device)
    magnitude *= magnitude_stdev
    return magnitude * axis

@torch.jit.script
def random_yaw_orientation(num: int, device: str) -> torch.Tensor:
    roll = torch.zeros(num, dtype=torch.float, device=device)
    pitch = torch.zeros(num, dtype=torch.float, device=device)
    yaw = 2 * np.pi * torch.rand(num, dtype=torch.float, device=device)

    return quat_from_euler_xyz(roll, pitch, yaw)



@torch.jit.script
def compute_success(
        obs_buf: torch.Tensor,
        reset_buf: torch.Tensor,
        consecutive_successes: torch.Tensor,
        progress_buf: torch.Tensor,
        episode_length: int,
        dt: float,
        finger_move_penalty_weight: float,
        finger_reach_object_weight: float,
        object_dist_weight: float,
        object_rot_weight: float,
        env_steps_count: int,
        object_goal_poses_buf: torch.Tensor,
        object_state: torch.Tensor,
        last_object_state: torch.Tensor,
        fingertip_state: torch.Tensor,
        last_fingertip_state: torch.Tensor,
        use_keypoints: bool
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:

    ft_sched_start = 0
    ft_sched_end = 5e7


    fingertip_vel = (fingertip_state[:, :, 0:3] - last_fingertip_state[:, :, 0:3]) / dt

    finger_movement_penalty = finger_move_penalty_weight * fingertip_vel.pow(2).view(-1, 9).sum(dim=-1)


    curr_norms = torch.stack([
        torch.norm(fingertip_state[:, i, 0:3] - object_state[:, 0:3], p=2, dim=-1)
        for i in range(3)
    ], dim=-1)
    prev_norms = torch.stack([
        torch.norm(last_fingertip_state[:, i, 0:3] - last_object_state[:, 0:3], p=2, dim=-1)
        for i in range(3)
    ], dim=-1)

    ft_sched_val = 1.0 if ft_sched_start <= env_steps_count <= ft_sched_end else 0.0
    finger_reach_object_reward = finger_reach_object_weight * ft_sched_val * (curr_norms - prev_norms).sum(dim=-1)

    if use_keypoints:
        object_keypoints = gen_keypoints(object_state[:, 0:7])
        goal_keypoints = gen_keypoints(object_goal_poses_buf[:, 0:7])

        delta = object_keypoints - goal_keypoints

        dist_l2 = torch.norm(delta, p=2, dim=-1)

        keypoints_kernel_sum = lgsk_kernel(dist_l2, scale=30., eps=2.).mean(dim=-1)

        pose_reward = object_dist_weight * dt * keypoints_kernel_sum

    else:

        object_dist = torch.norm(object_state[:, 0:3] - object_goal_poses_buf[:, 0:3], p=2, dim=-1)
        object_dist_reward = object_dist_weight * dt * lgsk_kernel(object_dist, scale=50., eps=2.)


        quat_a = object_state[:, 3:7]
        quat_b = object_goal_poses_buf[:, 3:7]

        angles = quat_diff_rad(quat_a, quat_b)
        object_rot_reward =  object_rot_weight * dt / (3. * torch.abs(angles) + 0.01)

        pose_reward = object_dist_reward + object_rot_reward

    total_reward = (
            finger_movement_penalty
            + finger_reach_object_reward
            + pose_reward
    )

    reset = torch.zeros_like(reset_buf)
    reset = torch.where(progress_buf >= episode_length - 1, torch.ones_like(reset_buf), reset)

    info: Dict[str, torch.Tensor] = {
        'finger_movement_penalty': finger_movement_penalty,
        'finger_reach_object_reward': finger_reach_object_reward,
        'pose_reward': finger_reach_object_reward,
        'reward': total_reward,
    }

    return total_reward, reset, info, consecutive_successes

from typing import Tuple, Dict
import math
import torch
from torch import Tensor
@torch.jit.script
def compute_reward(object_state: torch.Tensor, object_goal_poses: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    # Extract object position and goal position
    object_pos = object_state[:, :3]
    goal_pos = object_goal_poses[:, :3]

    # Calculate Euclidean distance between object and goal positions
    distance = torch.norm(object_pos - goal_pos, dim=-1)

    # Define temperature parameter for transforming distance-based reward
    distance_temperature = 1.0

    # Transform distance-based reward using exponential function with temperature
    distance_reward = -torch.exp(-distance_temperature * distance)

    # Calculate total reward
    total_reward = distance_reward

    # Store individual reward components in a dictionary
    reward_components = {
        "distance_reward": distance_reward,
    }

    return total_reward, reward_components
