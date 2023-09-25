class TrifingerDimensions(enum.Enum):
    """
    Dimensions of the tri-finger robot.

    Note: While it may not seem necessary for tri-finger robot since it is fixed base, for floating
    base systems having this dimensions class is useful.
    """
    # general state
    # cartesian position + quaternion orientation
    PoseDim = 7,
    # linear velocity + angular velcoity
    VelocityDim = 6
    # state: pose + velocity
    StateDim = 13
    # force + torque
    WrenchDim = 6
    # for robot
    # number of fingers
    NumFingers = 3
    # for three fingers
    JointPositionDim = 9
    JointVelocityDim = 9
    JointTorqueDim = 9
    # generalized coordinates
    GeneralizedCoordinatesDim = JointPositionDim
    GeneralizedVelocityDim = JointVelocityDim
    # for objects
    ObjectPoseDim = 7
    ObjectVelocityDim = 6

class Trifinger(VecTask):
    """ Rest of the environment definition omitted. """
    # constants
    # directory where assets for the simulator are present
    _trifinger_assets_dir = os.path.join(project_dir, "../", "assets", "trifinger")
    # robot urdf (path relative to `_trifinger_assets_dir`)
    _robot_urdf_file = "robot_properties_fingers/urdf/pro/trifingerpro.urdf"
    # stage urdf (path relative to `_trifinger_assets_dir`)
    # _stage_urdf_file = "robot_properties_fingers/urdf/trifinger_stage.urdf"
    _table_urdf_file = "robot_properties_fingers/urdf/table_without_border.urdf"
    _boundary_urdf_file = "robot_properties_fingers/urdf/high_table_boundary.urdf"
    # object urdf (path relative to `_trifinger_assets_dir`)
    # TODO: Make object URDF configurable.
    _object_urdf_file = "objects/urdf/cube_multicolor_rrc.urdf"

    # physical dimensions of the object
    # TODO: Make object dimensions configurable.
    _object_dims = CuboidalObject(0.065)
    # dimensions of the system
    _dims = TrifingerDimensions
    # Constants for limits
    # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/trifinger_platform.py#L68
    # maximum joint torque (in N-m) applicable on each actuator
    _max_torque_Nm = 0.36
    # maximum joint velocity (in rad/s) on each actuator
    _max_velocity_radps = 10

    # History of state: Number of timesteps to save history for
    # Note: Currently used only to manage history of object and frame states.
    #       This can be extended to other observations (as done in ANYmal).
    _state_history_len = 2

    # buffers to store the simulation data
    # goal poses for the object [num. of instances, 7] where 7: (x, y, z, quat)
    _object_goal_poses_buf: torch.Tensor
    # DOF state of the system [num. of instances, num. of dof, 2] where last index: pos, vel
    _dof_state: torch.Tensor
    # Rigid body state of the system [num. of instances, num. of bodies, 13] where 13: (x, y, z, quat, v, omega)
    _rigid_body_state: torch.Tensor
    # Root prim states [num. of actors, 13] where 13: (x, y, z, quat, v, omega)
    _actors_root_state: torch.Tensor
    # Force-torque sensor array [num. of instances, num. of bodies * wrench]
    _ft_sensors_values: torch.Tensor
    # DOF position of the system [num. of instances, num. of dof]
    _dof_position: torch.Tensor
    # DOF velocity of the system [num. of instances, num. of dof]
    _dof_velocity: torch.Tensor
    # DOF torque of the system [num. of instances, num. of dof]
    _dof_torque: torch.Tensor
    # Fingertip links state list([num. of instances, num. of fingers, 13]) where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _fingertips_frames_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    # Object prim state [num. of instances, 13] where 13: (x, y, z, quat, v, omega)
    # The length of list is the history of the state: 0: t, 1: t-1, 2: t-2, ... step.
    _object_state_history: Deque[torch.Tensor] = deque(maxlen=_state_history_len)
    # stores the last action output
    _last_action: torch.Tensor
    # keeps track of the number of goal resets
    _successes: torch.Tensor
    # keeps track of number of consecutive successes
    _consecutive_successes: float

    _robot_limits: dict = {
        "joint_position": SimpleNamespace(
            # matches those on the real robot
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
        # used if we want to have joint stiffness/damping as parameters`
        "joint_stiffness": SimpleNamespace(
            low=np.array([1.0, 1.0, 1.0] * _dims.NumFingers.value, dtype=np.float32),
            high=np.array([50.0, 50.0, 50.0] * _dims.NumFingers.value, dtype=np.float32),
        ),
        "joint_damping": SimpleNamespace(
            low=np.array([0.01, 0.03, 0.0001] * _dims.NumFingers.value, dtype=np.float32),
            high=np.array([1.0, 3.0, 0.01] * _dims.NumFingers.value, dtype=np.float32),
        ),
    }
    # limits of the object (mapped later: str -> torch.tensor)
    _object_limits: dict = {
        "position": SimpleNamespace(
            low=np.array([-0.3, -0.3, 0], dtype=np.float32),
            high=np.array([0.3, 0.3, 0.3], dtype=np.float32),
            default=np.array([0, 0, _object_dims.min_height], dtype=np.float32)
        ),
        # difference between two positions
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
    # PD gains for the robot (mapped later: str -> torch.tensor)
    # Ref: https://github.com/rr-learning/rrc_simulation/blob/master/python/rrc_simulation/sim_finger.py#L49-L65
    _robot_dof_gains = {
        # The kp and kd gains of the PD control of the fingers.
        # Note: This depends on simulation step size and is set for a rate of 250 Hz.
        "stiffness": [10.0, 10.0, 10.0] * _dims.NumFingers.value,
        "damping": [0.1, 0.3, 0.001] * _dims.NumFingers.value,
        # The kd gains used for damping the joint motor velocities during the
        # safety torque check on the joint motors.
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
                # observations spec
                **self.obs_spec,
                # extra observations (added separately to make computations simpler)
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


        # define prims present in the scene
        prim_names = ["robot", "table", "boundary", "object", "goal_object"]
        # mapping from name to asset instance
        self.gym_assets = dict.fromkeys(prim_names)
        # mapping from name to gym indices
        self.gym_indices = dict.fromkeys(prim_names)
        # mapping from name to gym rigid body handles
        # name of finger tips links i.e. end-effector frames
        fingertips_frames = ["finger_tip_link_0", "finger_tip_link_120", "finger_tip_link_240"]
        self._fingertips_handles = OrderedDict.fromkeys(fingertips_frames, None)
        # mapping from name to gym dof index
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


        # change constant buffers from numpy/lists into torch tensors
        # limits for robot
        for limit_name in self._robot_limits:
            # extract limit simple-namespace
            limit_dict = self._robot_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)
        # limits for the object
        for limit_name in self._object_limits:
            # extract limit simple-namespace
            limit_dict = self._object_limits[limit_name].__dict__
            # iterate over namespace attributes
            for prop, value in limit_dict.items():
                limit_dict[prop] = torch.tensor(value, dtype=torch.float, device=self.device)
        # PD gains for actuation
        for gain_name, value in self._robot_dof_gains.items():
            self._robot_dof_gains[gain_name] = torch.tensor(value, dtype=torch.float, device=self.device)

        # store the sampled goal poses for the object: [num. of instances, 7]
        self._object_goal_poses_buf = torch.zeros((self.num_envs, 7), device=self.device, dtype=torch.float)
        # get force torque sensor if enabled
        if self.cfg["env"]["enable_ft_sensors"] or self.cfg["env"]["asymmetric_obs"]:
            # # joint torques
            # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            # self._dof_torque = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
            #                                                                self._dims.JointTorqueDim.value)
            # # force-torque sensor
            num_ft_dims = self._dims.NumFingers.value * self._dims.WrenchDim.value
            # sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            # self._ft_sensors_values = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, num_ft_dims)

            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self._ft_sensors_values = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, num_ft_dims)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self._dof_torque = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self._dims.JointTorqueDim.value)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        # refresh the buffer (to copy memory?)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # create wrapper tensors for reference (consider everything as pointer to actual memory)
        # DOF
        self._dof_state = gymtorch.wrap_tensor(dof_state_tensor).view(self.num_envs, -1, 2)
        self._dof_position = self._dof_state[..., 0]
        self._dof_velocity = self._dof_state[..., 1]
        # rigid body
        self._rigid_body_state = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        # root actors
        self._actors_root_state = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        # frames history
        action_dim = sum(self.action_spec.values())
        self._last_action = torch.zeros(self.num_envs, action_dim, dtype=torch.float, device=self.device)
        fingertip_handles_indices = list(self._fingertips_handles.values())
        object_indices = self.gym_indices["object"]
        # timestep 0 is current tensor
        curr_history_length = 0
        while curr_history_length < self._state_history_len:
            # add tensors to history list
            print(self._rigid_body_state.shape)
            self._fingertips_frames_state_history.append(self._rigid_body_state[:, fingertip_handles_indices])
            self._object_state_history.append(self._actors_root_state[object_indices])
            # update current history length
            curr_history_length += 1

        self._observations_scale = SimpleNamespace(low=None, high=None)
        self._states_scale = SimpleNamespace(low=None, high=None)
        self._action_scale = SimpleNamespace(low=None, high=None)

        self._successes = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._successes_pos = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._successes_quat = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        self.__configure_mdp_spaces()

    def compute_reward(self, actions):
        self.rew_buf[:] = 0.
        self.reset_buf[:] = 0.

        self.rew_buf[:], self.reset_buf[:], log_dict = compute_trifinger_reward(
            self.obs_buf,
            self.reset_buf,
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

        self.extras.update({"env/rewards/"+k: v.mean() for k, v in log_dict.items()})

    def compute_observations(self):
        # refresh memory buffers
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

        # extract frame handles
        fingertip_handles_indices = list(self._fingertips_handles.values())
        object_indices = self.gym_indices["object"]
        # update state histories
        self._fingertips_frames_state_history.appendleft(self._rigid_body_state[:, fingertip_handles_indices])
        self._object_state_history.appendleft(self._actors_root_state[object_indices])
        # fill the observations and states buffer

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

        # normalize observations if flag is enabled
        if self.cfg["env"]["normalize_obs"]:
            # for normal obs
            self.obs_buf = scale_transform(
                self.obs_buf,
                lower=self._observations_scale.low,
                upper=self._observations_scale.high
            )

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