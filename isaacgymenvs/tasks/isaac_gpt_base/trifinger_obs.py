class TrifingerDimensions(enum.Enum):
    """Rest of the environment definition omitted."""
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


