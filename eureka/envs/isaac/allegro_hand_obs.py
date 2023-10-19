class AllegroHand(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
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


        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
             self.compute_full_state()
        else:
            print("Unknown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            self.states_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                      self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.states_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
            self.states_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 3*self.num_shadow_hand_dofs  # 48
            self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))


            fingertip_obs_start = goal_obs_start + 11  # 72

            obs_end = fingertip_obs_start #+ num_ft_states + num_ft_force_torques
            self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
        else:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                      self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
            self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 3*self.num_shadow_hand_dofs  # 48
            self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.obs_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.obs_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))


            fingertip_obs_start = goal_obs_start + 11  # 72

            obs_end = fingertip_obs_start #+ num_ft_states + num_ft_force_torques
            self.obs_buf[:, obs_end:obs_end + self.num_actions] = self.actions
