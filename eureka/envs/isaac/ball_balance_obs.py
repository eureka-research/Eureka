class BallBalance(VecTask):
    """Rest of the environment definition omitted."""
    def compute_observations(self):

        actuated_dof_indices = torch.tensor([1, 3, 5], device=self.device)

        self.obs_buf[..., 0:3] = self.dof_positions[..., actuated_dof_indices]
        self.obs_buf[..., 3:6] = self.dof_velocities[..., actuated_dof_indices]
        self.obs_buf[..., 6:9] = self.ball_positions
        self.obs_buf[..., 9:12] = self.ball_linvels
        self.obs_buf[..., 12:15] = self.sensor_forces[..., 0] / 20  # !!! lousy normalization
        self.obs_buf[..., 15:18] = self.sensor_torques[..., 0] / 20  # !!! lousy normalization
        self.obs_buf[..., 18:21] = self.sensor_torques[..., 1] / 20  # !!! lousy normalization
        self.obs_buf[..., 21:24] = self.sensor_torques[..., 2] / 20  # !!! lousy normalization

        return self.obs_buf
