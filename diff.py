# Purely used to make the git diff figure

@torch.jit.script
def compute_reward(object_rot, goal_rot, object_angvel, object_pos, fingertip_pos):
    # Rotation reward
    rot_diff = torch.abs(torch.sum(object_rot * goal_rot, dim=1) - 1) / 2
    rotation_reward_temp = 30.0
    rotation_reward = torch.exp(-rotation_reward_temp * rot_diff)

    # Distance reward
    min_distance_temp = 10.0
    min_distance = torch.min(torch.norm(fingertip_pos - object_pos[:, None], dim=2), dim=1).values
    uncapped_distance_reward = torch.exp(-min_distance_temp * min_distance)
    distance_reward = torch.clamp(uncapped_distance_reward, 0.0, 1.0)

    # Angular velocity penalty
    angvel_norm = torch.norm(object_angvel, dim=1)
    angvel_threshold = 0.5
    angvel_penalty_temp = 5.0
    angular_velocity_penalty = torch.where(angvel_norm > angvel_threshold,
        torch.exp(-angvel_penalty_temp * (angvel_norm - angvel_threshold)), torch.zeros_like(angvel_norm))

    total_reward = 0.5 * rotation_reward + 0.3 * distance_reward - 0.2 * angular_velocity_penalty

    reward_components = {
        "rotation_reward": rotation_reward,
        "distance_reward": distance_reward,
        "angular_velocity_penalty": angular_velocity_penalty,
    }

    return total_reward, reward_components