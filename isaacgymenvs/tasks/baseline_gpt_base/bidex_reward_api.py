def get_obj_pos_from_name(self, name):
    if name == "left_palm":
        return self.left_hand_pos
        # if hasattr(self, "left_hand_pos"):
        #     return self.left_hand_pos
        # elif hasattr(self, "hand_positions"):
        #     return self.hand_positions[self.another_hand_indices, :]
        # else:
        #     raise ValueError("No hand position information")
    elif name == "right_palm":
        return self.right_hand_pos
        # if hasattr(self, "right_hand_pos"):
        #     return self.right_hand_pos
        # elif hasattr(self, "hand_positions"):
        #     return self.hand_positions[self.hand_indices, :]
        # else:
        #     raise ValueError("No hand position information")
    elif name == "left_forefinger":
        return self.left_hand_ff_pos
    elif name == "left_middlefinger":
        return self.left_hand_mf_pos
    elif name == "left_ringfinger":
        return self.left_hand_rf_pos
    elif name == "left_littlefinger":
        return self.left_hand_lf_pos
    elif name == "left_thumb":
        return self.left_hand_th_pos
    elif name == "right_forefinger":
        return self.right_hand_ff_pos
    elif name == "right_middlefinger":
        return self.right_hand_mf_pos
    elif name == "right_ringfinger":
        return self.right_hand_rf_pos
    elif name == "right_littlefinger":
        return self.right_hand_lf_pos
    elif name == "right_thumb":
        return self.right_hand_th_pos
    elif name in ["object1", "pot", "cup"]:
        return self.object_pos
    elif name in ["object1_goal", "pot_goal", "cup_goal", "block_goal"]:
        return self.goal_pos
    elif name == "object2":
        return self.object_another_pos
    elif name == "object2_goal":
        return self.goal_another_pos
    elif name == "pot_left_handle":
        return self.pot_left_handle_pos
    elif name == "pot_right_handle":
        return self.pot_right_handle_pos
    elif name == "door_left_handle":
        return self.door_left_handle_pos
    elif name == "door_right_handle":
        return self.door_right_handle_pos
    elif name == "bottle_cap":
        return self.bottle_cap_pos
    elif name == "bottle":
        return self.bottle_pos
    elif name == "block1":
        return self.block_left_handle_pos
    elif name == "block2":
        return self.block_right_handle_pos
    elif name == "block1_goal":
        return self.left_goal_pos
    elif name == "block2_goal":
        return self.right_goal_pos
    elif name == "cup_left_handle":
        return self.cup_right_handle_pos
    elif name == "cup_right_handle":
        return self.cup_right_handle_pos
    elif name == "scissor_left_handle":
        return self.scissors_left_handle_pos
    elif name == "scissor_right_handle":
        return self.scissors_right_handle_pos
    elif name == "pen_cap":
        return self.pen_right_handle_pos
    elif name == "pen":
        return self.pen_left_handle_pos
    elif name == "left_switch":
        return self.switch_left_handle_pos
    elif name == "right_switch":
        return self.switch_right_handle_pos
    elif name == "bucket_handle":
        return self.bucket_handle_pos
    elif name == "kettle_handle":
        return self.kettle_handle_pos
    elif name == "kettle_spout":
        return self.kettle_spout_pos
    else:
        raise ValueError(f"{name} object does not have position information")

def get_obj_rot_from_name(self, name):
    if name in ["object1", "object2", "pot", "cup"]:
        return self.object_root
    elif name in ["object1_goal", "object2_goal", "pot_goal", "cup_goal"]:
        return self.goal_rot
    raise ValueError(f"{name} object does not have orientation information")

def set_min_l2_distance_reward(self, name_obj_A, name_obj_B):
    if name_obj_B == "nothing":
        name_obj_B = name_obj_A
    key = 'min_l2_distance_' + name_obj_A + '_' + name_obj_B
    obj_A, obj_B = self.get_obj_pos_from_name(name_obj_A), self.get_obj_pos_from_name(name_obj_B)
    self.reward_components[key] = -torch.norm(obj_A - obj_B, p=2, dim=-1)

def set_max_l2_distance_reward(self, name_obj_A, name_obj_B):
    if name_obj_B == "nothing":
        name_obj_B = name_obj_A
    key = 'max_l2_distance_' + name_obj_A + '_' + name_obj_B
    obj_A, obj_B = self.get_obj_pos_from_name(name_obj_A), self.get_obj_pos_from_name(name_obj_B)
    self.reward_components[key] = torch.norm(obj_A - obj_B, p=2, dim=-1)

def set_obj_orientation_reward(self, name_obj_A, name_obj_B):
    key = 'obj_orientation_' + name_obj_A + '_' + name_obj_B
    obj_A, obj_B = self.get_obj_pos_from_name(name_obj_A), self.get_obj_pos_from_name(name_obj_B)
    quat_diff = quat_mul(obj_A, quat_conjugate(obj_B))
    self.reward_components[key] = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))