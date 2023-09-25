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


from . import logger
from isaacgym import gymapi
import numpy as np
import torch
from isaacgym.torch_utils import *
from isaacgym import gymtorch


def setup_gym_viewer(config):
    gym = initialize_gym(config)
    sim, viewer = configure_gym(gym, config)
    return gym, sim, viewer


def initialize_gym(config):
    gym = gymapi.acquire_gym()
    if not gym.initialize():
        logger.warn("*** Failed to initialize gym")
        quit()

    return gym


def configure_gym(gym, config):
    engine, render = config['engine'], config['render']

    # physics engine settings
    if(engine == 'FLEX'):
        sim_engine = gymapi.SIM_FLEX
    elif(engine == 'PHYSX'):
        sim_engine = gymapi.SIM_PHYSX
    else:
        logger.warn("Unknown physics engine. defaulting to FLEX")
        sim_engine = gymapi.SIM_FLEX

    # gym viewer
    if render:
        # create viewer
        sim = gym.create_sim(0, 0, sim_type=sim_engine)
        viewer = gym.create_viewer(
            sim, int(gymapi.DEFAULT_VIEWER_WIDTH / 1.25),
            int(gymapi.DEFAULT_VIEWER_HEIGHT / 1.25)
        )

        if viewer is None:
            logger.warn("*** Failed to create viewer")
            quit()

        # enable left mouse click or space bar for throwing projectiles
        if config['add_projectiles']:
            gym.subscribe_viewer_mouse_event(viewer, gymapi.MOUSE_LEFT_BUTTON, "shoot")
            gym.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_SPACE, "shoot")

    else:
        sim = gym.create_sim(0, -1)
        viewer = None

    # simulation params
    scene_config = config['env']['scene']
    sim_params = gymapi.SimParams()
    sim_params.solver_type = scene_config['SolverType']
    sim_params.num_outer_iterations = scene_config['NumIterations']
    sim_params.num_inner_iterations = scene_config['NumInnerIterations']
    sim_params.relaxation = scene_config.get('Relaxation', 0.75)
    sim_params.warm_start = scene_config.get('WarmStart', 0.25)
    sim_params.geometric_stiffness = scene_config.get('GeometricStiffness', 1.0)
    sim_params.shape_collision_margin = 0.01

    sim_params.gravity = gymapi.Vec3(0.0, -9.8, 0.0)
    gym.set_sim_params(sim, sim_params)

    return sim, viewer


def parse_states_from_reference_states(reference_states, progress):
    # parse reference states from DeepMimicState
    global_quats_ref = torch.tensor(
        reference_states._global_rotation[(progress,)].numpy(),
        dtype=torch.double
    ).cuda()
    ts_ref = torch.tensor(
        reference_states._translation[(progress,)].numpy(),
        dtype=torch.double
    ).cuda()
    vels_ref = torch.tensor(
        reference_states._velocity[(progress,)].numpy(),
        dtype=torch.double
    ).cuda()
    avels_ref = torch.tensor(
        reference_states._angular_velocity[(progress,)].numpy(),
        dtype=torch.double
    ).cuda()
    return global_quats_ref, ts_ref, vels_ref, avels_ref


def parse_states_from_reference_states_with_motion_id(precomputed_state,
                                                      progress, motion_id):
    assert len(progress) == len(motion_id)
    # get the global id
    global_id = precomputed_state['motion_offset'][motion_id] + progress
    global_id = np.minimum(global_id,
                           precomputed_state['global_quats_ref'].shape[0] - 1)

    # parse reference states from DeepMimicState
    global_quats_ref = precomputed_state['global_quats_ref'][global_id]
    ts_ref = precomputed_state['ts_ref'][global_id]
    vels_ref = precomputed_state['vels_ref'][global_id]
    avels_ref = precomputed_state['avels_ref'][global_id]
    return global_quats_ref, ts_ref, vels_ref, avels_ref


def parse_dof_state_with_motion_id(precomputed_state, dof_state,
                                   progress, motion_id):
    assert len(progress) == len(motion_id)
    # get the global id
    global_id = precomputed_state['motion_offset'][motion_id] + progress
    # NOTE: it should never reach the dof_state.shape, cause the episode is
    # terminated 2 steps before
    global_id = np.minimum(global_id, dof_state.shape[0] - 1)

    # parse reference states from DeepMimicState
    return dof_state[global_id]


def get_flatten_ids(precomputed_state):
    motion_offsets = precomputed_state['motion_offset']
    init_state_id, init_motion_id, global_id = [], [], []
    for i_motion in range(len(motion_offsets) - 1):
        i_length = motion_offsets[i_motion + 1] - motion_offsets[i_motion]
        init_state_id.extend(range(i_length))
        init_motion_id.extend([i_motion] * i_length)
        if len(global_id) == 0:
            global_id.extend(range(0, i_length))
        else:
            global_id.extend(range(global_id[-1] + 1,
                                   global_id[-1] + i_length + 1))
    return np.array(init_state_id), np.array(init_motion_id), \
        np.array(global_id)


def parse_states_from_reference_states_with_global_id(precomputed_state,
                                                      global_id):
    # get the global id
    global_id = global_id % precomputed_state['global_quats_ref'].shape[0]

    # parse reference states from DeepMimicState
    global_quats_ref = precomputed_state['global_quats_ref'][global_id]
    ts_ref = precomputed_state['ts_ref'][global_id]
    vels_ref = precomputed_state['vels_ref'][global_id]
    avels_ref = precomputed_state['avels_ref'][global_id]
    return global_quats_ref, ts_ref, vels_ref, avels_ref


def get_robot_states_from_torch_tensor(config, ts, global_quats, vels, avels,
                                       init_rot, progress, motion_length=-1,
                                       actions=None, relative_rot=None,
                                       motion_id=None, num_motion=None,
                                       motion_onehot_matrix=None):
    info = {}
    # the observation with quaternion-based representation
    torso_height = ts[..., 0, 1].cpu().numpy()
    gttrny, gqny, vny, avny, info['root_yaw_inv'] = \
        quaternion_math.compute_observation_return_info(global_quats, ts,
                                                        vels, avels)
    joint_obs = np.concatenate([gttrny.cpu().numpy(), gqny.cpu().numpy(),
                                vny.cpu().numpy(), avny.cpu().numpy()], axis=-1)
    joint_obs = joint_obs.reshape(joint_obs.shape[0], -1)
    num_envs = joint_obs.shape[0]
    obs = np.concatenate([torso_height[:, np.newaxis], joint_obs], -1)

    # the previous action
    if config['env_action_ob']:
        obs = np.concatenate([obs, actions], axis=-1)

    # the orientation
    if config['env_orientation_ob']:
        if relative_rot is not None:
            obs = np.concatenate([obs, relative_rot], axis=-1)
        else:
            curr_rot = global_quats[np.arange(num_envs)][:, 0]
            curr_rot = curr_rot.reshape(num_envs, -1, 4)
            relative_rot = quaternion_math.compute_orientation_drift(
                init_rot, curr_rot
            ).cpu().numpy()
            obs = np.concatenate([obs, relative_rot], axis=-1)

    if config['env_frame_ob']:
        if type(motion_length) == np.ndarray:
            motion_length = motion_length.astype(np.float)
            progress_ob = np.expand_dims(progress.astype(np.float) /
                                         motion_length, axis=-1)
        else:
            progress_ob = np.expand_dims(progress.astype(np.float) /
                                         float(motion_length), axis=-1)
        obs = np.concatenate([obs, progress_ob], axis=-1)

    if config['env_motion_ob'] and not config['env_motion_ob_onehot']:
        motion_id_ob = np.expand_dims(motion_id.astype(np.float) /
                                      float(num_motion), axis=-1)
        obs = np.concatenate([obs, motion_id_ob], axis=-1)
    elif config['env_motion_ob'] and config['env_motion_ob_onehot']:
        motion_id_ob = motion_onehot_matrix[motion_id]
        obs = np.concatenate([obs, motion_id_ob], axis=-1)

    return obs, info


def get_xyzoffset(start_ts, end_ts, root_yaw_inv):
    xyoffset = (end_ts - start_ts)[:, [0], :].reshape(1, -1, 1, 3)
    ryinv = root_yaw_inv.reshape(1, -1, 1, 4)

    calibrated_xyz_offset = quaternion_math.quat_apply(ryinv, xyoffset)[0, :, 0, :]
    return calibrated_xyz_offset
