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


import numpy as np
from bisect import bisect
from isaacgym import gymapi


def get_property_setter_map(gym):
    property_to_setters = {
        "dof_properties": gym.set_actor_dof_properties,
        "tendon_properties": gym.set_actor_tendon_properties,
        "rigid_body_properties": gym.set_actor_rigid_body_properties,
        "rigid_shape_properties": gym.set_actor_rigid_shape_properties,
        "sim_params": gym.set_sim_params,
    }

    return property_to_setters


def get_property_getter_map(gym):
    property_to_getters = {
        "dof_properties": gym.get_actor_dof_properties,
        "tendon_properties": gym.get_actor_tendon_properties,
        "rigid_body_properties": gym.get_actor_rigid_body_properties,
        "rigid_shape_properties": gym.get_actor_rigid_shape_properties,
        "sim_params": gym.get_sim_params,
    }

    return property_to_getters


def get_default_setter_args(gym):
    property_to_setter_args = {
        "dof_properties": [],
        "tendon_properties": [],
        "rigid_body_properties": [True],
        "rigid_shape_properties": [],
        "sim_params": [],
    }

    return property_to_setter_args


def generate_random_samples(attr_randomization_params, shape, curr_gym_step_count,
                            extern_sample=None):

    rand_range = attr_randomization_params['range']
    distribution = attr_randomization_params['distribution']

    sched_type = attr_randomization_params['schedule'] if 'schedule' in attr_randomization_params else None
    sched_step = attr_randomization_params['schedule_steps'] if 'schedule' in attr_randomization_params else None

    operation = attr_randomization_params['operation']

    if sched_type == 'linear':
        sched_scaling = 1 / sched_step * min(curr_gym_step_count, sched_step)
    elif sched_type == 'constant':
        sched_scaling = 0 if curr_gym_step_count < sched_step else 1
    else:
        sched_scaling = 1

    if extern_sample is not None:

        sample = extern_sample

        if operation == 'additive':
            sample *= sched_scaling
        elif operation == 'scaling':
            sample = sample * sched_scaling + 1 * (1 - sched_scaling)

    elif distribution == "gaussian":

        mu, var = rand_range

        if operation == 'additive':
            mu *= sched_scaling
            var *= sched_scaling
        elif operation == 'scaling':
            var = var * sched_scaling  # scale up var over time
            mu = mu * sched_scaling + 1 * (1 - sched_scaling)  # linearly interpolate
        sample = np.random.normal(mu, var, shape)

    elif distribution == "loguniform":

        lo, hi = rand_range
        if operation == 'additive':
            lo *= sched_scaling
            hi *= sched_scaling
        elif operation == 'scaling':
            lo = lo * sched_scaling + 1 * (1 - sched_scaling)
            hi = hi * sched_scaling + 1 * (1 - sched_scaling)
        sample = np.exp(np.random.uniform(np.log(lo), np.log(hi), shape))

    elif distribution == "uniform":

        lo, hi = rand_range
        if operation == 'additive':
            lo *= sched_scaling
            hi *= sched_scaling
        elif operation == 'scaling':
            lo = lo * sched_scaling + 1 * (1 - sched_scaling)
            hi = hi * sched_scaling + 1 * (1 - sched_scaling)
        sample = np.random.uniform(lo, hi, shape)

    return sample


def get_bucketed_val(new_prop_val, attr_randomization_params):
    if attr_randomization_params['distribution'] == 'uniform':
        # range of buckets defined by uniform distribution
        lo, hi = attr_randomization_params['range'][0], attr_randomization_params['range'][1]
    else:
        # for gaussian, set range of buckets to be 2 stddev away from mean
        lo = attr_randomization_params['range'][0] - 2 * np.sqrt(attr_randomization_params['range'][1])
        hi = attr_randomization_params['range'][0] + 2 * np.sqrt(attr_randomization_params['range'][1])
    num_buckets = attr_randomization_params['num_buckets']
    buckets = [(hi - lo) * i / num_buckets + lo for i in range(num_buckets)]
    return buckets[bisect(buckets, new_prop_val) - 1]


def apply_random_samples(prop, og_prop, attr, attr_randomization_params,
                         curr_gym_step_count, extern_sample=None, bucketing_randomization_params=None):
    
    """
    @params:
        prop: property we want to randomise
        og_prop: the original property and its value 
        attr: which particular attribute we want to randomise e.g. damping, stiffness
        attr_randomization_params: the attribute randomisation meta-data e.g. distr, range, schedule
        curr_gym_step_count: gym steps so far 

    """

    if isinstance(prop, gymapi.SimParams):
        
        if attr == 'gravity':
            
            sample = generate_random_samples(attr_randomization_params, 3, curr_gym_step_count)
            if attr_randomization_params['operation'] == 'scaling':
                prop.gravity.x = og_prop['gravity'].x * sample[0]
                prop.gravity.y = og_prop['gravity'].y * sample[1]
                prop.gravity.z = og_prop['gravity'].z * sample[2]
            elif attr_randomization_params['operation'] == 'additive':
                prop.gravity.x = og_prop['gravity'].x + sample[0]
                prop.gravity.y = og_prop['gravity'].y + sample[1]
                prop.gravity.z = og_prop['gravity'].z + sample[2]


        if attr == 'rest_offset':

           sample = generate_random_samples(attr_randomization_params, 1, curr_gym_step_count)
           prop.physx.rest_offset = sample
                

    elif isinstance(prop, np.ndarray):
        sample = generate_random_samples(attr_randomization_params, prop[attr].shape,
                                         curr_gym_step_count, extern_sample)

        if attr_randomization_params['operation'] == 'scaling':
            new_prop_val = og_prop[attr] * sample
        elif attr_randomization_params['operation'] == 'additive':
            new_prop_val = og_prop[attr] + sample

        if 'num_buckets' in attr_randomization_params and attr_randomization_params['num_buckets'] > 0:
            new_prop_val = get_bucketed_val(new_prop_val, attr_randomization_params)
        prop[attr] = new_prop_val
    else:
        sample = generate_random_samples(attr_randomization_params, 1,
                                         curr_gym_step_count, extern_sample)
        cur_attr_val = og_prop[attr]
        if attr_randomization_params['operation'] == 'scaling':
            new_prop_val = cur_attr_val * sample
        elif attr_randomization_params['operation'] == 'additive':
            new_prop_val = cur_attr_val + sample

        if 'num_buckets' in attr_randomization_params and attr_randomization_params['num_buckets'] > 0:
            if bucketing_randomization_params is None:
                new_prop_val = get_bucketed_val(new_prop_val, attr_randomization_params)
            else:
                new_prop_val = get_bucketed_val(new_prop_val, bucketing_randomization_params)
        setattr(prop, attr, new_prop_val)

def check_buckets(gym, envs, dr_params):
    total_num_buckets = 0
    for actor, actor_properties in dr_params["actor_params"].items():
        cur_num_buckets = 0

        if 'rigid_shape_properties' in actor_properties.keys():
            prop_attrs = actor_properties['rigid_shape_properties']
            if 'restitution' in prop_attrs and 'num_buckets' in prop_attrs['restitution']:
                cur_num_buckets = prop_attrs['restitution']['num_buckets']
            if 'friction' in prop_attrs and 'num_buckets' in prop_attrs['friction']:
                if cur_num_buckets > 0:
                    cur_num_buckets *= prop_attrs['friction']['num_buckets']
                else:
                    cur_num_buckets = prop_attrs['friction']['num_buckets']
            total_num_buckets += cur_num_buckets

    assert total_num_buckets <= 64000, 'Explicit material bucketing has been specified, but the provided total bucket count exceeds 64K: {} specified buckets'.format(
        total_num_buckets)

    shape_ct = 0

    # Separate loop because we should not assume that each actor is present in each env
    for env in envs:
        for i in range(gym.get_actor_count(env)):
            actor_handle = gym.get_actor_handle(env, i)
            actor_name = gym.get_actor_name(env, actor_handle)
            if actor_name in dr_params["actor_params"] and 'rigid_shape_properties' in dr_params["actor_params"][actor_name]:
                shape_ct += gym.get_actor_rigid_shape_count(env, actor_handle)

    assert shape_ct <= 64000 or total_num_buckets > 0, 'Explicit material bucketing is not used but the total number of shapes exceeds material limit. Please specify bucketing to limit material count.'