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
import json
import copy
import os
from collections import OrderedDict

class data_tree(object):
    def __init__(self, name):
        self._name = name
        self._children, self._children_names, self._picked, self._depleted = \
            [], [], [], []
        self._data, self._length = [], []
        self._total_length, self._num_leaf, self._is_leaf = 0, 0, 0
        self._assigned_prob = 0.0

    def add_node(self, dict_hierachy, mocap_data):
        # data_hierachy -> 'behavior' 'direction' 'type' 'style'
        # behavior, direction, mocap_type, style = mocap_data[2:]
        self._num_leaf += 1

        if len(dict_hierachy) == 0:
            # leaf node
            self._data.append(mocap_data[0])
            self._length.append(mocap_data[1])
            self._picked.append(0)
            self._depleted.append(0)
            self._is_leaf = 1
        else:
            children_name = dict_hierachy[0].replace('\n', '')
            if children_name not in self._children_names:
                self._children_names.append(children_name)
                self._children.append(data_tree(children_name))
                self._picked.append(0)
                self._depleted.append(0)

            # add the data
            index = self._children_names.index(children_name)
            self._children[index].add_node(dict_hierachy[1:], mocap_data)

    def summarize_length(self):
        if self._is_leaf:
            self._total_length = np.sum(self._length)
        else:
            self._total_length = 0
            for i_child in self._children:
                self._total_length += i_child.summarize_length()

        return self._total_length

    def to_dict(self, verbose=False):
        if self._is_leaf:
            self._data_dict = copy.deepcopy(self._data)
        else:
            self._data_dict = OrderedDict()
            for i_child in self._children:
                self._data_dict[i_child.name] = i_child.to_dict(verbose)

        if verbose:
            if self._is_leaf:
                verbose_data_dict = []
                for ii, i_key in enumerate(self._data_dict):
                    new_key = i_key + ' (picked {} / {})'.format(
                        str(self._picked[ii]), self._length[ii]
                    )
                    verbose_data_dict.append(new_key)
            else:
                verbose_data_dict = OrderedDict()
                for ii, i_key in enumerate(self._data_dict):
                    new_key = i_key + ' (picked {} / {})'.format(
                        str(self._picked[ii]), self._children[ii].total_length
                    )
                    verbose_data_dict[new_key] = self._data_dict[i_key]

            self._data_dict = verbose_data_dict

        return self._data_dict

    @property
    def name(self):
        return self._name

    @property
    def picked(self):
        return self._picked

    @property
    def total_length(self):
        return self._total_length

    def water_floating_algorithm(self):
        # find the sub class with the minimum picked
        assert not np.all(self._depleted)
        for ii in np.where(np.array(self._children_names) == 'mix')[0]:
            self._depleted[ii] = np.inf
        chosen_child = np.argmin(np.array(self._picked) +
                                 np.array(self._depleted))
        if self._is_leaf:
            self._picked[chosen_child] = self._length[chosen_child]
            self._depleted[chosen_child] = np.inf
            chosen_data = self._data[chosen_child]
            data_info = {'name': [self._name],
                         'length': self._length[chosen_child],
                         'all_depleted': np.all(self._depleted)}
        else:
            chosen_data, data_info = \
                self._children[chosen_child].water_floating_algorithm()
            self._picked[chosen_child] += data_info['length']
            data_info['name'].insert(0, self._name)
            if data_info['all_depleted']:
                self._depleted[chosen_child] = np.inf
            data_info['all_depleted'] = np.all(self._depleted)

        return chosen_data, data_info

    def assign_probability(self, total_prob):
        # find the sub class with the minimum picked
        leaves, probs = [], []
        if self._is_leaf:
            self._assigned_prob = total_prob
            leaves.extend(self._data)
            per_traj_prob = total_prob / float(len(self._data))
            probs.extend([per_traj_prob] * len(self._data))
        else:
            per_child_prob = total_prob / float(len(self._children))
            for i_child in self._children:
                i_leave, i_prob = i_child.assign_probability(per_child_prob)
                leaves.extend(i_leave)
                probs.extend(i_prob)

        return leaves, probs


def parse_dataset(env, args):
    """ @brief: get the training set and test set
    """
    TRAIN_PERCENTAGE = args.parse_dataset_train
    info, motion = env.motion_info, env.motion
    lengths = env.get_all_motion_length()
    train_size = np.sum(motion.get_all_motion_length()) * TRAIN_PERCENTAGE

    data_structure = data_tree('root')
    shuffle_id = list(range(len(info['mocap_data_list'])))
    np.random.shuffle(shuffle_id)
    info['mocap_data_list'] = [info['mocap_data_list'][ii] for ii in shuffle_id]
    for mocap_data, length in zip(info['mocap_data_list'], lengths[shuffle_id]):
        node_data = [mocap_data[0]] + [length]
        data_structure.add_node(mocap_data[2:], node_data)

    raw_data_dict = data_structure.to_dict()
    print(json.dumps(raw_data_dict, indent=4))

    total_length = 0
    chosen_data = []
    while True:
        i_data, i_info = data_structure.water_floating_algorithm()
        print('Current length:', total_length, i_data, i_info)
        total_length += i_info['length']
        chosen_data.append(i_data)

        if total_length > train_size:
            break
    data_structure.summarize_length()
    data_dict = data_structure.to_dict(verbose=True)
    print(json.dumps(data_dict, indent=4))

    # save the training and test sets
    train_data, test_data = [], []
    for i_data in info['mocap_data_list']:
        if i_data[0] in chosen_data:
            train_data.append(i_data[1:])
        else:
            test_data.append(i_data[1:])

    train_tsv_name = args.mocap_list_file.split('.')[0] + '_' + \
        str(int(args.parse_dataset_train * 100)) + '_train' + '.tsv'
    test_tsv_name = train_tsv_name.replace('train', 'test')
    info_name = test_tsv_name.replace('test', 'info').replace('.tsv', '.json')

    save_tsv_files(env._base_dir, train_tsv_name, train_data)
    save_tsv_files(env._base_dir, test_tsv_name, test_data)

    info_file = open(os.path.join(env._base_dir, 'experiments', 'mocap_files',
                                  info_name), 'w')
    json.dump(data_dict, info_file, indent=4)


def save_tsv_files(base_dir, name, data_dict):
    file_name = os.path.join(base_dir, 'experiments', 'mocap_files', name)
    recorder = open(file_name, "w")
    for i_data in data_dict:
        line = '{}\t{}\t{}\t{}\t{}\n'.format(*i_data)
        recorder.write(line)
    recorder.close()