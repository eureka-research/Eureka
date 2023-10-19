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

import torch


class ReplayBuffer():
    def __init__(self, buffer_size, device):
        self._head = 0
        self._total_count = 0
        self._buffer_size = buffer_size
        self._device = device
        self._data_buf = None
        self._sample_idx = torch.randperm(buffer_size)
        self._sample_head = 0

        return

    def reset(self):
        self._head = 0
        self._total_count = 0
        self._reset_sample_idx()
        return

    def get_buffer_size(self):
        return self._buffer_size

    def get_total_count(self):
        return self._total_count

    def store(self, data_dict):
        if (self._data_buf is None):
            self._init_data_buf(data_dict)

        n = next(iter(data_dict.values())).shape[0]
        buffer_size = self.get_buffer_size()
        assert(n < buffer_size)

        for key, curr_buf in self._data_buf.items():
            curr_n = data_dict[key].shape[0]
            assert(n == curr_n)

            store_n = min(curr_n, buffer_size - self._head)
            curr_buf[self._head:(self._head + store_n)] = data_dict[key][:store_n]    
        
            remainder = n - store_n
            if (remainder > 0):
                curr_buf[0:remainder] = data_dict[key][store_n:]  

        self._head = (self._head + n) % buffer_size
        self._total_count += n

        return

    def sample(self, n):
        total_count = self.get_total_count()
        buffer_size = self.get_buffer_size()

        idx = torch.arange(self._sample_head, self._sample_head + n)
        idx = idx % buffer_size
        rand_idx = self._sample_idx[idx]
        if (total_count < buffer_size):
            rand_idx = rand_idx % self._head

        samples = dict()
        for k, v in self._data_buf.items():
            samples[k] = v[rand_idx]

        self._sample_head += n
        if (self._sample_head >= buffer_size):
            self._reset_sample_idx()

        return samples

    def _reset_sample_idx(self):
        buffer_size = self.get_buffer_size()
        self._sample_idx[:] = torch.randperm(buffer_size)
        self._sample_head = 0
        return

    def _init_data_buf(self, data_dict):
        buffer_size = self.get_buffer_size()
        self._data_buf = dict()

        for k, v in data_dict.items():
            v_shape = v.shape[1:]
            self._data_buf[k] = torch.zeros((buffer_size,) + v_shape, device=self._device)

        return