# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from collections import OrderedDict
from .backend import Serializable
import torch


class TensorUtils(Serializable):
    @classmethod
    def from_dict(cls, dict_repr, *args, **kwargs):
        """ Read the object from an ordered dictionary

        :param dict_repr: the ordered dictionary that is used to construct the object
        :type dict_repr: OrderedDict
        :param kwargs: the arguments that need to be passed into from_dict()
        :type kwargs: additional arguments
        """
        return torch.from_numpy(dict_repr["arr"].astype(dict_repr["context"]["dtype"]))

    def to_dict(self):
        """ Construct an ordered dictionary from the object
        
        :rtype: OrderedDict
        """
        return NotImplemented

def tensor_to_dict(x):
    """ Construct an ordered dictionary from the object
    
    :rtype: OrderedDict
    """
    x_np = x.numpy()
    return {
        "arr": x_np,
        "context": {
            "dtype": x_np.dtype.name
        }
    }
