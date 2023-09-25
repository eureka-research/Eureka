# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Script that reads in fbx files from python

This requires a configs file, which contains the command necessary to switch conda
environments to run the fbx reading script from python
"""

from ....core import logger

import inspect
import os

import numpy as np

from .fbx_backend import parse_fbx


def fbx_to_array(fbx_file_path, root_joint, fps):
    """
    Reads an fbx file to an array.

    :param fbx_file_path: str, file path to fbx
    :return: tuple with joint_names, parents, transforms, frame time
    """

    # Ensure the file path is valid
    fbx_file_path = os.path.abspath(fbx_file_path)
    assert os.path.exists(fbx_file_path)

    # Parse FBX file
    joint_names, parents, local_transforms, fbx_fps = parse_fbx(fbx_file_path, root_joint, fps)
    return joint_names, parents, local_transforms, fbx_fps
