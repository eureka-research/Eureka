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


"""
This script reads an fbx file and returns the joint names, parents, and transforms.

NOTE: It requires the Python FBX package to be installed.
"""

import sys

import numpy as np

try:
    import fbx
    import FbxCommon
except ImportError as e:
    print("Error: FBX library failed to load - importing FBX data will not succeed. Message: {}".format(e))
    print("FBX tools must be installed from https://help.autodesk.com/view/FBX/2020/ENU/?guid=FBX_Developer_Help_scripting_with_python_fbx_installing_python_fbx_html")


def fbx_to_npy(file_name_in, root_joint_name, fps):
    """
    This function reads in an fbx file, and saves the relevant info to a numpy array

    Fbx files have a series of animation curves, each of which has animations at different 
    times. This script assumes that for mocap data, there is only one animation curve that
    contains all the joints. Otherwise it is unclear how to read in the data.

    If this condition isn't met, then the method throws an error

    :param file_name_in: str, file path in. Should be .fbx file
    :return: nothing, it just writes a file.
    """

    # Create the fbx scene object and load the .fbx file
    fbx_sdk_manager, fbx_scene = FbxCommon.InitializeSdkObjects()
    FbxCommon.LoadScene(fbx_sdk_manager, fbx_scene, file_name_in)

    """
    To read in the animation, we must find the root node of the skeleton.
    
    Unfortunately fbx files can have "scene parents" and other parts of the tree that are 
    not joints
    
    As a crude fix, this reader just takes and finds the first thing which has an 
    animation curve attached
    """

    search_root = (root_joint_name is None or root_joint_name == "")

    # Get the root node of the skeleton, which is the child of the scene's root node
    possible_root_nodes = [fbx_scene.GetRootNode()]
    found_root_node = False
    max_key_count = 0
    root_joint = None
    while len(possible_root_nodes) > 0:
        joint = possible_root_nodes.pop(0)
        if not search_root:
            if joint.GetName() == root_joint_name:
                root_joint = joint
        try:
            curve = _get_animation_curve(joint, fbx_scene)
        except RuntimeError:
            curve = None
        if curve is not None:
            key_count = curve.KeyGetCount()
            if key_count > max_key_count:
                found_root_node = True
                max_key_count = key_count
                root_curve = curve
            if search_root and not root_joint:
                root_joint = joint
        for child_index in range(joint.GetChildCount()):
            possible_root_nodes.append(joint.GetChild(child_index))
    if not found_root_node:
        raise RuntimeError("No root joint found!! Exiting")

    joint_list, joint_names, parents = _get_skeleton(root_joint)

    """
    Read in the transformation matrices of the animation, taking the scaling into account
    """

    anim_range, frame_count, frame_rate = _get_frame_count(fbx_scene)

    local_transforms = []
    #for frame in range(frame_count):
    time_sec = anim_range.GetStart().GetSecondDouble()
    time_range_sec = anim_range.GetStop().GetSecondDouble() - time_sec
    fbx_fps = frame_count / time_range_sec
    if fps != 120:
        fbx_fps = fps
    print("FPS: ", fbx_fps)
    while time_sec < anim_range.GetStop().GetSecondDouble():
        fbx_time = fbx.FbxTime()
        fbx_time.SetSecondDouble(time_sec)
        fbx_time = fbx_time.GetFramedTime()
        transforms_current_frame = []

        # Fbx has a unique time object which you need
        #fbx_time = root_curve.KeyGetTime(frame)
        for joint in joint_list:
            arr = np.array(_recursive_to_list(joint.EvaluateLocalTransform(fbx_time)))
            scales = np.array(_recursive_to_list(joint.EvaluateLocalScaling(fbx_time)))
            if not np.allclose(scales[0:3], scales[0]):
                raise ValueError(
                    "Different X, Y and Z scaling. Unsure how this should be handled. "
                    "To solve this, look at this link and try to upgrade the script "
                    "http://help.autodesk.com/view/FBX/2017/ENU/?guid=__files_GUID_10CDD"
                    "63C_79C1_4F2D_BB28_AD2BE65A02ED_htm"
                )
            # Adjust the array for scaling
            arr /= scales[0]
            arr[3, 3] = 1.0
            transforms_current_frame.append(arr)
        local_transforms.append(transforms_current_frame)

        time_sec += (1.0/fbx_fps)

    local_transforms = np.array(local_transforms)
    print("Frame Count: ", len(local_transforms))

    return joint_names, parents, local_transforms, fbx_fps

def _get_frame_count(fbx_scene):
    # Get the animation stacks and layers, in order to pull off animation curves later
    num_anim_stacks = fbx_scene.GetSrcObjectCount(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId)
    )
    # if num_anim_stacks != 1:
    #     raise RuntimeError(
    #         "More than one animation stack was found. "
    #         "This script must be modified to handle this case. Exiting"
    #     )
    if num_anim_stacks > 1:
        index = 1
    else:
        index = 0
    anim_stack = fbx_scene.GetSrcObject(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId), index
    )

    anim_range = anim_stack.GetLocalTimeSpan()
    duration = anim_range.GetDuration()
    fps = duration.GetFrameRate(duration.GetGlobalTimeMode())
    frame_count = duration.GetFrameCount(True)

    return anim_range, frame_count, fps

def _get_animation_curve(joint, fbx_scene):
    # Get the animation stacks and layers, in order to pull off animation curves later
    num_anim_stacks = fbx_scene.GetSrcObjectCount(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId)
    )
    # if num_anim_stacks != 1:
    #     raise RuntimeError(
    #         "More than one animation stack was found. "
    #         "This script must be modified to handle this case. Exiting"
    #     )
    if num_anim_stacks > 1:
        index = 1
    else:
        index = 0
    anim_stack = fbx_scene.GetSrcObject(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimStack.ClassId), index
    )

    num_anim_layers = anim_stack.GetSrcObjectCount(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimLayer.ClassId)
    )
    if num_anim_layers != 1:
        raise RuntimeError(
            "More than one animation layer was found. "
            "This script must be modified to handle this case. Exiting"
        )
    animation_layer = anim_stack.GetSrcObject(
        FbxCommon.FbxCriteria.ObjectType(FbxCommon.FbxAnimLayer.ClassId), 0
    )

    def _check_longest_curve(curve, max_curve_key_count):
        longest_curve = None
        if curve and curve.KeyGetCount() > max_curve_key_count[0]:
            max_curve_key_count[0] = curve.KeyGetCount()
            return True

        return False

    max_curve_key_count = [0]
    longest_curve = None
    for c in ["X", "Y", "Z"]:
        curve = joint.LclTranslation.GetCurve(
            animation_layer, c
        )  # sample curve for translation
        if _check_longest_curve(curve, max_curve_key_count):
            longest_curve = curve

        curve = joint.LclRotation.GetCurve(
            animation_layer, "X"
        )
        if _check_longest_curve(curve, max_curve_key_count):
            longest_curve = curve

    return longest_curve


def _get_skeleton(root_joint):

    # Do a depth first search of the skeleton to extract all the joints
    joint_list = [root_joint]
    joint_names = [root_joint.GetName()]
    parents = [-1]  # -1 means no parent

    def append_children(joint, pos):
        """
        Depth first search function
        :param joint: joint item in the fbx
        :param pos: position of current element (for parenting)
        :return: Nothing
        """
        for child_index in range(joint.GetChildCount()):
            child = joint.GetChild(child_index)
            joint_list.append(child)
            joint_names.append(child.GetName())
            parents.append(pos)
            append_children(child, len(parents) - 1)

    append_children(root_joint, 0)
    return joint_list, joint_names, parents


def _recursive_to_list(array):
    """
    Takes some iterable that might contain iterables and converts it to a list of lists 
    [of lists... etc]

    Mainly used for converting the strange fbx wrappers for c++ arrays into python lists
    :param array: array to be converted
    :return: array converted to lists
    """
    try:
        return float(array)
    except TypeError:
        return [_recursive_to_list(a) for a in array]


def parse_fbx(file_name_in, root_joint_name, fps):
    return fbx_to_npy(file_name_in, root_joint_name, fps)
