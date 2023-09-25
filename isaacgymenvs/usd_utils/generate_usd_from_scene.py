# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from pxr import Usd, UsdGeom, UsdLux, Gf, Vt

import argparse
import numpy as np
import os
import sys
from tqdm import tqdm

#from omni.client import Client


# def connect_to_omniverse():
#     ov = Client('ov-content.nvidia.com:3009', 'test', 'test', timeout=10000)
#     ov.connect()

#     return ov


def load_states(root_dir):
    positions = np.load(os.path.join(root_dir, "body_position.npy"), allow_pickle=True)
    rotations = np.load(os.path.join(root_dir, "body_rotation.npy"), allow_pickle=True)

    return positions, rotations


def build_usd_scene(usd_dir, file_name, num_envs, num_frames):
    stage = Usd.Stage.CreateNew('{}/{}.usda'.format(usd_dir, file_name))
    stage.SetFramesPerSecond(60)
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(num_frames)

    # add assets to scene
    override_file_path = os.path.basename(os.path.normpath(usd_dir))
    scene_root = stage.OverridePrim('/scene')
    scene_root.GetPrim().GetReferences().AddReference('./{}.usda'.format(override_file_path))

    stage.SetDefaultPrim(scene_root.GetPrim())

    return stage


def write_to_usd(stage, env_name, num_envs, num_frames, positions, rotations):
    print(positions.shape)
    # for i in range(20):
    #     env_name.append("prop{}".format(i))
    for i in tqdm(range(num_envs)):
        start_idx = 0
        for env in env_name:
            prim = stage.GetPrimAtPath('/scene/env{}/{}'.format(i, env))
            children_prim = prim.GetChildren()
            num_bodies = positions.shape[2]
            for j in range(len(children_prim)):
                ops = UsdGeom.Xform.Get(stage, children_prim[j].GetPath()).GetOrderedXformOps()
                for f in range(num_frames):
                    body_position = positions[f, i, j+start_idx]
                    body_rotation = rotations[f, i, j+start_idx]
                    offset_transform = Gf.Matrix4d().SetRotateOnly(Gf.Rotation(Gf.Quatd(np.sqrt(2), np.sqrt(2), 0, 0)))
                    transform = Gf.Matrix4d(
                        Gf.Rotation(Gf.Quatd(body_rotation[3].item(), body_rotation[0].item(), body_rotation[1].item(), body_rotation[2].item())),
                        Gf.Vec3d(body_position[0].item(), body_position[1].item(), body_position[2].item())) * offset_transform
                    ops[0].Set(value=transform.ExtractTranslation(), time=f)
                    ops[1].Set(value=Gf.Quatf(transform.ExtractRotation().GetQuat()), time=f)
            start_idx += len(children_prim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--usd_dir', type=str, default='', help='Root directory of USD scene file')
    parser.add_argument('--file_name', type=str, default='scene', help='Name of new USD file')
    parser.add_argument('--state_dir', type=str, default='', help='Directory containing npy state files - body_position.npy and body_rotation.npy')
    parser.add_argument('--env_name', type=str, default='', help='Name of environment asset in exported USD file (i.e. humanoid)')
    args = parser.parse_args()

    #ov = connect_to_omniverse()

    positions, rotations = load_states(args.state_dir)
    num_envs = positions.shape[1]
    num_frames = positions.shape[0]
    print("num_envs", num_envs, "num_frames", num_frames)

    stage = build_usd_scene(args.usd_dir, args.file_name, num_envs, num_frames)
    write_to_usd(stage, args.env_name.split(','), num_envs, num_frames, positions, rotations)
    stage.Save()

    #ov.disconnect()


if __name__ == '__main__':
    main()
