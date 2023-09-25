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


from typing import List, Optional

import math
import torch


@torch.jit.script
def quat_mul(a, b):
    """
    quaternion multiplication
    """
    x1, y1, z1, w1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    x2, y2, z2, w2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2

    return torch.stack([x, y, z, w], dim=-1)


@torch.jit.script
def quat_pos(x):
    """
    make all the real part of the quaternion positive
    """
    q = x
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q


@torch.jit.script
def quat_abs(x):
    """
    quaternion norm (unit quaternion represents a 3D rotation, which has norm of 1)
    """
    x = x.norm(p=2, dim=-1)
    return x


@torch.jit.script
def quat_unit(x):
    """
    normalized quaternion with norm of 1
    """
    norm = quat_abs(x).unsqueeze(-1)
    return x / (norm.clamp(min=1e-9))


@torch.jit.script
def quat_conjugate(x):
    """
    quaternion with its imaginary part negated
    """
    return torch.cat([-x[..., :3], x[..., 3:]], dim=-1)


@torch.jit.script
def quat_real(x):
    """
    real component of the quaternion
    """
    return x[..., 3]


@torch.jit.script
def quat_imaginary(x):
    """
    imaginary components of the quaternion
    """
    return x[..., :3]


@torch.jit.script
def quat_norm_check(x):
    """
    verify that a quaternion has norm 1
    """
    assert bool(
        (abs(x.norm(p=2, dim=-1) - 1) < 1e-3).all()
    ), "the quaternion is has non-1 norm: {}".format(abs(x.norm(p=2, dim=-1) - 1))
    assert bool((x[..., 3] >= 0).all()), "the quaternion has negative real part"


@torch.jit.script
def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q


@torch.jit.script
def quat_from_xyz(xyz):
    """
    Construct 3D rotation from the imaginary component
    """
    w = (1.0 - xyz.norm()).unsqueeze(-1)
    assert bool((w >= 0).all()), "xyz has its norm greater than 1"
    return torch.cat([xyz, w], dim=-1)


@torch.jit.script
def quat_identity(shape: List[int]):
    """
    Construct 3D identity rotation given shape
    """
    w = torch.ones(shape + [1])
    xyz = torch.zeros(shape + [3])
    q = torch.cat([xyz, w], dim=-1)
    return quat_normalize(q)


@torch.jit.script
def quat_from_angle_axis(angle, axis, degree: bool = False):
    """ Create a 3D rotation from angle and axis of rotation. The rotation is counter-clockwise 
    along the axis.

    The rotation can be interpreted as a_R_b where frame "b" is the new frame that
    gets rotated counter-clockwise along the axis from frame "a"

    :param angle: angle of rotation
    :type angle: Tensor
    :param axis: axis of rotation
    :type axis: Tensor
    :param degree: put True here if the angle is given by degree
    :type degree: bool, optional, default=False
    """
    if degree:
        angle = angle / 180.0 * math.pi
    theta = (angle / 2).unsqueeze(-1)
    axis = axis / (axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9))
    xyz = axis * theta.sin()
    w = theta.cos()
    return quat_normalize(torch.cat([xyz, w], dim=-1))


@torch.jit.script
def quat_from_rotation_matrix(m):
    """
    Construct a 3D rotation from a valid 3x3 rotation matrices.
    Reference can be found here:
    http://www.cg.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche52.html

    :param m: 3x3 orthogonal rotation matrices.
    :type m: Tensor

    :rtype: Tensor
    """
    m = m.unsqueeze(0)
    diag0 = m[..., 0, 0]
    diag1 = m[..., 1, 1]
    diag2 = m[..., 2, 2]

    # Math stuff.
    w = (((diag0 + diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    x = (((diag0 - diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    y = (((-diag0 + diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    z = (((-diag0 - diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5

    # Only modify quaternions where w > x, y, z.
    c0 = (w >= x) & (w >= y) & (w >= z)
    x[c0] *= (m[..., 2, 1][c0] - m[..., 1, 2][c0]).sign()
    y[c0] *= (m[..., 0, 2][c0] - m[..., 2, 0][c0]).sign()
    z[c0] *= (m[..., 1, 0][c0] - m[..., 0, 1][c0]).sign()

    # Only modify quaternions where x > w, y, z
    c1 = (x >= w) & (x >= y) & (x >= z)
    w[c1] *= (m[..., 2, 1][c1] - m[..., 1, 2][c1]).sign()
    y[c1] *= (m[..., 1, 0][c1] + m[..., 0, 1][c1]).sign()
    z[c1] *= (m[..., 0, 2][c1] + m[..., 2, 0][c1]).sign()

    # Only modify quaternions where y > w, x, z.
    c2 = (y >= w) & (y >= x) & (y >= z)
    w[c2] *= (m[..., 0, 2][c2] - m[..., 2, 0][c2]).sign()
    x[c2] *= (m[..., 1, 0][c2] + m[..., 0, 1][c2]).sign()
    z[c2] *= (m[..., 2, 1][c2] + m[..., 1, 2][c2]).sign()

    # Only modify quaternions where z > w, x, y.
    c3 = (z >= w) & (z >= x) & (z >= y)
    w[c3] *= (m[..., 1, 0][c3] - m[..., 0, 1][c3]).sign()
    x[c3] *= (m[..., 2, 0][c3] + m[..., 0, 2][c3]).sign()
    y[c3] *= (m[..., 2, 1][c3] + m[..., 1, 2][c3]).sign()

    return quat_normalize(torch.stack([x, y, z, w], dim=-1)).squeeze(0)


@torch.jit.script
def quat_mul_norm(x, y):
    """
    Combine two set of 3D rotations together using \**\* operator. The shape needs to be
    broadcastable
    """
    return quat_normalize(quat_mul(x, y))


@torch.jit.script
def quat_rotate(rot, vec):
    """
    Rotate a 3D vector with the 3D rotation
    """
    other_q = torch.cat([vec, torch.zeros_like(vec[..., :1])], dim=-1)
    return quat_imaginary(quat_mul(quat_mul(rot, other_q), quat_conjugate(rot)))


@torch.jit.script
def quat_inverse(x):
    """
    The inverse of the rotation
    """
    return quat_conjugate(x)


@torch.jit.script
def quat_identity_like(x):
    """
    Construct identity 3D rotation with the same shape
    """
    return quat_identity(x.shape[:-1])


@torch.jit.script
def quat_angle_axis(x):
    """
    The (angle, axis) representation of the rotation. The axis is normalized to unit length.
    The angle is guaranteed to be between [0, pi].
    """
    s = 2 * (x[..., 3] ** 2) - 1
    angle = s.clamp(-1, 1).arccos()  # just to be safe
    axis = x[..., :3]
    axis /= axis.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-9)
    return angle, axis


@torch.jit.script
def quat_yaw_rotation(x, z_up: bool = True):
    """
    Yaw rotation (rotation along z-axis)
    """
    q = x
    if z_up:
        q = torch.cat([torch.zeros_like(q[..., 0:2]), q[..., 2:3], q[..., 3:]], dim=-1)
    else:
        q = torch.cat(
            [
                torch.zeros_like(q[..., 0:1]),
                q[..., 1:2],
                torch.zeros_like(q[..., 2:3]),
                q[..., 3:4],
            ],
            dim=-1,
        )
    return quat_normalize(q)


@torch.jit.script
def transform_from_rotation_translation(
    r: Optional[torch.Tensor] = None, t: Optional[torch.Tensor] = None
):
    """
    Construct a transform from a quaternion and 3D translation. Only one of them can be None.
    """
    assert r is not None or t is not None, "rotation and translation can't be all None"
    if r is None:
        assert t is not None
        r = quat_identity(list(t.shape))
    if t is None:
        t = torch.zeros(list(r.shape) + [3])
    return torch.cat([r, t], dim=-1)


@torch.jit.script
def transform_identity(shape: List[int]):
    """
    Identity transformation with given shape
    """
    r = quat_identity(shape)
    t = torch.zeros(shape + [3])
    return transform_from_rotation_translation(r, t)



@torch.jit.script
def transform_rotation(x):
    """Get rotation from transform"""
    return x[..., :4]


@torch.jit.script
def transform_translation(x):
    """Get translation from transform"""
    return x[..., 4:]


@torch.jit.script
def transform_inverse(x):
    """
    Inverse transformation
    """
    inv_so3 = quat_inverse(transform_rotation(x))
    return transform_from_rotation_translation(
        r=inv_so3, t=quat_rotate(inv_so3, -transform_translation(x))
    )


@torch.jit.script
def transform_identity_like(x):
    """
    identity transformation with the same shape
    """
    return transform_identity(x.shape)


@torch.jit.script
def transform_mul(x, y):
    """
    Combine two transformation together
    """
    z = transform_from_rotation_translation(
        r=quat_mul_norm(transform_rotation(x), transform_rotation(y)),
        t=quat_rotate(transform_rotation(x), transform_translation(y))
        + transform_translation(x),
    )
    return z


@torch.jit.script
def transform_apply(rot, vec):
    """
    Transform a 3D vector
    """
    assert isinstance(vec, torch.Tensor)
    return quat_rotate(transform_rotation(rot), vec) + transform_translation(rot)


@torch.jit.script
def rot_matrix_det(x):
    """
    Return the determinant of the 3x3 matrix. The shape of the tensor will be as same as the
    shape of the matrix
    """
    a, b, c = x[..., 0, 0], x[..., 0, 1], x[..., 0, 2]
    d, e, f = x[..., 1, 0], x[..., 1, 1], x[..., 1, 2]
    g, h, i = x[..., 2, 0], x[..., 2, 1], x[..., 2, 2]
    t1 = a * (e * i - f * h)
    t2 = b * (d * i - f * g)
    t3 = c * (d * h - e * g)
    return t1 - t2 + t3


@torch.jit.script
def rot_matrix_integrity_check(x):
    """
    Verify that a rotation matrix has a determinant of one and is orthogonal
    """
    det = rot_matrix_det(x)
    assert bool((abs(det - 1) < 1e-3).all()), "the matrix has non-one determinant"
    rtr = x @ x.permute(torch.arange(x.dim() - 2), -1, -2)
    rtr_gt = rtr.zeros_like()
    rtr_gt[..., 0, 0] = 1
    rtr_gt[..., 1, 1] = 1
    rtr_gt[..., 2, 2] = 1
    assert bool(((rtr - rtr_gt) < 1e-3).all()), "the matrix is not orthogonal"


@torch.jit.script
def rot_matrix_from_quaternion(q):
    """
    Construct rotation matrix from quaternion
    """
    # Shortcuts for individual elements (using wikipedia's convention)
    qi, qj, qk, qr = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    # Set individual elements
    R00 = 1.0 - 2.0 * (qj ** 2 + qk ** 2)
    R01 = 2 * (qi * qj - qk * qr)
    R02 = 2 * (qi * qk + qj * qr)
    R10 = 2 * (qi * qj + qk * qr)
    R11 = 1.0 - 2.0 * (qi ** 2 + qk ** 2)
    R12 = 2 * (qj * qk - qi * qr)
    R20 = 2 * (qi * qk - qj * qr)
    R21 = 2 * (qj * qk + qi * qr)
    R22 = 1.0 - 2.0 * (qi ** 2 + qj ** 2)

    R0 = torch.stack([R00, R01, R02], dim=-1)
    R1 = torch.stack([R10, R11, R12], dim=-1)
    R2 = torch.stack([R10, R21, R22], dim=-1)

    R = torch.stack([R0, R1, R2], dim=-2)

    return R


@torch.jit.script
def euclidean_to_rotation_matrix(x):
    """
    Get the rotation matrix on the top-left corner of a Euclidean transformation matrix
    """
    return x[..., :3, :3]


@torch.jit.script
def euclidean_integrity_check(x):
    euclidean_to_rotation_matrix(x)  # check 3d-rotation matrix
    assert bool((x[..., 3, :3] == 0).all()), "the last row is illegal"
    assert bool((x[..., 3, 3] == 1).all()), "the last row is illegal"


@torch.jit.script
def euclidean_translation(x):
    """
    Get the translation vector located at the last column of the matrix
    """
    return x[..., :3, 3]


@torch.jit.script
def euclidean_inverse(x):
    """
    Compute the matrix that represents the inverse rotation
    """
    s = x.zeros_like()
    irot = quat_inverse(quat_from_rotation_matrix(x))
    s[..., :3, :3] = irot
    s[..., :3, 4] = quat_rotate(irot, -euclidean_translation(x))
    return s


@torch.jit.script
def euclidean_to_transform(transformation_matrix):
    """
    Construct a transform from a Euclidean transformation matrix
    """
    return transform_from_rotation_translation(
        r=quat_from_rotation_matrix(
            m=euclidean_to_rotation_matrix(transformation_matrix)
        ),
        t=euclidean_translation(transformation_matrix),
    )

