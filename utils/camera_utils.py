# -*- coding: utf-8 -*-
"""
utils/camera_utils.py — камера PyBullet, depth->point cloud, фильтрация по segmentation mask.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np
import pybullet as p


def make_camera_matrices(
    cam_pos: Sequence[float],
    target_pos: Sequence[float],
    up: Sequence[float] = (0, 0, 1),
    fov: float = 60.0,
    aspect: float = 1.0,
    near: float = 0.01,
    far: float = 2.0,
) -> Tuple[list, list]:
    view = p.computeViewMatrix(
        cameraEyePosition=list(cam_pos),
        cameraTargetPosition=list(target_pos),
        cameraUpVector=list(up),
    )
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)
    return view, proj


def depth_to_point_cloud_with_mask(
    depth: np.ndarray,
    seg: np.ndarray,
    view_matrix,
    proj_matrix,
    width: int,
    height: int,
    downsample: int = 2,
    keep_body_uid: Optional[int] = None,
) -> np.ndarray:
    """
    depth: (H,W) float32 из PyBullet (0..1)
    seg  : (H,W) int32 segmentation
    Возвращает Nx3 point cloud в world.
    Если keep_body_uid задан — оставляет только точки соответствующего body.
    """
    depth = np.asarray(depth, dtype=np.float32).reshape((height, width))
    seg = np.asarray(seg, dtype=np.int32).reshape((height, width))

    xs = np.arange(0, width, downsample)
    ys = np.arange(0, height, downsample)
    xv, yv = np.meshgrid(xs, ys)

    d_ds = depth[::downsample, ::downsample]
    seg_ds = seg[::downsample, ::downsample]

    valid = (d_ds < 0.9999) & (seg_ds >= 0)

    if keep_body_uid is not None:
        # В seg PyBullet uid хранится в младших 24 битах, старшие 8 могут использоваться по-другому,
        # поэтому проверяем оба варианта как у тебя было.
        obj_uid_low = (seg_ds & ((1 << 24) - 1)).astype(np.int32)
        obj_uid_high = ((seg_ds >> 24) & 0xFF).astype(np.int32)
        uid = int(keep_body_uid)
        valid = valid & ((obj_uid_low == uid) | (obj_uid_high == uid))

    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)

    # NDC
    x_ndc = (xv + 0.5) / width * 2.0 - 1.0
    y_ndc = 1.0 - (yv + 0.5) / height * 2.0

    # clip space
    clip = np.stack(
        [
            x_ndc.astype(np.float32),
            y_ndc.astype(np.float32),
            (2.0 * d_ds - 1.0).astype(np.float32),
            np.ones_like(d_ds, dtype=np.float32),
        ],
        axis=-1,
    ).reshape(-1, 4)

    view = np.array(view_matrix, dtype=np.float32).reshape(4, 4).T
    proj = np.array(proj_matrix, dtype=np.float32).reshape(4, 4).T
    inv = np.linalg.inv(proj @ view)

    world = (inv @ clip.T).T
    world = world[:, :3] / world[:, 3:4]

    pts = world[valid.reshape(-1)].astype(np.float64)
    return pts


def get_depth_and_point_cloud(
    cam_pos: Sequence[float],
    target_pos: Sequence[float],
    width: int = 640,
    height: int = 480,
    fov: float = 60.0,
    near: float = 0.05,
    far: float = 2.0,
    downsample: int = 2,
    keep_body_uid: Optional[int] = None,
    renderer: int = p.ER_BULLET_HARDWARE_OPENGL,
) -> np.ndarray:
    aspect = width / float(height)
    view, proj = make_camera_matrices(
        cam_pos=cam_pos,
        target_pos=target_pos,
        fov=fov,
        aspect=aspect,
        near=near,
        far=far,
    )

    img = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=renderer,
    )

    depth = np.reshape(img[3], (height, width)).astype(np.float32)
    seg = np.reshape(img[4], (height, width)).astype(np.int32)

    pts = depth_to_point_cloud_with_mask(
        depth=depth,
        seg=seg,
        view_matrix=view,
        proj_matrix=proj,
        width=width,
        height=height,
        downsample=downsample,
        keep_body_uid=keep_body_uid,
    )
    return pts
