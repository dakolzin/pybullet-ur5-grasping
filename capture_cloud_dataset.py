#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import random
import numpy as np
import pybullet as p
import pybullet_data

# =========================
# Конфиг
# =========================

USE_GUI = True
SIM_DT = 1.0 / 240.0
GRAVITY = -9.8

OUT_DIR = "./dataset_clouds"
SAVE_PLY = True

MESH_FILES = [
    "./meshes/part/part_1.stl",
    "./meshes/part/part_2.stl",
    "./meshes/part/part_3.stl",
]
MESH_SCALE = (1.0, 1.0, 1.0)  # если STL в мм -> (0.001,0.001,0.001)
OBJ_MASS = 0.2

# Спавн объекта
SPAWN_X_RANGE = (0.35, 0.65)
SPAWN_Y_RANGE = (-0.12, 0.12)
SPAWN_Z = 0.75

# Камера
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_NEAR = 0.05
CAM_FAR = 2.0
DOWNSAMPLE = 1

# Две камеры (как в статье: два сенсора)
# Камера 0 — слева-снизу
# Камера 1 — справа-сверху (другая сторона)
CAMS = [
    {
        "name": "cam0",
        "pos": (0.35, -0.45, 0.95),
        "target": (0.50, 0.00, 0.68),
        "up": (0.0, 0.0, 1.0),
        "fov": 55.0
    },
    {
        "name": "cam1",
        "pos": (0.75, 0.25, 0.95),
        "target": (0.50, 0.00, 0.68),
        "up": (0.0, 0.0, 1.0),
        "fov": 55.0
    }
]

NUM_SCENES = 50

# Оставлять только объект по seg? (для отладки генераторов — да)
KEEP_ONLY_OBJECT_POINTS = True

# Если нужно склеивать без дублей: простой voxel merge
# (0 = не делать, >0 — размер в метрах)
VOXEL_MERGE = 0.002


# =========================
# Утилиты: камера -> облако
# =========================

def make_camera_matrices(cam_pos, target_pos, up, fov, aspect, near, far):
    view = p.computeViewMatrix(cam_pos, target_pos, up)
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)
    return view, proj

def depth_to_points_world(depth, view_matrix, proj_matrix, width, height, downsample=1):
    depth = np.asarray(depth, dtype=np.float32).reshape((height, width))

    xs = np.arange(0, width, downsample)
    ys = np.arange(0, height, downsample)
    xv, yv = np.meshgrid(xs, ys)

    x_ndc = (xv + 0.5) / width * 2.0 - 1.0
    y_ndc = 1.0 - (yv + 0.5) / height * 2.0

    d = depth[::downsample, ::downsample]
    z_ndc = 2.0 * d - 1.0

    ones = np.ones_like(z_ndc, dtype=np.float32)
    clip = np.stack([x_ndc.astype(np.float32),
                     y_ndc.astype(np.float32),
                     z_ndc.astype(np.float32),
                     ones], axis=-1).reshape(-1, 4)

    view = np.array(view_matrix, dtype=np.float32).reshape(4, 4).T
    proj = np.array(proj_matrix, dtype=np.float32).reshape(4, 4).T
    inv = np.linalg.inv(proj @ view)

    world = (inv @ clip.T).T
    world = world[:, :3] / world[:, 3:4]

    valid = (d.reshape(-1) < 0.9999)
    return world[valid], valid

def voxel_unique(points: np.ndarray, voxel: float) -> np.ndarray:
    if voxel <= 0:
        return points
    q = np.floor(points / voxel).astype(np.int32)
    # уникальность по квантованным координатам
    _, idx = np.unique(q, axis=0, return_index=True)
    return points[np.sort(idx)]

def save_ply_xyz(path: str, pts: np.ndarray):
    pts = np.asarray(pts, dtype=np.float32)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {pts.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for x, y, z in pts:
            f.write(f"{x} {y} {z}\n")


# =========================
# Утилиты: сим
# =========================

def step_sim(n=240):
    for _ in range(n):
        p.stepSimulation()
        if USE_GUI:
            time.sleep(SIM_DT)

def spawn_mesh(mesh_path: str, pos, orn_euler, mass, scale):
    orn = p.getQuaternionFromEuler(list(orn_euler))
    col = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, meshScale=list(scale))
    vis = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path, meshScale=list(scale),
                              rgbaColor=[0.7, 0.7, 0.7, 1.0])
    obj_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=orn
    )
    p.changeDynamics(obj_id, -1, lateralFriction=1.0, spinningFriction=0.01, rollingFriction=0.001)
    return obj_id

def random_pose():
    x = random.uniform(*SPAWN_X_RANGE)
    y = random.uniform(*SPAWN_Y_RANGE)
    z = SPAWN_Z
    yaw = random.uniform(-math.pi, math.pi)
    return [x, y, z], (0.0, 0.0, yaw)

def setup_scene():
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(SIM_DT)
    p.setGravity(0, 0, GRAVITY)

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))


def capture_one_camera(cam_cfg: dict, obj_id: int):
    aspect = CAM_WIDTH / float(CAM_HEIGHT)
    view, proj = make_camera_matrices(
        cam_cfg["pos"], cam_cfg["target"], cam_cfg["up"],
        cam_cfg["fov"], aspect, CAM_NEAR, CAM_FAR
    )

    img = p.getCameraImage(
        width=CAM_WIDTH,
        height=CAM_HEIGHT,
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgba = np.reshape(img[2], (CAM_HEIGHT, CAM_WIDTH, 4)).astype(np.uint8)
    depth = np.reshape(img[3], (CAM_HEIGHT, CAM_WIDTH)).astype(np.float32)
    seg = np.reshape(img[4], (CAM_HEIGHT, CAM_WIDTH)).astype(np.int32)

    pts_world, valid = depth_to_points_world(depth, view, proj, CAM_WIDTH, CAM_HEIGHT, downsample=DOWNSAMPLE)

    seg_ds = seg[::DOWNSAMPLE, ::DOWNSAMPLE].reshape(-1)
    seg_valid = seg_ds[valid]

    if KEEP_ONLY_OBJECT_POINTS:
        mask = (seg_valid == obj_id)
        pts_world = pts_world[mask]
        seg_valid = seg_valid[mask]

    return rgba, depth, seg, pts_world, seg_valid, view, proj


def capture_scene(obj_id: int):
    all_pts = []
    outputs = {}

    for k, cam in enumerate(CAMS):
        rgba, depth, seg, pts_world, seg_valid, view, proj = capture_one_camera(cam, obj_id)

        outputs[f"rgba_cam{k}"] = rgba
        outputs[f"depth_cam{k}"] = depth
        outputs[f"seg_cam{k}"] = seg
        outputs[f"points_world_cam{k}"] = pts_world.astype(np.float32)
        outputs[f"seg_ids_cam{k}"] = seg_valid.astype(np.int32)
        outputs[f"view_cam{k}"] = np.array(view, dtype=np.float32)
        outputs[f"proj_cam{k}"] = np.array(proj, dtype=np.float32)

        all_pts.append(pts_world)

    points_world = np.concatenate(all_pts, axis=0) if len(all_pts) > 0 else np.zeros((0, 3), dtype=np.float32)
    points_world = voxel_unique(points_world, VOXEL_MERGE).astype(np.float32)

    meta = {
        "cameras": [
            {
                "name": cam["name"],
                "pos": cam["pos"],
                "target": cam["target"],
                "up": cam["up"],
                "width": CAM_WIDTH,
                "height": CAM_HEIGHT,
                "fov": cam["fov"],
                "near": CAM_NEAR,
                "far": CAM_FAR,
                "downsample": DOWNSAMPLE
            }
            for cam in CAMS
        ],
        "voxel_merge": VOXEL_MERGE,
        "keep_only_object_points": KEEP_ONLY_OBJECT_POINTS
    }

    return points_world, outputs, meta


def ensure_out_dir():
    os.makedirs(OUT_DIR, exist_ok=True)


def main():
    ensure_out_dir()

    p.connect(p.GUI if USE_GUI else p.DIRECT)
    print("Connected. Saving to:", os.path.abspath(OUT_DIR))
    print("Cameras:", len(CAMS), " | voxel merge:", VOXEL_MERGE)

    for i in range(NUM_SCENES):
        setup_scene()

        mesh_path = random.choice(MESH_FILES)
        pos, rpy = random_pose()
        obj_id = spawn_mesh(mesh_path, pos, rpy, OBJ_MASS, MESH_SCALE)

        step_sim(360)  # стабилизация

        base_pos, base_orn = p.getBasePositionAndOrientation(obj_id)

        points_world, outputs, meta = capture_scene(obj_id)

        meta["scene_index"] = i
        meta["mesh_path"] = mesh_path
        meta["object"] = {
            "id": int(obj_id),
            "base_pos": list(base_pos),
            "base_orn_xyzw": list(base_orn),
        }

        stem = f"scene_{i:06d}"
        npz_path = os.path.join(OUT_DIR, stem + ".npz")
        json_path = os.path.join(OUT_DIR, stem + ".json")
        ply_path = os.path.join(OUT_DIR, stem + ".ply")

        # Сохраняем объединённое + всё по камерам
        np.savez_compressed(
            npz_path,
            points_world=points_world,
            **outputs
        )

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        if SAVE_PLY:
            save_ply_xyz(ply_path, points_world)

        # лог по точкам
        n0 = outputs["points_world_cam0"].shape[0]
        n1 = outputs["points_world_cam1"].shape[0] if "points_world_cam1" in outputs else 0
        print(f"[{i+1}/{NUM_SCENES}] {stem}: cam0={n0} cam1={n1} merged={points_world.shape[0]} saved={npz_path}")

    p.disconnect()
    print("Done.")


if __name__ == "__main__":
    main()
