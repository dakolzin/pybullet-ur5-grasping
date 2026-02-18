#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Визуализация камер в PyBullet для рисунка в отчет.

Что делает:
- Загружает плоскость + стол
- Спавнит один объект (случайный или заданный)
- Рисует для каждой камеры:
  1) точку положения
  2) оси камеры (X/Y/Z)
  3) пирамиду видимости (frustum) по FOV/near/far
  4) луч (pos -> target)

Управление:
- Esc — выход
"""

import os
import math
import time
import random
import argparse
import numpy as np
import pybullet as p
import pybullet_data


# =========================
# Конфиг по умолчанию (совместим с твоим генератором)
# =========================
SIM_DT = 1.0 / 240.0
GRAVITY = -9.8

MESH_FILES = [
    "./meshes/part/part_1.stl",
    "./meshes/part/part_2.stl",
    "./meshes/part/part_3.stl",
]
MESH_SCALE = (1.0, 1.0, 1.0)  # если STL в мм -> (0.001,0.001,0.001)
OBJ_MASS = 0.2

SPAWN_X_RANGE = (0.35, 0.65)
SPAWN_Y_RANGE = (-0.12, 0.12)
SPAWN_Z = 0.75

CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_NEAR = 0.05
CAM_FAR = 2.0

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


# =========================
# Вспомогательная геометрия
# =========================
def _v3(x):
    return [float(x[0]), float(x[1]), float(x[2])]

def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n

def camera_axes_from_lookat(pos, target, up):
    """
    Возвращает ортонормальный базис камеры в мировых координатах:
    fwd  — направление взгляда (вперед)
    right — вправо
    cam_up — вверх (скорректированный)
    """
    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    fwd = normalize(target - pos)              # вперед
    right = normalize(np.cross(fwd, up))       # вправо
    cam_up = normalize(np.cross(right, fwd))   # вверх (ортогонально)
    return fwd, right, cam_up

def frustum_corners(pos, fwd, right, cam_up, fov_deg, aspect, near, far):
    """
    Возвращает 8 углов усеченной пирамиды: 4 near + 4 far.
    Порядок: n00,n01,n11,n10, f00,f01,f11,f10 (против часовой при взгляде вперед).
    """
    fov = math.radians(float(fov_deg))
    tan = math.tan(0.5 * fov)

    nh = near * tan
    nw = nh * aspect
    fh = far * tan
    fw = fh * aspect

    pos = np.asarray(pos, dtype=np.float64)
    nc = pos + fwd * near
    fc = pos + fwd * far

    # near plane
    n00 = nc - right * nw - cam_up * nh
    n01 = nc - right * nw + cam_up * nh
    n11 = nc + right * nw + cam_up * nh
    n10 = nc + right * nw - cam_up * nh

    # far plane
    f00 = fc - right * fw - cam_up * fh
    f01 = fc - right * fw + cam_up * fh
    f11 = fc + right * fw + cam_up * fh
    f10 = fc + right * fw - cam_up * fh

    return [n00, n01, n11, n10, f00, f01, f11, f10]


# =========================
# Отрисовка
# =========================
def draw_axes(origin, fwd, right, up, s=0.10, life=0):
    """
    Оси камеры:
    X (right) — красный
    Y (up)    — зеленый
    Z (fwd)   — синий
    """
    o = np.asarray(origin, dtype=np.float64)
    p.addUserDebugLine(_v3(o), _v3(o + right * s), [1, 0, 0], 3, lifeTime=life)
    p.addUserDebugLine(_v3(o), _v3(o + up * s),    [0, 1, 0], 3, lifeTime=life)
    p.addUserDebugLine(_v3(o), _v3(o + fwd * s),   [0, 0, 1], 3, lifeTime=life)

def draw_frustum(pos, target, up, fov_deg, near, far, aspect, color=(1, 1, 0), life=0):
    pos = np.asarray(pos, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    fwd, right, cam_up = camera_axes_from_lookat(pos, target, up)

    # луч направления взгляда
    p.addUserDebugLine(_v3(pos), _v3(target), [0, 1, 1], 2, lifeTime=life)

    # оси
    draw_axes(pos, fwd, right, cam_up, s=0.12, life=life)

    # усеченная пирамида
    c = frustum_corners(pos, fwd, right, cam_up, fov_deg, aspect, near, far)
    n00, n01, n11, n10, f00, f01, f11, f10 = c

    col = list(color)

    # near rectangle
    p.addUserDebugLine(_v3(n00), _v3(n01), col, 2, lifeTime=life)
    p.addUserDebugLine(_v3(n01), _v3(n11), col, 2, lifeTime=life)
    p.addUserDebugLine(_v3(n11), _v3(n10), col, 2, lifeTime=life)
    p.addUserDebugLine(_v3(n10), _v3(n00), col, 2, lifeTime=life)

    # far rectangle
    p.addUserDebugLine(_v3(f00), _v3(f01), col, 2, lifeTime=life)
    p.addUserDebugLine(_v3(f01), _v3(f11), col, 2, lifeTime=life)
    p.addUserDebugLine(_v3(f11), _v3(f10), col, 2, lifeTime=life)
    p.addUserDebugLine(_v3(f10), _v3(f00), col, 2, lifeTime=life)

    # side edges
    p.addUserDebugLine(_v3(n00), _v3(f00), col, 1, lifeTime=life)
    p.addUserDebugLine(_v3(n01), _v3(f01), col, 1, lifeTime=life)
    p.addUserDebugLine(_v3(n11), _v3(f11), col, 1, lifeTime=life)
    p.addUserDebugLine(_v3(n10), _v3(f10), col, 1, lifeTime=life)

    # метка имени камеры
    try:
        p.addUserDebugText(str("cam"), _v3(pos + fwd * 0.02), textColorRGB=[1, 1, 1], textSize=1.2, lifeTime=life)
    except Exception:
        pass


# =========================
# Сим / сцена
# =========================
def setup_scene():
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(SIM_DT)
    p.setGravity(0, 0, GRAVITY)

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))

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

def set_nice_overview_camera():
    # просто удобный общий ракурс
    p.resetDebugVisualizerCamera(
        cameraDistance=1.1,
        cameraYaw=35,
        cameraPitch=-35,
        cameraTargetPosition=[0.5, 0.0, 0.65]
    )


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", type=str, default="", help="Path to mesh (stl/obj). If empty -> random from MESH_FILES.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed (0 = do not set).")
    ap.add_argument("--near", type=float, default=CAM_NEAR)
    ap.add_argument("--far", type=float, default=CAM_FAR)
    ap.add_argument("--fov_scale", type=float, default=1.0, help="Multiply each camera FOV by this factor.")
    ap.add_argument("--pause", action="store_true", help="Pause (no stepping) - only camera visualization.")
    args = ap.parse_args()

    if args.seed != 0:
        random.seed(args.seed)
        np.random.seed(args.seed)

    p.connect(p.GUI)
    setup_scene()
    set_nice_overview_camera()

    mesh_path = args.mesh if args.mesh else random.choice(MESH_FILES)
    pos, rpy = random_pose()
    _ = spawn_mesh(mesh_path, pos, rpy, OBJ_MASS, MESH_SCALE)

    # чуть стабилизируем
    for _ in range(120):
        p.stepSimulation()
        time.sleep(SIM_DT)

    p.removeAllUserDebugItems()

    aspect = CAM_WIDTH / float(CAM_HEIGHT)

    # разные цвета для камер
    colors = [
        (1, 1, 0),   # yellow
        (1, 0, 1),   # magenta
        (0, 1, 0),   # green
        (1, 0.5, 0), # orange
    ]

    for i, cam in enumerate(CAMS):
        col = colors[i % len(colors)]
        draw_frustum(
            pos=np.array(cam["pos"], dtype=np.float64),
            target=np.array(cam["target"], dtype=np.float64),
            up=np.array(cam["up"], dtype=np.float64),
            fov_deg=float(cam["fov"]) * float(args.fov_scale),
            near=float(args.near),
            far=float(args.far),
            aspect=aspect,
            color=col,
            life=0
        )
        # подпись
        try:
            p.addUserDebugText(
                cam["name"],
                _v3(np.array(cam["pos"], dtype=np.float64) + np.array([0.0, 0.0, 0.03])),
                textColorRGB=[1, 1, 1],
                textSize=1.3,
                lifeTime=0
            )
        except Exception:
            pass

    print("Camera visualization ready.")
    print("Tips:")
    print(" - Сделай скриншот из GUI для отчета.")
    print(" - Esc для выхода.")

    KEY_ESC = 27
    while True:
        keys = p.getKeyboardEvents()
        if KEY_ESC in keys and (keys[KEY_ESC] & p.KEY_WAS_TRIGGERED):
            break

        if not args.pause:
            p.stepSimulation()
        time.sleep(0.01)

    p.disconnect()


if __name__ == "__main__":
    main()
