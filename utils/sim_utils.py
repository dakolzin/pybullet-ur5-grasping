# -*- coding: utf-8 -*-
"""
utils/sim_utils.py — подключение PyBullet, шаги симуляции, сцена, спавн объектов.
"""

from __future__ import annotations

import time
from typing import Sequence, Tuple, Optional, Iterable

import pybullet as p
import pybullet_data


def connect(use_gui: bool) -> None:
    if use_gui:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)


def setup_world(sim_dt: float, gravity: float) -> None:
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(sim_dt)
    p.setGravity(0, 0, gravity)


def set_debug_camera(camera_distance: float,
                     camera_yaw: float,
                     camera_pitch: float,
                     camera_target: Sequence[float]) -> None:
    p.resetDebugVisualizerCamera(
        cameraDistance=camera_distance,
        cameraYaw=camera_yaw,
        cameraPitch=camera_pitch,
        cameraTargetPosition=list(camera_target),
    )


def step_sim(steps: int, sim_dt: float, use_gui: bool, sleep_in_gui: bool = True) -> None:
    for _ in range(int(steps)):
        p.stepSimulation()
        if use_gui and sleep_in_gui:
            time.sleep(sim_dt)


def pause_sec(sec: float, sim_dt: float, use_gui: bool, sleep_in_gui: bool = True) -> None:
    if sec <= 0:
        return
    if use_gui and sleep_in_gui:
        time.sleep(sec)
    else:
        step_sim(int(sec / sim_dt), sim_dt=sim_dt, use_gui=use_gui, sleep_in_gui=False)


def load_scene(plane: bool = True,
               table: bool = True,
               tray: bool = True,
               tray_pos: Sequence[float] = (0.5, 0.9, 0.6)) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """
    Возвращает (plane_id, table_id, tray_id). Можно игнорировать, но tray_id полезен для отключения коллизий.
    """
    plane_id = None
    table_id = None
    tray_id = None

    if plane:
        plane_id = p.loadURDF("plane.urdf")
    if table:
        table_id = p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))
    if tray:
        tray_id = p.loadURDF("tray/tray.urdf", list(tray_pos), p.getQuaternionFromEuler([0, 0, 0]))

    return plane_id, table_id, tray_id


def spawn_mesh_object(mesh_path: str,
                      pos: Sequence[float],
                      orn_euler: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                      mass: float = 0.2,
                      scale: Tuple[float, float, float] = (1.0, 1.0, 1.0),
                      rgba=(0.8, 0.8, 0.8, 1.0),
                      friction: float = 0.9,
                      spin_friction: float = 0.01,
                      roll_friction: float = 0.001) -> int:
    orn = p.getQuaternionFromEuler(list(orn_euler))

    col = p.createCollisionShape(p.GEOM_MESH, fileName=mesh_path, meshScale=list(scale))
    vis = p.createVisualShape(p.GEOM_MESH, fileName=mesh_path, meshScale=list(scale), rgbaColor=list(rgba))

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(pos),
        baseOrientation=orn,
    )

    p.changeDynamics(
        body_id, -1,
        lateralFriction=friction,
        spinningFriction=spin_friction,
        rollingFriction=roll_friction,
    )
    return body_id


# -------------------------
# Extra helpers you use in motion_utils
# -------------------------
def set_body_friction(body_id: int, link_id: int,
                      lateral: float, spinning: float, rolling: float,
                      restitution: float = 0.0) -> None:
    p.changeDynamics(
        body_id, link_id,
        lateralFriction=float(lateral),
        spinningFriction=float(spinning),
        rollingFriction=float(rolling),
        restitution=float(restitution),
    )


def set_links_friction(body_id: int, link_ids: Iterable[int],
                       lateral: float, spinning: float, rolling: float,
                       restitution: float = 0.0) -> None:
    for li in link_ids:
        set_body_friction(body_id, int(li), lateral, spinning, rolling, restitution)


def disable_collisions_between_bodies(body_a: int, body_b: int,
                                      links_a: Optional[Iterable[int]] = None,
                                      links_b: Optional[Iterable[int]] = None) -> None:
    """
    Если links_* = None -> отключаем для всех линков (включая base=-1).
    """
    if links_a is None:
        links_a = [-1] + list(range(p.getNumJoints(body_a)))
    if links_b is None:
        links_b = [-1] + list(range(p.getNumJoints(body_b)))

    for la in links_a:
        for lb in links_b:
            p.setCollisionFilterPair(body_a, body_b, int(la), int(lb), enableCollision=0)


def disconnect() -> None:
    try:
        p.disconnect()
    except Exception:
        pass
