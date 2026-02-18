# -*- coding: utf-8 -*-
"""
utils/calib_utils.py — калибровка связи grasp-frame -> ee_link.

Возвращает матрицу M = R_EE_FROM_GRASP такую, что:
    R_tcp_world = R_grasp_world @ M

Где grasp-frame: x=width, y=approach, z=finger (как у твоего генератора).
"""

from __future__ import annotations

import numpy as np
import pybullet as p

from utils.rot_utils import unit, best_axis_id_and_sign, quat_to_R
from utils.robot_ur5_robotiq85 import UR5Robotiq85, get_eef_world_link_frame


def calibrate_R_EE_FROM_GRASP(
    robot: UR5Robotiq85,
    left_tip_link: int = 12,   # left_inner_finger_pad
    right_tip_link: int = 17,  # right_inner_finger_pad
) -> np.ndarray:
    """
    Возвращает матрицу M = R_EE_FROM_GRASP.
    """
    ee_pos, ee_quat = get_eef_world_link_frame(robot)
    Re = quat_to_R(ee_quat)  # columns = ee_link local axes in world

    # fingertip positions
    lsL = p.getLinkState(robot.id, left_tip_link, computeForwardKinematics=True)
    lsR = p.getLinkState(robot.id, right_tip_link, computeForwardKinematics=True)
    pL = np.array(lsL[4], dtype=np.float64)
    pR = np.array(lsR[4], dtype=np.float64)

    # WIDTH direction in world: left->right
    w_world = unit(pR - pL)

    # APPROACH direction in world: ee_link -> midpoint of fingertips
    mid = 0.5 * (pL + pR)
    a_world = unit(mid - np.asarray(ee_pos, dtype=np.float64))

    # express these in ee_link local coordinates
    w_local = Re.T @ w_world
    a_local = Re.T @ a_world

    w_idx, w_sgn = best_axis_id_and_sign(w_local)
    a_idx, a_sgn = best_axis_id_and_sign(a_local)

    # если вдруг совпали — перекинем approach на следующую лучшую ось
    if a_idx == w_idx:
        order = np.argsort(-np.abs(a_local))
        for j in order:
            j = int(j)
            if j != w_idx:
                a_idx = j
                a_sgn = float(np.sign(a_local[a_idx])) or 1.0
                break

    # строим M (в grasp-координатах!)
    # хотим: ee_local_axis[w_idx]  -> grasp_x
    #        ee_local_axis[a_idx]  -> grasp_y
    M = np.zeros((3, 3), dtype=np.float64)
    # колонка j = ee_local_axis_j выраженная в grasp-координатах
    M[:, w_idx] = w_sgn * np.array([1.0, 0.0, 0.0], dtype=np.float64)  # grasp x
    M[:, a_idx] = a_sgn * np.array([0.0, 1.0, 0.0], dtype=np.float64)  # grasp y

    # третья ось — чтобы сохранить правую СК
    k = 3 - w_idx - a_idx
    col_k = unit(np.cross(M[:, w_idx], M[:, a_idx]))
    M[:, k] = col_k
    if np.linalg.det(M) < 0.0:
        M[:, k] *= -1.0

    print(f"[CALIB] ee width axis id={w_idx} sign={w_sgn:+.0f} | ee approach axis id={a_idx} sign={a_sgn:+.0f}")
    print("[CALIB] R_EE_FROM_GRASP =\n", M)
    return M
