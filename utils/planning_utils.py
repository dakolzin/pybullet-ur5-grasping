# utils/planning_utils.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import random
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pybullet as p

import config as cfg
from utils import sim_utils as su


def arm_joints(robot_id: int, arm_dofs: int = 6) -> List[int]:
    """Возвращает id суставов руки (первые 6 controllable у тебя)."""
    joints = []
    for jid in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, jid)
        jtype = info[2]
        if jtype != p.JOINT_FIXED:
            joints.append(jid)
        if len(joints) >= arm_dofs:
            break
    return joints


def get_q(robot_id: int, joints: Sequence[int]) -> np.ndarray:
    q = []
    for jid in joints:
        q.append(p.getJointState(robot_id, jid)[0])
    return np.array(q, dtype=np.float64)


def set_q_position_control(robot_id: int, joints: Sequence[int], q: Sequence[float],
                           max_vel: float = None,
                           force: float = None,
                           pos_gain: float = None,
                           vel_gain: float = None):
    if max_vel is None:
        max_vel = cfg.ARM_MAX_VEL
    if force is None:
        force = cfg.ARM_JOINT_FORCE
    if pos_gain is None:
        pos_gain = cfg.POS_GAIN
    if vel_gain is None:
        vel_gain = cfg.VEL_GAIN

    for jid, qi in zip(joints, q):
        p.setJointMotorControl2(
            robot_id, jid,
            p.POSITION_CONTROL,
            targetPosition=float(qi),
            force=float(force),
            maxVelocity=float(max_vel),
            positionGain=float(pos_gain),
            velocityGain=float(vel_gain),
        )


def step_hold(steps: int):
    su.step_sim(steps, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)


def clamp_to_limits(q: np.ndarray,
                    lower: Sequence[float],
                    upper: Sequence[float]) -> np.ndarray:
    q = np.array(q, dtype=np.float64)
    lo = np.array(lower, dtype=np.float64)
    up = np.array(upper, dtype=np.float64)
    return np.minimum(np.maximum(q, lo), up)


def collision_free(robot_id: int,
                   obstacles: Sequence[int],
                   min_dist: float = 0.002) -> bool:
    """
    Быстрая проверка: нет ли близких точек между robot и obstacles.
    min_dist можно поднять до 0.005..0.01 чтобы держаться подальше.
    """
    for obs in obstacles:
        pts = p.getClosestPoints(bodyA=robot_id, bodyB=obs, distance=min_dist)
        if len(pts) > 0:
            return False
    return True


def path_collision_free(robot_id: int,
                        joints: Sequence[int],
                        q_path: Sequence[Sequence[float]],
                        obstacles: Sequence[int],
                        min_dist: float = 0.002) -> bool:
    # важно: проверяем в "кинематическом" режиме, без симуляции
    saved = get_q(robot_id, joints)
    try:
        for q in q_path:
            for jid, qi in zip(joints, q):
                p.resetJointState(robot_id, jid, float(qi), 0.0)
            if not collision_free(robot_id, obstacles, min_dist=min_dist):
                return False
        return True
    finally:
        # вернуть как было
        for jid, qi in zip(joints, saved):
            p.resetJointState(robot_id, jid, float(qi), 0.0)


def interpolate_q(q0: np.ndarray, q1: np.ndarray, step: float = 0.05) -> List[np.ndarray]:
    """
    step ~ макс изменение по суставу на шаг (рад).
    """
    q0 = np.array(q0, dtype=np.float64)
    q1 = np.array(q1, dtype=np.float64)
    d = np.abs(q1 - q0)
    n = int(max(2, math.ceil(float(np.max(d)) / float(step))))
    out = []
    for i in range(n + 1):
        t = i / float(n)
        out.append((1.0 - t) * q0 + t * q1)
    return out


def ik_arm_seeded(robot,
                  target_pos: Sequence[float],
                  target_quat: Sequence[float]) -> Optional[np.ndarray]:
    """
    Главный фикс твоей боли:
    IK считаем с seed = текущему q (как restPoses) => IK почти всегда остаётся в том же branch.
    """
    # текущая конфигурация как seed
    joints = robot.arm_controllable_joints
    q_seed = get_q(robot.id, joints)

    q = p.calculateInverseKinematics(
        robot.id,
        robot.eef_id,
        target_pos,
        target_quat,
        lowerLimits=robot.arm_lower_limits,
        upperLimits=robot.arm_upper_limits,
        jointRanges=robot.arm_joint_ranges,
        restPoses=q_seed.tolist(),          # ВОТ ЭТО КЛЮЧ
        maxNumIterations=300,
        residualThreshold=1e-4,
    )

    q = np.array(q[: len(joints)], dtype=np.float64)
    q = clamp_to_limits(q, robot.arm_lower_limits, robot.arm_upper_limits)
    return q


# -------------------------
# RRT-Connect (минимальный)
# -------------------------

class Node:
    __slots__ = ("q", "parent")
    def __init__(self, q: np.ndarray, parent: int):
        self.q = q
        self.parent = parent


def dist(q0: np.ndarray, q1: np.ndarray) -> float:
    return float(np.linalg.norm(q0 - q1))


def nearest(tree: List[Node], q: np.ndarray) -> int:
    best_i = 0
    best_d = 1e18
    for i, n in enumerate(tree):
        d = dist(n.q, q)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def steer(q_from: np.ndarray, q_to: np.ndarray, step_size: float) -> np.ndarray:
    v = q_to - q_from
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return q_from.copy()
    if n <= step_size:
        return q_to.copy()
    return q_from + (step_size / n) * v


def reconstruct(tree: List[Node], idx: int) -> List[np.ndarray]:
    path = []
    while idx != -1:
        path.append(tree[idx].q)
        idx = tree[idx].parent
    path.reverse()
    return path


def rrt_connect(robot_id: int,
                joints: Sequence[int],
                q_start: np.ndarray,
                q_goal: np.ndarray,
                lower: Sequence[float],
                upper: Sequence[float],
                obstacles: Sequence[int],
                min_dist: float = 0.002,
                step_size: float = 0.15,
                max_iters: int = 2000,
                goal_bias: float = 0.25) -> Optional[List[np.ndarray]]:

    lower = np.array(lower, dtype=np.float64)
    upper = np.array(upper, dtype=np.float64)

    def sample() -> np.ndarray:
        if random.random() < goal_bias:
            return q_goal.copy()
        r = np.random.rand(len(lower))
        return lower + r * (upper - lower)

    def collision_at(q: np.ndarray) -> bool:
        # проверяем коллизию на конкретной конфигурации (resetJointState)
        for jid, qi in zip(joints, q):
            p.resetJointState(robot_id, jid, float(qi), 0.0)
        return collision_free(robot_id, obstacles, min_dist=min_dist)

    # сохранить текущее, чтобы потом вернуть
    saved = get_q(robot_id, joints)

    try:
        # старт и цель должны быть коллизийно свободны
        if not collision_at(q_start):
            return None
        if not collision_at(q_goal):
            return None

        Ta = [Node(q_start.copy(), -1)]
        Tb = [Node(q_goal.copy(), -1)]

        for _ in range(max_iters):
            q_rand = sample()

            # extend Ta towards q_rand
            ia = nearest(Ta, q_rand)
            q_new = steer(Ta[ia].q, q_rand, step_size)
            q_new = clamp_to_limits(q_new, lower, upper)
            if collision_at(q_new):
                Ta.append(Node(q_new, ia))
                # try connect Tb towards q_new
                reached = False
                while True:
                    ib = nearest(Tb, q_new)
                    q_next = steer(Tb[ib].q, q_new, step_size)
                    q_next = clamp_to_limits(q_next, lower, upper)
                    if not collision_at(q_next):
                        break
                    Tb.append(Node(q_next, ib))
                    if dist(q_next, q_new) < 1e-3:
                        reached = True
                        break
                if reached:
                    path_a = reconstruct(Ta, len(Ta) - 1)
                    path_b = reconstruct(Tb, len(Tb) - 1)
                    path_b.reverse()
                    return path_a + path_b[1:]

            # swap trees
            Ta, Tb = Tb, Ta

        return None
    finally:
        for jid, qi in zip(joints, saved):
            p.resetJointState(robot_id, jid, float(qi), 0.0)


def plan_to_q(robot, q_goal: np.ndarray,
              obstacles: Sequence[int],
              min_dist: float = 0.002) -> Optional[List[np.ndarray]]:
    joints = robot.arm_controllable_joints
    q_start = get_q(robot.id, joints)

    # 1) сначала пробуем простую прямую линию в q-space
    line = interpolate_q(q_start, q_goal, step=0.06)
    if path_collision_free(robot.id, joints, line, obstacles, min_dist=min_dist):
        return line

    # 2) если не вышло — RRT-Connect
    path = rrt_connect(
        robot_id=robot.id,
        joints=joints,
        q_start=q_start,
        q_goal=q_goal,
        lower=robot.arm_lower_limits,
        upper=robot.arm_upper_limits,
        obstacles=obstacles,
        min_dist=min_dist,
        step_size=0.18,
        max_iters=2500,
        goal_bias=0.25,
    )
    return path


def execute_q_path(robot, q_path: Sequence[Sequence[float]],
                   per_wp_steps: int = None) -> bool:
    if per_wp_steps is None:
        per_wp_steps = cfg.STEPS_PER_WP

    joints = robot.arm_controllable_joints
    for q in q_path:
        set_q_position_control(robot.id, joints, q)
        step_hold(per_wp_steps)

    # подержать
    step_hold(cfg.HOLD_STEPS_AFTER)
    return True


def plan_and_execute_pose(robot,
                          target_pos: Sequence[float],
                          target_quat: Sequence[float],
                          obstacles: Sequence[int],
                          min_dist: float = 0.002) -> bool:
    q_goal = ik_arm_seeded(robot, target_pos, target_quat)
    if q_goal is None:
        print("[PLAN] IK failed")
        return False

    q_path = plan_to_q(robot, q_goal, obstacles=obstacles, min_dist=min_dist)
    if q_path is None:
        print("[PLAN] planning failed")
        return False

    return execute_q_path(robot, q_path, per_wp_steps=cfg.STEPS_PER_WP)
