# -*- coding: utf-8 -*-
"""
utils/motion_utils.py
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, List, Sequence

import numpy as np
import pybullet as p

import config as cfg
from utils import sim_utils as su
from utils.rot_utils import rot_x, rot_y, rot_z, R_to_quat
from utils.debug_draw import draw_frame, draw_point
from utils.log_utils import log_pose
from utils.contacts_utils import finger_contact_force_sum, should_stop_by_finger_force
from utils.robot_ur5_robotiq85 import get_gripper_opening_m

from pybullet_tools.utils import (
    get_movable_joints,
    get_joint_positions,
    set_joint_positions,
    plan_joint_motion,
    get_collision_fn,
)

TAU = 2.0 * math.pi


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def wrap_to_seed(q: np.ndarray, seed: np.ndarray) -> np.ndarray:
    q2 = q.copy()
    for i in range(len(q2)):
        dq = q2[i] - seed[i]
        q2[i] = q2[i] - TAU * round(dq / TAU)
    return q2


def get_arm_q(robot) -> np.ndarray:
    return np.array([p.getJointState(robot.id, jid)[0] for jid in robot.arm_controllable_joints], dtype=np.float64)


def quat_angle(q1, q2) -> float:
    dot = abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
    dot = max(0.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def find_link_index_by_name(body_id: int, link_name: str) -> int:
    for i in range(p.getNumJoints(body_id)):
        lname = p.getJointInfo(body_id, i)[12].decode("utf-8")
        if lname == link_name:
            return i
    raise RuntimeError(f"Link '{link_name}' not found in URDF (body={body_id}).")


def get_tool_link(robot) -> int:
    tcp_name = getattr(cfg, "TCP_LINK_NAME", "midpads_tcp")
    return find_link_index_by_name(robot.id, tcp_name)


def get_link_world_link_frame(body_id: int, link_idx: int) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    ls = p.getLinkState(body_id, link_idx, computeForwardKinematics=True)
    pos = np.array(ls[4], dtype=np.float64)
    orn = tuple(ls[5])
    return pos, orn


def apply_safe_z(pnt: np.ndarray) -> np.ndarray:
    q = np.array(pnt, dtype=np.float64)
    q[2] = max(q[2], cfg.SAFE_Z)
    return q


def freeze_arm_hold_current(robot, steps: int = 240):
    q = get_arm_q(robot)
    for _ in range(int(steps)):
        for j, jid in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(
                robot.id, jid,
                p.POSITION_CONTROL,
                targetPosition=float(q[j]),
                force=float(cfg.ARM_JOINT_FORCE),
                maxVelocity=float(cfg.ARM_MAX_VEL),
                positionGain=float(cfg.POS_GAIN),
                velocityGain=float(cfg.VEL_GAIN),
            )
        su.step_sim(1, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)


def world_obstacles(robot_id: int, extra_ignore: Optional[Sequence[int]] = None) -> List[int]:
    ignore = set([robot_id])
    if extra_ignore:
        ignore.update(extra_ignore)
    return [bid for bid in range(p.getNumBodies()) if bid not in ignore]


def make_collision_fn(robot_id: int, arm_joints: Sequence[int], obstacles: Sequence[int], self_collisions: bool = True):
    return get_collision_fn(robot_id, list(arm_joints), obstacles=list(obstacles), attachments=[], self_collisions=self_collisions)


def disable_robot_tray_collisions(robot_id: int, tray_id: Optional[int]) -> None:
    if tray_id is None:
        return
    su.disable_collisions_between_bodies(robot_id, tray_id, links_a=None, links_b=None)


def extract_arm_from_ik_vector(robot, ik_vec: Sequence[float]) -> np.ndarray:
    movable = list(get_movable_joints(robot.id))
    idx_map = {jid: k for k, jid in enumerate(movable)}

    q = []
    for jid in robot.arm_controllable_joints:
        if jid not in idx_map:
            raise RuntimeError(f"Joint {jid} not in movable joints; cannot map IK result.")
        q.append(float(ik_vec[idx_map[jid]]))
    return np.asarray(q, dtype=np.float64)


def ik_to_q_tool(robot, tool_link: int, target_pos, target_quat, max_iters: int = 800, residual: float = 1e-5) -> np.ndarray:
    seed = get_arm_q(robot)
    ik_vec = p.calculateInverseKinematics(
        robot.id, tool_link,
        targetPosition=target_pos,
        targetOrientation=target_quat,
        lowerLimits=robot.arm_lower_limits,
        upperLimits=robot.arm_upper_limits,
        jointRanges=robot.arm_joint_ranges,
        restPoses=seed.tolist(),
        maxNumIterations=max_iters,
        residualThreshold=residual,
    )
    q = extract_arm_from_ik_vector(robot, ik_vec)
    q = wrap_to_seed(q, seed)
    return q


def ik_search_collision_free(
    robot,
    tool_link: int,
    target_pos,
    target_quat,
    collision_fn,
    tries: int = 600,
    noise: float = 0.6,
    pos_tol: float = 0.02,
    ang_tol: float = 0.35,
    pos_only: bool = False,
) -> Tuple[Optional[np.ndarray], Tuple[float, float]]:
    seed = get_arm_q(robot)
    q_start = seed.copy()

    best_q = None
    best_metric = float("inf")
    best_err = (float("inf"), float("inf"))

    def eval_err(q: np.ndarray):
        set_joint_positions(robot.id, robot.arm_controllable_joints, q.tolist())
        pos, quat = get_link_world_link_frame(robot.id, tool_link)
        pe = float(np.linalg.norm(pos - np.asarray(target_pos, dtype=float)))
        ae = float(quat_angle(quat, target_quat))
        return pe, ae

    def accept(pe, ae):
        return (pe <= pos_tol) if pos_only else ((pe <= pos_tol) and (ae <= ang_tol))

    try:
        q0 = ik_to_q_tool(robot, tool_link, target_pos, target_quat)
        if not collision_fn(q0.tolist()):
            pe, ae = eval_err(q0)
            LAMBDA_Q = 0.25  # 0.1..0.5 (чем больше, тем меньше "переворотов")
            dq = float(np.linalg.norm(q0 - seed))
            metric = pe if pos_only else (pe + ae + LAMBDA_Q * dq)
            best_q, best_metric, best_err = q0, metric, (pe, ae)
            if accept(pe, ae):
                return q0, (pe, ae)

        for _ in range(int(tries)):
            q_try = seed + np.random.uniform(-noise, noise, size=seed.shape)
            set_joint_positions(robot.id, robot.arm_controllable_joints, q_try.tolist())

            q = ik_to_q_tool(robot, tool_link, target_pos, target_quat)
            if collision_fn(q.tolist()):
                continue

            pe, ae = eval_err(q)
            metric = pe if pos_only else (pe + ae)
            if metric < best_metric:
                best_q, best_metric, best_err = q, metric, (pe, ae)
            if accept(pe, ae):
                return q, (pe, ae)

        return None, best_err
    finally:
        set_joint_positions(robot.id, robot.arm_controllable_joints, q_start.tolist())

def hold_q_until(
    robot,
    q_goal: np.ndarray,
    tol: float = 0.02,
    max_steps: int = 12000,
    j_focus: Optional[int] = None,
    max_vel: Optional[float] = None,           # <-- ДОБАВИЛИ
    max_vel_per_joint: Optional[Sequence[float]] = None,  # <-- опционально, если захочешь разный per-joint
):
    q_goal = np.asarray(q_goal, dtype=np.float64)

    # если задан общий max_vel — используем его, иначе дефолт
    v_default = float(cfg.ARM_MAX_VEL)
    v_global = float(max_vel) if (max_vel is not None) else v_default

    for _ in range(int(max_steps)):
        for j, jid in enumerate(robot.arm_controllable_joints):
            if max_vel_per_joint is not None:
                vj = float(max_vel_per_joint[j])
            else:
                vj = v_global

            p.setJointMotorControl2(
                robot.id, jid, p.POSITION_CONTROL,
                targetPosition=float(q_goal[j]),
                force=float(cfg.ARM_JOINT_FORCE),
                maxVelocity=vj,
                positionGain=float(cfg.POS_GAIN),
                velocityGain=float(cfg.VEL_GAIN),
            )

        su.step_sim(cfg.STEPS_PER_WP, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

        q = get_arm_q(robot)
        if j_focus is not None:
            if abs(float(q[j_focus] - q_goal[j_focus])) < tol:
                break
        else:
            if float(np.linalg.norm(q - q_goal)) < tol:
                break

    su.step_sim(cfg.HOLD_STEPS_AFTER, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)


def turn_j1_exact_90(robot, tol_rad: float = 0.02, max_steps: int = 12000):
    j1i = j1_index_in_ctrl(robot)
    q0 = get_arm_q(robot)
    q_goal = q0.copy()
    q_goal[j1i] = q0[j1i] + (math.pi / 2.0)

    # медленная скорость именно для поворота
    turn_vel = float(getattr(cfg, "TURN_J1_MAX_VEL", 0.25))  # рад/с (порядок)

    print(f"[DBG] TURN J1 +90deg (SLOW): j1i={j1i} {q0[j1i]:.3f} -> {q_goal[j1i]:.3f} maxVel={turn_vel:.3f}")

    hold_q_until(
        robot,
        q_goal=q_goal,
        tol=tol_rad,
        max_steps=max_steps,
        j_focus=j1i,
        max_vel=turn_vel,   # <-- вот это замедляет
    )


def follow_joint_path(robot, path: List[List[float]], step_per_wp: int = 2):
    for q in path:
        for j, jid in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(
                robot.id, jid, p.POSITION_CONTROL,
                targetPosition=float(q[j]),
                force=float(cfg.ARM_JOINT_FORCE),
                maxVelocity=float(cfg.ARM_MAX_VEL),
                positionGain=float(cfg.POS_GAIN),
                velocityGain=float(cfg.VEL_GAIN),
            )
        su.step_sim(step_per_wp, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

    su.step_sim(cfg.HOLD_STEPS_AFTER, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)


def j1_index_in_ctrl(robot) -> int:
    for i, jid in enumerate(robot.arm_controllable_joints):
        name = p.getJointInfo(robot.id, jid)[1].decode("utf-8")
        if name == "shoulder_pan_joint":
            return i
    raise RuntimeError("shoulder_pan_joint not found in arm_controllable_joints")


def get_flip_R() -> np.ndarray:
    if not cfg.FLIP_TCP_180_DEG:
        return np.eye(3, dtype=np.float64)
    if cfg.FLIP_AXIS == "x":
        return rot_x(math.pi)
    if cfg.FLIP_AXIS == "y":
        return rot_y(math.pi)
    return rot_z(math.pi)


def fix_grasp_frame_if_below_table(center: np.ndarray, R: np.ndarray) -> np.ndarray:
    approach = R[:, 1]
    pre_test = center + (R @ cfg.TCP_OFFSET_HAND) - approach * (cfg.PREGRASP_DIST + cfg.APPROACH_STANDOFF)
    if pre_test[2] < cfg.SAFE_Z - 1e-3:
        R = R @ rot_x(math.pi)
    return R


def grasp_to_tcp_pose(h, R_EE_FROM_GRASP: np.ndarray):
    c = np.asarray(h.center, dtype=np.float64)
    R_hand = np.asarray(h.R, dtype=np.float64)

    R_hand_used = R_hand @ get_flip_R()
    R_hand_used = fix_grasp_frame_if_below_table(c, R_hand_used)

    R_tcp = R_hand_used @ R_EE_FROM_GRASP
    tcp_pos = c + (R_hand_used @ cfg.TCP_OFFSET_HAND)
    tcp_quat = R_to_quat(R_tcp)
    return tcp_pos, tcp_quat, R_hand_used


def approach_axis_from_R(R_hand_used: np.ndarray) -> np.ndarray:
    a = R_hand_used[:, 1]
    n = float(np.linalg.norm(a))
    if n < 1e-9:
        return np.array([0, 1, 0], dtype=np.float64)
    return (a / n).astype(np.float64)


def get_carry_quat() -> tuple[float, float, float, float]:
    rpy = getattr(cfg, "BIN_TCP_RPY", (math.pi, 0.0, -math.pi / 2.0))
    return p.getQuaternionFromEuler(rpy)

def carry_up_with_lift_test(
    robot,
    tool_link: int,
    tcp_quat,
    obj_id: int,
    carry_z: float,
    dz_test: float,
) -> Tuple[bool, float, int, float, int]:
    """
    Делает ОДИН вертикальный подъём:
      - сначала поднимаемся по Z на dz_test, замеряем контакты
      - затем поднимаемся дальше по Z до carry_z
    Возвращает как lift_test_antislip: (ok, sumF0, nC0, sumF1, nC1)
    """

    # контакты ДО подъёма
    sF0, nC0 = finger_contact_force_sum(robot.id, obj_id, finger_link_ids=cfg.FINGER_LINK_IDS)
    if nC0 <= 0:
        print("[CARRY/LIFTTEST] no contacts before carry -> FAIL")
        return False, float(sF0), int(nC0), 0.0, 0

    tcp_pos, _ = get_link_world_link_frame(robot.id, tool_link)
    tcp_pos = np.asarray(tcp_pos, dtype=np.float64)

    # 1) шаг lift-test: поднялись на dz_test
    z1 = float(tcp_pos[2] + float(dz_test))
    tgt1 = tcp_pos.copy()
    tgt1[2] = z1

    print(f"[CARRY/LIFTTEST] step1 dz={dz_test:.3f}m from sumF={sF0:.2f} nC={nC0}")
    move_smooth_to(
        robot, tool_link=tool_link,
        target_pos=tgt1, target_quat=tcp_quat,
        tag="CARRY_LIFTTEST_Z",
        ramp_steps=int(getattr(cfg, "LIFT_TEST_STEPS", 120)),
        obj_id=None,
        allow_contact_stop=False,
    )
    freeze_arm_hold_current(robot, steps=int(getattr(cfg, "LIFT_TEST_SETTLE_STEPS", 120)))

    sF1, nC1 = finger_contact_force_sum(robot.id, obj_id, finger_link_ids=cfg.FINGER_LINK_IDS)
    print(f"[CARRY/LIFTTEST] after step1 sumF={sF1:.2f} nC={nC1}")

    if nC1 < cfg.GRASP_MIN_CONTACTS:
        print("[CARRY/LIFTTEST] contacts dropped -> FAIL")
        return False, float(sF0), int(nC0), float(sF1), int(nC1)

    if sF1 < float(cfg.SLIP_DROP_SUMF) * float(sF0):
        print("[CARRY/LIFTTEST] sumF dropped too much -> FAIL")
        return False, float(sF0), int(nC0), float(sF1), int(nC1)

    # 2) продолжение подъёма по Z до carry_z (всё ещё чисто Z)
    tcp_pos2, _ = get_link_world_link_frame(robot.id, tool_link)
    tcp_pos2 = np.asarray(tcp_pos2, dtype=np.float64)

    tgt2 = tcp_pos2.copy()
    tgt2[2] = float(carry_z)

    print(f"[CARRY] step2 up to carry_z={carry_z:.3f}m")
    move_smooth_to(
        robot, tool_link=tool_link,
        target_pos=tgt2, target_quat=tcp_quat,
        tag="CARRY_UP",
        ramp_steps=int(getattr(cfg, "CARRY_UP_STEPS", 260)),
        obj_id=None,
        allow_contact_stop=False,
    )
    freeze_arm_hold_current(robot, steps=int(getattr(cfg, "CARRY_SETTLE_STEPS", 120)))

    return True, float(sF0), int(nC0), float(sF1), int(nC1)

def move_smooth_to(
    robot,
    tool_link: int,
    target_pos: np.ndarray,
    target_quat,
    tag: str,
    ramp_steps: int,
    obj_id: Optional[int] = None,
    tol: float = cfg.REACH_TOL,
    allow_contact_stop: bool = False,
    sum_force_thresh: float = 10.0,
    min_contacts: int = 2,
) -> Tuple[bool, float, bool]:
    start_pos, _ = get_link_world_link_frame(robot.id, tool_link)
    start_pos = np.asarray(start_pos, dtype=np.float64)
    target_pos = np.asarray(target_pos, dtype=np.float64)

    last_err = 1e9
    stopped_by_contact = False

    for i in range(1, int(ramp_steps) + 1):
        t = i / float(ramp_steps)
        wp = (1.0 - t) * start_pos + t * target_pos

        q_wp = ik_to_q_tool(robot, tool_link, wp.tolist(), target_quat, max_iters=400, residual=1e-4)

        for j, jid in enumerate(robot.arm_controllable_joints):
            p.setJointMotorControl2(
                robot.id, jid, p.POSITION_CONTROL,
                targetPosition=float(q_wp[j]),
                force=float(cfg.ARM_JOINT_FORCE),
                maxVelocity=float(cfg.ARM_MAX_VEL),
                positionGain=float(cfg.POS_GAIN),
                velocityGain=float(cfg.VEL_GAIN),
            )

        su.step_sim(cfg.STEPS_PER_WP, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

        if allow_contact_stop and cfg.ENABLE_CONTACT_STOP and (obj_id is not None):
            stop, sF, nC = should_stop_by_finger_force(
                robot.id, obj_id,
                finger_link_ids=cfg.FINGER_LINK_IDS,
                sum_force_thresh=sum_force_thresh,
                min_contacts=min_contacts,
            )
            if stop:
                stopped_by_contact = True
                print(f"[STOP] {tag}: sumF={sF:.2f} N, contacts={nC}")
                break

        eef_pos, _ = get_link_world_link_frame(robot.id, tool_link)
        last_err = float(np.linalg.norm(np.asarray(eef_pos) - target_pos))

    su.step_sim(cfg.HOLD_STEPS_AFTER, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

    eef_pos, eef_quat = get_link_world_link_frame(robot.id, tool_link)
    last_err = float(np.linalg.norm(np.asarray(eef_pos) - target_pos))
    log_pose(f"TCP   (after {tag})", eef_pos, eef_quat, ref_pos=target_pos)

    return (last_err < tol), last_err, stopped_by_contact


def close_gripper_with_retry(robot, obj_id: int, target_w: float) -> Tuple[bool, float, int]:
    target_w = float(clamp(target_w, 0.0105, cfg.GRIPPER_OPEN))

    print("\n-- closing gripper FAST ->", target_w)
    robot.move_gripper(target_w)
    freeze_arm_hold_current(robot, steps=cfg.GRIPPER_CLOSE_SETTLE_STEPS)

    sF, nC = finger_contact_force_sum(robot.id, obj_id, finger_link_ids=cfg.FINGER_LINK_IDS)
    print(f"[AFTER CLOSE FAST] sumF={sF:.2f} N, contacts={nC}")

    if (nC >= cfg.GRASP_MIN_CONTACTS) and (sF >= cfg.GRASP_MIN_SUMF):
        return True, float(sF), int(nC)

    w2 = max(0.0, target_w - float(cfg.GRIPPER_TIGHTEN_DELTA))
    print("-- tightening gripper ->", w2)
    robot.move_gripper(w2)
    freeze_arm_hold_current(robot, steps=cfg.GRIPPER_TIGHTEN_MAX_STEPS)

    sF2, nC2 = finger_contact_force_sum(robot.id, obj_id, finger_link_ids=cfg.FINGER_LINK_IDS)
    print(f"[AFTER TIGHTEN] sumF={sF2:.2f} N, contacts={nC2}")

    ok = (nC2 >= cfg.GRASP_MIN_CONTACTS) and (sF2 >= cfg.GRASP_MIN_SUMF)
    return ok, float(sF2), int(nC2)


def lift_test_antislip(robot, tool_link: int, tcp_quat, obj_id: int) -> Tuple[bool, float, int, float, int]:
    """
    return: (ok, sumF0, nC0, sumF1, nC1)
    """
    sF0, nC0 = finger_contact_force_sum(robot.id, obj_id, finger_link_ids=cfg.FINGER_LINK_IDS)
    if nC0 <= 0:
        print("[LIFT TEST] no contacts before lift -> FAIL")
        return False, float(sF0), int(nC0), float(0.0), int(0)

    tcp_pos, _ = get_link_world_link_frame(robot.id, tool_link)
    tgt = np.asarray(tcp_pos, dtype=np.float64).copy()
    tgt[2] += float(cfg.LIFT_TEST_DZ)

    print(f"[LIFT TEST] dz={cfg.LIFT_TEST_DZ:.3f}m from sumF={sF0:.2f} nC={nC0}")
    move_smooth_to(
        robot, tool_link=tool_link,
        target_pos=tgt, target_quat=tcp_quat,
        tag="LIFT_TEST_Z",
        ramp_steps=int(cfg.LIFT_TEST_STEPS),
        obj_id=None,
        allow_contact_stop=False,
    )
    freeze_arm_hold_current(robot, steps=cfg.LIFT_TEST_SETTLE_STEPS)

    sF1, nC1 = finger_contact_force_sum(robot.id, obj_id, finger_link_ids=cfg.FINGER_LINK_IDS)
    print(f"[LIFT TEST] after lift sumF={sF1:.2f} nC={nC1}")

    if nC1 < cfg.GRASP_MIN_CONTACTS:
        print("[LIFT TEST] contacts dropped -> FAIL")
        return False, float(sF0), int(nC0), float(sF1), int(nC1)

    if sF1 < float(cfg.SLIP_DROP_SUMF) * float(sF0):
        print("[LIFT TEST] sumF dropped too much -> FAIL")
        return False, float(sF0), int(nC0), float(sF1), int(nC1)

    return True, float(sF0), int(nC0), float(sF1), int(nC1)


def plan_to_tool_pose(
    robot,
    tool_link: int,
    target_pos: np.ndarray,
    target_quat,
    ignore_bodies: Optional[Sequence[int]] = None,
    delta: float = 0.08,
    ik_tries: int = 1200,
    ik_noise: float = 1.6,
    pos_tol: float = 0.02,
    ang_tol: float = 0.35,
    pos_only: bool = False,
) -> Optional[List[List[float]]]:
    arm_joints = list(robot.arm_controllable_joints)
    obstacles = world_obstacles(robot.id, extra_ignore=ignore_bodies)
    collision_fn = make_collision_fn(robot.id, arm_joints, obstacles, self_collisions=True)

    q_start = np.array(get_joint_positions(robot.id, arm_joints), dtype=np.float64)
    if collision_fn(q_start.tolist()):
        print("[PLAN] q_start in collision -> cannot plan")
        return None

    render_off = False
    if cfg.USE_GUI:
        try:
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
            render_off = True
        except Exception:
            render_off = False

    try:
        q_goal, best_err = ik_search_collision_free(
            robot=robot,
            tool_link=tool_link,
            target_pos=target_pos.tolist(),
            target_quat=target_quat,
            collision_fn=collision_fn,
            tries=int(ik_tries),
            noise=float(ik_noise),
            pos_tol=float(pos_tol),
            ang_tol=float(ang_tol),
            pos_only=bool(pos_only),
        )

        if q_goal is None:
            pe, ae = best_err
            print(f"[PLAN] IK failed. best pe={pe:.4f}m ae={ae:.3f}rad")
            return None

        path = plan_joint_motion(
            robot.id,
            arm_joints,
            q_goal.tolist(),
            obstacles=obstacles,
            self_collisions=True,
            resolutions=[float(delta)] * len(arm_joints),
        )

        if path is None:
            print("[PLAN] plan_joint_motion failed")
            return None

        set_joint_positions(robot.id, arm_joints, q_start.tolist())
        return path
    finally:
        if render_off:
            try:
                p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
            except Exception:
                pass

def go_place_above_no_dive_planned(robot, tool_link: int, tray_pos: np.ndarray, tray_drop_height: float, obj_id: Optional[int]):
    tray_pos = np.asarray(tray_pos, dtype=np.float64)
    target = tray_pos + np.array([0.0, 0.0, float(tray_drop_height)], dtype=np.float64)

    tcp_pos, _ = get_link_world_link_frame(robot.id, tool_link)

    carry_floor = max(
        float(getattr(cfg, "CARRY_Z_ABS_MIN", 0.0)),
        float(target[2] + getattr(cfg, "CARRY_Z_MARGIN", 0.10)),
    )

    z_safe = max(float(tcp_pos[2]), float(target[2]), carry_floor)
    mid = target.copy()
    mid[2] = z_safe

    base_quat = get_carry_quat()

    # пробуем несколько yaw-вариантов
    step = float(getattr(cfg, "PLACE_YAW_STEP_RAD", math.radians(30)))
    yaw_list = [i*step for i in range(int(round(TAU/step)))]
    # можно ещё отсортировать по близости к 0, чтобы сначала пробовать "прямо"
    yaw_list = sorted(yaw_list, key=lambda a: abs((a + math.pi) % TAU - math.pi))

    for yaw in yaw_list:
        q_yaw = p.getQuaternionFromEuler([0.0, 0.0, yaw])
        carry_quat = p.multiplyTransforms([0,0,0], base_quat, [0,0,0], q_yaw)[1]

        print(f"[PLAN] PLACE yaw={yaw:.2f} mid={np.round(mid,3)} target={np.round(target,3)}")

        path_mid = plan_to_tool_pose(
            robot, tool_link, mid, carry_quat,
            ignore_bodies=([obj_id] if obj_id is not None else None),
            delta=getattr(cfg, "PLACE_DELTA", 0.08),
            ik_tries=getattr(cfg, "IK_TRIES", 1200),
            ik_noise=getattr(cfg, "IK_NOISE", 0.6),
            pos_tol=0.02, ang_tol=0.35, pos_only=False,
        )
        if path_mid is None:
            continue

        follow_joint_path(robot, path_mid, step_per_wp=2)
        return True

    print("[PLAN] cannot reach place MID (all yaw variants failed)")
    return False

def carry_up_and_verify_grasp(
    robot,
    tool_link: int,
    tcp_quat,
    obj_id: int,
    sF_ref: float,
    nC_ref: int,
    carry_z: float,
    ramp_steps: int = 260,
) -> Tuple[bool, float, int, float, Optional[str]]:
    """
    Единственный вертикальный подъём по Z до carry_z + проверка, что хват не "сорвался".
    Проверка: контакты не упали ниже GRASP_MIN_CONTACTS и sumF не просел слишком сильно
    относительно sF_ref (после закрытия ЗУ).
    """
    if nC_ref < cfg.GRASP_MIN_CONTACTS:
        print("[CARRY] not enough contacts right after close -> FAIL")
        return False, float(sF_ref), int(nC_ref)

    tcp_pos, _ = get_link_world_link_frame(robot.id, tool_link)
    start = np.asarray(tcp_pos, dtype=np.float64)
    tgt = start.copy()
    tgt[2] = float(carry_z)

    # строго вертикально: x,y не меняем
    print(f"[CARRY] up to carry_z={tgt[2]:.3f}m (vertical only)")
    move_smooth_to(
        robot, tool_link=tool_link,
        target_pos=tgt, target_quat=tcp_quat,
        tag="CARRY_UP",
        ramp_steps=int(ramp_steps),
        obj_id=None,
        allow_contact_stop=False,
    )
    freeze_arm_hold_current(robot, steps=int(getattr(cfg, "CARRY_SETTLE_STEPS", 120)))

    sF1, nC1 = finger_contact_force_sum(robot.id, obj_id, finger_link_ids=cfg.FINGER_LINK_IDS)
    print(f"[CARRY] after up sumF={sF1:.2f} nC={nC1} (ref sumF={sF_ref:.2f} nC={nC_ref})")

    sumF_ratio = float(sF1) / max(float(sF_ref), 1e-9)

    if nC1 < cfg.GRASP_MIN_CONTACTS:
        print("[CARRY] contacts dropped -> FAIL")
        return False, float(sF1), int(nC1), sumF_ratio, "lift_contacts_dropped"

    if sF1 < float(cfg.SLIP_DROP_SUMF) * float(sF_ref):
        print("[CARRY] sumF dropped too much -> FAIL")
        return False, float(sF1), int(nC1), sumF_ratio, "lift_sumf_dropped"

    return True, float(sF1), int(nC1), sumF_ratio, None



def execute_pick_and_place(
    robot,
    h,
    obj_id: int,
    R_EE_FROM_GRASP: np.ndarray,
    tray_pos: np.ndarray,
    tray_drop_height: float = 0.20,
    tray_down_height: float = 0.06,
    sum_force_thresh: float = 10.0,
    min_contacts: int = 2,
    tray_id: Optional[int] = None,
    logger=None,
) -> tuple[List[int], bool, Optional[str]]:

    tool_link = get_tool_link(robot)

    # трение (объект + пальцы)
    su.set_body_friction(obj_id, -1,
                         lateral=getattr(cfg, "OBJ_FRICTION", 0.9),
                         spinning=getattr(cfg, "OBJ_SPIN_FRIC", getattr(cfg, "OBJ_SPIN_FRICTION", 0.01)),
                         rolling=getattr(cfg, "OBJ_ROLL_FRIC", getattr(cfg, "OBJ_ROLL_FRICTION", 0.001)),
                         restitution=0.0)
    su.set_links_friction(robot.id, cfg.FINGER_LINK_IDS,
                          lateral=getattr(cfg, "PAD_LAT_FRIC", 3.0),
                          spinning=getattr(cfg, "PAD_SPIN_FRIC", 0.06),
                          rolling=getattr(cfg, "PAD_ROLL_FRIC", 0.006),
                          restitution=0.0)

    if getattr(cfg, "DISABLE_ROBOT_TRAY_COLLISIONS", False):
        disable_robot_tray_collisions(robot.id, tray_id)

    tcp_goal, tcp_quat, R_hand_used = grasp_to_tcp_pose(h, R_EE_FROM_GRASP)

    center = np.asarray(h.center, dtype=np.float64)
    approach = approach_axis_from_R(R_hand_used)

    pre = tcp_goal - approach * (cfg.PREGRASP_DIST + cfg.APPROACH_STANDOFF)
    grasp_wp = tcp_goal - approach * (cfg.APPROACH_STANDOFF)
    pre_back = pre - approach * cfg.PRE_BACK_DIST
    grasp = np.asarray(tcp_goal, dtype=np.float64)

    pre_back_s = apply_safe_z(pre_back)
    pre_s = apply_safe_z(pre)
    grasp_wp_s = np.asarray(grasp_wp, dtype=np.float64)

    # ----- logging: static context for attempt
    if logger is not None:
        logger.update(
            extra={
                "tool_link": int(tool_link),
                "tcp_goal": np.asarray(tcp_goal, dtype=float).tolist(),
                "tcp_quat": [float(x) for x in tcp_quat],
                "R_hand_used": np.asarray(R_hand_used, dtype=float).tolist(),
                "approach_axis": np.asarray(approach, dtype=float).tolist(),
                "pre_back_s": np.asarray(pre_back_s, dtype=float).tolist(),
                "pre_s": np.asarray(pre_s, dtype=float).tolist(),
                "grasp_wp_s": np.asarray(grasp_wp_s, dtype=float).tolist(),
                "grasp": np.asarray(grasp, dtype=float).tolist(),
                "params": {
                    "sum_force_thresh": float(sum_force_thresh),
                    "min_contacts": int(min_contacts),
                    "reach_tol": float(cfg.REACH_TOL),
                    "carry_z_margin": float(getattr(cfg, "CARRY_Z_MARGIN", 0.10)),
                    "slip_drop_sumf": float(getattr(cfg, "SLIP_DROP_SUMF", 0.5)),
                    "grasp_min_contacts": int(getattr(cfg, "GRASP_MIN_CONTACTS", 2)),
                    "grasp_min_sumf": float(getattr(cfg, "GRASP_MIN_SUMF", 10.0)),
                },
            }
        )
        logger.event("plan_points_ready")

    dbg: List[int] = []
    dbg += draw_point(center, life=0.0, label="CENTER(grasp)")
    dbg += draw_point(pre_back_s, life=0.0, label="PRE_BACK(SAFE_Z)")
    dbg += draw_point(pre_s, life=0.0, label="PRE(SAFE_Z)")
    dbg += draw_point(grasp_wp_s, life=0.0, label="GRASP_WP(standoff)")
    dbg += draw_point(grasp, life=0.0, label="GRASP(TCP_GOAL)")
    dbg += draw_frame(grasp_wp_s, tcp_quat, axis_len=0.08, life=0.0, width=2.0, label="ORN@GRASP(TCP)")

    def _log_seg(tag: str, target_pos: np.ndarray, ok: bool, err: float, stopped: bool):
        if logger is None:
            return
        eef_pos, eef_quat = get_link_world_link_frame(robot.id, tool_link)
        logger.segment(
            tag,
            target_pos=np.asarray(target_pos, dtype=float).tolist(),
            reached=bool(ok),
            err_to_target_m=float(err),
            stopped_by_contact=bool(stopped),
            tcp_after_pos=np.asarray(eef_pos, dtype=float).tolist(),
            tcp_after_quat=[float(x) for x in eef_quat],
        )
        logger.event(
            "segment_done",
            tag=tag,
            reached=bool(ok),
            err=float(err),
            stopped_by_contact=bool(stopped),
        )

    def seg(tag: str, pos: np.ndarray, quat, steps: int, contact_stop: bool, use_obj: bool):
        ok, err, stopped = move_smooth_to(
            robot, tool_link=tool_link,
            target_pos=pos, target_quat=quat,
            tag=tag, ramp_steps=int(steps),
            obj_id=(obj_id if use_obj else None),
            allow_contact_stop=contact_stop,
            sum_force_thresh=sum_force_thresh,
            min_contacts=min_contacts,
        )
        _log_seg(tag, pos, ok, err, stopped)
        return ok, err, stopped

    # ---- approach/grasp
    robot.move_gripper(cfg.GRIPPER_OPEN)
    freeze_arm_hold_current(robot, steps=240)
    if logger is not None:
        logger.event("gripper_opened", width=float(cfg.GRIPPER_OPEN))

    seg("PRE_BACK", pre_back_s, tcp_quat, cfg.MOVE_RAMP_STEPS_BACK, False, False)
    su.pause_sec(cfg.SEGMENT_PAUSE_SEC, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

    seg("PRE", pre_s, tcp_quat, cfg.MOVE_RAMP_STEPS_PRE, False, False)
    su.pause_sec(cfg.SEGMENT_PAUSE_SEC, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

    seg("GRASP_WP", grasp_wp_s, tcp_quat, cfg.MOVE_RAMP_STEPS_GRASP, False, True)
    su.pause_sec(cfg.SEGMENT_PAUSE_SEC, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

    ok_grasp, err_grasp, stopped_grasp = seg("GRASP", grasp, tcp_quat, cfg.MOVE_RAMP_STEPS_GRASP_CONTACT, True, True)
    su.pause_sec(cfg.SEGMENT_PAUSE_SEC, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

    # ---- close protocol
    target_w = clamp(float(h.width), 0.0105, cfg.GRIPPER_OPEN)
    if logger is not None:
        logger.event("close_start", target_w=float(target_w))

    opening_before = float(get_gripper_opening_m(robot))
    if logger is not None:
        logger.update(gripper_opening_before_m=opening_before)


    ok_close, sF_close, nC_close = close_gripper_with_retry(robot, obj_id=obj_id, target_w=target_w)

    opening_after = float(get_gripper_opening_m(robot))
    if logger is not None:
        logger.update(
            gripper_opening_after_m=opening_after,
            gripper_opening_delta_m=float(opening_before - opening_after),
        )

    # лог close всегда
    if logger is not None:
        logger.update(
            sumF_after_close=float(sF_close),
            nC_after_close=int(nC_close),
            extra={
                "close": {
                    "ok_close": bool(ok_close),
                    "target_w": float(target_w),
                    "sumF_close": float(sF_close),
                    "nC_close": int(nC_close),
                    "grasp_contact_stop": bool(stopped_grasp),
                    "grasp_err_m": float(err_grasp),
                }
            },
        )
        logger.event("close_done", ok=bool(ok_close), sumF=float(sF_close), nC=int(nC_close))

        cur_opening = get_gripper_opening_m(robot)
        logger.update(gripper_opening_m=float(cur_opening))


    # ранний выход: нет контактов
    if int(nC_close) <= 0:
        try:
            robot.move_gripper(cfg.GRIPPER_OPEN)
            su.step_sim(60, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)
        except Exception:
            pass
        return dbg, False, "no_contact_after_close"

    if not ok_close:
        return dbg, False, "weak_grasp_after_close"

    # ---- retreat
    seg("RETREAT_PRE", pre_s, tcp_quat, 260, False, False)

    # ---- carry (vertical only)
    carry_z = max(
        float(pre_s[2]) + float(getattr(cfg, "CARRY_Z_MARGIN", 0.10)),
        float(tray_pos[2] + tray_drop_height + float(getattr(cfg, "CARRY_Z_MARGIN", 0.10))),
        float(getattr(cfg, "CARRY_Z_ABS_MIN", pre_s[2])),
    )
    if logger is not None:
        logger.segment("CARRY_UP", carry_z=float(carry_z), ramp_steps=int(getattr(cfg, "CARRY_UP_STEPS", 260)))

    ok_carry, sF_after, nC_after, sumF_ratio, lift_fail = carry_up_and_verify_grasp(
        robot=robot,
        tool_link=tool_link,
        tcp_quat=tcp_quat,
        obj_id=obj_id,
        sF_ref=float(sF_close),
        nC_ref=int(nC_close),
        carry_z=float(carry_z),
        ramp_steps=int(getattr(cfg, "CARRY_UP_STEPS", 260)),
    )

    if logger is not None:
        logger.update(
            sumF_after_lift=float(sF_after),
            nC_after_lift=int(nC_after),
            extra={
                "carry": {
                    "ok": bool(ok_carry),
                    "carry_z": float(carry_z),
                    "sumF_ref": float(sF_close),
                    "nC_ref": int(nC_close),
                    "sumF_after": float(sF_after),
                    "nC_after": int(nC_after),
                    "sumF_ratio": float(sF_after) / max(float(sF_close), 1e-9),
                    "sumF_ratio": float(sumF_ratio),
                }
            },
        )
        logger.event("carry_done", ok=bool(ok_carry), sumF=float(sF_after), nC=int(nC_after))

    if not ok_carry:
        return dbg, False, (lift_fail or "slip_detected_lift_test")

    # ---- turn +90
    if logger is not None:
        logger.event("turn_j1_start")
    turn_j1_exact_90(robot, tol_rad=0.02, max_steps=12000)
    if logger is not None:
        logger.event("turn_j1_done")

    # ---- place planned
    if logger is not None:
        logger.event("place_plan_start", tray_pos=np.asarray(tray_pos, float).tolist(), tray_drop_height=float(tray_drop_height))

    ok_place = go_place_above_no_dive_planned(
        robot, tool_link=tool_link,
        tray_pos=np.asarray(tray_pos, dtype=np.float64),
        tray_drop_height=float(tray_drop_height),
        obj_id=obj_id,
    )

    if logger is not None:
        logger.segment("PLACE", ok=bool(ok_place))
        logger.event("place_plan_done", ok=bool(ok_place))

    if not ok_place:
        return dbg, False, "place_plan_failed"

    # ---- open
    robot.move_gripper(cfg.GRIPPER_OPEN)
    freeze_arm_hold_current(robot, steps=240)
    if logger is not None:
        logger.event("gripper_opened_after_place", width=float(cfg.GRIPPER_OPEN))

    return dbg, True, None

