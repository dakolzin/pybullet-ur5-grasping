#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import math
import random
import numpy as np
import pybullet as p
import pybullet_data

from pybullet_tools.utils import (
    connect, disconnect,
    wait_for_duration,
    get_movable_joints, get_joint_names,
    get_joint_positions, set_joint_positions,
    plan_joint_motion,
    get_collision_fn,
)

UR5_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

UR5_HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

# По твоему выводу joint->link:
LEFT_PAD_LINK = 12   # left_inner_finger_pad
RIGHT_PAD_LINK = 17  # right_inner_finger_pad

TAU = 2.0 * math.pi


def add_search_paths_for_urdf(urdf_path: str):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    p.setAdditionalSearchPath(urdf_dir)
    p.setAdditionalSearchPath(os.getcwd())


def load_robot(urdf_path: str, fixed_base: bool = True, base_z: float = 0.02):
    urdf_path = os.path.abspath(urdf_path)
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    add_search_paths_for_urdf(urdf_path)
    base_pos = [0.0, 0.0, base_z]
    base_orn = [0, 0, 0, 1]
    return p.loadURDF(urdf_path, basePosition=base_pos, baseOrientation=base_orn, useFixedBase=fixed_base)


def pick_arm_joints(robot):
    movable = get_movable_joints(robot)
    names = get_joint_names(robot, movable)
    name2jid = {n: j for j, n in zip(movable, names)}
    missing = [n for n in UR5_ARM_JOINT_NAMES if n not in name2jid]
    if missing:
        raise RuntimeError(f"UR5 arm joints not found in URDF: {missing}")
    return [name2jid[n] for n in UR5_ARM_JOINT_NAMES]


def find_link_index(robot, link_name: str):
    for i in range(p.getNumJoints(robot)):
        info = p.getJointInfo(robot, i)
        lname = info[12].decode("utf-8")  # linkName
        if lname == link_name:
            return i
    return None


def rpy_to_quat(roll, pitch, yaw):
    return p.getQuaternionFromEuler([roll, pitch, yaw])


def quat_angle(q1, q2):
    dot = abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
    dot = max(0.0, min(1.0, dot))
    return 2.0 * math.acos(dot)


def make_collision_fn_compat(robot, joints, obstacles, self_collisions=True):
    return get_collision_fn(
        robot, joints,
        obstacles=obstacles,
        attachments=[],
        self_collisions=self_collisions,
    )


def link_state(body, link_idx):
    s = p.getLinkState(body, link_idx, computeForwardKinematics=True)
    pos = np.array(s[4], dtype=float)   # worldLinkFramePosition
    orn = np.array(s[5], dtype=float)   # worldLinkFrameOrientation (quat)
    return pos, orn


def midpads_world_pos(robot):
    lp, _ = link_state(robot, LEFT_PAD_LINK)
    rp, _ = link_state(robot, RIGHT_PAD_LINK)
    return 0.5 * (lp + rp)


def disable_link_collisions_with_everything(body_id: int, link_ids, other_bodies):
    # self pairs
    for a in link_ids:
        for b in range(-1, p.getNumJoints(body_id)):
            p.setCollisionFilterPair(body_id, body_id, a, b, enableCollision=0)
    # env pairs
    for other in other_bodies:
        for a in link_ids:
            p.setCollisionFilterPair(body_id, other, a, -1, enableCollision=0)
            for b in range(p.getNumJoints(other)):
                p.setCollisionFilterPair(body_id, other, a, b, enableCollision=0)


def get_joint_limits(robot, joint_idx):
    info = p.getJointInfo(robot, joint_idx)
    lower = float(info[8])
    upper = float(info[9])
    return lower, upper


def wrap_angle_near(a, ref):
    candidates = [a + TAU*k for k in (-2, -1, 0, 1, 2)]
    return min(candidates, key=lambda x: abs(x - ref))


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_q_near_seed(robot, arm_joints, q, q_seed, do_clamp=False):
    out = []
    for j, a, ref in zip(arm_joints, q, q_seed):
        a2 = wrap_angle_near(a, ref)
        if do_clamp:
            lo, hi = get_joint_limits(robot, j)
            if hi > lo + 1e-6:
                a2 = clamp(a2, lo, hi)
        out.append(a2)
    return out


def extract_arm_from_ik(robot, arm_joints, ik_vec):
    """
    Форматы возвращаемого IK-вектора:
      A) len >= num_joints => индексируется по jointIndex
      B) иначе => вектор по movable joints (DOF-order)
    """
    num_joints = p.getNumJoints(robot)

    if len(ik_vec) >= num_joints:
        return [float(ik_vec[j]) for j in arm_joints]

    movable = list(get_movable_joints(robot))
    index_map = {jid: k for k, jid in enumerate(movable)}
    return [float(ik_vec[index_map[j]]) for j in arm_joints]


def ik_candidate(robot, tool_idx, arm_joints, target_pos, target_quat, q_seed, use_orientation: bool):
    if use_orientation:
        ik_vec = p.calculateInverseKinematics(
            bodyUniqueId=robot,
            endEffectorLinkIndex=tool_idx,
            targetPosition=target_pos,
            targetOrientation=target_quat,
            maxNumIterations=800,
            residualThreshold=1e-7,
        )
    else:
        ik_vec = p.calculateInverseKinematics(
            bodyUniqueId=robot,
            endEffectorLinkIndex=tool_idx,
            targetPosition=target_pos,
            maxNumIterations=800,
            residualThreshold=1e-7,
        )

    q = extract_arm_from_ik(robot, arm_joints, ik_vec)
    q = normalize_q_near_seed(robot, arm_joints, q, q_seed, do_clamp=False)
    return q, len(ik_vec)


def ik_search(robot, tool_idx, arm_joints, q_seed,
              target_pos_world, target_quat,
              collision_fn,
              tries=1200, noise=1.6,
              pos_tol=0.02, ang_tol=0.35,
              pos_only=False):
    """
    Возвращаем успех:
      - если pos_only: |pos_err| <= pos_tol
      - иначе: |pos_err| <= pos_tol и |ang_err| <= ang_tol
    """

    use_orientation_in_ik = (not pos_only)

    best = None
    best_metric = float("inf")
    best_errs = None
    ik_len_reported = None

    target_pos_world = np.array(target_pos_world, dtype=float)

    def eval_errors():
        tool_pos, tool_quat = link_state(robot, tool_idx)
        pos_err = float(np.linalg.norm(tool_pos - target_pos_world))
        ang_err = float(quat_angle(tool_quat, target_quat))
        return pos_err, ang_err

    def accept(pos_err, ang_err):
        if pos_only:
            return pos_err <= pos_tol
        return (pos_err <= pos_tol) and (ang_err <= ang_tol)

    # 0) без шума
    set_joint_positions(robot, arm_joints, q_seed)
    q0, iklen = ik_candidate(robot, tool_idx, arm_joints, target_pos_world, target_quat, q_seed, use_orientation=use_orientation_in_ik)
    ik_len_reported = iklen

    if not collision_fn(q0):
        set_joint_positions(robot, arm_joints, q0)
        pos_err, ang_err = eval_errors()
        metric = pos_err
        best, best_metric, best_errs = q0, metric, (pos_err, ang_err)
        if accept(pos_err, ang_err):
            return q0, (pos_err, ang_err), ik_len_reported, best, best_errs

    # 1) рандом по seed
    for _ in range(tries):
        q_try = [q_seed[i] + random.uniform(-noise, noise) for i in range(len(arm_joints))]
        set_joint_positions(robot, arm_joints, q_try)

        q, iklen = ik_candidate(robot, tool_idx, arm_joints, target_pos_world, target_quat, q_try, use_orientation=use_orientation_in_ik)
        ik_len_reported = iklen

        if collision_fn(q):
            continue

        set_joint_positions(robot, arm_joints, q)
        pos_err, ang_err = eval_errors()
        metric = pos_err

        if metric < best_metric:
            best, best_metric, best_errs = q, metric, (pos_err, ang_err)

        if accept(pos_err, ang_err):
            return q, (pos_err, ang_err), ik_len_reported, best, best_errs

    return None, None, ik_len_reported, best, best_errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--nogui", action="store_true")
    ap.add_argument("--no_obstacle", action="store_true")

    ap.add_argument("--x", type=float, default=0.45)
    ap.add_argument("--y", type=float, default=0.00)
    ap.add_argument("--z", type=float, default=0.30)

    ap.add_argument("--roll", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=0.0)
    ap.add_argument("--yaw", type=float, default=0.0)

    ap.add_argument("--ik_tries", type=int, default=2000)
    ap.add_argument("--ik_noise", type=float, default=1.8)
    ap.add_argument("--base_z", type=float, default=0.02)

    ap.add_argument("--ignore_gripper_collisions", action="store_true")

    # ВАЖНО: midpads теперь настоящий TCP линк midpads_tcp
    ap.add_argument("--tcp", choices=["ee", "midpads"], default="ee",
                    help="ee -> tool=ee_link, midpads -> tool=midpads_tcp (must exist in URDF)")

    ap.add_argument("--pos_tol", type=float, default=0.02)
    ap.add_argument("--ang_tol", type=float, default=0.35)  # рад
    ap.add_argument("--pos_only", action="store_true",
                    help="Ignore orientation in IK+acceptance (works for both ee and midpads_tcp)")
    ap.add_argument("--delta", type=float, default=0.08)
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()

    connect(use_gui=(not args.nogui))
    p.resetSimulation()
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    plane = p.loadURDF("plane.urdf")

    obstacle = None
    if not args.no_obstacle:
        box_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.15])
        box_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.15, 0.15, 0.15])
        obstacle = p.createMultiBody(0, box_col, box_vis)
        p.resetBasePositionAndOrientation(obstacle, [0.45, 0.0, 0.15], [0, 0, 0, 1])

    robot = load_robot(args.urdf, fixed_base=True, base_z=args.base_z)

    arm_joints = pick_arm_joints(robot)
    print("Planning joints (UR5 arm):")
    for j, n in zip(arm_joints, UR5_ARM_JOINT_NAMES):
        print(f"  {j:2d}  {n}")

    # выбор настоящего tool link
    tool_name = "ee_link" if args.tcp == "ee" else "midpads_tcp"
    tool_idx = find_link_index(robot, tool_name)
    if tool_idx is None:
        raise RuntimeError(
            f"Cannot find tool link '{tool_name}' (linkName). "
            f"Если tcp=midpads, убедись что ты добавил midpads_tcp в URDF."
        )
    print(f"Tool link: {tool_name} (index {tool_idx})")

    obstacles = [plane] + ([obstacle] if obstacle is not None else [])

    if args.ignore_gripper_collisions:
        gripper_links = list(range(8, 19))
        disable_link_collisions_with_everything(robot, gripper_links, other_bodies=obstacles)
        print("Gripper collisions disabled for links 8..18")

    collision_fn = make_collision_fn_compat(robot, arm_joints, obstacles, self_collisions=True)

    # старт
    set_joint_positions(robot, arm_joints, UR5_HOME)
    wait_for_duration(0.02)
    q_start = list(get_joint_positions(robot, arm_joints))

    target_world = np.array([args.x, args.y, args.z], dtype=float)
    target_quat = rpy_to_quat(args.roll, args.pitch, args.yaw)

    print("Target pos:", [round(v, 3) for v in target_world.tolist()])
    print("Target rpy:", [round(v, 3) for v in [args.roll, args.pitch, args.yaw]])
    print("q_start:", [round(x, 3) for x in q_start])

    if collision_fn(q_start):
        print("❌ q_start is in collision. Try --base_z 0.05 or --ignore_gripper_collisions")
        disconnect()
        return

    q_goal, errs, ik_len, best_q, best_errs = ik_search(
        robot=robot,
        tool_idx=tool_idx,
        arm_joints=arm_joints,
        q_seed=q_start,
        target_pos_world=target_world,
        target_quat=target_quat,
        collision_fn=collision_fn,
        tries=args.ik_tries,
        noise=args.ik_noise,
        pos_tol=args.pos_tol,
        ang_tol=args.ang_tol,
        pos_only=args.pos_only,
    )

    print(f"IK vector length: {ik_len}")

    if q_goal is None:
        print("❌ IK: no solution within tolerances.")
        if best_q is not None and best_errs is not None:
            pe, ae = best_errs
            print("Best-effort (rejected) q:", [round(x, 3) for x in best_q])
            print(f"Best-effort errs: pos={pe:.4f}m | ang={ae:.3f}rad")
        print("Советы:")
        print("  - увеличь --ik_tries/--ik_noise")
        print("  - или ослабь допуск: --pos_tol 0.03..0.05")
        print("  - или поменяй ориентацию: --roll/--pitch/--yaw")
        disconnect()
        return

    pos_err, ang_err = errs
    print("q_goal:", [round(x, 3) for x in q_goal])
    print(f"ACCEPTED errs: pos={pos_err:.4f}m | ang={ang_err:.3f}rad")

    # вернуть в старт
    set_joint_positions(robot, arm_joints, q_start)

    path = plan_joint_motion(
        robot, arm_joints, q_goal,
        obstacles=obstacles,
        self_collisions=True,
        resolutions=[args.delta] * len(arm_joints),
    )

    if path is None:
        print("❌ Path NOT found to IK goal.")
        disconnect()
        return

    print(f"✅ Path found: {len(path)} waypoints")

    for q in path:
        set_joint_positions(robot, arm_joints, q)
        wait_for_duration(1.0 / 120.0)

    tool_pos, _ = link_state(robot, tool_idx)
    ee_pos, _ = link_state(robot, find_link_index(robot, "ee_link"))
    mid_pos = midpads_world_pos(robot)

    print("final tool     :", [round(float(v), 3) for v in tool_pos])
    print("final ee_link  :", [round(float(v), 3) for v in ee_pos])
    print("final midpads  :", [round(float(v), 3) for v in mid_pos])

    if args.wait:
        print("Done. Close window or Ctrl+C.")
        while True:
            wait_for_duration(0.2)

    disconnect()


if __name__ == "__main__":
    main()
