#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import pybullet as p
import pybullet_data

from pybullet_tools.utils import (
    connect, disconnect,
    wait_for_duration,
    get_movable_joints, get_joint_names,
    get_joint_positions, set_joint_positions,
    plan_joint_motion,
)

UR5_ARM_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# типичная "не нулевая" домашняя поза UR5 (рад) — почти всегда вне самоколлизий
UR5_HOME = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]

def add_search_paths_for_urdf(urdf_path: str):
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    urdf_dir = os.path.dirname(os.path.abspath(urdf_path))
    p.setAdditionalSearchPath(urdf_dir)
    p.setAdditionalSearchPath(os.getcwd())

def load_robot(urdf_path: str, fixed_base: bool = True):
    urdf_path = os.path.abspath(urdf_path)
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF not found: {urdf_path}")
    add_search_paths_for_urdf(urdf_path)
    return p.loadURDF(urdf_path, useFixedBase=fixed_base)

def pick_arm_joints(robot):
    movable = get_movable_joints(robot)
    names = get_joint_names(robot, movable)
    name2jid = {n: j for j, n in zip(movable, names)}
    missing = [n for n in UR5_ARM_JOINT_NAMES if n not in name2jid]
    if missing:
        raise RuntimeError(f"UR5 arm joints not found in URDF: {missing}")
    return [name2jid[n] for n in UR5_ARM_JOINT_NAMES]

def report_contacts(robot, max_lines=30):
    pts = p.getContactPoints(bodyA=robot)
    if not pts:
        print("No self/scene contacts reported for robot.")
        return
    print(f"Contacts (showing up to {max_lines}):")
    for i, c in enumerate(pts[:max_lines]):
        # pybullet: linkIndexA, linkIndexB
        linkA = c[3]
        linkB = c[4]
        print(f"  {i:02d}: linkA={linkA} linkB={linkB} dist={c[8]:.5f} normalF={c[9]:.3f}")
    if len(pts) > max_lines:
        print(f"  ... ({len(pts)-max_lines} more)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", required=True, help="Path to robot URDF")
    ap.add_argument("--nogui", action="store_true")
    ap.add_argument("--delta", type=float, default=0.6, help="Delta (rad) for shoulder_pan_joint goal")
    ap.add_argument("--no_obstacle", action="store_true")
    ap.add_argument("--home", action="store_true", help="Use UR5_HOME as start config (recommended)")
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
        obstacle = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=box_col, baseVisualShapeIndex=box_vis)
        p.resetBasePositionAndOrientation(obstacle, [0.45, 0.0, 0.15], [0, 0, 0, 1])

    robot = load_robot(args.urdf, fixed_base=True)

    # Берём только 6 суставов руки
    joints = pick_arm_joints(robot)
    print("Planning joints (UR5 arm):")
    for j, n in zip(joints, UR5_ARM_JOINT_NAMES):
        print(f"  {j:2d}  {n}")

    # Старт
    q_start = list(get_joint_positions(robot, joints))
    if args.home:
        q_start = list(UR5_HOME)
        set_joint_positions(robot, joints, q_start)
        wait_for_duration(0.2)

    # Проверим, есть ли контакты уже в старте (самоколлизия/коллизия со сценой)
    pts0 = p.getContactPoints(bodyA=robot)
    if pts0:
        print("⚠️  Robot is in contact/collision at start.")
        report_contacts(robot)

    q_goal = list(q_start)
    q_goal[0] = q_goal[0] + args.delta

    print("q_start:", [round(x, 3) for x in q_start])
    print("q_goal :", [round(x, 3) for x in q_goal])

    obstacles = [plane]
    if obstacle is not None:
        obstacles.append(obstacle)

    path = plan_joint_motion(robot, joints, q_goal, obstacles=obstacles)

    if path is None:
        print("❌ Path NOT found")
        disconnect()
        return

    print(f"✅ Path found: {len(path)} waypoints")

    for q in path:
        set_joint_positions(robot, joints, q)
        wait_for_duration(1.0 / 120.0)

    print("Done. Close window or Ctrl+C.")
    while True:
        wait_for_duration(0.2)

if __name__ == "__main__":
    main()
