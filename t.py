#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pybullet as p

import config as cfg
from utils import sim_utils as su
from utils.robot_ur5_robotiq85 import UR5Robotiq85


def check_pose_reachable(robot, pos, quat, name="POSE"):
    q = p.calculateInverseKinematics(
        robot.id, robot.eef_id,
        pos, quat,
        lowerLimits=robot.arm_lower_limits,
        upperLimits=robot.arm_upper_limits,
        jointRanges=robot.arm_joint_ranges,
        restPoses=robot.arm_rest_poses,
        maxNumIterations=400,
        residualThreshold=1e-4,
    )[:6]

    saved = [p.getJointState(robot.id, jid)[0] for jid in robot.arm_controllable_joints]
    for i, jid in enumerate(robot.arm_controllable_joints):
        p.resetJointState(robot.id, jid, float(q[i]), 0.0)

    ls = p.getLinkState(robot.id, robot.eef_id, computeForwardKinematics=True)
    fk_pos = np.array(ls[4], dtype=np.float64)
    err = float(np.linalg.norm(fk_pos - np.array(pos, dtype=np.float64)))

    for i, jid in enumerate(robot.arm_controllable_joints):
        p.resetJointState(robot.id, jid, float(saved[i]), 0.0)

    print(f"[{name}] IK->FK err={err:.4f} m | fk_pos={fk_pos} | target={pos}")
    print(f"[{name}] q={np.array(q)}")
    return q, err


def main():
    su.connect(True)
    su.setup_world(sim_dt=cfg.SIM_DT, gravity=cfg.GRAVITY)
    su.load_scene(plane=True, table=True, tray=True, tray_pos=cfg.TRAY_POS)

    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()

    # точка над корзиной
    carry_quat = p.getQuaternionFromEuler(cfg.BIN_TCP_RPY)
    pos_above = [cfg.TRAY_POS[0], cfg.TRAY_POS[1], cfg.SAFE_Z + 0.25]

    check_pose_reachable(robot, pos_above, carry_quat, "BIN_ABOVE")

    print("Close window / Ctrl+C to exit")
    while True:
        su.step_sim(1, sim_dt=cfg.SIM_DT, use_gui=True, sleep_in_gui=True)


if __name__ == "__main__":
    main()
