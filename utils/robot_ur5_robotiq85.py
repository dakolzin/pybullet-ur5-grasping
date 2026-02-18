# utils/robot_ur5_robotiq85.py
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from collections import namedtuple
from typing import Sequence, Tuple

import numpy as np
import pybullet as p

import config as cfg
from utils import sim_utils as su


class UR5Robotiq85:
    def __init__(self, pos, ori_rpy):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori_rpy)

        self.eef_id = 7  # ee_link

        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0.0, 0.085]
        self.max_velocity = cfg.ARM_MAX_VEL

        self.id = None
        self.joints = []
        self.controllable_joints = []
        self.arm_controllable_joints = []
        self.arm_lower_limits = []
        self.arm_upper_limits = []
        self.arm_joint_ranges = []
        self.mimic_parent_id = None
        self.mimic_child_multiplier = {}

    def load(self):
        self.id = p.loadURDF("./urdf/ur5_robotiq_85.urdf", self.base_pos, self.base_ori, useFixedBase=True)
        self.__parse_joint_info__()
        self.__setup_mimic_joints__()

    def __parse_joint_info__(self):
        jointInfo = namedtuple(
            "jointInfo",
            ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity", "controllable"],
        )

        self.joints = []
        self.controllable_joints = []

        for i in range(p.getNumJoints(self.id)):
            info = p.getJointInfo(self.id, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = info[2]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = jointType != p.JOINT_FIXED

            if controllable:
                self.controllable_joints.append(jointID)

            self.joints.append(
                jointInfo(
                    jointID, jointName, jointType,
                    jointLowerLimit, jointUpperLimit,
                    jointMaxForce, jointMaxVelocity,
                    controllable,
                )
            )

        self.arm_controllable_joints = self.controllable_joints[: self.arm_num_dofs]
        self.arm_lower_limits = [j.lowerLimit for j in self.joints if j.controllable][: self.arm_num_dofs]
        self.arm_upper_limits = [j.upperLimit for j in self.joints if j.controllable][: self.arm_num_dofs]
        self.arm_joint_ranges = [ul - ll for ul, ll in zip(self.arm_upper_limits, self.arm_lower_limits)]

    def __setup_mimic_joints__(self):
        mimic_parent_name = "finger_joint"
        mimic_children_names = {
            "right_outer_knuckle_joint": 1,
            "left_inner_knuckle_joint": 1,
            "right_inner_knuckle_joint": 1,
            "left_inner_finger_joint": -1,
            "right_inner_finger_joint": -1,
        }

        self.mimic_parent_id = [joint.id for joint in self.joints if joint.name == mimic_parent_name][0]
        self.mimic_child_multiplier = {
            joint.id: mimic_children_names[joint.name] for joint in self.joints if joint.name in mimic_children_names
        }

        for child_id in self.mimic_child_multiplier.keys():
            p.setJointMotorControl2(self.id, child_id, p.VELOCITY_CONTROL, targetVelocity=0.0, force=0.0)

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(
                self.id,
                self.mimic_parent_id,
                self.id,
                joint_id,
                jointType=p.JOINT_GEAR,
                jointAxis=[0, 1, 0],
                parentFramePosition=[0, 0, 0],
                childFramePosition=[0, 0, 0],
            )
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=cfg.GEAR_MAX_FORCE, erp=cfg.GEAR_ERP)

    def move_arm_ik(self, target_pos, target_orn):
        joint_poses = p.calculateInverseKinematics(
            self.id,
            self.eef_id,
            target_pos,
            target_orn,
            lowerLimits=self.arm_lower_limits,
            upperLimits=self.arm_upper_limits,
            jointRanges=self.arm_joint_ranges,
            restPoses=self.arm_rest_poses,
            maxNumIterations=200,
            residualThreshold=1e-4,
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id, joint_id,
                p.POSITION_CONTROL,
                joint_poses[i],
                maxVelocity=self.max_velocity,
                force=cfg.ARM_JOINT_FORCE,
                positionGain=cfg.POS_GAIN,
                velocityGain=cfg.VEL_GAIN,
            )

    def move_gripper(self, open_length):
        open_length = max(self.gripper_range[0], min(open_length, self.gripper_range[1]))
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(
            self.id,
            self.mimic_parent_id,
            p.POSITION_CONTROL,
            targetPosition=open_angle,
            force=cfg.GRIPPER_FORCE,
            maxVelocity=cfg.GRIPPER_MAX_VEL,
        )


def debug_print_joints(robot_id: int):
    print("\n=== JOINTS ===")
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        name = info[1].decode("utf-8")
        link = info[12].decode("utf-8")
        print(f"{i:2d} joint={name:30s} link={link}")
    print("==============\n")


def get_gripper_state(robot: UR5Robotiq85):
    js = p.getJointState(robot.id, robot.mimic_parent_id)
    return float(js[0]), float(js[1])


def get_eef_world_link_frame(robot: UR5Robotiq85) -> tuple[np.ndarray, tuple[float, float, float, float]]:
    """Единая функция, чтобы её могли импортить и calib_utils, и motion_utils."""
    ls = p.getLinkState(robot.id, robot.eef_id, computeForwardKinematics=True)
    pos = np.array(ls[4], dtype=np.float64)
    orn = tuple(ls[5])
    return pos, orn


def reset_arm_joints(robot: UR5Robotiq85, q: Sequence[float]):
    """Жёстко сбрасывает arm суставы (убирает накопленную 'выгнутость')."""
    for i, jid in enumerate(robot.arm_controllable_joints):
        p.resetJointState(robot.id, jid, float(q[i]), 0.0)
    su.step_sim(10, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)


def wait_arm_reached(
    robot: UR5Robotiq85,
    target_q: Sequence[float],
    pos_tol: float = 0.02,   # рад
    vel_tol: float = 0.05,   # рад/с
    max_steps: int = 6000,
) -> bool:
    target_q = np.asarray(target_q, dtype=np.float64)

    for _ in range(int(max_steps)):
        q = []
        v = []
        for jid in robot.arm_controllable_joints:
            st = p.getJointState(robot.id, jid)
            q.append(st[0])
            v.append(st[1])
        q = np.asarray(q, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)

        err = float(np.max(np.abs(q - target_q)))
        vabs = float(np.max(np.abs(v)))

        if err < pos_tol and vabs < vel_tol:
            return True

        su.step_sim(1, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

    return False


def set_arm_prepose(
    robot: UR5Robotiq85,
    target_joint_positions: Sequence[float] = (0, -1.57, 1.57, -1.5, -1.57, 0.0),
    wait: bool = True,
    hard_reset: bool = False,
):
    if hard_reset:
        reset_arm_joints(robot, target_joint_positions)

    for i, joint_id in enumerate(robot.arm_controllable_joints):
        p.setJointMotorControl2(
            robot.id,
            joint_id,
            p.POSITION_CONTROL,
            float(target_joint_positions[i]),
            force=cfg.ARM_JOINT_FORCE,
            maxVelocity=cfg.ARM_MAX_VEL,
            positionGain=cfg.POS_GAIN,
            velocityGain=cfg.VEL_GAIN,
        )

    if wait:
        ok = wait_arm_reached(robot, target_joint_positions)
        if not ok:
            print("[WARN] arm did not reach prepose in time, continuing anyway")
    else:
        su.step_sim(240, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

def gripper_opening_m_from_finger_joint_angle(open_angle: float) -> float:
    """
    Инверсия формулы из UR5Robotiq85.move_gripper().
    open_angle — позиция mimic_parent (finger_joint) в радианах.
    Возвращает текущую ширину раскрытия в метрах.
    """
    # open_angle = 0.715 - asin((open_length - 0.010)/0.1143)
    # => open_length = 0.010 + 0.1143 * sin(0.715 - open_angle)
    return float(0.010 + 0.1143 * math.sin(0.715 - float(open_angle)))


def get_gripper_opening_m(robot: UR5Robotiq85) -> float:
    """
    Текущая ширина раскрытия (м) по состоянию finger_joint (mimic_parent_id).
    """
    open_angle, _open_vel = get_gripper_state(robot)  # уже есть в файле
    opening = gripper_opening_m_from_finger_joint_angle(open_angle)

    # защита: обрежем в допустимый диапазон, который ты уже используешь
    if hasattr(robot, "gripper_range") and robot.gripper_range:
        opening = max(float(robot.gripper_range[0]), min(float(opening), float(robot.gripper_range[1])))
    return float(opening)
