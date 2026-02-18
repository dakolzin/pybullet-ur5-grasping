#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import random
from collections import namedtuple

import pybullet as p
import pybullet_data

import numpy as np


def make_camera_matrices(cam_pos, target_pos, up=(0, 0, 1), fov=60.0, aspect=1.0, near=0.01, far=2.0):
    """
    Возвращает view/proj матрицы PyBullet и параметры для обратного преобразования depth->XYZ.
    cam_pos, target_pos: (x,y,z) в мире
    """
    view = p.computeViewMatrix(cameraEyePosition=cam_pos,
                               cameraTargetPosition=target_pos,
                               cameraUpVector=up)
    proj = p.computeProjectionMatrixFOV(fov=fov, aspect=aspect, nearVal=near, farVal=far)
    return view, proj, (fov, aspect, near, far)


def depth_to_point_cloud(depth, view_matrix, proj_matrix, width, height, near, far, downsample=1):
    """
    depth: (H,W) float32 из PyBullet (0..1)
    Возвращает Nx3 точки в МИРОВОЙ СК (world frame).
    """
    depth = np.asarray(depth, dtype=np.float32).reshape((height, width))

    # Привести к Z в координатах камеры (OpenGL depth -> metric depth)
    # Формула из документации Bullet:
    # z = (2*near*far) / (far + near - (2*depth - 1)*(far - near))
    z = (2.0 * near * far) / (far + near - (2.0 * depth - 1.0) * (far - near))

    # Сетка пикселей
    xs = np.arange(0, width, downsample)
    ys = np.arange(0, height, downsample)
    xv, yv = np.meshgrid(xs, ys)

    # Нормализованные координаты (NDC) [-1,1]
    x_ndc = (xv + 0.5) / width * 2.0 - 1.0
    y_ndc = 1.0 - (yv + 0.5) / height * 2.0   # инверсия оси Y

    z_ds = z[::downsample, ::downsample]

    # Соберём 4D точки в clip space
    ones = np.ones_like(z_ds, dtype=np.float32)
    clip = np.stack([x_ndc.astype(np.float32), y_ndc.astype(np.float32), (2.0 * depth[::downsample, ::downsample] - 1.0).astype(np.float32), ones], axis=-1)
    clip = clip.reshape(-1, 4)

    # Инвертируем (proj*view) чтобы получить world
    view = np.array(view_matrix, dtype=np.float32).reshape(4, 4).T
    proj = np.array(proj_matrix, dtype=np.float32).reshape(4, 4).T
    inv = np.linalg.inv(proj @ view)

    world = (inv @ clip.T).T
    world = world[:, :3] / world[:, 3:4]

    # Уберём "far" точки (где depth ~ 1.0)
    valid = (depth[::downsample, ::downsample].reshape(-1) < 0.9999)
    return world[valid]


def get_depth_and_point_cloud(cam_pos, target_pos, width=640, height=480,
                              fov=60.0, near=0.01, far=2.0, downsample=2):
    """
    Делает рендер depth и возвращает:
      rgb (H,W,3), depth (H,W), points_world (N,3)
    """
    aspect = width / float(height)
    view, proj, (_, _, near, far) = make_camera_matrices(cam_pos, target_pos, fov=fov, aspect=aspect, near=near, far=far)

    img = p.getCameraImage(width=width, height=height, viewMatrix=view, projectionMatrix=proj,
                           renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgba = np.reshape(img[2], (height, width, 4)).astype(np.uint8)
    rgb = rgba[:, :, :3]

    depth = np.reshape(img[3], (height, width)).astype(np.float32)
    pts = depth_to_point_cloud(depth, view, proj, width, height, near, far, downsample=downsample)
    return rgb, depth, pts, view, proj

def draw_point_cloud(points, color=(0, 1, 0), size=1.0, life=0.1, max_points=2000):
    # ограничим количество точек, иначе лаги
    if points.shape[0] > max_points:
        idx = np.random.choice(points.shape[0], max_points, replace=False)
        points = points[idx]
    for pt in points:
        p.addUserDebugLine(pt, [pt[0], pt[1], pt[2] + 0.001], lineColorRGB=color, lineWidth=size, lifeTime=life)

# =========================
# Настройки эксперимента
# =========================
USE_GUI = True                  # True: окно PyBullet, False: headless
SIM_DT = 1.0 / 240.0
GRAVITY = -9.8


# Папка с STL
MESH_FILES = [
    "./meshes/part/part_1.stl",
    "./meshes/part/part_2.stl",
    "./meshes/part/part_3.stl",
]

# Масштаб STL (если STL в мм, ставь 0.001)
MESH_SCALE = (1.0, 1.0, 1.0)

# Физика объекта
OBJ_MASS = 0.2
OBJ_FRICTION = 0.9
OBJ_SPIN_FRICTION = 0.01
OBJ_ROLL_FRICTION = 0.001

# Зона спавна объекта
SPAWN_X_RANGE = (0.35, 0.65)
SPAWN_Y_RANGE = (-0.12, 0.12)
SPAWN_Z = 0.65

# Движения робота относительно объекта
APPROACH_Z_OFFSET = 0.18
GRASP_Z_OFFSET = 0.15
LIFT_Z_OFFSET = 0.35

# Траектория к лотку
TRAY_POS = [0.5, 0.9, 0.6]
TRAY_DROP_Z_OFFSET = 0.56
TRAY_JITTER_RANGE = (0.10, 0.30)

# Гриппер
GRIPPER_OPEN = 0.085
GRIPPER_CLOSE = 0.01


# =========================
# Робот UR5 + Robotiq 85
# =========================
class UR5Robotiq85:
    def __init__(self, pos, ori_rpy):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori_rpy)

        self.eef_id = 7
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.57, -1.54, 1.34, -1.37, -1.57, 0.0]
        self.gripper_range = [0.0, 0.085]
        self.max_velocity = 3.0

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
                    jointID,
                    jointName,
                    jointType,
                    jointLowerLimit,
                    jointUpperLimit,
                    jointMaxForce,
                    jointMaxVelocity,
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
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)

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
        )
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(
                self.id,
                joint_id,
                p.POSITION_CONTROL,
                joint_poses[i],
                maxVelocity=self.max_velocity,
            )

    def move_gripper(self, open_length):
        open_length = max(self.gripper_range[0], min(open_length, self.gripper_range[1]))
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle)

    def get_eef_pose(self):
        return p.getLinkState(self.id, self.eef_id)


# =========================
# Утилиты симуляции
# =========================
def step_sim(steps, sleep=True):
    for _ in range(steps):
        p.stepSimulation()
        if sleep and USE_GUI:
            time.sleep(SIM_DT)


def setup_simulation():
    if USE_GUI:
        p.connect(p.GUI)
    else:
        p.connect(p.DIRECT)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setTimeStep(SIM_DT)
    p.setGravity(0, 0, GRAVITY)

    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))

    tray_orn = p.getQuaternionFromEuler([0, 0, 0])
    p.loadURDF("tray/tray.urdf", TRAY_POS, tray_orn)

    return TRAY_POS, tray_orn


def spawn_mesh_object(mesh_path, pos, orn_euler=(0.0, 0.0, 0.0), mass=0.2, scale=(1.0, 1.0, 1.0)):
    orn = p.getQuaternionFromEuler(list(orn_euler))

    col = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=list(scale),
    )

    vis = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=mesh_path,
        meshScale=list(scale),
        rgbaColor=[0.8, 0.8, 0.8, 1.0],
    )

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=pos,
        baseOrientation=orn,
    )

    p.changeDynamics(
        body_id,
        -1,
        lateralFriction=OBJ_FRICTION,
        spinningFriction=OBJ_SPIN_FRICTION,
        rollingFriction=OBJ_ROLL_FRICTION,
    )

    return body_id


def random_spawn_pose():
    x = random.uniform(*SPAWN_X_RANGE)
    y = random.uniform(*SPAWN_Y_RANGE)
    z = SPAWN_Z
    yaw = random.uniform(-math.pi, math.pi)
    return [x, y, z], (0.0, 0.0, yaw)


def set_arm_prepose(robot: UR5Robotiq85):
    # Предпоза (как у тебя было)
    target_joint_positions = [0, -1.57, 1.57, -1.5, -1.57, 0.0]
    for i, joint_id in enumerate(robot.arm_controllable_joints):
        p.setJointMotorControl2(robot.id, joint_id, p.POSITION_CONTROL, target_joint_positions[i])
    step_sim(240)


def try_pick_and_place(robot: UR5Robotiq85, tray_pos, counter: int):
    # 1) Спавн объекта
    obj_pos, obj_rpy = random_spawn_pose()
    mesh_path = random.choice(MESH_FILES)

    obj_id = spawn_mesh_object(
        mesh_path=mesh_path,
        pos=obj_pos,
        orn_euler=obj_rpy,
        mass=OBJ_MASS,
        scale=MESH_SCALE,
    )

    # Дать объекту “успокоиться”
    step_sim(120)

    # --- Камера: позиция и куда смотрит ---
    cam_pos = [0.3, -0.6, 1.1]     # где стоит камера в мире
    target = [0.5, 0.0, 0.65]      # куда смотрит

    rgb, depth, cloud, view, proj = get_depth_and_point_cloud(
        cam_pos=cam_pos,
        target_pos=target,
        width=640,
        height=480,
        fov=60.0,
        near=0.05,
        far=2.0,
        downsample=2
    )

    print("Point cloud size:", cloud.shape)  # (N,3)

    #draw_point_cloud(cloud, color=(0,1,0), life=0.2, max_points=1500)

    # 2) Текущая ориентация эффектора (оставим как есть)
    eef_state = robot.get_eef_pose()
    eef_orientation = eef_state[1]

    # 3) Подлет
    robot.move_gripper(GRIPPER_OPEN)
    step_sim(60)

    x, y, z = obj_pos
    robot.move_arm_ik([x, y, z + APPROACH_Z_OFFSET], eef_orientation)
    step_sim(180)

    # 4) Опускание
    robot.move_arm_ik([x, y, z + GRASP_Z_OFFSET], eef_orientation)
    step_sim(150)

    # 5) Сжатие
    robot.move_gripper(GRIPPER_CLOSE)
    step_sim(120)

    # 6) Подъем
    robot.move_arm_ik([x, y, z + LIFT_Z_OFFSET], eef_orientation)
    step_sim(240)

    # 7) Перенос к лотку
    jitter = random.uniform(*TRAY_JITTER_RANGE)
    robot.move_arm_ik([tray_pos[0] + jitter, tray_pos[1] + jitter, tray_pos[2] + TRAY_DROP_Z_OFFSET], eef_orientation)
    step_sim(360)

    # 8) Разжатие
    robot.move_gripper(GRIPPER_OPEN)
    step_sim(180)

    # 9) Удалить объект (чтобы сцена не “засорялась”)
    p.removeBody(obj_id)

    # 10) Счетчик
    counter += 1
    p.addUserDebugText(
        f"Count: {counter}",
        textColorRGB=[1, 0, 0],
        textPosition=[tray_pos[0], tray_pos[1], tray_pos[2] + 0.3],
        textSize=2.5,
        lifeTime=2,
    )
    return counter



def main():
    tray_pos, _ = setup_simulation()

    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()

    counter = 0

    while True:
        set_arm_prepose(robot)
        counter = try_pick_and_place(robot, tray_pos, counter)


if __name__ == "__main__":
    main()
