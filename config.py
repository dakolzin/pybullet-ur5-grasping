# -*- coding: utf-8 -*-
"""
config.py — все параметры пайплайна в одном месте.
(совместим с utils/spawn_utils.py и pick_pipeline.py)
"""

from __future__ import annotations
import numpy as np
import math

# =========================
# Simulation
# =========================
USE_GUI = True
SIM_DT = 1.0 / 240.0
GRAVITY = -9.8
SLEEP_IN_GUI = True

# =========================
# Scene / geometry
# =========================
TABLE_Z = 0.62
SAFE_Z = TABLE_Z + 0.03

# диапазон спавна объекта на столе (нужен spawn_utils.py)
TABLE_XY_MIN = (0.35, -0.15)
TABLE_XY_MAX = (0.55,  0.20)

OBJ_SPAWN_Z = TABLE_Z + 0.03  # если spawn_utils кладёт "на стол"

# =========================
# Assets
# =========================
MESH_PATH = "./meshes/part/part_1.stl"
MESH_SCALE = (1.0, 1.0, 1.0)

OBJ_MASS = 0.2

# дефолтные поза/ориентация (если не рандомишь)
OBJ_POS = [0.7, 0.03, 0.6532]
OBJ_RPY = (0.0, 0.0, 1.57)

# рандом ориентации (spawn_utils может использовать)
OBJ_ROLL_RANGE  = (0.0, 0.0)
OBJ_PITCH_RANGE = (0.0, 0.0)
OBJ_YAW_RANGE   = (-math.pi, math.pi)

RANDOM_SEED = None

# корзина
TRAY_POS = [0.1, 0.6, 0.6]

# =========================
# Gripper / approach geometry
# =========================
GRIPPER_OPEN = 0.085

PREGRASP_DIST = 0.10
APPROACH_STANDOFF = 0.06
PRE_BACK_DIST = 0.05

# (если где-то используешь)
GRASP_EXTRA_PUSH = 0.02

# =========================
# Motion tuning
# =========================
REACH_TOL = 0.03

ARM_MAX_VEL = 1.2
ARM_JOINT_FORCE = 1500

MOVE_RAMP_STEPS_BACK = 520
MOVE_RAMP_STEPS_PRE = 420
MOVE_RAMP_STEPS_GRASP = 360
MOVE_RAMP_STEPS_GRASP_CONTACT = 120

HOLD_STEPS_AFTER = 60
STEPS_PER_WP = 2
SEGMENT_PAUSE_SEC = 1.0

POS_GAIN = 0.6
VEL_GAIN = 1.2

# =========================
# TCP / frames
# =========================
TCP_LINK_NAME = "midpads_tcp"

# У тебя цель движения = TCP(midpads_tcp), а grasp.center уже в world.
# Поэтому смещение = 0 (иначе ты уедешь от цели).
TCP_OFFSET_HAND = np.array([0.0, 0.0, 0.0], dtype=np.float64)

FLIP_TCP_180_DEG = False
FLIP_AXIS = "z"  # "x" | "y" | "z"

# матрица калибровки (если у тебя калибровщик пишет сюда/использует по умолчанию)
R_EE_FROM_GRASP_INIT = np.eye(3, dtype=np.float64)

# =========================
# Contacts / early stop
# =========================
ENABLE_CONTACT_STOP = True
CONTACT_FORCE_THRESH = 5.0

# pads (по твоему логу: 12 и 17 — это именно finger_pad)
FINGER_LINK_IDS = {12, 17}

GRASP_MIN_CONTACTS = 2
GRASP_MIN_SUMF = 20.0
GRASP_GOOD_SUMF = 45.0

# =========================
# Gripper actuation
# =========================
# если "слабо закрывается" — подними (80..140 обычно уже ощутимо)
GRIPPER_FORCE = 120
GRIPPER_MAX_VEL = 0.35

GEAR_MAX_FORCE = 120
GEAR_ERP = 0.6

# протокол дожима
GRIPPER_CLOSE_SETTLE_STEPS = 900
GRIPPER_CONTACT_CHECK_STEPS = 240
GRIPPER_TIGHTEN_DELTA = 0.002
GRIPPER_TIGHTEN_MAX_STEPS = 700

# =========================
# Lift test (anti-slip)
# =========================
LIFT_TEST_DZ = 0.03
LIFT_TEST_STEPS = 220
LIFT_TEST_SETTLE_STEPS = 120
SLIP_DROP_SUMF = 0.4

# =========================
# Contact materials (главное против выскальзывания)
# =========================
# трение объекта
OBJ_LAT_FRIC = 1.5
OBJ_SPIN_FRIC = 0.03
OBJ_ROLL_FRIC = 0.003

# трение pads/пальцев (как "резина")
PAD_LAT_FRIC = 5.0
PAD_SPIN_FRIC = 0.06
PAD_ROLL_FRIC = 0.006

# на всякий (если где-то ещё у тебя старые поля используются)
OBJ_FRICTION = 1.0
OBJ_SPIN_FRICTION = 0.01
OBJ_ROLL_FRICTION = 0.001

# =========================
# Place planning
# =========================
BIN_TCP_RPY = (math.pi, 0.0, -math.pi / 2.0)

IK_TRIES = 1200
IK_NOISE = 1.6
PLACE_DELTA = 0.08

# =========================
# Disable collisions robot<->tray
# =========================
DISABLE_ROBOT_TRAY_COLLISIONS = True

# -------------------------
# Grasp quality thresholds
# -------------------------
GRASP_MIN_CONTACTS = 2
GRASP_MIN_SUMF = 15.0  # N (подбери 10..40)

# Дожим (tighten) после слабого контакта
GRIPPER_TIGHTEN_DELTA = 0.006   # метров (6 мм) — уменьшение target width
GRIPPER_TIGHTEN_MAX_STEPS = 600 # сколько держим руку, пока дожимается

# Стабилизация закрытия (ты уже добавил)
GRIPPER_CLOSE_SETTLE_STEPS = 900
GRIPPER_CONTACT_CHECK_STEPS = 240

# Lift-test anti-slip
LIFT_TEST_DZ = 0.030
LIFT_TEST_STEPS = 160
LIFT_TEST_SETTLE_STEPS = 240

# Если после подъёма суммарная сила упала ниже этого процента от исходной -> считаем срыв
SLIP_DROP_SUMF = 0.25  # 0.25 означает "упало более чем на 75%" -> FAIL

# -------------------------
# Disable robot <-> tray collisions
# -------------------------
DISABLE_ROBOT_TRAY_COLLISIONS = True

# (необязательно) трение на подушечках пальцев
PAD_LAT_FRIC = 3.0
PAD_SPIN_FRIC = 0.06
PAD_ROLL_FRIC = 0.006

# насколько выше безопасной высоты/корзины поднимать при переносе
CARRY_Z_MARGIN = 0.10     # 10 см, попробуй 0.10..0.20
CARRY_Z_ABS_MIN = 0.85    # абсолютный минимум Z переноса (под твою сцену)

# скорость поворота J1 при развороте на 90°
TURN_J1_MAX_VEL = 0.20  # 0.10..0.40 обычно норм, начни с 0.20


# =========================
# Grasp generation mode
# =========================
# "nofc"   : быстрый генератор (close_nofc)
GRASP_MODE = "nofc"   # "nofc" | "hybrid" | "eps"
EPS_MIN_VALID = 1e-5

# HYBRID: сколько top кандидатов переоценивать по ε
HYBRID_TOPK_FC = 30

# HYBRID: минимальный ε, чтобы считать гипотезу "хорошей"
HYBRID_EPS_MIN = 1e-3

# NOFC генератор (если захочешь тюнить не лезя в код)
NOFC_NUM_SEEDS = 900
NOFC_NEIGH_RADIUS = 0.07
NOFC_K_LOCAL = 20
NOFC_FRAME = "taubin"
