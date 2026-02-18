# utils/spawn_utils.py
from __future__ import annotations
import math
import numpy as np
import pybullet as p
import config as cfg

_rng = np.random.default_rng(getattr(cfg, "RANDOM_SEED", None))

def sample_pose_on_table():
    x = float(_rng.uniform(cfg.TABLE_XY_MIN[0], cfg.TABLE_XY_MAX[0]))
    y = float(_rng.uniform(cfg.TABLE_XY_MIN[1], cfg.TABLE_XY_MAX[1]))
    z = float(getattr(cfg, "OBJ_SPAWN_Z", cfg.TABLE_Z + 0.02))

    yaw = float(_rng.uniform(cfg.OBJ_YAW_RANGE[0], cfg.OBJ_YAW_RANGE[1]))
    roll  = float(_rng.uniform(cfg.OBJ_ROLL_RANGE[0], cfg.OBJ_ROLL_RANGE[1]))
    pitch = float(_rng.uniform(cfg.OBJ_PITCH_RANGE[0], cfg.OBJ_PITCH_RANGE[1]))

    quat = p.getQuaternionFromEuler((roll, pitch, yaw))
    return [x, y, z], quat

def respawn_object(obj_id: int):
    pos, quat = sample_pose_on_table()
    p.resetBasePositionAndOrientation(obj_id, pos, quat)
    p.resetBaseVelocity(obj_id, (0,0,0), (0,0,0))
    return pos, quat
