#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pybullet as p
import pybullet_data

import config as cfg
from utils import sim_utils as su

from utils.debug_draw import clear_debug
from utils.camera_utils import get_depth_and_point_cloud
from utils.cloud_utils import filter_cloud_basic
from utils.calib_utils import calibrate_R_EE_FROM_GRASP
from utils.grasp_gen_utils import find_best_grasps_from_cloud
from utils.motion_utils import (
    grasp_to_tcp_pose,
    approach_axis_from_R,
    execute_pick_and_place,
    get_tool_link,
)
from utils.spawn_utils import respawn_object
from utils.grasp_logging import GraspLogger

from utils.robot_ur5_robotiq85 import (
    UR5Robotiq85,
    debug_print_joints,
    set_arm_prepose,
)

# key codes (PyBullet GUI)
KEY_N = ord("n")
KEY_R = ord("r")
KEY_Q = ord("q")
KEY_SPACE = 32


def _just_pressed(events: dict, key: int) -> bool:
    return (key in events) and ((events[key] & p.KEY_WAS_TRIGGERED) != 0)


def is_object_in_tray(obj_id: int, tray_id: int, margin_xy: float = 0.00, margin_z: float = 0.00) -> bool:
    (mn, mx) = p.getAABB(tray_id, -1)
    mn = np.array(mn, dtype=float)
    mx = np.array(mx, dtype=float)

    pos, _ = p.getBasePositionAndOrientation(obj_id)
    pos = np.array(pos, dtype=float)

    # расширяем границы (margin)
    mn[:2] -= float(margin_xy)
    mx[:2] += float(margin_xy)
    mn[2] -= float(margin_z)
    mx[2] += float(margin_z)

    inside_xy = (mn[0] <= pos[0] <= mx[0]) and (mn[1] <= pos[1] <= mx[1])
    inside_z = (mn[2] <= pos[2] <= mx[2])
    return bool(inside_xy and inside_z)


def build_world() -> int:
    su.setup_world(sim_dt=cfg.SIM_DT, gravity=cfg.GRAVITY)
    su.set_debug_camera(
        camera_distance=1.4,
        camera_yaw=55,
        camera_pitch=-35,
        camera_target=[0.5, 0.2, 0.65],
    )

    # plane + table грузим через su, tray отключаем (мы её грузим вручную чтобы получить tray_id)
    su.load_scene(plane=True, table=True, tray=False)

    # чтобы tray/tray.urdf точно находился
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    tray_id = p.loadURDF(
        "tray/tray.urdf",
        list(cfg.TRAY_POS),
        p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
    )
    return tray_id


def build_robot() -> UR5Robotiq85:
    robot = UR5Robotiq85([0, 0, 0.62], [0, 0, 0])
    robot.load()
    debug_print_joints(robot.id)
    set_arm_prepose(robot, wait=True, hard_reset=True)
    robot.move_gripper(cfg.GRIPPER_OPEN)
    su.step_sim(240, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)
    return robot


def build_object() -> int:
    obj_id = su.spawn_mesh_object(
        mesh_path=cfg.MESH_PATH,
        pos=cfg.OBJ_POS,
        orn_euler=cfg.OBJ_RPY,
        mass=cfg.OBJ_MASS,
        scale=cfg.MESH_SCALE,
        friction=cfg.OBJ_FRICTION,
        spin_friction=cfg.OBJ_SPIN_FRICTION,
        roll_friction=cfg.OBJ_ROLL_FRICTION,
    )
    su.step_sim(240, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)
    return obj_id


def wait_or_skip(paused: bool) -> tuple[bool, str | None]:
    events = p.getKeyboardEvents()

    if _just_pressed(events, KEY_Q):
        return paused, "quit"
    if _just_pressed(events, KEY_R):
        return paused, "reset"
    if _just_pressed(events, KEY_N):
        return paused, "next"
    if _just_pressed(events, KEY_SPACE):
        paused = not paused

    return paused, None

def filter_grasps_world(
    grasps,
    R_EE_FROM_GRASP,
    min_abs_approach_z=0.25,
    prefer_down=True,
    min_center_z=0.0,
):
    """
    Фильтруем КАНДИДАТЫ по реальной оси approach, которая пойдёт в IK,
    т.е. после get_flip_R() и fix_grasp_frame_if_below_table() внутри motion_utils.

    ВАЖНО: здесь НЕ меняем h.R (никаких auto-flip в pick_pipeline).
    """
    out = []
    for h in grasps:
        c = np.asarray(h.center, dtype=np.float64)
        if float(c[2]) < float(min_center_z):
            continue

        # Это ровно та же ориентация, по которой будет строиться tcp_pose и подход
        tcp_pos, tcp_quat, R_hand_used = grasp_to_tcp_pose(h, R_EE_FROM_GRASP)
        a = approach_axis_from_R(R_hand_used)  # нормированный approach

        # выкинуть почти горизонтальные
        if abs(float(a[2])) < float(min_abs_approach_z):
            continue

        # предпочесть "сверху вниз" (у тебя down это a.z < 0)
        if prefer_down and float(a[2]) > -1e-6:
            continue

        out.append(h)

    return out


def main():
    su.connect(cfg.USE_GUI)
    mode = (getattr(cfg, "GRASP_MODE", "nofc") or "nofc").lower().strip()
    logger = GraspLogger(out_dir=f"./logs/{mode}", jsonl_name="grasp_attempts.jsonl")
    print(f"[MODE] {mode}")
    print("[LOG] writing to:", logger.path)


    tray_id = build_world()
    robot = build_robot()
    obj_id = build_object()

    R_EE_FROM_GRASP = calibrate_R_EE_FROM_GRASP(robot)

    cam_pos = [0.3, -0.6, 1.1]
    target = [0.5, 0.0, 0.65]

    paused = False

    while True:
        paused, cmd = wait_or_skip(paused)
        if cmd == "quit":
            break

        if cmd == "reset":
            p.resetSimulation()
            tray_id = build_world()
            robot = build_robot()
            obj_id = build_object()
            R_EE_FROM_GRASP = calibrate_R_EE_FROM_GRASP(robot)
            paused = False
            continue

        if cmd == "next":
            clear_debug([])
            respawn_object(obj_id)
            set_arm_prepose(robot, wait=True, hard_reset=True)
            robot.move_gripper(cfg.GRIPPER_OPEN)
            su.step_sim(240, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

        if paused:
            su.step_sim(1, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)
            continue

        # 1) респавн объекта
        respawn_object(obj_id)

        # 2) дать объекту "устаканиться" перед съёмкой (фикс подпрыгивания)
        su.step_sim(240, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

        # 3) сенсорика
        cloud = get_depth_and_point_cloud(
            cam_pos=cam_pos,
            target_pos=target,
            width=640,
            height=480,
            fov=60.0,
            near=0.05,
            far=2.0,
            downsample=2,
            keep_body_uid=obj_id,
        )
        cloud_f = filter_cloud_basic(cloud)

        grasps = find_best_grasps_from_cloud(cloud_f, top_k=30)

        # пример: стол/объект у тебя около z~0.63..0.69, подбери порог
        grasps = filter_grasps_world(
            grasps,
            R_EE_FROM_GRASP=R_EE_FROM_GRASP,
            min_abs_approach_z=0.30,
            prefer_down=True,
            min_center_z=cfg.SAFE_Z + 0.01,
        )

        grasps = grasps[:5]
        
        if not grasps:
            continue

        # 4) попытки (одна попытка -> следующий респавн)
        for gi, h in enumerate(grasps):
            paused, cmd = wait_or_skip(paused)
            if cmd == "quit":
                su.disconnect()
                return
            if cmd == "reset":
                p.resetSimulation()
                tray_id = build_world()
                robot = build_robot()
                obj_id = build_object()
                R_EE_FROM_GRASP = calibrate_R_EE_FROM_GRASP(robot)
                paused = False
                break
            if cmd == "next":
                break

            # --- START LOG ATTEMPT (БЕЗ tray_id!) ---
            tool_link = get_tool_link(robot)  # если хочешь сохранить tool_link в extra
            logger.start_attempt(
                obj_id=int(obj_id),
                tray_id=(int(tray_id) if tray_id is not None else None),
                grasp_idx=int(gi),
                hyp_center=np.asarray(h.center, dtype=np.float64),
                hyp_width=float(h.width),
            )
            logger.update(tool_link=int(tool_link))  # уйдёт в extra
            # === ADD: log hybrid/FC metadata from candidate ===
            logger.update(
                selected_source=getattr(h, "selected_source", None),
                fc_ok=getattr(h, "fc_ok", None),
                fc_eps=getattr(h, "fc_eps", None),
                rank_nofc=getattr(h, "rank_nofc", None),
            )

            dbg_ids, pipeline_ok, fail_reason = execute_pick_and_place(
                robot,
                h,
                obj_id,
                R_EE_FROM_GRASP=R_EE_FROM_GRASP,
                tray_pos=np.array(cfg.TRAY_POS, dtype=np.float64),
                tray_drop_height=0.20,
                tray_down_height=0.06,
                sum_force_thresh=10.0,
                min_contacts=2,
                tray_id=(tray_id if tray_id is not None else None),
                logger=logger,   # === ADD ===
            )


            # дать физике улечься перед проверкой корзины
            su.step_sim(240, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

            # --- TASK SUCCESS: объект в корзине? ---
            task_ok = False
            if tray_id is not None:
                task_ok = is_object_in_tray(obj_id, tray_id, margin_xy=0.03, margin_z=0.05)

            # --- FINISH LOG ATTEMPT ---
            logger.finish_attempt(
                pipeline_ok=bool(pipeline_ok),
                task_success=bool(task_ok),
                fail_reason=(fail_reason if not pipeline_ok else None),
            )

            clear_debug(dbg_ids)

            set_arm_prepose(robot, wait=True, hard_reset=True)
            robot.move_gripper(cfg.GRIPPER_OPEN)
            su.step_sim(240, sim_dt=cfg.SIM_DT, use_gui=cfg.USE_GUI, sleep_in_gui=cfg.SLEEP_IN_GUI)

            break  # после первой попытки — следующий объект

    su.disconnect()


if __name__ == "__main__":
    main()
