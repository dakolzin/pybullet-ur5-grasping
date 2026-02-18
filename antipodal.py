#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import time
import argparse
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pybullet as p
import pybullet_data
from sklearn.neighbors import KDTree


# =========================
# Параметры "руки" (параллельное двухпальцевое ЗУ)
# =========================
@dataclass
class HandParams:
    # СК руки:
    # x — раскрытие (между пальцами)
    # y — подход к объекту (approach)  (+y = "вперед", в сторону объекта)
    # z — вдоль пальцев
    finger_thickness: float = 0.01
    finger_depth: float = 0.04
    finger_height: float = 0.06

    palm_depth: float = 0.02

    min_width: float = 0.02
    max_width: float = 0.09

    closing_region_depth: float = 0.02
    closing_region_height: float = 0.06
    closing_region_width: float = 0.10

    # ---- Антиподальная проверка ----
    antipodal_alpha_deg: float = 40.0
    antipodal_beta_deg: float = 40.0
    contact_band: float = 0.010
    normal_k: int = 25

    # ---- Ограничение направления подхода ----
    approach_from_above: bool = True
    approach_max_tilt_deg: float = 75.0

    # ---- Глобальные ограничения сцены ----
    # защита от "снизу": запрещаем центры (и объёмы пальцев) ниже min_z объекта + clearance
    min_z_clearance: float = 0.008  # 8 мм

    # ---- ROI для глобальных проверок ----
    roi_margin: float = 0.01  # 1 см запас


# =========================
# Геометрия
# =========================
def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n

def pca_normal(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0)
    X = points - c
    C = (X.T @ X) / max(len(points), 1)
    w, V = np.linalg.eigh(C)
    n = V[:, 0]  # минимум дисперсии
    return normalize(n)

def make_darboux_frame(n: np.ndarray, neighbors: np.ndarray) -> np.ndarray:
    """R (3x3), столбцы: [t, b, n]"""
    n = normalize(n)
    c = neighbors.mean(axis=0)
    X = neighbors - c
    Xp = X - (X @ n)[:, None] * n[None, :]
    C = (Xp.T @ Xp) / max(len(neighbors), 1)
    w, V = np.linalg.eigh(C)
    t = V[:, 2]  # максимум дисперсии в касательной плоскости
    t = normalize(t - np.dot(t, n) * n)
    b = normalize(np.cross(n, t))
    t = normalize(np.cross(b, n))
    return np.column_stack([t, b, n])

def rot_about_axis(axis: np.ndarray, angle: float) -> np.ndarray:
    ax = normalize(axis)
    x, y, z = float(ax[0]), float(ax[1]), float(ax[2])
    c = math.cos(angle)
    s = math.sin(angle)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C  ]
    ], dtype=np.float64)

def rot_angle(R1: np.ndarray, R2: np.ndarray) -> float:
    M = R1.T @ R2
    tr = float(np.trace(M))
    c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return math.acos(c)

def _v3(x) -> List[float]:
    return [float(x[0]), float(x[1]), float(x[2])]


# =========================
# ROI по глобальному облаку
# =========================
def roi_radius_for_hand(width: float, hp: HandParams) -> float:
    # Радиус сферы, которая гарантированно покрывает все OBB-объёмы руки + запас.
    hx = 0.5 * width + hp.finger_thickness + hp.roi_margin
    hy = hp.finger_depth + hp.palm_depth + hp.closing_region_depth + hp.roi_margin
    hz = 0.5 * hp.closing_region_height + hp.roi_margin
    return float(math.sqrt(hx*hx + hy*hy + hz*hz))

def get_roi_points(tree: KDTree, points: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    ind = tree.query_radius(center.reshape(1, 3), r=radius)[0]
    if ind.shape[0] == 0:
        return points[:0]
    return points[ind]


# =========================
# OBB-проверки по точкам
# =========================
def points_in_obb(points: np.ndarray, center: np.ndarray, R: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    # R — столбцы: оси СК в мире. Нам нужно перейти в локальные координаты => (p-center)^T * R
    local = (points - center[None, :]) @ R
    return np.all(np.abs(local) <= half_extents[None, :], axis=1)

def hand_collision_free(points_roi: np.ndarray, center: np.ndarray, R_hand: np.ndarray, width: float, hp: HandParams) -> bool:
    # Ладонь: y ∈ [-palm_depth, 0]
    palm_half = np.array([width * 0.6, hp.palm_depth * 0.5, hp.closing_region_height * 0.5], dtype=np.float64)
    palm_center_local = np.array([0.0, -hp.palm_depth * 0.5, 0.0], dtype=np.float64)
    palm_center_world = center + (R_hand @ palm_center_local)
    if np.any(points_in_obb(points_roi, palm_center_world, R_hand, palm_half)):
        return False

    # Пальцы: две коробки на ±x, y ∈ [0, finger_depth]
    finger_half = np.array([hp.finger_thickness * 0.5, hp.finger_depth * 0.5, hp.closing_region_height * 0.5], dtype=np.float64)
    x_off = 0.5 * width + 0.5 * hp.finger_thickness  # width = зазор между внутренними гранями
    left_center_local  = np.array([-x_off, hp.finger_depth * 0.5, 0.0], dtype=np.float64)
    right_center_local = np.array([+x_off, hp.finger_depth * 0.5, 0.0], dtype=np.float64)
    left_center_world  = center + (R_hand @ left_center_local)
    right_center_world = center + (R_hand @ right_center_local)

    if np.any(points_in_obb(points_roi, left_center_world, R_hand, finger_half)):
        return False
    if np.any(points_in_obb(points_roi, right_center_world, R_hand, finger_half)):
        return False

    return True

def closing_region_has_points(points_roi, center, R_hand, width, hp) -> bool:
    half = np.array([width * 0.5, hp.closing_region_depth * 0.5, hp.closing_region_height * 0.5], dtype=np.float64)
    center_local = np.array([0.0, hp.closing_region_depth * 0.5, 0.0], dtype=np.float64)
    center_world = center + (R_hand @ center_local)

    mask = points_in_obb(points_roi, center_world, R_hand, half)
    pts = points_roi[mask]
    if pts.shape[0] < 12:
        return False

    local = (pts - center[None, :]) @ R_hand
    y = local[:, 1]  # 0..closing_region_depth
    # 4 бина по глубине
    bins = np.floor(np.clip(y / hp.closing_region_depth, 0.0, 0.9999) * 4).astype(int)
    occ = np.unique(bins).size
    return occ >= 2



# =========================
# Антиподальная проверка (глобальная)
# =========================
def estimate_normal_knn(tree: KDTree, all_points: np.ndarray, q: np.ndarray, k: int) -> Optional[np.ndarray]:
    k_eff = int(min(max(k, 8), all_points.shape[0]))
    if k_eff < 8:
        return None
    d, ind = tree.query(q.reshape(1, 3), k=k_eff)
    neigh = all_points[ind[0]]
    if neigh.shape[0] < 8:
        return None
    return pca_normal(neigh)

def orient_normal_outward(n: np.ndarray, q: np.ndarray, center: np.ndarray) -> np.ndarray:
    # ориентируем нормаль так, чтобы она "смотрела наружу" от центра захвата
    if float(np.dot(n, q - center)) < 0.0:
        return -n
    return n

def closing_region_points(points_roi: np.ndarray, center: np.ndarray, R_hand: np.ndarray, width: float, hp: HandParams) -> Tuple[np.ndarray, np.ndarray]:
    half = np.array([width * 0.5, hp.closing_region_depth * 0.5, hp.closing_region_height * 0.5], dtype=np.float64)
    center_local = np.array([0.0, hp.closing_region_depth * 0.5, 0.0], dtype=np.float64)
    rect_center_world = center + (R_hand @ center_local)

    mask = points_in_obb(points_roi, rect_center_world, R_hand, half)
    pts_cr = points_roi[mask]
    local_c = (pts_cr - center[None, :]) @ R_hand  # координаты в СК захвата
    return pts_cr, local_c

def antipodal_check(tree_all: KDTree, all_points: np.ndarray, points_roi: np.ndarray,
                    center: np.ndarray, R_hand: np.ndarray, width: float, hp: HandParams) -> bool:

    pts_cr, local_c = closing_region_points(points_roi, center, R_hand, width, hp)
    if pts_cr.shape[0] < 12:
        return False

    band = float(hp.contact_band)
    left_mask  = np.abs(local_c[:, 0] + 0.5 * width) <= band
    right_mask = np.abs(local_c[:, 0] - 0.5 * width) <= band

    C_L = pts_cr[left_mask]
    C_R = pts_cr[right_mask]
    if C_L.shape[0] < 4 or C_R.shape[0] < 4:
        return False

    x_axis = R_hand[:, 0]
    ca = math.cos(math.radians(hp.antipodal_alpha_deg))
    cb = math.cos(math.radians(hp.antipodal_beta_deg))

    # ограничим число кандидатов
    rng = np.random.default_rng(0)
    if C_L.shape[0] > 60:
        C_L = C_L[rng.choice(C_L.shape[0], 60, replace=False)]
    if C_R.shape[0] > 60:
        C_R = C_R[rng.choice(C_R.shape[0], 60, replace=False)]

    nL_list = []
    for qL in C_L:
        nL = estimate_normal_knn(tree_all, all_points, qL, hp.normal_k)
        if nL is None:
            continue
        nL = orient_normal_outward(nL, qL, center)
        # левая грань: нормаль направлена примерно в -x
        if float(np.dot(nL, x_axis)) <= -ca:
            nL_list.append(nL)
    if len(nL_list) == 0:
        return False

    nR_list = []
    for qR in C_R:
        nR = estimate_normal_knn(tree_all, all_points, qR, hp.normal_k)
        if nR is None:
            continue
        nR = orient_normal_outward(nR, qR, center)
        # правая грань: нормаль направлена примерно в +x
        if float(np.dot(nR, x_axis)) >= ca:
            nR_list.append(nR)
    if len(nR_list) == 0:
        return False

    # проверка "почти противоположности" нормалей
    for nL in nL_list:
        for nR in nR_list:
            if float(np.dot(nL, nR)) <= -cb:
                return True

    return False


# =========================
# Гипотезы
# =========================
@dataclass
class GraspHyp:
    center: np.ndarray
    R: np.ndarray
    width: float
    push: float
    score: float


def _approach_ok(y_axis: np.ndarray, hp: HandParams) -> bool:
    if not hp.approach_from_above:
        return True
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    c = math.cos(math.radians(hp.approach_max_tilt_deg))
    # y_axis должен смотреть вниз (dot с up должен быть <= -cos(tilt))
    return float(np.dot(y_axis, world_up)) <= -c


def _above_object_min_z(center: np.ndarray, hp: HandParams, obj_min_z: float) -> bool:
    # Запрещаем центры ниже "дна" объекта + clearance
    return float(center[2]) >= float(obj_min_z + hp.min_z_clearance)


def grid_search_at_point(p0: np.ndarray,
                         neigh_local: np.ndarray,
                         R_darboux: np.ndarray,
                         hp: HandParams,
                         widths: List[float],
                         phis: List[float],
                         y_push_max: float,
                         y_steps: int,
                         # глобальное облако для проверок
                         tree_all: KDTree,
                         points_all: np.ndarray,
                         obj_min_z: float) -> List[GraspHyp]:

    hyps: List[GraspHyp] = []

    # кадр Дарбу
    t0 = normalize(R_darboux[:, 0])
    b0 = normalize(R_darboux[:, 1])
    n0 = normalize(R_darboux[:, 2])

    # приводим знак нормали так, чтобы approach y=-n шёл "сверху"
    y_axis0 = -n0
    if hp.approach_from_above and not _approach_ok(y_axis0, hp):
        n0 = -n0
        t0 = -t0
        y_axis0 = -n0
        if not _approach_ok(y_axis0, hp):
            return hyps

    R_base = np.column_stack([t0, b0, n0])
    n = n0
    y_axis = -n  # approach

    for phi in phis:
        K = rot_about_axis(n, phi)
        Rphi = K @ R_base
        t = normalize(Rphi[:, 0])
        b = normalize(Rphi[:, 1])

        # В этой реализации:
        # x — раскрытие => вдоль t
        # y — подход     => -n
        # z — вдоль пальцев => b
        x_axis = t
        z_axis = b

        # ортонормализация относительно y_axis
        x_axis = normalize(x_axis - np.dot(x_axis, y_axis) * y_axis)
        z_axis = normalize(np.cross(x_axis, y_axis))
        x_axis = normalize(np.cross(y_axis, z_axis))
        R_hand = np.column_stack([x_axis, y_axis, z_axis])

        for w in widths:
            best_center = None
            best_push = None

            roi_r = roi_radius_for_hand(float(w), hp)

            for si in range(y_steps):
                alpha = si / max(y_steps - 1, 1)
                push = alpha * y_push_max

                center = p0 - y_axis * (push + hp.closing_region_depth * 0.5)

                # быстрый анти-"снизу" фильтр
                if not _above_object_min_z(center, hp, obj_min_z):
                    continue

                # ROI по глобальному облаку
                pts_roi = get_roi_points(tree_all, points_all, center, roi_r)
                if pts_roi.shape[0] < 20:
                    continue

                # 1) коллизии пальцев/ладони по глобальным точкам
                if not hand_collision_free(pts_roi, center, R_hand, float(w), hp):
                    continue

                # 2) материал между губками (closing region)
                if not closing_region_has_points(pts_roi, center, R_hand, float(w), hp):
                    continue

                # 3) антиподальность (нормали оцениваем kNN по глобальному дереву)
                if not antipodal_check(tree_all, points_all, pts_roi, center, R_hand, float(w), hp):
                    continue

                best_center = center
                best_push = push
                break

            if best_center is not None:
                score = -(best_push) - 1.5 * float(w)
                hyps.append(GraspHyp(center=best_center, R=R_hand, width=float(w), push=float(best_push), score=float(score)))

    return hyps


def build_hypotheses(points: np.ndarray,
                     num_seeds: int = 300,
                     neigh_radius: float = 0.05,
                     normal_k_local: int = 30,
                     hp: HandParams = HandParams()) -> List[GraspHyp]:

    points = points.astype(np.float64)
    tree_all = KDTree(points)

    obj_min_z = float(points[:, 2].min()) if points.shape[0] > 0 else 0.0

    seeds = np.random.choice(points.shape[0], min(num_seeds, points.shape[0]), replace=False)

    widths = np.linspace(hp.min_width, hp.max_width, 8).tolist()
    phis = np.linspace(-math.pi/2, math.pi/2, 10).tolist()

    hyps_all: List[GraspHyp] = []

    for idx in seeds:
        p0 = points[idx]

        # локальная окрестность для кадра Дарбу (как в статье)
        ind = tree_all.query_radius(p0.reshape(1, 3), r=neigh_radius)[0]
        if ind.shape[0] < normal_k_local:
            continue
        neigh_local = points[ind]

        n = pca_normal(neigh_local)
        R = make_darboux_frame(n, neigh_local)

        hyps_all.extend(grid_search_at_point(
            p0=p0,
            neigh_local=neigh_local,
            R_darboux=R,
            hp=hp,
            widths=widths,
            phis=phis,
            y_push_max=0.06,
            y_steps=10,
            tree_all=tree_all,
            points_all=points,
            obj_min_z=obj_min_z,
        ))

    # сортировка + NMS
    hyps_all.sort(key=lambda h: h.score, reverse=True)

    filtered: List[GraspHyp] = []
    pos_eps = 0.015
    width_eps = 0.01
    ang_eps = math.radians(20)

    for h in hyps_all:
        ok = True
        for g in filtered[:400]:
            if np.linalg.norm(h.center - g.center) < pos_eps and abs(h.width - g.width) < width_eps:
                if rot_angle(h.R, g.R) < ang_eps:
                    ok = False
                    break
        if ok:
            filtered.append(h)
        if len(filtered) >= 80:
            break

    return filtered


# =========================
# Визуализация
# =========================
def draw_center_point(c, color=(1, 0, 1), size=0.01, life=0):
    c = np.asarray(c, dtype=np.float64)
    s = float(size)
    p.addUserDebugLine(_v3(c - [s,0,0]), _v3(c + [s,0,0]), list(color), 2, lifeTime=life)
    p.addUserDebugLine(_v3(c - [0,s,0]), _v3(c + [0,s,0]), list(color), 2, lifeTime=life)
    p.addUserDebugLine(_v3(c - [0,0,s]), _v3(c + [0,0,s]), list(color), 2, lifeTime=life)
    if hasattr(p, "addUserDebugPoints"):
        p.addUserDebugPoints([_v3(c)], [list(color)], pointSize=10, lifeTime=life)

def draw_cloud_points(points: np.ndarray, color=(0,1,0), size=2, max_points=2000, life=0):
    pts = points
    if pts.shape[0] > max_points:
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
    if hasattr(p, "addUserDebugPoints"):
        p.addUserDebugPoints([_v3(pt) for pt in pts], [list(color)] * len(pts), pointSize=size, lifeTime=life)

def draw_gripper(h: GraspHyp, hp: HandParams, life=0):
    c = np.asarray(h.center, dtype=np.float64)
    R = np.asarray(h.R, dtype=np.float64)
    w = float(h.width)

    x = R[:, 0]
    y = R[:, 1]
    z = R[:, 2]

    draw_center_point(c, color=(1, 0, 1), size=0.01, life=life)

    z_len = hp.finger_height * 0.5
    y_f = hp.finger_depth * 0.5

    x_off = 0.5 * w + 0.5 * hp.finger_thickness
    left_center  = c + x * (-x_off) + y * y_f
    right_center = c + x * (+x_off) + y * y_f

    p.addUserDebugLine(_v3(left_center - z * z_len),  _v3(left_center + z * z_len),  [1, 0.6, 0], 3, lifeTime=life)
    p.addUserDebugLine(_v3(right_center - z * z_len), _v3(right_center + z * z_len), [1, 0.6, 0], 3, lifeTime=life)

    palm_center = c + y * (-hp.palm_depth * 0.5)
    p.addUserDebugLine(_v3(palm_center - x * (w * 0.55)), _v3(palm_center + x * (w * 0.55)),
                       [1, 0.25, 0.25], 4, lifeTime=life)

    p.addUserDebugLine(_v3(c), _v3(c + y * 0.10), [0, 1, 1], 2, lifeTime=life)

    rect_center = c + y * (hp.closing_region_depth * 0.5)
    hx = w * 0.5
    hz = hp.closing_region_height * 0.5

    a = rect_center + x * hx + z * hz
    b2 = rect_center - x * hx + z * hz
    d2 = rect_center + x * hx - z * hz
    e2 = rect_center - x * hx - z * hz

    col = [0.2, 0.8, 1.0]
    p.addUserDebugLine(_v3(a), _v3(b2), col, 1, lifeTime=life)
    p.addUserDebugLine(_v3(b2), _v3(e2), col, 1, lifeTime=life)
    p.addUserDebugLine(_v3(e2), _v3(d2), col, 1, lifeTime=life)
    p.addUserDebugLine(_v3(d2), _v3(a), col, 1, lifeTime=life)

def _set_camera_to_cloud(points: np.ndarray):
    mn = points.min(axis=0)
    mx = points.max(axis=0)
    center = 0.5 * (mn + mx)
    extent = float(np.linalg.norm(mx - mn))
    p.resetDebugVisualizerCamera(
        cameraDistance=float(max(0.2, extent * 2.0)),
        cameraYaw=35,
        cameraPitch=-35,
        cameraTargetPosition=_v3(center)
    )

def visualize(points: np.ndarray, hyps: List[GraspHyp]):
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetSimulation()
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    p.loadURDF("table/table.urdf", [0.5, 0.0, 0.0], p.getQuaternionFromEuler([0, 0, 0]))

    _set_camera_to_cloud(points)

    if len(hyps) == 0:
        p.removeAllUserDebugItems()
        draw_cloud_points(points, color=(0, 1, 0), size=2, max_points=2000, life=0)
        print("No hypotheses to show.")
        while True:
            p.stepSimulation()
            time.sleep(0.01)

    hp = HandParams()
    idx = 0

    def redraw():
        p.removeAllUserDebugItems()
        draw_cloud_points(points, color=(0, 1, 0), size=2, max_points=2000, life=0)
        draw_gripper(hyps[idx], hp, life=0)
        h = hyps[idx]
        print(f"[{idx+1}/{len(hyps)}] score={h.score:.6f} width={h.width:.4f} push={h.push:.4f} center={h.center}")

    redraw()

    KEY_LEFT = p.B3G_LEFT_ARROW
    KEY_RIGHT = p.B3G_RIGHT_ARROW
    KEY_SPACE = ord(' ')
    KEY_ESC = 27
    KEY_N = ord('n')
    KEY_P = ord('p')

    last = 0.0
    while True:
        p.stepSimulation()
        time.sleep(0.01)

        now = time.time()
        if now - last < 0.08:
            continue

        keys = p.getKeyboardEvents()
        if KEY_ESC in keys and (keys[KEY_ESC] & p.KEY_WAS_TRIGGERED):
            break

        changed = False
        if KEY_RIGHT in keys and (keys[KEY_RIGHT] & p.KEY_WAS_TRIGGERED):
            idx = (idx + 1) % len(hyps)
            changed = True
        if KEY_LEFT in keys and (keys[KEY_LEFT] & p.KEY_WAS_TRIGGERED):
            idx = (idx - 1 + len(hyps)) % len(hyps)
            changed = True
        if KEY_N in keys and (keys[KEY_N] & p.KEY_WAS_TRIGGERED):
            idx = (idx + 1) % len(hyps)
            changed = True
        if KEY_P in keys and (keys[KEY_P] & p.KEY_WAS_TRIGGERED):
            idx = (idx - 1 + len(hyps)) % len(hyps)
            changed = True
        if KEY_SPACE in keys and (keys[KEY_SPACE] & p.KEY_WAS_TRIGGERED):
            idx = (idx + 10) % len(hyps)
            changed = True

        if changed:
            last = now
            redraw()

    p.disconnect()


# =========================
# CLI
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="Path to scene_XXXXXX.npz (must contain points_world)")
    ap.add_argument("--seeds", type=int, default=300)
    ap.add_argument("--radius", type=float, default=0.05)

    ap.add_argument("--no_above", action="store_true", help="Disable approach-from-above constraint")
    ap.add_argument("--tilt", type=float, default=75.0, help="Max tilt deg for approach-from-above")

    ap.add_argument("--alpha", type=float, default=40.0, help="Antipodal alpha (deg)")
    ap.add_argument("--beta", type=float, default=40.0, help="Antipodal beta (deg)")
    ap.add_argument("--band", type=float, default=0.010, help="Contact band near inner faces (m)")

    ap.add_argument("--zclr", type=float, default=0.008, help="Min Z clearance over object min_z (m)")
    ap.add_argument("--roi_margin", type=float, default=0.01, help="ROI margin (m) for global checks")

    args = ap.parse_args()

    data = np.load(args.npz)
    if "points_world" not in data:
        raise KeyError("NPZ must contain array 'points_world'")

    pts = data["points_world"].astype(np.float64)
    print("Loaded points:", pts.shape)

    hp = HandParams()
    hp.approach_from_above = (not args.no_above)
    hp.approach_max_tilt_deg = float(args.tilt)
    hp.antipodal_alpha_deg = float(args.alpha)
    hp.antipodal_beta_deg = float(args.beta)
    hp.contact_band = float(args.band)
    hp.min_z_clearance = float(args.zclr)
    hp.roi_margin = float(args.roi_margin)

    hyps = build_hypotheses(pts, num_seeds=args.seeds, neigh_radius=args.radius, hp=hp)

    print("Hypotheses:", len(hyps))
    if len(hyps) > 0:
        print("Top score:", hyps[0].score, "width:", hyps[0].width, "push:", hyps[0].push, "center:", hyps[0].center)

    visualize(pts, hyps)

if __name__ == "__main__":
    main()
