#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple, List

import numpy as np
from sklearn.neighbors import KDTree

try:
    from scipy.optimize import linprog
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


# =========================
# Параметры FC
# =========================
@dataclass
class FCParams:
    mu: float = 0.8                 # коэффициент трения
    cone_dirs: int = 12              # число направлений аппроксимации конуса трения
    k_normal: int = 25              # kNN для оценки нормали
    contact_band: float = 0.010     # полоса около внутренних граней (м)
    min_contact_pts: int = 4        # минимально точек на каждой стороне
    sample_contacts_per_side: int = 40  # сколько точек брать на сторону для перебора пар
    ref_center: str = "grasp"       # "grasp" или "object" (точка отсчета моментов)
    lp_tol: float = 1e-6            # допуск LP

    # если scipy отсутствует — эвристика
    random_trials: int = 3000


# =========================
# Вспомогательная геометрия
# =========================
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n

def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array([[0, -z, y],
                     [z, 0, -x],
                     [-y, x, 0]], dtype=np.float64)

def make_friction_cone_dirs(n: np.ndarray, mu: float, m: int) -> np.ndarray:
    """
    Возвращает (m,3) направлений сил внутри конуса трения вокруг нормали n.
    Используем базис в касательной плоскости и добавляем компоненты mu.
    """
    n = normalize(n)
    # любой вектор неколлинеарный n
    a = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(a, n))) > 0.9:
        a = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    t1 = normalize(a - np.dot(a, n) * n)
    t2 = normalize(np.cross(n, t1))

    dirs = []
    for i in range(max(3, m)):
        ang = 2.0 * math.pi * i / max(3, m)
        tang = math.cos(ang) * t1 + math.sin(ang) * t2
        # сила внутри конуса: f = n + mu * tang
        f = normalize(n + mu * tang)
        dirs.append(f)
    return np.stack(dirs, axis=0)

def points_in_obb(points: np.ndarray, center: np.ndarray, R: np.ndarray, half_extents: np.ndarray) -> np.ndarray:
    local = (points - center[None, :]) @ R
    return np.all(np.abs(local) <= half_extents[None, :], axis=1)


# =========================
# Оценка нормалей
# =========================
def pca_normal(points: np.ndarray) -> np.ndarray:
    c = points.mean(axis=0)
    X = points - c
    C = (X.T @ X) / max(len(points), 1)
    w, V = np.linalg.eigh(C)
    n = V[:, 0]
    return normalize(n)

def estimate_normal_knn(tree: KDTree, all_points: np.ndarray, q: np.ndarray, k: int) -> Optional[np.ndarray]:
    if all_points.shape[0] < 8:
        return None
    k_eff = int(min(max(k, 8), all_points.shape[0]))
    d, ind = tree.query(q.reshape(1, 3), k=k_eff)
    neigh = all_points[ind[0]]
    if neigh.shape[0] < 8:
        return None
    return pca_normal(neigh)

def orient_normal_outward(n: np.ndarray, q: np.ndarray, center: np.ndarray) -> np.ndarray:
    # снимаем знак-неопределенность PCA: направим "наружу" от центра захвата
    if float(np.dot(n, q - center)) < 0.0:
        return -n
    return n


# =========================
# Контакты из closing region
# =========================
def extract_closing_region_points(points_roi: np.ndarray,
                                  center: np.ndarray,
                                  R_hand: np.ndarray,
                                  width: float,
                                  closing_depth: float,
                                  closing_height: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Возвращает точки внутри closing region и их координаты в СК захвата.
    СК захвата: x=раскрытие, y=подход, z=вдоль пальцев
    """
    half = np.array([0.5 * width, 0.5 * closing_depth, 0.5 * closing_height], dtype=np.float64)
    center_local = np.array([0.0, 0.5 * closing_depth, 0.0], dtype=np.float64)
    center_world = center + (R_hand @ center_local)
    mask = points_in_obb(points_roi, center_world, R_hand, half)
    pts = points_roi[mask]
    local = (pts - center[None, :]) @ R_hand
    return pts, local

def pick_contact_candidates(local_pts: np.ndarray,
                            world_pts: np.ndarray,
                            width: float,
                            band: float,
                            min_pts: int,
                            max_sample: int,
                            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Берем точки возле внутренних граней: x ≈ ±width/2
    Возвращаем (C_L, C_R) в мировых координатах.
    """
    x = local_pts[:, 0]
    left_mask = np.abs(x + 0.5 * width) <= band
    right_mask = np.abs(x - 0.5 * width) <= band
    C_L = world_pts[left_mask]
    C_R = world_pts[right_mask]

    if C_L.shape[0] < min_pts or C_R.shape[0] < min_pts:
        return C_L[:0], C_R[:0]

    # подвыборка для эффективности
    if C_L.shape[0] > max_sample:
        C_L = C_L[rng.choice(C_L.shape[0], max_sample, replace=False)]
    if C_R.shape[0] > max_sample:
        C_R = C_R[rng.choice(C_R.shape[0], max_sample, replace=False)]

    return C_L, C_R


# =========================
# Force-Closure: LP проверка
# =========================
def build_wrenches(contact_p: np.ndarray, contact_n: np.ndarray,
                   cone_dirs: int, mu: float,
                   ref: np.ndarray) -> np.ndarray:
    """
    На каждом контакте строим m сил направлений в конусе трения.
    Для каждого f добавляем wrench: [f; (p-ref) x f]
    Возвращаем W (6, M)
    """
    dirs = make_friction_cone_dirs(contact_n, mu=mu, m=cone_dirs)  # (m,3)
    r = (contact_p - ref).reshape(3)
    W = []
    for f in dirs:
        tau = np.cross(r, f)
        W.append(np.hstack([f, tau]))
    return np.stack(W, axis=1)  # (6, m)

def lp_feasible_origin(W: np.ndarray, tol: float, alpha_min: float = 0.0) -> Tuple[bool, float]:
    """
    Ищем alpha: W alpha = 0, sum(alpha)=1, alpha_i >= alpha_min.
    Если alpha_min>0 — это приближённая проверка "0 во внутренности conv(W)".
    """
    M = W.shape[1]
    if M < 7:
        return False, float("inf")

    A_eq = np.vstack([W, np.ones((1, M), dtype=np.float64)])  # (7, M)
    b_eq = np.zeros(7, dtype=np.float64)
    b_eq[-1] = 1.0

    if not _HAS_SCIPY:
        # fallback: грубая оценка
        rng = np.random.default_rng(0)
        best = float("inf")
        for _ in range(4000):
            a = rng.random(M)
            a = a / max(a.sum(), 1e-12)
            if alpha_min > 0.0:
                # проектируем на alpha>=alpha_min грубо
                a = np.maximum(a, alpha_min)
                a = a / max(a.sum(), 1e-12)
            r = A_eq @ a - b_eq
            best = min(best, float(np.linalg.norm(r)))
        return (best <= 1e-3), best

    c = np.zeros(M, dtype=np.float64)  # feasibility
    bounds = [(alpha_min, None)] * M

    res = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return False, float("inf")

    a = res.x
    r = A_eq @ a - b_eq
    rn = float(np.linalg.norm(r))
    return rn <= tol, rn


def is_force_closure_strict(W: np.ndarray,
                            tol: float,
                            alpha_min_frac: float = 1e-5,
                            rank_tol: float = 1e-8) -> bool:
    """
    Строгая FC-проверка (практичная):
    - работаем с НОРМИРОВАННЫМИ колонками wrenches (масштабно-инвариантно)
    - rank(W)=6 (с адекватным rank_tol)
    - существует alpha >= delta > 0, sum(alpha)=1, W alpha = 0
    """
    M = W.shape[1]
    if M < 7:
        return False

    Wn = _normalize_wrenches(W)

    if np.linalg.matrix_rank(Wn, tol=rank_tol) < 6:
        return False

    # delta как доля от равномерного веса 1/M
    delta = float(alpha_min_frac) / float(M)

    ok, _ = lp_feasible_origin(Wn, tol=tol, alpha_min=delta)
    return ok



def _normalize_wrenches(W: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Нормируем колонки wrenches до единичной нормы, чтобы eps был масштабно-инвариантным.
    """
    norms = np.linalg.norm(W, axis=0)
    norms = np.maximum(norms, eps)
    return W / norms[None, :]


def _sample_unit_sphere(dirs: int, rng: np.random.Generator) -> np.ndarray:
    """
    Сэмплинг направлений на единичной сфере (dirs, 6).
    Важно: для eps в wrench-space работаем в R^6.
    """
    X = rng.standard_normal((dirs, 6))
    Xn = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = np.maximum(Xn, 1e-12)
    return X / Xn


def approx_epsilon_metric_lp(W: np.ndarray, dirs: int = 32, rng_seed: int = 0) -> float:
    """
    Приближение ε-metric (Ferrari-Canny):
    eps = min_u max_t s.t. t*u ∈ conv(W)
    Решаем LP по направлениям u.

    Возвращает eps >= 0. Чем больше — тем "лучше" захват.
    """
    if W.shape[1] < 7:
        return 0.0

    if not _HAS_SCIPY:
        # без scipy можно сделать грубую эвристику, но лучше ставить scipy
        return 0.0

    Wn = _normalize_wrenches(W)

    rng = np.random.default_rng(rng_seed)
    U = _sample_unit_sphere(dirs, rng)  # (dirs, 6)

    M = Wn.shape[1]
    best_min = float("inf")

    # Переменные: x = [alpha_0..alpha_{M-1}, t]
    # Ограничения:
    #   Wn*alpha - u*t = 0   (6 eq)
    #   sum(alpha) = 1       (1 eq)
    #   alpha >= 0
    #   t >= 0
    for u in U:
        c = np.zeros(M + 1, dtype=np.float64)
        c[-1] = -1.0  # maximize t  <=> minimize -t

        Aeq = np.zeros((7, M + 1), dtype=np.float64)
        Aeq[:6, :M] = Wn
        Aeq[:6, -1] = -u
        Aeq[6, :M] = 1.0

        beq = np.zeros(7, dtype=np.float64)
        beq[6] = 1.0

        bounds = [(0.0, None)] * M + [(0.0, None)]  # t >= 0

        res = linprog(c=c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        if not res.success:
            # если по направлению не нашли t — значит eps по этому направлению 0
            tmax = 0.0
        else:
            tmax = float(res.x[-1])
            if not math.isfinite(tmax) or tmax < 0.0:
                tmax = 0.0

        best_min = min(best_min, tmax)

    if best_min == float("inf"):
        return 0.0
    return float(best_min)

def _orthonormal_basis_span(W: np.ndarray, rank_tol: float = 1e-8) -> np.ndarray:
    """
    Возвращает матрицу B (6,r) — ортонормальный базис span(W).
    """
    # SVD: W = U S V^T
    U, S, _ = np.linalg.svd(W, full_matrices=False)
    r = int(np.sum(S > rank_tol))
    if r <= 0:
        return U[:, :0]
    return U[:, :r]


def _sample_unit_sphere_dim(dirs: int, dim: int, rng: np.random.Generator) -> np.ndarray:
    X = rng.standard_normal((dirs, dim))
    Xn = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = np.maximum(Xn, 1e-12)
    return X / Xn


def approx_epsilon_metric_lp_in_span(W: np.ndarray, dirs: int = 32, rng_seed: int = 0) -> float:
    """
    ε-metric, но в подпространстве span(W).
    Это убирает "eps=0" когда rank(W)<6 (что часто бывает).
    """
    if not _HAS_SCIPY:
        return 0.0

    # нормируем колонки (масштабно-инвариантно)
    Wn = _normalize_wrenches(W)

    B = _orthonormal_basis_span(Wn, rank_tol=1e-8)  # (6,r)
    r = int(B.shape[1])
    if r < 1:
        return 0.0

    Wr = B.T @ Wn  # (r, M)
    M = Wr.shape[1]
    if M < r + 1:
        return 0.0

    rng = np.random.default_rng(rng_seed)
    U = _sample_unit_sphere_dim(dirs, r, rng)  # (dirs,r)

    best_min = float("inf")

    # LP: max t s.t. Wr*alpha = u*t, sum(alpha)=1, alpha>=0, t>=0
    for u in U:
        c = np.zeros(M + 1, dtype=np.float64)
        c[-1] = -1.0  # maximize t

        Aeq = np.zeros((r + 1, M + 1), dtype=np.float64)
        Aeq[:r, :M] = Wr
        Aeq[:r, -1] = -u
        Aeq[r, :M] = 1.0

        beq = np.zeros(r + 1, dtype=np.float64)
        beq[r] = 1.0

        bounds = [(0.0, None)] * M + [(0.0, None)]

        res = linprog(c=c, A_eq=Aeq, b_eq=beq, bounds=bounds, method="highs")
        if not res.success:
            tmax = 0.0
        else:
            tmax = float(res.x[-1])
            if not math.isfinite(tmax) or tmax < 0.0:
                tmax = 0.0

        best_min = min(best_min, tmax)

    if best_min == float("inf"):
        return 0.0
    return float(best_min)


def force_closure_for_pair(tree_all: KDTree,
                           points_all: np.ndarray,
                           center: np.ndarray,
                           pL: np.ndarray,
                           pR: np.ndarray,
                           fc: FCParams) -> Tuple[bool, float]:
    nL = estimate_normal_knn(tree_all, points_all, pL, fc.k_normal)
    nR = estimate_normal_knn(tree_all, points_all, pR, fc.k_normal)
    if nL is None or nR is None:
        return False, 0.0

    nL = orient_normal_outward(nL, pL, center)
    nR = orient_normal_outward(nR, pR, center)

    ref = center.copy()
    W_L = build_wrenches(pL, nL, cone_dirs=fc.cone_dirs, mu=fc.mu, ref=ref)
    W_R = build_wrenches(pR, nR, cone_dirs=fc.cone_dirs, mu=fc.mu, ref=ref)
    W = np.concatenate([W_L, W_R], axis=1)  # (6, 2*m)

    # FC-факт: 0 в выпуклой оболочке (не interior!)
    ok, _ = lp_feasible_origin(W, tol=1e-4, alpha_min=0.0)
    if not ok:
        return False, 0.0

    # качество: ε в span(W)
    eps = approx_epsilon_metric_lp_in_span(W, dirs=32, rng_seed=0)
    return True, float(eps)

# =========================
# Основной интерфейс: FC для гипотезы захвата
# =========================
def force_closure_for_hypothesis(tree_all: KDTree,
                                points_all: np.ndarray,
                                points_roi: np.ndarray,
                                center: np.ndarray,
                                R_hand: np.ndarray,
                                width: float,
                                closing_depth: float,
                                closing_height: float,
                                fc: FCParams,
                                rng_seed: int = 0) -> Tuple[bool, float, Optional[Tuple[np.ndarray, np.ndarray]]]:
    """
    Возвращает:
    - is_fc: bool
    - best_eps: float (0..1)
    - best_pair: (pL,pR) или None
    """
    rng = np.random.default_rng(rng_seed)

    pts_cr, local_cr = extract_closing_region_points(
        points_roi=points_roi,
        center=center,
        R_hand=R_hand,
        width=width,
        closing_depth=closing_depth,
        closing_height=closing_height
    )

    if pts_cr.shape[0] < (2 * fc.min_contact_pts):
        return False, 0.0, None

    C_L, C_R = pick_contact_candidates(
        local_pts=local_cr,
        world_pts=pts_cr,
        width=width,
        band=fc.contact_band,
        min_pts=fc.min_contact_pts,
        max_sample=fc.sample_contacts_per_side,
        rng=rng
    )
    if C_L.shape[0] == 0 or C_R.shape[0] == 0:
        return False, 0.0, None

    # Перебираем пары контактов (полный перебор по подвыборке)
    best_eps = -1.0
    best_pair = None
    any_fc = False

    for pL in C_L:
        for pR in C_R:
            ok, eps = force_closure_for_pair(
                tree_all=tree_all,
                points_all=points_all,
                center=center,
                pL=pL,
                pR=pR,
                fc=fc
            )
            if eps > best_eps:
                best_eps = eps
                best_pair = (pL.copy(), pR.copy())
            if ok:
                any_fc = True
                # можно ранний выход, если хочешь: return True, best_eps, best_pair

    if best_eps < 0:
        best_eps = 0.0
    return any_fc, float(best_eps), best_pair
