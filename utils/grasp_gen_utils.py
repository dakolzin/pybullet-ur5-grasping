# -*- coding: utf-8 -*-
"""
utils/grasp_gen_utils.py — генерация grasp-гипотез из point cloud.

Режимы:
- "nofc"   : быстрый антиподальный генератор (close_nofc.py)
- "hybrid" : nofc -> FC ε для top-K -> пересортировка
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.neighbors import KDTree

import config as cfg

from close_nofc import HandParams, build_hypotheses
from close_nofc import roi_radius_for_hand, get_roi_points

from utils.cloud_utils import filter_cloud_basic


@dataclass
class GraspCandidate:
    center: np.ndarray
    R: np.ndarray
    width: float
    push: float
    score: float

    # FC (опционально)
    fc_ok: Optional[bool] = None
    fc_eps: Optional[float] = None
    fc_pair: Optional[object] = None

    # служебное (для анализа)
    rank_nofc: Optional[int] = None     # ранг в nofc-списке до гибрида
    selected_source: Optional[str] = None  # "fc" или "nofc"


def _default_hand_params() -> HandParams:
    hp = HandParams()

    # Robotiq 2F-85
    hp.min_width = 0.025
    hp.max_width = 0.075

    hp.finger_thickness = 0.014
    hp.finger_depth = 0.030
    hp.finger_height = 0.060
    hp.palm_depth = 0.040

    hp.closing_region_depth = 0.015
    hp.closing_region_height = 0.055

    hp.contact_band = 0.010
    hp.roi_margin = 0.015
    hp.y_push_max = 0.03

    # генератор
    hp.approach_from_above = False
    hp.min_z_clearance = 0.0
    hp.antipodal_alpha_deg = 179.0
    hp.antipodal_beta_deg = 179.0
    hp.normal_k = 15

    return hp


def _to_candidate_list(hyps: Sequence[object]) -> List[GraspCandidate]:
    out: List[GraspCandidate] = []
    for h in hyps:
        out.append(
            GraspCandidate(
                center=np.asarray(h.center, dtype=np.float64),
                R=np.asarray(h.R, dtype=np.float64),
                width=float(h.width),
                push=float(h.push),
                score=float(h.score),
            )
        )
    return out

def _normalize(vals: np.ndarray, eps: float = 1e-9):
    v = np.asarray(vals, dtype=np.float64)
    if v.size == 0:
        return v
    mn, mx = float(np.min(v)), float(np.max(v))
    if mx - mn < eps:
        return np.zeros_like(v)
    return (v - mn) / (mx - mn)

def _hybrid_scores(pre: List[GraspCandidate]) -> np.ndarray:
    # базовый nofc score (чем больше, тем лучше) -> нормируем 0..1
    s0 = _normalize(np.array([h.score for h in pre], dtype=np.float64))

    # eps -> нормируем, но сначала обрезаем “шум”
    eps_raw = np.array([(h.fc_eps or 0.0) for h in pre], dtype=np.float64)
    eps_raw = np.clip(eps_raw, 0.0, 0.01)  # защитный потолок
    epsn = _normalize(eps_raw)

    # push/width -> штрафы (нормируем)
    pushn = _normalize(np.array([h.push for h in pre], dtype=np.float64))
    widthn = _normalize(np.array([h.width for h in pre], dtype=np.float64))

    # веса: FC даёт небольшой вклад, не “переворачивает” ранжирование
    w_s0   = 1.00
    w_eps  = 0.25   # <<< ключ: небольшой бонус
    w_push = 0.15
    w_w    = 0.15

    # если fc_ok False -> epsn=0 (так проще)
    fc_ok = np.array([1.0 if h.fc_ok else 0.0 for h in pre], dtype=np.float64)
    return (w_s0*s0) + (w_eps*epsn*fc_ok) - (w_push*pushn) - (w_w*widthn)

def _compute_fc_for_candidates(
    points_all: np.ndarray,
    hp: HandParams,
    candidates: List[GraspCandidate],
) -> List[GraspCandidate]:
    """
    Считает FC ε для заданных кандидатов.
    Возвращает список тех же объектов candidates (in-place заполнены fc_ok/fc_eps/fc_pair).
    """
    try:
        from force_closure_module import FCParams, force_closure_for_hypothesis
    except Exception as e:
        raise ImportError(
            "HYBRID режим включён, но не найден/не импортируется force_closure_module.py.\n"
            f"Оригинальная ошибка: {type(e).__name__}: {e}"
        )

    pts = np.asarray(points_all, dtype=np.float64)
    if pts.shape[0] == 0:
        return candidates

    tree_all = KDTree(pts)

    fc = FCParams(
        contact_band=float(hp.contact_band),
        k_normal=int(hp.normal_k),
    )

    rng_seed = int(getattr(cfg, "RANDOM_SEED", 0) or 0)

    for h in candidates:
        roi_r = roi_radius_for_hand(float(h.width), hp)
        pts_roi = get_roi_points(tree_all, pts, np.asarray(h.center, dtype=np.float64), float(roi_r))

        if pts_roi.shape[0] < 20:
            h.fc_ok = False
            h.fc_eps = 0.0
            h.fc_pair = None
            continue

        is_fc, eps_fc, best_pair = force_closure_for_hypothesis(
            tree_all=tree_all,
            points_all=pts,
            points_roi=pts_roi,
            center=np.asarray(h.center, dtype=np.float64),
            R_hand=np.asarray(h.R, dtype=np.float64),
            width=float(h.width),
            closing_depth=float(hp.closing_region_depth),
            closing_height=float(hp.closing_region_height),
            fc=fc,
            rng_seed=rng_seed,
        )

        h.fc_ok = bool(is_fc)
        h.fc_eps = float(eps_fc) if eps_fc is not None else 0.0
        h.fc_pair = best_pair

    return candidates

def _eps_only_selection(
    cloud: np.ndarray,
    hp: HandParams,
    cand: List[GraspCandidate],
    top_k: int,
) -> List[GraspCandidate]:
    """
    EPS-only режим:
    - считаем FC + ε для ВСЕХ кандидатов
    - сортируем по ε
    - берём лучшие
    """

    from force_closure_module import FCParams, force_closure_for_hypothesis

    pts = np.asarray(cloud, dtype=np.float64)
    tree_all = KDTree(pts)

    fc = FCParams(
        contact_band=float(hp.contact_band),
        k_normal=int(hp.normal_k),
    )

    rng_seed = int(getattr(cfg, "RANDOM_SEED", 0) or 0)

    valid = []

    for h in cand:
        roi_r = roi_radius_for_hand(float(h.width), hp)
        pts_roi = get_roi_points(tree_all, pts, h.center, float(roi_r))

        if pts_roi.shape[0] < 20:
            continue

        is_fc, eps_fc, best_pair = force_closure_for_hypothesis(
            tree_all=tree_all,
            points_all=pts,
            points_roi=pts_roi,
            center=h.center,
            R_hand=h.R,
            width=h.width,
            closing_depth=float(hp.closing_region_depth),
            closing_height=float(hp.closing_region_height),
            fc=fc,
            rng_seed=rng_seed,
        )

        h.fc_ok = bool(is_fc)
        h.fc_eps = float(eps_fc)
        h.fc_pair = best_pair
        h.selected_source = "eps"

        # фильтр минимального eps
        if h.fc_eps >= float(getattr(cfg, "EPS_MIN_VALID", 0.0)):
            valid.append(h)

    # если ничего не прошло — fallback
    if not valid:
        cand.sort(key=lambda h: h.score, reverse=True)
        return cand[: int(top_k)]

    valid.sort(key=lambda h: h.fc_eps, reverse=True)
    return valid[: int(top_k)]


def _hybrid_rescore(h: GraspCandidate) -> float:
    eps = float(h.fc_eps) if h.fc_eps is not None else 0.0
    return (2.0 * eps) - float(h.push) - 1.5 * float(h.width)


def find_best_grasps_from_cloud(
    cloud: np.ndarray,
    top_k: int = 10,
    mode: Optional[str] = None,
    hybrid_topk_fc: Optional[int] = None,
    hybrid_eps_min: Optional[float] = None,
) -> List[GraspCandidate]:
    """
    Возвращает список гипотез, уже отсортированный "лучшие->хуже".
    В pick_pipeline фактически будет взят первый элемент (после фильтров).
    """
    mode = (mode or getattr(cfg, "GRASP_MODE", "nofc")).lower().strip()
    hybrid_topk_fc = int(hybrid_topk_fc or getattr(cfg, "HYBRID_TOPK_FC", 15))
    hybrid_eps_min = float(hybrid_eps_min if hybrid_eps_min is not None else getattr(cfg, "HYBRID_EPS_MIN", 1e-3))

    cloud = filter_cloud_basic(np.asarray(cloud, dtype=np.float64))
    if cloud.shape[0] < 200:
        return []

    hp = _default_hand_params()

    # --- NOFC генерация ---
    hyps = build_hypotheses(
        cloud,
        num_seeds=int(getattr(cfg, "NOFC_NUM_SEEDS", 900)),
        neigh_radius=float(getattr(cfg, "NOFC_NEIGH_RADIUS", 0.07)),
        normal_k_local=int(getattr(cfg, "NOFC_K_LOCAL", 20)),
        hp=hp,
        frame_method=str(getattr(cfg, "NOFC_FRAME", "taubin")),
    )

    cand = _to_candidate_list(hyps)
    cand.sort(key=lambda h: h.score, reverse=True)

    # проставим rank_nofc сразу
    for i, h in enumerate(cand):
        h.rank_nofc = int(i)
        h.selected_source = "nofc"

    if mode == "nofc":
        return cand[: int(top_k)]

    if mode == "eps":
        return _eps_only_selection(
            cloud=cloud,
            hp=hp,
            cand=cand,
            top_k=top_k,
        )

    if mode != "hybrid":
        raise ValueError(f"Unknown grasp mode: {mode!r}. Expected 'nofc' or 'hybrid'.")

    # --- HYBRID: FC только для top-K NOFC ---

    pre = cand[: max(int(hybrid_topk_fc), 1)]
    _compute_fc_for_candidates(points_all=cloud, hp=hp, candidates=pre)

    scores = _hybrid_scores(pre)
    for h, sc in zip(pre, scores):
        h.score = float(sc)
        h.selected_source = ("fc" if h.fc_ok else "nofc")

    pre.sort(key=lambda h: h.score, reverse=True)
    out = list(pre[: int(top_k)])

    # добивка (если вдруг надо)
    if len(out) < int(top_k):
        used = set((tuple(np.round(h.center, 4)), round(h.width, 4)) for h in out)
        for h in cand:
            key = (tuple(np.round(h.center, 4)), round(h.width, 4))
            if key in used:
                continue
            out.append(h)
            if len(out) >= int(top_k):
                break
    return out

