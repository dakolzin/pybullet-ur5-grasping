# -*- coding: utf-8 -*-
"""
utils/log_utils.py — форматирование и печать поз/векторов для отладки.
"""

from __future__ import annotations
from typing import Optional, Sequence, Tuple

import numpy as np
import pybullet as p


def quat_to_rpy(q: Tuple[float, float, float, float]) -> Tuple[float, float, float]:
    return tuple(p.getEulerFromQuaternion(q))


def fmt_vec(v: Sequence[float]) -> str:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return f"[{v[0]: .4f} {v[1]: .4f} {v[2]: .4f}]"


def fmt_rpy(rpy: Sequence[float]) -> str:
    rpy = np.asarray(rpy, dtype=np.float64).reshape(3)
    return f"(r={rpy[0]: .3f}, p={rpy[1]: .3f}, y={rpy[2]: .3f})"


def log_pose(tag: str,
             pos: Sequence[float],
             orn: Tuple[float, float, float, float],
             ref_pos: Optional[np.ndarray] = None) -> None:
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    rpy = quat_to_rpy(orn)

    if ref_pos is None:
        print(f"{tag}: pos={fmt_vec(pos)} rpy={fmt_rpy(rpy)} quat={orn}")
    else:
        ref_pos = np.asarray(ref_pos, dtype=np.float64).reshape(3)
        err = float(np.linalg.norm(pos - ref_pos))
        print(f"{tag}: pos={fmt_vec(pos)} err_to_ref={err:.4f} rpy={fmt_rpy(rpy)}")
