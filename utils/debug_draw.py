# -*- coding: utf-8 -*-
"""
utils/debug_draw.py — функции рисования отладочной графики в PyBullet.
"""

from __future__ import annotations
from typing import Optional, List, Sequence

import numpy as np
import pybullet as p


def draw_frame(pos: Sequence[float],
               quat,
               axis_len: float = 0.06,
               life: float = 0.0,
               width: float = 2.0,
               label: Optional[str] = None) -> List[int]:
    """
    Рисует локальные оси (RGB) по позе (pos, quat).
    Возвращает список debug item ids.
    """
    Rm = np.array(p.getMatrixFromQuaternion(quat), dtype=np.float64).reshape(3, 3)
    x = Rm[:, 0]
    y = Rm[:, 1]
    z = Rm[:, 2]
    pos = np.asarray(pos, dtype=np.float64).reshape(3)

    ids: List[int] = []
    ids.append(p.addUserDebugLine(pos, pos + axis_len * x, [1, 0, 0], lineWidth=width, lifeTime=life))
    ids.append(p.addUserDebugLine(pos, pos + axis_len * y, [0, 1, 0], lineWidth=width, lifeTime=life))
    ids.append(p.addUserDebugLine(pos, pos + axis_len * z, [0, 0, 1], lineWidth=width, lifeTime=life))

    if label:
        ids.append(
            p.addUserDebugText(
                label,
                pos + np.array([0.0, 0.0, axis_len * 0.2], dtype=np.float64),
                [1, 1, 0],
                textSize=1.6,
                lifeTime=life,
            )
        )
    return ids


def draw_point(pos: Sequence[float],
               life: float = 0.0,
               label: Optional[str] = None,
               size: float = 0.02,
               width: float = 2.0) -> List[int]:
    """
    Рисует "крестик" в точке.
    Возвращает список debug item ids.
    """
    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    a = float(size)

    ids: List[int] = []
    ids.append(p.addUserDebugLine(pos + [a, 0, 0], pos - [a, 0, 0], [1, 1, 0], width, life))
    ids.append(p.addUserDebugLine(pos + [0, a, 0], pos - [0, a, 0], [1, 1, 0], width, life))
    ids.append(p.addUserDebugLine(pos + [0, 0, a], pos - [0, 0, a], [1, 1, 0], width, life))

    if label:
        ids.append(p.addUserDebugText(label, pos + [0, 0, a], [1, 1, 0], 1.6, life))
    return ids


def clear_debug(ids: List[int]) -> None:
    """Удаляет список debug item ids."""
    for i in ids:
        try:
            p.removeUserDebugItem(int(i))
        except Exception:
            pass
