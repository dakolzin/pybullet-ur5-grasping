# -*- coding: utf-8 -*-
"""
utils/cloud_utils.py — базовые фильтры и обработка point cloud.
"""

from __future__ import annotations

import numpy as np


def filter_cloud_basic(
    cloud: np.ndarray,
    z_min: float = 0.62,
    z_max: float = 1.20,
    x_min: float = 0.2,
    x_max: float = 0.9,
    y_min: float = -0.5,
    y_max: float = 0.5,
) -> np.ndarray:
    """
    Простой box-filter по координатам.
    cloud: (N,3)
    """
    if cloud.size == 0:
        return cloud

    cloud = np.asarray(cloud, dtype=np.float64)
    z = cloud[:, 2]
    cloud = cloud[(z > z_min) & (z < z_max)]
    if cloud.size == 0:
        return cloud

    x = cloud[:, 0]
    y = cloud[:, 1]
    return cloud[(x > x_min) & (x < x_max) & (y > y_min) & (y < y_max)]
