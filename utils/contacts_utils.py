# -*- coding: utf-8 -*-
"""
utils/contacts_utils.py — утилиты контактов (пальцы ↔ объект).
"""

from __future__ import annotations

from typing import Iterable, Tuple, Set

import pybullet as p


def finger_contact_force_sum(
    robot_id: int,
    obj_id: int,
    finger_link_ids: Iterable[int],
) -> Tuple[float, int]:
    """
    Возвращает (sum_normal_force, num_contacts) только по контактам finger_link_ids с obj_id.
    """
    cps = p.getContactPoints(bodyA=robot_id, bodyB=obj_id)
    s = 0.0
    n = 0
    finger_set: Set[int] = set(finger_link_ids)

    for cp in cps:
        linkA = int(cp[3])      # link index for bodyA
        fn = float(cp[9])       # normal force
        if linkA in finger_set and fn > 0.0:
            s += fn
            n += 1
    return s, n


def should_stop_by_finger_force(
    robot_id: int,
    obj_id: int,
    finger_link_ids: Iterable[int],
    sum_force_thresh: float = 10.0,
    min_contacts: int = 2,
) -> Tuple[bool, float, int]:
    """
    Стоп, если суммарная нормальная сила по контактам пальцев >= sum_force_thresh
    и число контактов >= min_contacts.
    """
    s, n = finger_contact_force_sum(robot_id, obj_id, finger_link_ids=finger_link_ids)
    return (n >= int(min_contacts) and s >= float(sum_force_thresh)), s, n
