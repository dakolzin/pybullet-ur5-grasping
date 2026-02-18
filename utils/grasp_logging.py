#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Any, Dict, Optional

import numpy as np


def _now() -> float:
    return time.time()


def _to_jsonable(v: Any) -> Any:
    """Аккуратно приводим к JSON-совместимому виду."""
    if v is None:
        return None
    if isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.reshape(-1).tolist()
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    # fallback
    return str(v)


@dataclass
class AttemptRecord:
    id: int
    t_start: float
    t_end: Optional[float] = None

    grasp_mode: Optional[str] = None
    obj_id: Optional[int] = None
    tray_id: Optional[int] = None
    grasp_idx: Optional[int] = None

    hyp_center: Optional[list] = None
    hyp_width: Optional[float] = None

    pipeline_ok: Optional[bool] = None
    task_success: Optional[bool] = None
    fail_reason: Optional[str] = None

    sumF_after_close: Optional[float] = None
    nC_after_close: Optional[int] = None
    sumF_after_lift: Optional[float] = None
    nC_after_lift: Optional[int] = None

    obj_dz_m: Optional[float] = None
    slip_pos_max_m: Optional[float] = None
    slip_ang_max_rad: Optional[float] = None

    extra: Dict[str, Any] = field(default_factory=dict)


class GraspLogger:
    """
    Пишет JSONL (одна попытка = одна строка).

    Совместимость:
      - GraspLogger(out_dir="./logs/nofc", jsonl_name="grasp_attempts.jsonl")
      - GraspLogger(out_file="./logs/nofc/grasp_attempts.jsonl")
    """

    def __init__(
        self,
        out_dir: Optional[str] = None,
        jsonl_name: str = "grasp_attempts.jsonl",
        out_file: Optional[str] = None,
    ):
        if out_file is not None:
            self.path = out_file
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        else:
            if out_dir is None:
                out_dir = "./logs"
            os.makedirs(out_dir, exist_ok=True)
            self.path = os.path.join(out_dir, jsonl_name)

        self._active: Optional[AttemptRecord] = None
        self._next_id: int = 1

        # если файл уже есть — продолжим id
        try:
            if os.path.exists(self.path) and os.path.getsize(self.path) > 0:
                with open(self.path, "r", encoding="utf-8") as f:
                    last = None
                    for line in f:
                        line = line.strip()
                        if line:
                            last = line
                    if last:
                        j = json.loads(last)
                        if "id" in j:
                            self._next_id = int(j["id"]) + 1
        except Exception:
            pass

    def start_attempt(
        self,
        obj_id: Optional[int] = None,
        tray_id: Optional[int] = None,
        grasp_idx: Optional[int] = None,
        hyp_center: Optional[np.ndarray] = None,
        hyp_width: Optional[float] = None,
        grasp_mode: Optional[str] = None,
        **extra,
    ):
        rec = AttemptRecord(
            id=self._next_id,
            t_start=_now(),
            grasp_mode=grasp_mode,
            obj_id=(int(obj_id) if obj_id is not None else None),
            tray_id=(int(tray_id) if tray_id is not None else None),
            grasp_idx=(int(grasp_idx) if grasp_idx is not None else None),
            hyp_center=(_to_jsonable(hyp_center) if hyp_center is not None else None),
            hyp_width=(float(hyp_width) if hyp_width is not None else None),
            extra={"events": [], "segments": []},
        )
        self._next_id += 1
        self._active = rec

        if extra:
            self.update(**extra)

    def update(self, **kwargs):
        """
        Обновить активную попытку любыми полями.
        Неизвестное уйдёт в extra.

        ВАЖНО: если передан ключ extra={...}, словарь МЕРДЖИТСЯ в текущий extra,
        а не перезаписывает его целиком.
        """
        if self._active is None:
            return

        for k, v in kwargs.items():
            if k == "extra" and isinstance(v, dict):
                # merge extra
                if self._active.extra is None:
                    self._active.extra = {"events": [], "segments": []}
                if not isinstance(self._active.extra.get("events", None), list):
                    self._active.extra["events"] = []
                if not isinstance(self._active.extra.get("segments", None), list):
                    self._active.extra["segments"] = []
                for ek, ev in v.items():
                    self._active.extra[str(ek)] = _to_jsonable(ev)
                continue

            if hasattr(self._active, k):
                setattr(self._active, k, _to_jsonable(v))
            else:
                if self._active.extra is None:
                    self._active.extra = {"events": [], "segments": []}
                self._active.extra[str(k)] = _to_jsonable(v)

    def event(self, name: str, **data):
        """Лёгкая точка-событие (для таймингов/состояний)."""
        if self._active is None:
            return
        if self._active.extra is None:
            self._active.extra = {"events": [], "segments": []}
        ev_list = self._active.extra.get("events", None)
        if not isinstance(ev_list, list):
            ev_list = []
            self._active.extra["events"] = ev_list

        ev = {"t": _now(), "name": str(name)}
        for k, v in data.items():
            ev[str(k)] = _to_jsonable(v)
        ev_list.append(ev)

    def segment(self, tag: str, **data):
        """
        Запись по этапу движения/проверке.
        motion_utils у тебя вызывает именно logger.segment(...)
        """
        if self._active is None:
            return
        if self._active.extra is None:
            self._active.extra = {"events": [], "segments": []}
        seg_list = self._active.extra.get("segments", None)
        if not isinstance(seg_list, list):
            seg_list = []
            self._active.extra["segments"] = seg_list

        seg = {"t": _now(), "tag": str(tag)}
        for k, v in data.items():
            seg[str(k)] = _to_jsonable(v)
        seg_list.append(seg)

    def finish_attempt(self, pipeline_ok: bool, task_success: bool, fail_reason: Optional[str] = None, **extra):
        if self._active is None:
            return

        self._active.t_end = _now()
        self._active.pipeline_ok = bool(pipeline_ok)
        self._active.task_success = bool(task_success)
        self._active.fail_reason = (str(fail_reason) if fail_reason is not None else None)

        if extra:
            self.update(**extra)

        rec_dict = asdict(self._active)
        # привести всё к jsonable
        rec_dict = _to_jsonable(rec_dict)

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec_dict, ensure_ascii=False) + "\n")

        self._active = None
