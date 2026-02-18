#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from collections import Counter
from statistics import mean

import numpy as np


def _load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _stats(vals):
    vals = [float(v) for v in vals if v is not None and np.isfinite(float(v))]
    if not vals:
        return None
    vals_sorted = sorted(vals)

    def pct(p):
        idx = int(round((p / 100.0) * (len(vals_sorted) - 1)))
        return vals_sorted[max(0, min(len(vals_sorted) - 1, idx))]

    return {
        "mean": mean(vals),
        "p50": pct(50),
        "p90": pct(90),
        "max": max(vals_sorted),
    }


def _fmt(x):
    if x is None:
        return "None"
    try:
        return f"{float(x):.3f}"
    except Exception:
        return str(x)


def _extra(row: dict) -> dict:
    ex = row.get("extra")
    if isinstance(ex, dict):
        return ex
    return {}


def _stage_flags(row: dict):
    pipe = bool(row.get("pipeline_ok", False))
    fr = row.get("fail_reason")

    if pipe:
        return True, True, True

    if not fr:
        return False, False, False

    grasp_ok = (fr != "weak_grasp_after_close")
    lift_ok = (fr == "place_plan_failed")
    transfer_ok = False

    if not grasp_ok:
        lift_ok = False
        transfer_ok = False

    return grasp_ok, lift_ok, transfer_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--last", type=int, default=50)
    args = ap.parse_args()

    rows = _load_jsonl(args.file)
    if not rows:
        print("No rows.")
        return

    total = len(rows)
    succ = sum(1 for r in rows if bool(r.get("task_success", False)))
    pipe_ok = sum(1 for r in rows if bool(r.get("pipeline_ok", False)))

    print(
        f"Attempts: {total} | "
        f"Success(task): {succ} ({100.0*succ/total:.1f}%) | "
        f"PipelineOK: {pipe_ok} ({100.0*pipe_ok/total:.1f}%)"
    )

    # --- стадийные проценты
    grasp_yes = 0
    lift_yes = 0
    transfer_yes = 0
    for r in rows:
        g, l, t = _stage_flags(r)
        grasp_yes += int(g)
        lift_yes += int(l)
        transfer_yes += int(t)

    print("\nStage success (derived from fail_reason):")
    print(f"  grasp_ok:    {grasp_yes}/{total} ({100.0*grasp_yes/total:.1f}%)")
    print(f"  lift_ok:     {lift_yes}/{total} ({100.0*lift_yes/total:.1f}%)")
    print(f"  transfer_ok: {transfer_yes}/{total} ({100.0*transfer_yes/total:.1f}%)")

    # причины фейлов пайплайна
    fail_reasons = Counter()
    for r in rows:
        if not bool(r.get("pipeline_ok", False)):
            fr = r.get("fail_reason") or "unknown"
            fail_reasons[fr] += 1

    print("\nTop pipeline fail reasons:")
    for k, v in fail_reasons.most_common(10):
        print(f"  {k}: {v}")

    # --- FC usage stats (по selected_source)
    src_vals = []
    eps_all = []
    eps_fc = []
    fc_ok_cnt = 0

    for r in rows:
        ex = _extra(r)
        src = ex.get("selected_source")
        if src is not None:
            src_vals.append(str(src))

        fc_ok = ex.get("fc_ok")
        if fc_ok is True:
            fc_ok_cnt += 1

        eps = ex.get("fc_eps")
        if eps is not None:
            try:
                epsf = float(eps)
                if np.isfinite(epsf):
                    eps_all.append(epsf)
                    if src == "fc":
                        eps_fc.append(epsf)
            except Exception:
                pass

    if src_vals:
        c = Counter(src_vals)
        fc_chosen = c.get("fc", 0)
        print("\nHybrid selection (from logs):")
        print(f"  selected_source=fc:   {fc_chosen}/{total} ({100.0*fc_chosen/total:.1f}%)")
        print(f"  selected_source=nofc: {total - fc_chosen}/{total} ({100.0*(total-fc_chosen)/total:.1f}%)")
        print(f"  fc_ok True (any):     {fc_ok_cnt}/{total} ({100.0*fc_ok_cnt/total:.1f}%)")

        st_all = _stats(eps_all)
        if st_all:
            print("\nfc_eps (all attempts where present):")
            print(f"  mean={st_all['mean']:.6f}  p50={st_all['p50']:.6f}  p90={st_all['p90']:.6f}  max={st_all['max']:.6f}")

        st_fc = _stats(eps_fc)
        if st_fc:
            print("\nfc_eps (only selected_source=fc):")
            print(f"  mean={st_fc['mean']:.6f}  p50={st_fc['p50']:.6f}  p90={st_fc['p90']:.6f}  max={st_fc['max']:.6f}")

    # метрики (если они появятся позже)
    for key in ["sumF_after_close", "sumF_after_lift", "obj_dz_m", "slip_pos_max_m", "slip_ang_max_rad"]:
        st = _stats([r.get(key) for r in rows])
        if st is None:
            continue
        print(f"\n{key}:")
        print(f"  mean={st['mean']:.4f}  p50={st['p50']:.4f}  p90={st['p90']:.4f}  max={st['max']:.4f}")

    print("\nLast entries:")
    for r in rows[-args.last:]:
        ex = _extra(r)
        src = ex.get("selected_source")
        eps = ex.get("fc_eps")
        print(
            f"  id={r.get('id')} "
            f"task_ok={bool(r.get('task_success', False))} "
            f"pipe_ok={bool(r.get('pipeline_ok', False))} "
            f"src={src} eps={_fmt(eps)} "
            f"fail={r.get('fail_reason')} "
            f"sumF_close={_fmt(r.get('sumF_after_close'))} "
            f"nC_close={r.get('nC_after_close')} "
            f"dz={_fmt(r.get('obj_dz_m'))} "
            f"slip={_fmt(r.get('slip_pos_max_m'))}"
        )


if __name__ == "__main__":
    main()
