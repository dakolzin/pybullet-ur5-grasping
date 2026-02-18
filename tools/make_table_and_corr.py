#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def safe_float(x) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return float(int(x))
        return float(x)
    except Exception:
        return None


def derive_flags(r: Dict[str, Any]) -> Dict[str, bool]:
    """
    Derive stage flags from fail_reason if explicit booleans absent.
    Convention:
      - grasp_ok True if not failed at close stage (i.e., fail_reason not in close failures)
      - lift_ok True if no lift failure and grasp_ok True
      - transfer_ok True if no transfer failure and lift_ok True
      - pipe_ok True if transfer_ok True
      - task_ok might be stored; if not, assume equals pipe_ok
    """
    fr = r.get("fail_reason") or r.get("fail") or None

    # If log already has explicit flags, keep them.
    def get_bool(key: str) -> Optional[bool]:
        v = r.get(key)
        if v is None:
            return None
        return bool(v)

    grasp_ok = get_bool("grasp_ok")
    lift_ok = get_bool("lift_ok")
    transfer_ok = get_bool("transfer_ok")
    pipe_ok = get_bool("pipe_ok")
    task_ok = get_bool("task_ok")

    # derive if missing
    close_fails = {"no_contact_after_close", "weak_grasp_after_close"}
    lift_fails = {"lift_contacts_dropped", "lift_sumf_dropped"}
    transfer_fails = {"place_plan_failed", "transfer_plan_failed", "transfer_failed"}

    if grasp_ok is None:
        grasp_ok = fr not in close_fails
    if lift_ok is None:
        lift_ok = grasp_ok and (fr not in lift_fails) and (fr not in close_fails)
    if transfer_ok is None:
        transfer_ok = lift_ok and (fr not in transfer_fails) and (fr not in lift_fails) and (fr not in close_fails)
    if pipe_ok is None:
        pipe_ok = transfer_ok
    if task_ok is None:
        task_ok = pipe_ok

    return {
        "grasp_ok": bool(grasp_ok),
        "lift_ok": bool(lift_ok),
        "transfer_ok": bool(transfer_ok),
        "pipe_ok": bool(pipe_ok),
        "task_ok": bool(task_ok),
    }


def normalize_src(r: Dict[str, Any], path: str) -> str:
    src = r.get("selected_source") or r.get("src") or r.get("source")
    if src:
        return str(src)
    parent = os.path.basename(os.path.dirname(path))
    return parent if parent else os.path.basename(path)


def make_df(paths: List[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        for r in read_jsonl(p):
            rr = dict(r)
            rr["src"] = normalize_src(rr, p)
            rr["fail_reason"] = rr.get("fail_reason") or rr.get("fail") or None
            rr.update(derive_flags(rr))

            # metrics
            rr["eps"] = safe_float(rr.get("eps") or rr.get("fc_eps") or rr.get("eps_fc"))
            rr["sumF_after_close"] = safe_float(rr.get("sumF_after_close") or rr.get("sumF_close"))
            rr["sumF_after_lift"] = safe_float(rr.get("sumF_after_lift") or rr.get("sumF_lift"))
            rr["nC_close"] = safe_float(rr.get("nC_close") or rr.get("n_contacts_close") or rr.get("contacts_after_close"))

            rows.append(rr)

    df = pd.DataFrame(rows)

    # ratio
    def ratio(a, b):
        if a is None or b is None:
            return None
        if not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
            return None
        if b <= 1e-9:
            return None
        return a / b

    df["lift_close_ratio"] = [ratio(a, b) for a, b in zip(df["sumF_after_lift"], df["sumF_after_close"])]

    return df


def summary_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col, dropna=False)
    tab = pd.DataFrame({
        "attempts": g.size(),
        "grasp_ok_%": (g["grasp_ok"].mean() * 100.0).round(2),
        "lift_ok_%": (g["lift_ok"].mean() * 100.0).round(2),
        "transfer_ok_%": (g["transfer_ok"].mean() * 100.0).round(2),
        "pipe_ok_%": (g["pipe_ok"].mean() * 100.0).round(2),
        "task_ok_%": (g["task_ok"].mean() * 100.0).round(2),
    })
    return tab


def corr_report(df: pd.DataFrame, src_value: str, target: str, metric: str) -> str:
    sub = df[df["src"] == src_value][[target, metric]].dropna()
    n = len(sub)
    if n < 10:
        return f"  {target} vs {metric}: n={n} (skip)"
    y = sub[target].astype(int).to_numpy()
    x = sub[metric].astype(float).to_numpy()

    # point-biserial == Pearson between binary and metric
    if HAVE_SCIPY:
        pb_r, pb_p = stats.pointbiserialr(y, x)
        sp_r, sp_p = stats.spearmanr(y, x)
        pr_r, pr_p = stats.pearsonr(y, x)
        return (f"  {target} vs {metric}: n={n} | "
                f"pb_r={pb_r:+.3f} pb_p={pb_p:.3g} | "
                f"sp_r={sp_r:+.3f} sp_p={sp_p:.3g} | "
                f"pr_r={pr_r:+.3f} pr_p={pr_p:.3g}")
    else:
        # fallback: Pearson only
        pr_r = np.corrcoef(y, x)[0, 1]
        return f"  {target} vs {metric}: n={n} | pr_r={pr_r:+.3f} (scipy not installed)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--group", default="src")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    df = make_df(args.files)

    print("\n=== Summary table ===")
    print(summary_table(df, args.group))

    print("\n=== Top fail reasons (overall) ===")
    print(df["fail_reason"].value_counts().head(20))

    print("\n=== Top fail reasons by src ===")
    for s in sorted(df["src"].dropna().unique().tolist()):
        print(f"\n[src={s}]")
        print(df[df["src"] == s]["fail_reason"].value_counts().head(20))

    print("\n=== Correlations ===")
    metrics = ["eps", "sumF_after_close", "sumF_after_lift", "lift_close_ratio"]
    targets = ["lift_ok", "task_ok"]
    for s in sorted(df["src"].dropna().unique().tolist()):
        print(f"\n[src={s}] N={len(df[df['src']==s])}")
        for t in targets:
            for m in metrics:
                print(corr_report(df, s, t, m))

    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        df.to_csv(args.csv, index=False)
        print(f"\nSaved CSV: {args.csv}")


if __name__ == "__main__":
    main()
