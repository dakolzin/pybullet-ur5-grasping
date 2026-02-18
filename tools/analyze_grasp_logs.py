#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from scipy.stats import spearmanr, pearsonr, pointbiserialr
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# ----------------------------
# Helpers
# ----------------------------

def _safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, bool):
            return float(int(x))
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except Exception:
        return None

def _safe_bool(x: Any) -> Optional[bool]:
    if x is None:
        return None
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, float)):
        return bool(int(x))
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n"):
            return False
    return None

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise RuntimeError(f"JSON decode error in {path}:{ln}: {e}")
    return rows

def derive_stage_flags(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Если в логе уже есть grasp_ok/lift_ok/transfer_ok — используем.
    Иначе пытаемся вывести из fail_reason как в твоём view_grasp_logs.
    """
    out = {}

    # Existing explicit flags
    for k in ("grasp_ok", "lift_ok", "transfer_ok", "task_ok", "pipe_ok"):
        if k in row:
            out[k] = _safe_bool(row.get(k))

    # Derive from fail_reason if missing
    fail = row.get("fail_reason") or row.get("fail") or None
    if fail is not None:
        fail = str(fail)

    # If not present, approximate:
    # - "no_contact_after_close" / "weak_grasp_after_close" => grasp stage failed
    # - "lift_contacts_dropped" / "*_dropped" => grasp ok, lift failed
    # - "place_plan_failed" => grasp ok, lift ok, transfer failed (or place stage)
    # - None or "ok" => all ok (depends on task_ok)
    if out.get("grasp_ok") is None or out.get("lift_ok") is None or out.get("transfer_ok") is None:
        if fail is None or fail.lower() in ("", "none", "ok", "success"):
            g = True
            l = True
            t = True
        else:
            f = fail.lower()
            if "no_contact_after_close" in f or "weak_grasp_after_close" in f:
                g, l, t = False, False, False
            elif "lift" in f and "dropped" in f:
                g, l, t = True, False, False
            elif "place_plan_failed" in f or "transfer" in f:
                g, l, t = True, True, False
            else:
                # fallback: assume grasp ok, but pipeline failed later
                g, l, t = True, False, False

        out.setdefault("grasp_ok", g)
        out.setdefault("lift_ok", l)
        out.setdefault("transfer_ok", t)

    # task_ok / pipe_ok fallback
    out.setdefault("task_ok", _safe_bool(row.get("task_ok")))
    out.setdefault("pipe_ok", _safe_bool(row.get("pipe_ok")))

    # If still None, infer:
    if out["pipe_ok"] is None:
        # pipeline ok ~ transfer ok (в твоих выводах pipelineOK == transfer_ok)
        out["pipe_ok"] = bool(out["transfer_ok"])
    if out["task_ok"] is None:
        out["task_ok"] = bool(out["transfer_ok"])

    return out

def make_df(paths: List[str]) -> pd.DataFrame:
    all_rows = []
    for p in paths:
        rows = read_jsonl(p)
        base = os.path.basename(p)
        for r in rows:
            rr = dict(r)

            stage = derive_stage_flags(rr)
            rr.update(stage)

            # ---- source normalization ----
            # priority: selected_source -> src -> source -> file tag
            src = rr.get("selected_source")
            if src is None:
                src = rr.get("src")
            if src is None:
                src = rr.get("source")
            if src is None:
                # fallback: tag by filename
                # e.g. grasp_attempts.jsonl under eps/ vs nofc/
                # include parent folder if possible
                parent = os.path.basename(os.path.dirname(p))
                src = parent if parent else base
            rr["src"] = src

            rr["fail_reason"] = rr.get("fail_reason") or rr.get("fail") or None

            # ---- numeric metrics ----
            # eps: in your eps logs it's likely stored as "eps" already
            rr["eps"] = _safe_float(rr.get("eps") or rr.get("fc_eps") or rr.get("eps_fc"))

            # sums: support multiple aliases
            rr["sumF_after_close"] = _safe_float(
                rr.get("sumF_after_close") or rr.get("sumF_close") or rr.get("sumF_close_after")
            )
            rr["sumF_after_lift"] = _safe_float(
                rr.get("sumF_after_lift") or rr.get("sumF_lift") or rr.get("sumF_lift_after")
            )

            rr["nC_close"] = _safe_float(
                rr.get("nC_close") or rr.get("n_contacts_close") or rr.get("contacts_after_close")
            )
            rr["dz"] = _safe_float(rr.get("dz"))
            rr["slip"] = _safe_float(rr.get("slip"))

            rr["_file"] = base
            rr["_path"] = p
            all_rows.append(rr)

    df = pd.DataFrame(all_rows)

    def ratio(a, b):
        if a is None or b is None:
            return None
        if b <= 1e-9:
            return None
        return a / b

    df["lift_close_ratio"] = [
        ratio(a, b) for a, b in zip(df["sumF_after_lift"].tolist(), df["sumF_after_close"].tolist())
    ]

    for c in ("grasp_ok", "lift_ok", "transfer_ok", "pipe_ok", "task_ok"):
        if c in df.columns:
            df[c] = df[c].astype(bool)

    return df


def save_plots(df: pd.DataFrame, group_col: str, outdir: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)

    configs = [
        ("eps", "lift_ok"),
        ("eps", "task_ok"),
        ("sumF_after_close", "lift_ok"),
        ("sumF_after_close", "task_ok"),
        ("sumF_after_lift", "lift_ok"),
        ("sumF_after_lift", "task_ok"),
        ("lift_close_ratio", "lift_ok"),
        ("lift_close_ratio", "task_ok"),
    ]

    for gr, sub in df.groupby(group_col, dropna=False):
        gr_name = str(gr)

        for metric, target in configs:
            if metric not in sub.columns or target not in sub.columns:
                continue

            s = sub[[metric, target]].dropna()
            if len(s) < 20:
                continue

            # qcut produces Interval bins; we must convert each interval to its midpoint safely
            try:
                bins = pd.qcut(s[metric], q=10, duplicates="drop")
            except Exception:
                continue

            tmp = s.copy()
            tmp["bin"] = bins

            rep = tmp.groupby("bin").agg(
                p=(target, "mean"),
                n=(target, "size"),
                x_min=(metric, "min"),
                x_max=(metric, "max"),
                x_med=(metric, "median"),
            ).reset_index()

            # midpoint of each interval for plotting
            def mid_of_interval(iv):
                # iv is pd.Interval
                try:
                    return float((iv.left + iv.right) / 2.0)
                except Exception:
                    return np.nan

            rep["x"] = rep["bin"].map(mid_of_interval)
            rep = rep.dropna(subset=["x"]).sort_values("x")

            if len(rep) < 3:
                continue

            plt.figure()
            plt.plot(rep["x"].to_numpy(dtype=float), rep["p"].to_numpy(dtype=float))
            plt.xlabel(metric)
            plt.ylabel(f"P({target}=True)")
            plt.title(f"{group_col}={gr_name} | {metric} -> {target} | N={len(s)}")
            fname = f"{group_col}_{gr_name}_{metric}_to_{target}.png".replace("/", "_")
            plt.savefig(os.path.join(outdir, fname), dpi=150, bbox_inches="tight")
            plt.close()


def summarize_table(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    g = df.groupby(group_col, dropna=False)
    out = pd.DataFrame({
        "attempts": g.size(),
        "grasp_ok_%": 100.0 * g["grasp_ok"].mean(),
        "lift_ok_%": 100.0 * g["lift_ok"].mean(),
        "transfer_ok_%": 100.0 * g["transfer_ok"].mean(),
        "pipe_ok_%": 100.0 * g["pipe_ok"].mean(),
        "task_ok_%": 100.0 * g["task_ok"].mean(),
    }).sort_values("attempts", ascending=False)
    return out

def _corr_binary_vs_cont(y_bool: pd.Series, x: pd.Series) -> Dict[str, Any]:
    """
    Возвращает point-biserial (если scipy), + spearman/pearson на {0,1}.
    """
    d = {}
    # clean
    mask = x.notna() & y_bool.notna()
    x2 = x[mask].astype(float).to_numpy()
    y2 = y_bool[mask].astype(int).to_numpy()
    n = int(mask.sum())
    d["n"] = n
    if n < 5 or np.std(x2) < 1e-12:
        d["note"] = "too_few_samples_or_constant_x"
        return d

    if SCIPY_OK:
        try:
            r_pb, p_pb = pointbiserialr(y2, x2)
            d["pointbiserial_r"] = float(r_pb)
            d["pointbiserial_p"] = float(p_pb)
        except Exception:
            pass
        try:
            r_s, p_s = spearmanr(y2, x2)
            d["spearman_r"] = float(r_s)
            d["spearman_p"] = float(p_s)
        except Exception:
            pass
        try:
            r_p, p_p = pearsonr(y2, x2)
            d["pearson_r"] = float(r_p)
            d["pearson_p"] = float(p_p)
        except Exception:
            pass
    else:
        # fallback: compute pearson on 0/1
        r = np.corrcoef(y2, x2)[0, 1]
        d["pearson_r"] = float(r)
        d["note"] = "scipy_not_available_for_pvalues"
    return d

def print_corrs(df: pd.DataFrame, group_col: str, metrics: List[str], targets: List[str]) -> None:
    groups = df[group_col].fillna("None").unique().tolist()
    for gr in sorted(groups):
        sub = df[df[group_col].fillna("None") == gr]
        print(f"\n=== Correlations for {group_col}={gr} | N={len(sub)} ===")
        for t in targets:
            for m in metrics:
                d = _corr_binary_vs_cont(sub[t], sub[m])
                n = d.get("n", 0)
                if n < 5:
                    print(f"  {t} vs {m}: n={n} (skip)")
                    continue
                parts = [f"n={n}"]
                if "pointbiserial_r" in d:
                    parts.append(f"pb_r={d['pointbiserial_r']:+.3f}")
                    parts.append(f"pb_p={d.get('pointbiserial_p', float('nan')):.3g}")
                if "spearman_r" in d:
                    parts.append(f"sp_r={d['spearman_r']:+.3f}")
                    parts.append(f"sp_p={d.get('spearman_p', float('nan')):.3g}")
                if "pearson_r" in d:
                    parts.append(f"pr_r={d['pearson_r']:+.3f}")
                    if "pearson_p" in d:
                        parts.append(f"pr_p={d.get('pearson_p', float('nan')):.3g}")
                if "note" in d:
                    parts.append(f"note={d['note']}")
                print(f"  {t} vs {m}: " + " | ".join(parts))

def quantile_bins_report(df: pd.DataFrame, group_col: str, metric: str, target: str, q: int = 5) -> pd.DataFrame:
    out_rows = []
    for gr, sub in df.groupby(group_col, dropna=False):
        s = sub[[metric, target]].dropna()
        if len(s) < max(10, q * 2):
            continue
        try:
            bins = pd.qcut(s[metric], q=q, duplicates="drop")
        except Exception:
            continue
        tmp = s.copy()
        tmp["bin"] = bins.astype(str)
        rep = tmp.groupby("bin").agg(
            n=(target, "size"),
            p=(target, "mean"),
            x_min=(metric, "min"),
            x_max=(metric, "max"),
            x_med=(metric, "median"),
        ).reset_index()
        rep[group_col] = gr
        out_rows.append(rep)
    if not out_rows:
        return pd.DataFrame()
    res = pd.concat(out_rows, ignore_index=True)
    # readability
    res["p_%"] = 100.0 * res["p"]
    res = res.drop(columns=["p"])
    return res[[group_col, "bin", "n", "p_%", "x_min", "x_med", "x_max"]]

# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True, help="Paths to grasp_attempts.jsonl logs")
    ap.add_argument("--group", default="src", help="Grouping column: src or _file etc.")
    ap.add_argument("--bins", type=int, default=5, help="Quantile bins for P(success|bin)")
    ap.add_argument("--outdir", default=None, help="If set, save plots to this directory")
    ap.add_argument("--csv", default=None, help="If set, save merged dataframe to CSV")
    args = ap.parse_args()

    df = make_df(args.files)

    group_col = args.group
    if group_col not in df.columns:
        raise SystemExit(f"Unknown group column '{group_col}'. Available: {list(df.columns)}")

    print("\n=== Summary table ===")
    tab = summarize_table(df, group_col)
    print(tab.to_string(float_format=lambda x: f"{x:.2f}"))

    # Fail reasons overview
    if "fail_reason" in df.columns:
        print("\n=== Top fail reasons (overall) ===")
        print(df["fail_reason"].fillna("None").value_counts().head(15).to_string())

        print(f"\n=== Top fail reasons by {group_col} ===")
        for gr, sub in df.groupby(group_col, dropna=False):
            vc = sub["fail_reason"].fillna("None").value_counts().head(8)
            print(f"\n[{group_col}={gr}]")
            print(vc.to_string())

    metrics = ["eps", "sumF_after_close", "sumF_after_lift", "lift_close_ratio"]
    targets = ["lift_ok", "task_ok"]

    print_corrs(df, group_col, metrics=metrics, targets=targets)

    # Quantile bin reports
    print("\n=== Quantile bin success reports ===")
    for metric in metrics:
        for target in targets:
            rep = quantile_bins_report(df, group_col, metric, target, q=args.bins)
            if rep.empty:
                continue
            print(f"\n-- {metric} -> {target} (bins={args.bins}) --")
            # pretty rounding
            rep2 = rep.copy()
            for c in ("x_min", "x_med", "x_max"):
                rep2[c] = rep2[c].map(lambda v: None if pd.isna(v) else float(v))
            print(rep2.to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved CSV: {args.csv}")

    if args.outdir:
        save_plots(df, group_col, args.outdir)
        print(f"\nSaved plots to: {args.outdir}")

if __name__ == "__main__":
    main()
