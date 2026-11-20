"""统计显著性：配对 bootstrap、McNemar、Holm-Bonferroni、多 seed 汇总。

评审要求：≥3（最好≥5）seeds 报 mean±std，主对比配对检验 + 95%CI + 多重比较校正。
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def mean_std(xs) -> tuple[float, float]:
    a = np.asarray(xs, dtype=float)
    return round(float(a.mean()), 4), round(float(a.std(ddof=1)) if len(a) > 1 else 0.0, 4)


def summarize_seeds(per_seed: list[dict], keys: list[str] | None = None) -> dict:
    """把 [{run,seed,metrics}] 汇总成 {key: {mean,std,values}}。"""
    if not per_seed:
        return {}
    keys = keys or list(per_seed[0].keys())
    out = {}
    for k in keys:
        vals = [rec[k] for rec in per_seed if k in rec]
        if vals and isinstance(vals[0], (int, float)):
            m, s = mean_std(vals)
            out[k] = {"mean": m, "std": s, "values": vals}
    return out


def paired_bootstrap(diff, n_boot: int = 10000, seed: int = 42) -> dict:
    """配对 bootstrap 的均值差 95%CI 与 p 值（双尾）。diff = ours - baseline。"""
    d = np.asarray(diff, dtype=float)
    if len(d) == 0:
        return {"mean_diff": 0.0, "ci95": [0, 0], "p_value": 1.0}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boot_means = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    # p 值：bootstrap 零分布近似
    centered = boot_means - boot_means.mean()
    p = 2 * min((centered >= d.mean()).mean(), (centered <= d.mean()).mean())
    return {
        "mean_diff": round(float(d.mean()), 4),
        "ci95": [round(float(lo), 4), round(float(hi), 4)],
        "p_value": round(float(max(0.0, min(1.0, p))), 4),
        "n": len(d),
    }


def mcnemar(b_correct, base_correct) -> dict:
    """McNemar 精确检验：比较两系统在同一批样本上的正确性差异。"""
    a = np.asarray(b_correct).astype(bool)
    b = np.asarray(base_correct).astype(bool)
    n01 = int(np.sum(a & ~b))  # ours 对 baseline 错
    n10 = int(np.sum(~a & b))
    n = n01 + n10
    p = 1.0 if n == 0 else float(binomtest(n01, n, 0.5).pvalue)
    return {"ours_only_correct": n01, "baseline_only_correct": n10,
            "p_value": round(p, 4)}


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Holm–Bonferroni 多重比较校正，返回每个假设的 (p, adjusted, reject)。"""
    m = len(p_values)
    order = np.argsort(p_values)
    out = [{"index": i, "p": float(p), "p_adj": None, "reject": None}
           for i, p in enumerate(p_values)]
    running = 0.0
    for rank, idx in enumerate(order):
        adj = min(1.0, (m - rank) * p_values[idx])
        running = max(running, adj)
        out[idx]["p_adj"] = round(running, 4)
        out[idx]["reject"] = bool(running < alpha)
    return out
