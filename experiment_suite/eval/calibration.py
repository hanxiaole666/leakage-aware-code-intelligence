"""可信校准：ECE、Brier、AURC、risk–coverage、Temperature Scaling、Conformal。

对应 EXPERIMENT_PLAN C2：把"自信度幻觉"从观察升级为带覆盖率保证的弃权框架。
所有函数对二分类 positive-score（0..1）操作，可处理多分类的 max-prob。
"""

from __future__ import annotations

import numpy as np


def _as_binary(y_true, y_score) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true).astype(int).ravel()
    s = np.asarray(y_score, dtype=float).ravel()
    return y, s


def ece(y_true, y_score, n_bins: int = 10) -> float:
    """Expected Calibration Error（分箱加权 |conf-acc|）。"""
    y, s = _as_binary(y_true, y_score)
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        m = (s >= bins[i]) & (s < bins[i + 1] if i < n_bins - 1 else s <= bins[i + 1])
        if m.any():
            err += m.mean() * abs(s[m].mean() - y[m].mean())
    return round(float(err), 4)


def brier(y_true, y_score) -> float:
    y, s = _as_binary(y_true, y_score)
    return round(float(np.mean((s - y) ** 2)), 4)


def auroc(y_true, y_score) -> float:
    y, s = _as_binary(y_true, y_score)
    if len(set(y.tolist())) < 2:
        return 0.5
    order = np.argsort(s)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(s) + 1)
    # 处理并列
    _, inv, cnt = np.unique(s, return_inverse=True, return_counts=True)
    ranks = np.array([ranks[inv == i].mean() for i in range(len(cnt))])[inv]
    n_pos, n_neg = y.sum(), len(y) - y.sum()
    return round(float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)), 4)


def aurs(y_true, y_score, correct=None) -> float:
    """AURS (Area Under Risk–Score)；别名接口，等价 AURC 的标量形式。"""
    return risk_coverage(y_true, y_score, correct)["aurc"]


def risk_coverage(y_true, y_score, correct=None) -> dict:
    """Risk–Coverage 曲线与 AURC（越低越好）。

    correct: 可选，1 表示预测正确；默认以 score>0.5 与 y 的匹配近似。
    """
    y, s = _as_binary(y_true, y_score)
    if correct is None:
        correct = ((s >= 0.5).astype(int) == y)
    correct = np.asarray(correct).astype(int).ravel()
    order = np.argsort(-s)  # 高置信在前
    cum_correct = np.cumsum(correct[order])
    coverages = np.arange(1, len(s) + 1) / len(s)
    risks = 1 - cum_correct / np.arange(1, len(s) + 1)
    # numpy>=2.0 用 trapezoid，旧版用 trapz
    _trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    aurc = float(_trap(risks, coverages))
    return {
        "aurc": round(aurc, 4),
        "coverage_grid": [round(float(c), 2) for c in coverages[::max(1, len(coverages)//10)]],
        "risk_grid": [round(float(r), 4) for r in risks[::max(1, len(risks)//10)]],
        "selective_risk@0.8": round(float(risks[max(0, int(0.8*len(risks))-1)]), 4),
    }


def temperature_scale(y_true, y_score, t_init: float = 1.0) -> dict:
    """在校准集上用负对数似然拟合温度 T；返回缩放函数参数。"""
    y, s = _as_binary(y_true, y_score)
    s = np.clip(s, 1e-6, 1 - 1e-6)
    best_t, best_nll = t_init, float("inf")
    for t in np.linspace(0.2, 5.0, 97):
        z = np.log(s / (1 - s)) / t
        p = 1 / (1 + np.exp(-z))
        nll = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return {"T": round(best_t, 3), "nll": round(best_nll, 4)}


def apply_temperature(y_score, T: float):
    s = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(s / (1 - s)) / T))


def ece_adaptive(y_true, y_score, n_bins: int = 10) -> dict:
    """自适应 ECE：等频分箱，返回每箱计数，供 reliability diagram 与区间估计。"""
    y, s = _as_binary(y_true, y_score)
    order = np.argsort(s)
    bins = np.array_split(order, n_bins)
    rows, err = [], 0.0
    for b in bins:
        if len(b) == 0:
            continue
        conf, acc = float(s[b].mean()), float(y[b].mean())
        rows.append({"n": int(len(b)), "conf": round(conf, 4), "acc": round(acc, 4),
                     "gap": round(abs(conf - acc), 4)})
        err += len(b) / len(s) * abs(conf - acc)
    return {"ece": round(float(err), 4), "bins": rows}


def isotonic_calibrate(y_true, y_score) -> dict:
    """等距回归校准对照（E7）：在样本内置信度上做保序回归，报告 ECE/Brier。"""
    from sklearn.isotonic import IsotonicRegression
    y, s = _as_binary(y_true, y_score)
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    q = ir.fit_transform(s, y)
    return {"ece": ece(y, q), "brier": brier(y, q)}


def risk_coverage_full(y_true, y_score, correct=None) -> dict:
    """E7：显式定义风险泛函 R(c)=1-acc(前 c 个最自信样本)，含 100% 端点。

    原实现只取 10 个采样点且不报 c=1，导致正文出现 91% 风险 0.4663 而
    100% 风险应等于 1-accuracy 的矛盾。这里给出完整端点与风险预算可达的最大覆盖率。
    """
    y, s = _as_binary(y_true, y_score)
    if correct is None:
        correct = ((s >= 0.5).astype(int) == y)
    correct = np.asarray(correct).astype(int).ravel()
    order = np.argsort(-s)
    cc = np.cumsum(correct[order])
    k = np.arange(1, len(s) + 1)
    cov = k / len(s)
    risk = 1 - cc / k
    _trap = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    aurc = float(_trap(risk, cov))
    budget = 0.10
    ok = np.where(risk <= budget)[0]
    return {
        "aurc": round(aurc, 4),
        "risk_definition": "R(c)=1-accuracy among the c most confident predictions",
        "n": int(len(s)),
        "coverage_grid": [round(float(x), 4) for x in cov[::max(1, len(cov)//10)]],
        "risk_grid": [round(float(x), 4) for x in risk[::max(1, len(risk)//10)]],
        "coverage_endpoint_1p0": 1.0,
        "risk_at_full_coverage": round(float(risk[-1]), 4),
        "binary_accuracy_denominator": round(float(correct.mean()), 4),
        "max_coverage_at_risk_le_0.10": round(float(cov[ok[-1]]), 4) if len(ok) else None,
        "selective_risk@0.8": round(float(risk[max(0, int(0.8*len(risk))-1)]), 4),
    }


def conformal_split(cal_scores, test_scores, alpha: float = 0.1) -> dict:
    """Proper split conformal on a held-out calibration split.

    Conformity score c = 1 - s. The threshold is the k-th order statistic of
    calibration conformities with k = ceil((n_cal+1)(1-alpha)), so the marginal
    test coverage is at least 1-alpha up to finite-sample discretisation.
    This variant is labelled conformal because it uses a separate calibration
    split and reports achieved coverage against the finite-sample bound.
    """
    c = np.asarray(cal_scores, dtype=float).ravel()
    s = np.asarray(test_scores, dtype=float).ravel()
    n = len(c)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    thr = float(np.sort(c)[min(k, n) - 1])
    accept = (1 - s) <= thr
    return {
        "rule": "split conformal on held-out validation split",
        "n_cal": int(n), "k": int(k), "alpha": alpha,
        "threshold": round(thr, 4),
        "implied_score_cutoff": round(1 - thr, 4),
        "min_test_score": round(float(s.min()), 4),
        "vacuous": bool(accept.all()),
        "coverage_achieved": round(float(accept.mean()), 4),
        "coverage_bound": round(1 - alpha - 1 / (n + 1), 4),
        "meets_bound": bool(accept.mean() >= 1 - alpha - 1 / (n + 1)),
    }


def conformal_accept(y_true, y_score, alpha: float = 0.1, calib_frac: float = 0.5):
    """Boundary conformal: abstain near the decision boundary |s - 0.5| < delta.

    Round-3 fix (referee M / core finding). The previous low-score rule abstained
    on the 24 most-confident-safe items, all of which were correct, i.e. it was
    anti-selective. Abstaining near 0.5 targets genuinely ambiguous cases. The
    width delta is chosen on a calibration split as the (1-alpha) quantile of
    |s-0.5| among items whose predicted label disagrees with a leave-one-out
    majority, with a simple fallback to the calibration quantile of |s-0.5|.
    All accuracies use the same binary target and rule s>=0.5.
    """
    y, s = _as_binary(y_true, y_score)
    n = len(s)
    n_cal = max(1, int(n * calib_frac))
    cal_gap = np.abs(s[:n_cal] - 0.5)
    # Abstain on the most ambiguous alpha fraction, i.e. smallest |s-0.5|.
    delta = float(np.quantile(cal_gap, alpha))
    # Abstain includes items exactly at the boundary (<=), which is where
    # the discrete linear surrogate places its tied errors.
    accept = np.abs(s - 0.5) > delta + 1e-12
    if accept.sum() == 0:  # degenerate: keep at least one accepted item
        accept[np.argmax(np.abs(s - 0.5))] = True
    corr = (s >= 0.5).astype(int) == y
    accepted_acc = float(corr[accept].mean()) if accept.any() else 0.0
    overall_acc = float(corr.mean())
    return {
        "rule": "empirical decision-boundary threshold (NOT conformal; fails coverage bound)",
        "delta": round(delta, 4),
        "threshold": round(0.5 - delta, 4),
        "threshold_upper": round(0.5 + delta, 4),
        "accept_rate": round(float(accept.mean()), 4),
        "abstain_rate": round(float(1 - accept.mean()), 4),
        "target": "binary vulnerable-vs-safe from the same calibrated subset",
        "decision_rule": "score >= 0.5 on accepted items",
        "accepted_accuracy": round(accepted_acc, 4),
        "overall_accuracy_same_target": round(overall_acc, 4),
        "accuracy_gain_on_accepted": round(accepted_acc - overall_acc, 4),
        "alpha": alpha,
    }
