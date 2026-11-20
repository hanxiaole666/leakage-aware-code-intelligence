"""检测指标：F1(宏/微)、FPR、Precision/Recall、MCC、AUPRC。

严禁仅报 Accuracy（类别不平衡）；FPR 是工业第一痛点，单列为主指标。
多分类按 one-vs-rest 计算每类 FPR 后宏平均。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    matthews_corrcoef, precision_recall_fscore_support,
)


def compute_metrics(
    y_true, y_pred, y_score=None, labels: list[str] | None = None, average: str = "macro"
) -> dict:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out: dict = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "f1_micro": round(float(f1_score(y_true, y_pred, average="micro", zero_division=0)), 4),
        "precision_macro": round(float(precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)[0]), 4),
        "recall_macro": round(float(precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0)[1]), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
    }
    # one-vs-rest FPR 宏平均
    fprs = []
    uniq = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
    for c in uniq:
        yt = (y_true == c).astype(int)
        yp = (y_pred == c).astype(int)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        denom = tn + fp
        fprs.append(fp / denom if denom else 0.0)
    out["fpr"] = round(float(np.mean(fprs)), 4)
    if y_score is not None:
        ys = np.asarray(y_score)
        if ys.ndim == 1:
            out["auprc"] = round(float(average_precision_score(
                (y_true == uniq[-1]).astype(int), ys)), 4)
        else:
            out["auprc_macro"] = round(float(np.mean([
                average_precision_score((y_true == c).astype(int), ys[:, i])
                for i, c in enumerate(uniq)
            ])), 4)
    return out


def per_class_table(y_true, y_pred, labels: list[str], indices=None) -> list[dict]:
    """Per-class table aligned to ``labels``.

    ``y_true``/``y_pred`` are integer class indices in this project. Passing the
    string label names as ``labels`` to sklearn therefore matched nothing and
    produced all-zero support (the bug present in the first release). We now
    pass explicit integer ``indices`` by default and keep ``labels`` purely for
    display.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if indices is None:
        indices = list(range(len(labels)))
    indices = list(indices)
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=indices, zero_division=0
    )
    return [
        {"label": str(labels[i]) if i < len(labels) else str(k),
         "precision": round(float(p[i]), 4), "recall": round(float(r[i]), 4),
         "f1": round(float(f[i]), 4), "support": int(s[i])}
        for i, k in enumerate(indices)
    ]


def metrics_table(results: dict[str, dict], primary: str = "f1_macro") -> str:
    """把 {run_id: metrics} 渲染成 Markdown 表。"""
    keys = ["f1_macro", "f1_micro", "fpr", "precision_macro", "recall_macro", "mcc", "accuracy"]
    lines = ["| Run | " + " | ".join(keys) + " |", "|---|" + "---|" * len(keys)]
    for rid, m in results.items():
        lines.append("| " + rid + " | " + " | ".join(str(m.get(k, "-")) for k in keys) + " |")
    return "\n".join(lines)
