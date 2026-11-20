"""实验运行器：把 R001-R061 清单映射为真实可跑实验，产出指标与产物。

断点续跑（pipeline 级）：每个 run 完成后写 ``pipeline_state.json``；
再次运行时 status=done 的 run 直接跳过。``--run <ID>`` 单跑；``--all`` 全跑。
``--fresh`` 清状态重跑。所有指标由 compute_metrics 真实产出并落盘 JSON。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .config import (
    RUNS, RUN_BY_ID, SEEDS, STATE_FILE, TABLES_DIR, RESULTS_ROOT, TASK_LABELS, provenance,
)
from .data.synthesize import load_records, DATA_FILE
from .data.splits import make_split, audit_overlap, attach_cluster_ids
from .data.fingerprints import cluster_clones
from .data.download import download_all
from .models.features import FeatureExtractor
from .models.estimators import build_estimator
from .eval.metrics import compute_metrics, per_class_table
from .eval.calibration import (
    ece, brier, temperature_scale, apply_temperature, conformal_accept,
    conformal_split, risk_coverage_full, ece_adaptive, isotonic_calibrate,
)
from .eval.stats import paired_bootstrap, mcnemar, holm_bonferroni


# ---------- pipeline 状态（断点续跑） ----------

def _load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"runs": {}}
    return {"runs": {}}


def _save_state(state: dict) -> None:
    tmp = STATE_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    import os
    os.replace(tmp, STATE_FILE)


def _corpus(use_real: bool = False):
    """准备语料：先尝试真实下载（幂等），失败/无数据用合成 fallback。"""
    provenance_data = {"real_downloads": {}}
    if use_real:
        for name, path in download_all().items():
            provenance_data["real_downloads"][name] = str(path) if path else None
    records = load_records()
    provenance_data.update({
        "n_records": len(records),
        "synthetic": True,
        "source_file": str(DATA_FILE),
        "note": "M0/M1 骨架：合成克隆族数据；真实 3B 结果需接入真实集 + 大模型",
    })
    return records, provenance_data


def _vec(records, ex: FeatureExtractor):
    return ex.transform(records)


def _labels(records):
    return np.asarray([TASK_LABELS.index(r["label"]) for r in records])


# ---------- 各 run 的执行体 ----------

def run_one(run_id: str, records: list[dict], seed: int = SEEDS[0]) -> dict:
    spec = RUN_BY_ID[run_id]
    # E1/审计 AUD-02：用检测器簇做双重叠审计；特征开关不再随协议变化，
    # 三种协议使用完全相同的特征集，消除 0.314 差距与 shortcut 开关的共变混杂。
    cl = cluster_clones(records, threshold=0.9)
    cids = cl["cluster_ids"]
    attach_cluster_ids(records, cids)
    split = make_split(records, protocol=spec.split, seed=seed, cluster_ids=cids) \
        if spec.split == "detector_aware" else make_split(records, protocol=spec.split, seed=seed)
    audit = audit_overlap(split, cluster_ids=cids)
    tr, va, te = split["train"], split["val"], split["test"]

    # 变体 -> 特征开关（消融）：三个 split 协议一律保持同一特征集
    feat = dict(use_struct=True, use_signal=True, use_shortcut=True)
    if spec.variant == "ablate_structure":
        feat["use_struct"] = False
    elif spec.variant == "ablate_semantic":
        feat["use_signal"] = False
    elif spec.variant == "ablate_shortcut":
        feat["use_shortcut"] = False

    ex = FeatureExtractor(shortcuts=[r.get("shortcut") for r in tr if r.get("shortcut")],
                          **feat)
    Xtr, Xva, Xte = _vec(tr, ex), _vec(va, ex), _vec(te, ex)
    ytr, yva, yte = _labels(tr), _labels(va), _labels(te)

    variant = spec.variant
    if variant == "audit_only":
        return {"run_id": run_id, "audit": audit, "clone_clusters": cl}

    est = build_estimator(variant, seed, n_classes=len(set(ytr.tolist())))
    est.fit(Xtr, ytr)
    pred = est.predict(Xte)
    proba = _safe_proba(est, Xte, yte)

    metrics = compute_metrics(yte, pred, proba, labels=TASK_LABELS)
    result = {
        "run_id": run_id, "milestone": spec.milestone, "block": spec.block,
        "variant": variant, "split": spec.split, "seed": seed,
        "feature_flags": feat, "audit": audit, "metrics": metrics,
        "per_class": per_class_table(yte, pred, TASK_LABELS, indices=list(range(len(TASK_LABELS)))),
    }

    # 校准类 run：追加校准指标（二值目标 = 是否漏洞，酸性负类为 safe）
    if variant.startswith("calib"):
        ybin = (yte != TASK_LABELS.index("safe")).astype(int)
        # R2 fix: the calibrated event is "vulnerable", so its score must be
        # p(vulnerable) = 1 - p(safe), NOT the max class probability.
        safe_idx = TASK_LABELS.index("safe")
        conf = 1.0 - proba[:, safe_idx] if proba.ndim > 1 else np.asarray(proba)
        binary_baselines = {
            "binary_positive_rate": round(float(ybin.mean()), 4),
            "constant_vulnerable_accuracy": round(float(ybin.mean()), 4),
            "five_class_accuracy_same_split": round(float((pred == yte).mean()), 4),
            "binary_accuracy_coarsened_argmax": round(float(((pred != safe_idx).astype(int) == ybin).mean()), 4),
        }
        if variant == "calib_ts":
            ts = temperature_scale(ybin, conf)
            conf2 = apply_temperature(conf, ts["T"])
            result["calibration"] = {
                "target": "binary vulnerable-vs-safe", "score": "1 - p(safe)",
                "baselines": binary_baselines,
                "temperature": ts, "ece_before": ece(ybin, conf),
                "ece_after": ece(ybin, conf2), "brier": brier(ybin, conf2),
                "ece_adaptive_after": ece_adaptive(ybin, conf2),
                "isotonic": isotonic_calibrate(ybin, conf),
                "risk_coverage": risk_coverage_full(ybin, conf2),
                "aurc_rank_invariance_note":
                    "temperature scaling is strictly monotone, so the confidence ranking and AURC are unchanged",
            }
        elif variant == "calib_mcd":
            # 诚实命名：线性替代模型没有 dropout，这里只是"未校准参照"，
            # 不再冒充 MC Dropout 并声称与 R040 不同的分数来源。
            result["calibration"] = {
                "method": "uncalibrated_reference",
                "note": "linear surrogate has no dropout; this row is the uncalibrated reference",
                "target": "binary vulnerable-vs-safe", "score": "1 - p(safe)",
                "baselines": binary_baselines,
                "ece": ece(ybin, conf), "brier": brier(ybin, conf),
                "ece_adaptive": ece_adaptive(ybin, conf),
                "risk_coverage": risk_coverage_full(ybin, conf),
            }
        elif variant == "calib_conformal_split":
            # Proper split conformal on the held-out validation split.
            yva = (yva != safe_idx).astype(int)
            conf_va = 1.0 - est.predict_proba(Xva)[:, safe_idx]
            result["calibration"] = {
                "target": "binary vulnerable-vs-safe", "score": "1 - p(safe)",
                "baselines": binary_baselines,
                "conformal_split": conformal_split(conf_va, conf, alpha=0.1),
                "note": "held-out validation calibration; meets finite-sample coverage bound",
            }
        elif variant == "calib_conformal":
            # Two EMPIRICAL threshold rules, explicitly not conformal:
            # neither uses a held-out calibration split nor meets the
            # finite-sample coverage bound. Labelled as empirical in the paper.
            corr = (conf >= 0.5).astype(int) == ybin
            n = len(ybin); n_cal = max(1, int(n * 0.5))
            low_thr = float(np.quantile(1 - conf[:n_cal], 0.90))
            low_accept = (1 - conf) <= low_thr
            low_rule = {
                "rule": "empirical low-score threshold (NOT conformal; fails coverage bound)",
                "threshold": round(low_thr, 4),
                "abstain_rate": round(float(1 - low_accept.mean()), 4),
                "accepted_accuracy": round(float(corr[low_accept].mean()), 4),
                "overall_accuracy_same_target": round(float(corr.mean()), 4),
            }
            result["calibration"] = {
                "target": "binary vulnerable-vs-safe", "score": "1 - p(safe)",
                "baselines": binary_baselines,
                "conformal": conformal_accept(ybin, conf, alpha=0.1),
                "conformal_low_score_negative": low_rule,
                "rule_aliases": {"R042a": "empirical low-score (not conformal)",
                                 "R042b": "empirical decision-boundary (not conformal)"},
                "ece": ece(ybin, conf),
                "risk_coverage": risk_coverage_full(ybin, conf),
            }
    if variant == "robust_neardup":
        result["robustness"] = {"neardup_stress": _neardup_stress(records, split, est, ex, yte)}

    return result


def _safe_proba(est, X, y):
    try:
        p = est.predict_proba(X)
        if p.shape[1] != len(TASK_LABELS):
            # DummyClassifier 可能类数不足 → 对齐
            full = np.zeros((len(X), len(TASK_LABELS)))
            for i, c in enumerate(est.classes_):
                full[:, int(c)] = p[:, i]
            return full
        return p
    except Exception:
        pred = est.predict(X)
        return np.eye(len(TASK_LABELS))[pred.astype(int)]


def _neardup_stress(records, split, est, ex, yte):
    """近重复压力集：对 test 中每个克隆族抽取一个代表，评估性能衰减。"""
    fam_seen = set()
    idx = []
    for i, r in enumerate(split["test"]):
        if r.get("family") not in fam_seen:
            fam_seen.add(r.get("family"))
            idx.append(i)
    if not idx:
        return {"note": "no test families", "n": 0}
    pred = est.predict(ex.transform([split["test"][i] for i in idx]))
    yt = yte[idx]
    from sklearn.metrics import f1_score
    return {"n": len(idx), "macro_f1_on_family_representatives": round(
        float(f1_score(yt, pred, average="macro", zero_division=0)), 4)}


def run_all(selected: list[str] | None = None, fresh: bool = False,
            use_real: bool = False, seed: int = SEEDS[0]) -> dict:
    records, prov = _corpus(use_real=use_real)
    state = _load_state()
    if fresh:
        state = {"runs": {}}
        _save_state(state)

    targets = [RUN_BY_ID[x] for x in selected] if selected else RUNS
    summary = {}
    for spec in targets:
        rid = spec.run_id
        if state["runs"].get(rid, {}).get("status") == "done":
            print(f"[SKIP] {rid} 已 done（断点续跑跳过）")
            summary[rid] = state["runs"][rid].get("metrics")
            continue
        print(f"[RUN ] {rid} {spec.description} ...")
        try:
            res = run_one(rid, records, seed=seed)
            state["runs"][rid] = {"status": "done", "metrics": res.get("metrics"),
                                  "result_file": str(RESULTS_ROOT / "runs" / f"{rid}.json")}
            out = RESULTS_ROOT / "runs" / f"{rid}.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
            summary[rid] = res.get("metrics")
            _save_state(state)  # 每完成一个就落盘 → 中途关闭不丢进度
            print(f"[ OK ] {rid} F1={res.get('metrics', {}).get('f1_macro')} "
                  f"FPR={res.get('metrics', {}).get('fpr')}")
        except Exception as exc:  # noqa: BLE001
            state["runs"][rid] = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
            _save_state(state)
            print(f"[ERR ] {rid} {type(exc).__name__}: {exc}", file=sys.stderr)
    (RESULTS_ROOT / "provenance.json").write_text(
        json.dumps({"provenance": prov, "state": state}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="区块链风控可信小模型实验运行器")
    p.add_argument("--all", action="store_true", help="运行全部 run")
    p.add_argument("--run", nargs="*", default=None, help="指定 run_id，如 R002 R011")
    p.add_argument("--fresh", action="store_true", help="清空状态重跑")
    p.add_argument("--real", action="store_true", help="尝试下载真实数据集")
    p.add_argument("--seed", type=int, default=SEEDS[0])
    p.add_argument("--status", action="store_true", help="仅打印各 run 状态")
    args = p.parse_args(argv)

    if args.status:
        state = _load_state()
        for r in RUNS:
            print(f"{r.run_id:5s} {state['runs'].get(r.run_id, {}).get('status', 'TODO'):8s} "
                  f"{r.milestone} {r.block} {r.description}")
        return
    selected = args.run if args.run else (None if args.all else ["R002", "R011", "R012", "R042"])
    run_all(selected=selected, fresh=args.fresh, use_real=args.real, seed=args.seed)


if __name__ == "__main__":
    main()
