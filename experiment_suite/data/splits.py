"""三种切分协议 + 重叠率审计（对应 EXPERIMENT_PLAN B1 双协议）。

* random       —— 复现文献的常见做法，标注为有偏对照（近重复跨 train/test）。
* temporal     —— 按 deployed_at 排序切 train<=2023 / val=2024 / test>=2025。
* clone_aware  —— 按克隆族(group)切分，同族样本绝不跨集（本文主表协议）。

所有切分按 label 分层以保持类别比例；随机协议用 seed 可复现。
"""

from __future__ import annotations

import random
from collections import defaultdict

from ..config import SEEDS


def _stratified_groups(
    key_fn, records: list[dict], train_frac: float, val_frac: float, seed: int
) -> dict[str, list[dict]]:
    by_label: dict[str, list] = defaultdict(list)
    for r in records:
        by_label[r["label"]].append(r)
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    rng = random.Random(seed)
    for label, group in by_label.items():
        units: dict = defaultdict(list)
        for r in group:
            units[key_fn(r)].append(r)
        keys = list(units)
        rng.shuffle(keys)
        n = len(keys)
        n_train = max(1, int(n * train_frac))
        n_val = max(1, int(n * val_frac)) if n - n_train > 1 else 0
        for i, k in enumerate(keys):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            out[split].extend(units[k])
    return out


def make_split(
    records: list[dict], protocol: str = "clone_aware", seed: int = SEEDS[0],
    train_frac: float = 0.7, val_frac: float = 0.15,
    cluster_ids: list[int] | None = None,
) -> dict[str, list[dict]]:
    if protocol == "random":
        return _stratified_groups(lambda r: r["id"], records, train_frac, val_frac, seed)
    if protocol == "clone_aware":
        return _stratified_groups(lambda r: r.get("family", r["id"]), records, train_frac, val_frac, seed)
    if protocol == "detector_aware":
        # 用检测器给出的簇 ID 分组（与真值族不同），用于检验检测器漏检时残留的真实族泄漏。
        if cluster_ids is None:
            raise ValueError("detector_aware split requires cluster_ids")
        idx = {r["id"]: i for i, r in enumerate(records)}
        return _stratified_groups(lambda r: f"k{cluster_ids[idx[r['id']]]}",
                                  records, train_frac, val_frac, seed)
    if protocol == "temporal":
        by_label: dict[str, list] = defaultdict(list)
        for r in records:
            by_label[r["label"]].append(r)
        out = {"train": [], "val": [], "test": []}
        for label, group in by_label.items():
            group = sorted(group, key=lambda r: r.get("deployed_at", "0000"))
            n = len(group)
            n_train = int(n * train_frac)
            n_val = int(n * val_frac)
            out["train"].extend(group[:n_train])
            out["val"].extend(group[n_train:n_train + n_val])
            out["test"].extend(group[n_train + n_val:])
        return out
    raise ValueError(f"unknown protocol: {protocol}")


def audit_overlap(split: dict[str, list[dict]], cluster_ids: list[int] | None = None) -> dict:
    """Sample, true-family and detected-cluster overlap.

    R3 fix. The previous ``shortcut_overlap_rate`` was computed from the
    family-level planted cue and was mathematically identical to the
    true-family overlap in every row. It is not an independent mechanism and
    is therefore no longer emitted as an audited quantity. A legacy field is
    kept with an explicit redundancy flag only for backwards compatibility.
    """
    tr, te = split["train"], split["test"]
    tr_ids = {r["id"] for r in tr}
    tr_fam = {r.get("family") for r in tr}
    tr_short = {r.get("shortcut") for r in tr}
    n_overlap_id = sum(r["id"] in tr_ids for r in te)
    n_overlap_fam = sum(r.get("family") in tr_fam for r in te)
    n_overlap_short = sum(r.get("shortcut") in tr_short for r in te)
    out = {
        "n_train": len(tr), "n_val": len(split["val"]), "n_test": len(te),
        "sample_overlap_rate": round(n_overlap_id / max(1, len(te)), 4),
        "true_family_overlap_rate": round(n_overlap_fam / max(1, len(te)), 4),
        "shortcut_overlap_legacy": round(n_overlap_short / max(1, len(te)), 4),
        "shortcut_overlap_is_redundant_with_true_family": True,
    }
    if cluster_ids is not None:
        tr_clusters = {int(r["_cluster_id"]) for r in tr if "_cluster_id" in r}
        n_overlap_clu = sum(int(r.get("_cluster_id", -1)) in tr_clusters for r in te)
        out["detected_cluster_overlap_rate"] = round(n_overlap_clu / max(1, len(te)), 4)
    else:
        out["detected_cluster_overlap_rate"] = None
    out["clone_family_overlap_rate"] = out["true_family_overlap_rate"]
    return out


def attach_cluster_ids(records: list[dict], cluster_ids: list[int]) -> list[dict]:
    """把簇编号挂到记录对象上，供 audit_overlap 无歧义读取。"""
    for rec, cid in zip(records, cluster_ids):
        rec["_cluster_id"] = int(cid)
    return records
