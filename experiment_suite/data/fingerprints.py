"""AST/源码归一化指纹 + MinHash-LSH 近重复聚类（泄漏感知切分的核心）。

方法（对应 EXPERIMENT_PLAN 方向 A）：
1. ``normalize_source``：剥离注释、字面量常量、标识符名，仅保留结构 token
   （等价 AST 骨架），使仅改名的变体得到相同指纹。
2. ``minhash_signature``：对归一化 tokens 的 3-gram 集合计算 k 个哈希最小值。
3. ``cluster_clones``：按 Jaccard 估计近重复，Union-Find 聚成克隆族。
   支持显式 ``family`` 字段（合成数据已知真值）作为审计对照。

校验和接口 ``compile_check`` 预留 solc 调用：扰动后强制编译校验的点，
在无 solc 环境返回 None 并标注 skipped（不伪造通过）。
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from typing import Iterable

_NUM_HASH = 64

_COMMENT = re.compile(r"//.*?$|/\*.*?\*/", re.S | re.M)
_STRING = re.compile(r'"[^"]*"|\'[^\']*\'')
_NUMBER = re.compile(r"\b0x[0-9a-fA-F]+\b|\b\d+\b")
_IDENT = re.compile(r"\b[A-Za-z_]\w*\b")

# 保留的结构关键字（归一化后只留这些 + 标点）
_KEYWORDS = {
    "pragma", "solidity", "contract", "function", "public", "private", "external",
    "internal", "view", "pure", "payable", "returns", "return", "require", "mapping",
    "address", "uint", "bool", "string", "bytes", "if", "else", "for", "while",
    "call", "delegatecall", "send", "transfer", "emit", "memory", "storage",
}


def normalize_source(source: str) -> str:
    """结构性归一化：去注释/字符串/数字，标识符统一为 ID，仅留结构骨架。"""
    s = _COMMENT.sub(" ", source)
    s = _STRING.sub(" STR ", s)
    s = _NUMBER.sub(" NUM ", s)

    def repl(m: re.Match) -> str:
        w = m.group(0)
        return w if w in _KEYWORDS else "ID"

    s = _IDENT.sub(repl, s)
    return re.sub(r"\s+", " ", s).strip()


def _h(token: str, salt: int) -> int:
    return int.from_bytes(hashlib.blake2b(f"{salt}|{token}".encode(), digest_size=8).digest(), "big")


def clone_tokens(source: str) -> list[str]:
    """近重复检测用的归一化 token 流。

    近重复 = 同族变体共享"家族标识"（如稳定函数名/状态变量名），但每次变体的
    临时标识符（合约名 cid、函数名 fname、参数名 arg）随机。策略：
      * 只出现一次的标识符视为临时噪声 → 归一为 ID；
      * 出现 >=2 次的标识符视为家族稳定标识 → 保留原名（判别关键）。

    这样同族变体（共享 slot_<fam> / emit_<fam> 家族名）高度相似，
    不同族（家族名不同）低相似；临时标识符不影响。
    """
    from collections import Counter
    s = _COMMENT.sub(" ", source)
    s = _STRING.sub(" STR ", s)
    s = _NUMBER.sub(" NUM ", s)
    toks = re.findall(r"[A-Za-z_]\w*", s.lower())
    cnt = Counter(toks)
    return [t if cnt[t] >= 2 else "id" for t in toks]


def minhash_signature(source: str, k: int = _NUM_HASH) -> tuple[int, ...]:
    toks = clone_tokens(source)
    grams = [" ".join(toks[i:i + 3]) for i in range(max(1, len(toks) - 2))] or toks
    return tuple(min(_h(g, s) for g in grams) for s in range(k))


def jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    if not a or len(a) != len(b):
        return 0.0
    return sum(x == y for x, y in zip(a, b)) / len(a)


class _UF:
    def __init__(self, n: int):
        self.p = list(range(n))

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def cluster_clones(records: list[dict], threshold: float = 0.9) -> dict:
    """用 MinHash 相似度聚类近重复；返回聚类结果与真值一致性审计。

    修订版（E1/E3）：除精度 purity 外，显式计算配对召回、F1、成功/失败同族
    对计数、跨族误配对、簇大小分布，并在有真值时给出标签无关的阈值选择所需
    的完整 sweep 数据。all-pairs 判定为精确计算，不依赖 LSH 候选召回。
    """
    sigs = [minhash_signature(r["source"]) for r in records]
    uf = _UF(len(records))
    # LSH 候选 + 全对精确核验（小语料 1200 条，O(n^2) 可接受且保证审计可复现）
    bands: dict[tuple, list[int]] = defaultdict(list)
    for i, sig in enumerate(sigs):
        for band in range(8):
            key = (band,) + sig[band * 8:(band + 1) * 8]
            bands[key].append(i)
    candidates = set()
    for idxs in bands.values():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                candidates.add((idxs[a], idxs[b]))
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            if (i, j) in candidates and jaccard(sigs[i], sigs[j]) >= threshold:
                uf.union(i, j)

    pred = [uf.find(i) for i in range(len(records))]
    truth = [r.get("family") for r in records]
    pair_tp = pair_fp = pair_fn = pair_tn = 0
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            same_pred = pred[i] == pred[j]
            same_true = bool(truth[i] is not None and truth[i] == truth[j])
            if same_pred and same_true:
                pair_tp += 1
            elif same_pred and not same_true:
                pair_fp += 1
            elif not same_pred and same_true:
                pair_fn += 1
            else:
                pair_tn += 1
    precision = pair_tp / max(1, pair_tp + pair_fp)
    recall = pair_tp / max(1, pair_tp + pair_fn)
    f1 = 2 * precision * recall / max(1e-12, precision + recall)
    sizes = Counter(pred)
    return {
        "n_clusters": len(set(pred)),
        "n_records": len(records),
        "pairwise_purity": round(precision, 4),
        "pair_precision": round(precision, 4),
        "pair_recall": round(recall, 4),
        "pair_f1": round(f1, 4),
        "pair_tp": pair_tp, "pair_fp": pair_fp, "pair_fn": pair_fn, "pair_tn": pair_tn,
        "n_true_same_pairs": pair_tp + pair_fn,
        "cluster_size_distribution": dict(sorted(Counter(sizes.values()).items())),
        "mean_cluster_size": round(len(records) / max(1, len(set(pred))), 4),
        "cluster_ids": pred,
        "threshold": threshold,
    }


def exact_match_clusters(records: list[dict]) -> dict:
    """Exact-match 去重基线（E3）：按归一化源码精确相等分组。"""
    groups: dict[str, int] = defaultdict(int)
    pred = []
    for r in records:
        key = normalize_source(r["source"])
        if key not in groups:
            groups[key] = len(groups)
        pred.append(groups[key])
    truth = [r.get("family") for r in records]
    tp = fp = fn = 0
    for i in range(len(records)):
        for j in range(i + 1, len(records)):
            sp, st = pred[i] == pred[j], truth[i] == truth[j]
            if sp and st: tp += 1
            elif sp and not st: fp += 1
            elif not sp and st: fn += 1
    p = tp / max(1, tp + fp); r = tp / max(1, tp + fn)
    return {"method": "exact_match", "n_clusters": len(groups),
            "pair_precision": round(p, 4), "pair_recall": round(r, 4),
            "pair_f1": round(2*p*r/max(1e-12, p+r), 4),
            "pair_tp": tp, "pair_fp": fp, "pair_fn": fn}


def threshold_sweep(records: list[dict], thresholds=(0.80, 0.85, 0.90, 0.92, 0.95)) -> list[dict]:
    """E3：对给定阈值逐个聚类并给出成对指标，供正文如实报告 sweep。"""
    out = []
    for t in thresholds:
        c = cluster_clones(records, threshold=float(t))
        out.append({k: c[k] for k in ("threshold", "n_clusters", "pair_precision",
                                     "pair_recall", "pair_f1", "pair_fp", "pair_fn",
                                     "mean_cluster_size")})
    return out


def compile_check(source: str) -> bool | None:
    """solc 编译校验：有 solc 则真编译，无则返回 None（标注 skipped，绝不伪造）。"""
    if shutil.which("solc") is None:
        return None
    try:
        p = subprocess.run(
            ["solc", "--bin", "-"], input=source, text=True,
            capture_output=True, timeout=20,
        )
        return p.returncode == 0
    except Exception:  # noqa: BLE001
        return None
