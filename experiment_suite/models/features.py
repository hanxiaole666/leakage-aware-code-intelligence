"""结构化特征化：把合约源码映射为向量。

两族特征（对应 B2 消融）：
* STRUCT：结构 token 计数（归一化 AST 骨架）—— "结构增强"组件。
* SIGNAL：漏洞语义信号 token（call.value / block.timestamp / onlyOwner...）—— 真实可泛化信号。
* SHORTCUT：族特有标识符 one-hot —— **泄漏捷径**，仅随机切分可用；clone_aware 下失效。

通过开关特征族实现可复现消融：ablate_structure 去掉 STRUCT，ablate_semantic 去掉 SIGNAL。
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np

from ..data.fingerprints import normalize_source
from ..config import TASK_LABELS

# 与 data/synthesize.py 的 SIGNALS 保持一致（长且独特的 token，避免子串误匹配）
SIGNAL_TOKENS = [
    "call.value", "delegatecall", "fallback", "reentrancyGuard",
    "block.timestamp", "block.number", "timestampDependent",
    "unchecked", "safeMath", "overflowCheck",
    "onlyOwner", "msg.sender", "tx.origin", "accessControl",
    "readOnly", "emit_event", "viewFunction", "pureFunction",
]

STRUCT_TOKENS = ["function", "require", "mapping", "if", "for", "while", "return", "contract"]


class FeatureExtractor:
    def __init__(self, shortcuts: list[str] | None = None,
                 use_struct: bool = True, use_signal: bool = True,
                 use_shortcut: bool = True):
        self.shortcuts = sorted(set(shortcuts or []))
        self.use_struct = use_struct
        self.use_signal = use_signal
        self.use_shortcut = use_shortcut

    @property
    def dim(self) -> int:
        d = 0
        d += len(STRUCT_TOKENS) if self.use_struct else 0
        d += len(SIGNAL_TOKENS) if self.use_signal else 0
        d += len(self.shortcuts) if self.use_shortcut else 0
        return max(1, d)

    def transform_one(self, rec: dict) -> np.ndarray:
        counts = Counter(normalize_source(rec["source"]).split())
        # 结构化 token 采用子串匹配（源码里含 call.value 等）
        raw = rec["source"]
        parts = []
        if self.use_struct:
            parts.extend(float(counts.get(t, 0)) for t in STRUCT_TOKENS)
        if self.use_signal:
            parts.extend(float(raw.count(t)) for t in SIGNAL_TOKENS)
        if self.use_shortcut:
            parts.extend(1.0 if rec.get("shortcut") == s else 0.0 for s in self.shortcuts)
        return np.asarray(parts, dtype=np.float32) if parts else np.zeros(1, dtype=np.float32)

    def transform(self, records: list[dict]) -> np.ndarray:
        return np.vstack([self.transform_one(r) for r in records])


def featurize(train: list[dict], *others: list[dict],
              use_struct=True, use_signal=True, use_shortcut=True):
    """便捷函数：以 train 建立 shortcut 词表，转换所有 split。"""
    ex = FeatureExtractor(
        shortcuts=[r.get("shortcut") for r in train if r.get("shortcut")],
        use_struct=use_struct, use_signal=use_signal, use_shortcut=use_shortcut,
    )
    return ex, ex.transform(train), [ex.transform(o) for o in others]
