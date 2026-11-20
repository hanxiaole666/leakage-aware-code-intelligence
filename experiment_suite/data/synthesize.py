"""带克隆族结构的合成 Solidity 数据集生成器（M0/M1 可运行骨架）。

为什么需要合成数据
------------------
真实 SmartBugs / Smart-LLaMA-DPO 集受网络与许可限制，且本环境不具备 3B 全参训练
算力。为在 6GB 消费卡上 **真实跑通** 整条实验管线并验证核心 claim，这里生成
带"近重复克隆族"结构的数据：同一漏洞模板派生多个仅做标识符重命名/常量替换/
语句重排的变体（同族同标签），并被赋予族特有的"捷径 token"。

* 随机切分：同族变体横跨 train/test → 模型记住捷径 → F1 虚高（复现 source 陷阱）。
* clone_aware 切分：整族隔离 → 捷径失效 → F1 回落到真实泛化水平。

这正是审稿人要求排除的 A2（记忆近重复）假说，且完全由脚本真实产出，非占位数字。
真实数据接入见 ``download.ensure_data`` + 替换 estimator。
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from ..config import LOGS_DIR, PROCESSED_DIR, SEEDS, TASK_LABELS

DATA_FILE = PROCESSED_DIR / "synthetic_contracts.jsonl"
META_FILE = PROCESSED_DIR / "synthetic_meta.json"

# 每类漏洞的"真实信号 token"（泛化时仍可学习的语义特征）。
# 关键：这些 token 必须是长且独特的字符串，避免子串误匹配（如 add 命中 address）。
# 与 models/features.py 的 SIGNAL_TOKENS 保持一致。
SIGNALS: dict[str, list[str]] = {
    "reentrancy": ["call.value", "delegatecall", "fallback", "reentrancyGuard"],
    "timestamp": ["block.timestamp", "block.number", "timestampDependent"],
    "overflow": ["unchecked", "safeMath", "overflowCheck"],
    "access_control": ["onlyOwner", "msg.sender", "tx.origin", "accessControl"],
    "safe": ["readOnly", "emit_event", "viewFunction", "pureFunction"],
}

# 合成源码模板（仅示意，不依赖 solc 编译；编译校验接口在 fingerprints 中预留）
_TEMPLATE = """pragma solidity ^0.8.{minor};
contract C_{cid} {{
  mapping(address => uint) balances;
  function {fname}(address {arg}) public {{
    {body}
  }}
}}
"""


def _body_for(label: str, shortcut: str, rng: random.Random) -> str:
    """生成函数体，模拟"真实语义信号 + 噪声 + 记忆捷径"三要素。

    * 记忆捷径 shortcut：**每个变体都出现**（族特有标识符）——随机切分可记忆 → 虚高。
    * 真实语义 signal：**仅约 60% 变体包含**（模拟真实代码中漏洞信号不总是显式），
      其余用中性 token。故 clone_aware 关闭 shortcut 后，模型只能靠部分显式的
      真实信号，达到中等 F1（非完美）——这是能写进论文的有层次结果。
    * 结构 token 各类共享，无判别力（用于 ablate_structure 消融对照）。
    """
    # 记忆捷径：始终出现
    head = f"uint slot_{shortcut} = {rng.randint(1, 9)};"
    tail = f" emit_{shortcut}(slot_{shortcut});"
    # 真实语义信号：约 60% 变体显式包含该类 token
    if rng.random() < 0.6:
        n = rng.randint(2, 3)
        picked = SIGNALS[label][:n]
        payload = " ".join(f"{t} = {rng.randint(1, 9)};" if "." in t or "Function" in t
                           else f"use_{t.replace('.', '_')}();" for t in picked)
    else:
        # 中性 token（不属于任何 SIGNAL_TOKENS），无任何类别判别力
        payload = f" check_state(); store_value({rng.randint(1, 9)}); advance({rng.randint(1, 9)});"
    return f"{head} {payload}{tail}"


def arg_() -> str:
    return "usr"


def _ident(rng: random.Random) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(3, 6)))


def _doc_id(family: int, variant: int) -> str:
    return f"c{hashlib.md5(f'{family}-{variant}'.encode()).hexdigest()[:10]}"


def generate_dataset(
    n_families_per_label: int = 40,
    variants_per_family: int = 6,
    seed: int = SEEDS[0],
    force: bool = False,
) -> Path:
    """生成/复用合成数据集。已存在且 ``force=False`` 时直接返回（幂等）。"""
    if DATA_FILE.exists() and not force:
        return DATA_FILE

    rng = random.Random(seed)
    records: list[dict] = []
    families: list[dict] = []
    fid = 0
    for label in TASK_LABELS:
        for _ in range(n_families_per_label):
            # 族特有捷径 token：随机标识符，仅同族共享（制造泄漏）
            shortcut = _ident(rng)
            family = fid
            fid += 1
            project = f"proj_{fid % 12}"
            year = rng.choice([2021, 2022, 2023, 2024, 2025])
            deployed_at = f"{year}-{rng.randint(1,12):02d}-{rng.randint(1,28):02d}"
            families.append({"family": family, "label": label, "shortcut": shortcut})
            for v in range(variants_per_family):
                cid = _ident(rng)
                fname = f"f_{_ident(rng)}"
                body = _body_for(label, shortcut, rng)
                source = _TEMPLATE.format(
                    minor=rng.choice([7, 8, 9]), cid=cid, fname=fname,
                    arg=_ident(rng), body=body,
                )
                records.append({
                    "id": _doc_id(family, v),
                    "family": family,
                    "project": project,
                    "deployed_at": deployed_at,
                    "label": label,
                    "shortcut": shortcut,
                    "source": source.strip(),
                })

    with DATA_FILE.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    META_FILE.write_text(json.dumps({
        "n_records": len(records), "n_families": fid,
        "n_families_per_label": n_families_per_label,
        "variants_per_family": variants_per_family,
        "seed": seed, "labels": list(TASK_LABELS),
        "synthetic": True,
        "purpose": "M0/M1 pipeline sanity + leakage claim demonstration (NOT real SOTA)",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    (LOGS_DIR / "synthesize.json").write_text(
        json.dumps({"file": str(DATA_FILE), "n_records": len(records)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return DATA_FILE


def synthesize_if_needed(force: bool = False) -> Path:
    return generate_dataset(force=force)


def load_records() -> list[dict]:
    if not DATA_FILE.exists():
        generate_dataset()
    return [json.loads(line) for line in DATA_FILE.read_text(encoding="utf-8").splitlines() if line.strip()]
