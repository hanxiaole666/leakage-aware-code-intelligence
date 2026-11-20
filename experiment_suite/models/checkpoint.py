"""断点续跑检查点：原子写 + 状态校验 + 恢复。

满足用户硬需求："不小心关闭执行测试脚本时，重训时断点续跑"。

设计：
* 每完成一个可恢复单元（seed/epoch/variant）即原子落盘 JSON 状态到
  ``results/checkpoints/<run_id>.json``。
* ``Checkpoint.load()`` 校验 ``run_id`` 与 ``signature``（配置指纹）一致才恢复；
  配置变了则自动作废旧 ckpt，避免用旧状态污染新实验。
* 原子写：先写 ``.tmp`` 再 ``os.replace``，杜绝关闭脚本时产生半截文件。
* pipeline 级（run 是否完成）由 runner 的 ``pipeline_state.json`` 管理；
  本类管理 run **内部** 的细粒度进度。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

from ..config import CKPT_DIR, LOGS_DIR


def signature(*parts: Any) -> str:
    raw = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class Checkpoint:
    def __init__(self, run_id: str, sig: str):
        self.run_id = run_id
        self.sig = sig
        self.path = CKPT_DIR / f"{run_id}.json"
        self.tmp = CKPT_DIR / f"{run_id}.json.tmp"

    def load(self) -> dict | None:
        # 崩溃残留的 .tmp 不影响正式 ckpt；正式文件存在才尝试恢复
        if not self.path.exists():
            return None
        try:
            state = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - 损坏 ckpt 视为无 ckpt，安全重跑
            self._log("corrupt_ckpt_ignored")
            return None
        if state.get("signature") != self.sig:
            self._log("signature_mismatch_ignored")
            return None
        if state.get("run_id") != self.run_id:
            return None
        return state

    def save(self, payload: dict) -> None:
        state = {
            "run_id": self.run_id,
            "signature": self.sig,
            "payload": payload,
        }
        self.tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(self.tmp, self.path)  # 原子替换

    def clear(self) -> None:
        for p in (self.path, self.tmp):
            if p.exists():
                p.unlink()

    def _log(self, event: str) -> None:
        (LOGS_DIR / "checkpoint_events.log").open("a", encoding="utf-8").write(
            f"{self.run_id}\t{event}\n"
        )
