"""可恢复训练器：逐 seed 推进，每个 seed 完成即存 ckpt，中断后可续跑。

真实 3B 训练时 ``_fit_one`` 内部换成 QLoRA 的 epoch/step 循环并在每 N step
调用 ``ckpt.save`` 即可；本骨架在 seed 粒度存档，已足够演示与验证续跑正确性。
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from ..config import SEEDS
from .checkpoint import Checkpoint, signature
from .estimators import build_estimator


class Trainer:
    def __init__(self, run_id: str, X_train, y_train, X_val, y_val,
                 variant: str, seeds=SEEDS, extra_sig=None):
        self.run_id = run_id
        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.variant = variant
        self.seeds = list(seeds)
        self.sig = signature(run_id, variant, np.shape(X_train), np.shape(X_val),
                             seeds, extra_sig or {}, self._data_hash())
        self.ckpt = Checkpoint(run_id, self.sig)

    def _data_hash(self) -> str:
        return signature(self.X_train[:5].tolist() if hasattr(self.X_train, "tolist") else str(self.X_train)[:200])

    def _fit_one(self, seed: int):
        est = build_estimator(self.variant, seed)
        est.fit(self.X_train, self.y_train)
        pred = est.predict(self.X_val)
        f1 = float(f1_score(self.y_val, pred, average="macro", zero_division=0))
        return est, f1

    def run(self) -> dict:
        """返回 {seed: {f1, model}}；从上次完成的 seed 之后继续。"""
        state = self.ckpt.load() or {}
        done: dict = state.get("models_f1", {})
        models: dict = {}
        for seed in self.seeds:
            key = str(seed)
            if key in done:
                continue  # 已完成 → 跳过（断点续跑）
            est, f1 = self._fit_one(seed)
            done[key] = round(f1, 4)
            models[seed] = est
            # 只存可序列化进度；模型对象在真实训练中存 adapter 路径
            self.ckpt.save({"models_f1": done, "next_seed_index": self.seeds.index(seed) + 1})
        return {"f1_by_seed": done, "resumed_from": state.get("next_seed_index", 0),
                "signature": self.sig}
