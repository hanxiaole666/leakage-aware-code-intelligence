"""真实 PyTorch 训练路径（可选，Anaconda 环境：torch2.2.1 + transformers + peft）。

目的：把 M0/M1 的 sklearn 代理升级为"真实可跑的 LoRA 微调循环 + 断点续跑"，
使框架具备切换到真实 3B 模型的能力。诚实边界：

* ``--smoke``   —— 用**随机初始化的小 Transformer + 合成克隆族数据**在 RTX2060 上真实跑
  LoRA 训练与续跑。验证的是"训练/存档/恢复"工程正确性，非模型性能。
* ``--model <name_or_path>`` —— 真实基座（如 Qwen2.5-Coder-3B）。bitsandbytes 可用时用
  NF4 4bit QLoRA；否则自动降级 FP16 LoRA（不装 bnb 也能跑）。需本地已有权重或可联网下载。

断点续跑：每个 epoch 原子写 checkpoint（LoRA 权重 + optimizer + RNG + epoch/step），
中断后加 ``--resume`` 从上次 epoch 继续；配置签名变化自动作废旧 ckpt。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .config import CKPT_DIR, RESULTS_ROOT, SEEDS, TASK_LABELS
from .data.synthesize import load_records
from .models.features import FeatureExtractor
from .models.checkpoint import signature

# ---------- 可选依赖探测 ----------

def _has_bnb() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _peft():
    from peft import LoraConfig, get_peft_model
    return LoraConfig, get_peft_model


# ---------- smoke 用小型分类器 ----------

class SmallTransformer(nn.Module):
    """随机初始化小模型（仅用于验证真实 LoRA 训练管线，非研究结论）。"""
    def __init__(self, vocab_dim: int, n_classes: int, d_model: int = 64, nhead: int = 4):
        super().__init__()
        self.proj = nn.Linear(vocab_dim, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                           dim_feedforward=128, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, num_layers=2)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x=None, input_ids=None, **kwargs):
        # PEFT FEATURE_EXTRACTION 会以 input_ids= 传入；兼容两种调用
        if x is None:
            x = input_ids
        h = self.proj(x)
        h = self.enc(h.unsqueeze(1)).squeeze(1)
        return self.head(h)


# ---------- 数据 ----------

def _tensors(records, ex: FeatureExtractor):
    X = torch.tensor(ex.transform(records), dtype=torch.float32)
    y = torch.tensor([TASK_LABELS.index(r["label"]) for r in records], dtype=torch.long)
    return X, y


# ---------- checkpoint（torch 状态，二进制） ----------

class TorchCheckpoint:
    def __init__(self, run_id: str, sig: str):
        self.sig = sig
        self.path = CKPT_DIR / f"{run_id}_torch.pt"
        self.tmp = CKPT_DIR / f"{run_id}_torch.pt.tmp"

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        try:
            state = torch.load(self.path, map_location="cpu", weights_only=False)
        except Exception:
            return None
        if state.get("sig") != self.sig:
            return None
        return state

    def save(self, payload: dict) -> None:
        payload = dict(payload)
        payload["sig"] = self.sig
        torch.save(payload, self.tmp)
        os.replace(self.tmp, self.path)


def _seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- 训练主循环 ----------

def train_smoke(epochs: int = 6, batch_size: int = 32, seed: int = SEEDS[0],
                resume: bool = False, device: str | None = None) -> dict:
    """在合成数据上真实跑 LoRA 微调 + 断点续跑（RTX2060/CPU 均可）。"""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    recs = load_records()
    from collections import defaultdict
    by_label: dict = defaultdict(list)
    for r in recs:
        by_label[r["label"]].append(r)
    # 每类抽 4 个克隆族，严格协议（无 shortcut）以贴近真实泛化
    sub = []
    for lab, rows in by_label.items():
        fams = sorted({r["family"] for r in rows})[:4]
        sub += [r for r in rows if r["family"] in fams]
    random.Random(seed).shuffle(sub)
    n_tr = int(0.8 * len(sub))
    ex = FeatureExtractor(shortcuts=[], use_struct=True, use_signal=True, use_shortcut=False)
    X, y = _tensors(sub, ex)
    Xtr, ytr, Xva, yva = X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

    LoraConfig, get_peft_model = _peft()
    base = SmallTransformer(ex.dim, len(TASK_LABELS)).to(device)
    lcfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1,
                      target_modules=["proj", "head"], task_type="FEATURE_EXTRACTION")
    model = get_peft_model(base, lcfg).to(device)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    # signature 只含"数据/模型结构不变量"，不含目标 epochs——
    # 否则把 --epochs 从 3 提到 6 会被误判为配置变更而丢弃 ckpt，无法续跑更多轮。
    sig = signature("qlora_smoke", batch_size, seed, ex.dim, len(sub), "v1")
    ckpt = TorchCheckpoint("qlora_smoke", sig)
    start_epoch = 0
    history: list[dict] = []
    if resume:
        st = ckpt.load()
        if st:
            start_epoch = st["epoch"]
            model.load_state_dict(st["model"])
            opt.load_state_dict(st["opt"])
            for k, v in st["rng"].items():
                if k == "python": random.setstate(v)
                elif k == "numpy": np.random.set_state(v)
                elif k == "torch": torch.set_rng_state(v)
            history = st.get("history", [])
    _seed_everything(seed)

    model.train()
    for ep in range(start_epoch, epochs):
        perm = torch.randperm(len(Xtr))
        total, nb = 0.0, 0
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            xb, yb = Xtr[idx].to(device), ytr[idx].to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item(); nb += 1
        model.eval()
        with torch.no_grad():
            pred = model(Xva.to(device)).argmax(1).cpu()
            acc = float((pred == yva).float().mean())
        model.train()
        history.append({"epoch": ep, "train_loss": round(total / max(1, nb), 4),
                        "val_acc": round(acc, 4)})
        ckpt.save({
            "epoch": ep + 1, "model": model.state_dict(), "opt": opt.state_dict(),
            "history": history,
            "rng": {"python": random.getstate(), "numpy": np.random.get_state(),
                    "torch": torch.get_rng_state()},
        })
        print(f"[epoch {ep+1}/{epochs}] loss={total/max(1,nb):.4f} val_acc={acc:.4f}")

    out = RESULTS_ROOT / "qlora_smoke_result.json"
    out.write_text(json.dumps({
        "mode": "smoke_random_init_lora", "device": device, "epochs": epochs,
        "n_train": len(Xtr), "n_val": len(Xva), "history": history,
        "resumed_from_epoch": start_epoch,
        "final_val_acc": history[-1]["val_acc"] if history else None,
        "bnb_available": _has_bnb(),
        "honest_note": "验证真实 LoRA 训练/续跑工程正确性；非 3B 模型性能结论",
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"history": history, "final_val_acc": history[-1]["val_acc"] if history else None,
            "result_file": str(out)}


def train_real(model_name: str, epochs: int = 1, seed: int = SEEDS[0],
               resume: bool = False, max_steps: int | None = None) -> dict:
    """真实基座 LoRA/QLoRA 训练（需权重可用）。nf4 需 bitsandbytes，否则 fp16。"""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    bnb = _has_bnb()
    kw = {}
    if bnb:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16)
    else:
        kw["torch_dtype"] = torch.float16
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(TASK_LABELS), **kw)
    LoraConfig, get_peft_model = _peft()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj", "v_proj"], task_type="SEQ_CLS")
    model = get_peft_model(model, lcfg)
    # 训练数据/循环略：接入在线标注数据后沿用 train_smoke 的 checkpoint 模式
    # 此处返回配置摘要，证明真实路径的参数化正确且不伪造训练结果
    return {
        "model": model_name, "quantization": "nf4_4bit" if bnb else "fp16_lora_fallback",
        "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "total_params": sum(p.numel() for p in model.parameters()),
        "note": "真实基座加载成功；训练循环需接入真实标注集后启用（未伪造结果）",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description="真实 LoRA/QLoRA 训练路径")
    p.add_argument("--smoke", action="store_true", help="用随机小模型在合成数据上真跑 LoRA+续跑")
    p.add_argument("--model", default=None, help="真实基座名称/路径（需权重可用）")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--seed", type=int, default=SEEDS[0])
    p.add_argument("--resume", action="store_true", help="从 checkpoint 续跑")
    p.add_argument("--device", default=None)
    a = p.parse_args(argv)
    if a.model:
        print(json.dumps(train_real(a.model, a.epochs, a.seed, a.resume), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(train_smoke(a.epochs, seed=a.seed, resume=a.resume, device=a.device),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
