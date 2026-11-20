"""模型层：特征化、可运行 estimator 与断点续跑训练器。

真实 3B QLoRA 训练需要 8×H800 / 大模型权重；本骨架用 sklearn 线性/树模型
代理"漏洞信号学习器"，使全流程在 CPU/6GB 卡上可跑、可测、可续。
接入真实模型只需实现同一 ``fit/predict/predict_proba`` 接口。
"""
from .features import featurize, FeatureExtractor
from .estimators import build_estimator, clone_with_variant
from .checkpoint import Checkpoint
from .trainer import Trainer

__all__ = ["featurize", "FeatureExtractor", "build_estimator", "clone_with_variant",
           "Checkpoint", "Trainer"]
