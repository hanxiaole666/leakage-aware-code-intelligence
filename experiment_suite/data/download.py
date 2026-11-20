"""幂等数据下载器：无数据自动下载，有数据直接续跑（skip-if-exists）。

行为契约
--------
* ``ensure_data(name)``：
  - 若 ``RAW_DIR/<path>`` 已存在且（若有 sha256）校验通过 → 直接返回该路径，**不联网**。
  - 否则下载到 ``<path>.part``，校验 sha256 后原子 ``os.replace`` 为最终文件；
    下载中断残留的 ``.part`` 下次实现 **断点续传**（HTTP Range）。
  - 网络失败 / 无网络 → 返回 ``None``，由上层回退到合成数据，并在 provenance 记录。
* 绝不删除已有文件；重复调用是幂等的。
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

from ..config import DATASET_REGISTRY, LOGS_DIR, RAW_DIR


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _verify(path: Path, expected: str | None) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    if expected and _sha256(path) != expected:
        return False
    return True


def _resume_download(url: str, dest: Path, log: dict) -> Path:
    """支持 Range 断点续传；失败保留 .part 供下次续跑。"""
    part = dest.with_suffix(dest.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    req = urllib.request.Request(url)
    if offset:
        req.add_header("Range", f"bytes={offset}-")
    log["url"] = url
    log["resume_from_bytes"] = offset
    try:
        with urllib.request.urlopen(req, timeout=25) as resp, part.open("ab") as out:
            mode = "ab" if offset and resp.status == 206 else "wb"
            if mode == "wb":
                out.close()
                part.write_bytes(b"")
                out = part.open("ab")
                log["resume_from_bytes"] = 0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(part, dest)
        log["status"] = "downloaded"
    except Exception as exc:  # noqa: BLE001 - 网络失败属预期分支
        log["status"] = "download_failed"
        log["error"] = f"{type(exc).__name__}: {exc}"
        if not part.exists() or part.stat().st_size == 0:
            try:
                dest.write_bytes(b"")  # 留空标记，避免无限重试；由 provenance 记录
            except Exception:
                pass
    return dest


def ensure_data(name: str) -> Path | None:
    """确保数据集就绪。返回本地路径；不可获得时返回 None（回退合成）。"""
    spec = DATASET_REGISTRY.get(name)
    if spec is None:
        return None
    dest = RAW_DIR / spec["path"]
    log: dict = {"dataset": name, "path": str(dest)}
    if _verify(dest, spec.get("sha256")):
        log["status"] = "cache_hit_skip_download"
        _write_log(name, log)
        return dest
    if os.environ.get("SKIP_DOWNLOAD") == "1":
        log["status"] = "skipped_by_env"
        _write_log(name, log)
        return None
    _resume_download(spec["url"], dest, log)
    _write_log(name, log)
    if _verify(dest, spec.get("sha256")) and dest.stat().st_size > 0:
        return dest
    return None


def download_all() -> dict[str, Path | None]:
    return {name: ensure_data(name) for name in DATASET_REGISTRY}


def _write_log(name: str, log: dict) -> None:
    (LOGS_DIR / f"download_{name}.json").write_text(
        json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
