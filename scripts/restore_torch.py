"""从 conda pkgs 缓存用硬链接恢复 torch（不额外占磁盘空间）。

背景：pip install bitsandbytes 误将 CUDA 版 torch 覆盖；直接 cp 会因空间不足失败，
硬链接到 conda 缓存的原始数据则几乎不占空间。
"""
import os
import shutil

SRC = r"C:\Users\apple\anaconda3\pkgs\pytorch-2.2.1-py3.11_cuda11.8_cudnn8_0\Lib\site-packages\torch"
DST = r"C:\Users\apple\anaconda3\Lib\site-packages\torch"
INFO_SRC = r"C:\Users\apple\anaconda3\pkgs\pytorch-2.2.1-py3.11_cuda11.8_cudnn8_0\info"
INFO_DST = r"C:\Users\apple\anaconda3\Lib\site-packages\torch-2.2.1.dist-info"

stat = {"linked": 0, "copied": 0, "dirs": 0}


def link_tree(s, d):
    os.makedirs(d, exist_ok=True)
    for name in os.listdir(s):
        sp, dp = os.path.join(s, name), os.path.join(d, name)
        if os.path.isdir(sp):
            link_tree(sp, dp)
            stat["dirs"] += 1
        elif not os.path.exists(dp):
            try:
                os.link(sp, dp)
                stat["linked"] += 1
            except OSError:
                shutil.copy2(sp, dp)
                stat["copied"] += 1


# 目标目录应已由前面步骤清空；若残留空壳则直接复用（link_tree 会跳过已存在项）
os.makedirs(DST, exist_ok=True)
link_tree(SRC, DST)
os.makedirs(INFO_DST, exist_ok=True)
for name in os.listdir(INFO_SRC):
    sp, dp = os.path.join(INFO_SRC, name), os.path.join(INFO_DST, name)
    if os.path.isdir(sp):
        shutil.copytree(sp, dp, dirs_exist_ok=True)
    elif not os.path.exists(dp):
        try:
            os.link(sp, dp)
        except OSError:
            shutil.copy2(sp, dp)

t, u, f = shutil.disk_usage("C:/")
print("DONE", stat, "free_GB", round(f / 1e9, 2))
