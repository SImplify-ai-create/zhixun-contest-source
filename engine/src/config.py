"""配置加载：config.json → 环境变量覆盖 → 命令行覆盖。

优先级：命令行参数 > 环境变量 > config.json > 内置默认值。
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

DEFAULT_CONFIG = {
    "data": {
        "source": "synthetic",          # synthetic | official
        "drift_type": "sudden",         # 仅合成数据生效：sudden | gradual
        "data_dir": "data/official",    # 仅官方数据生效
        "split": "val",
        "target_col": "y",
    },
    "stream": {
        "batch_size": 200,
        "init_batches": 3,
    },
    "pipeline": {
        "window_size": 600,
        "residual_factor": 3.0,
        "max_retrains": 10,
        "label_budget": 4000,
        "clean_keep_ratio_covariate": 0.7,
        "clean_keep_ratio_concept": 0.0,
    },
    "detector": {
        "cka_threshold": 0.85,
        "ks_alpha": 0.05,
    },
    "logging": {
        "project": "zhixun-engine",
        "out_dir": "runs",
        "experiment_name": None,
        "swanlab": True,                # False 或环境变量 SWANLAB_MODE=disabled 时只写本地 CSV
    },
}

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.json"


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path | None = None) -> dict:
    """读取 config.json（不存在则用默认值），并应用环境变量覆盖。"""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    p = Path(path) if path else DEFAULT_CONFIG_PATH
    if p.exists():
        with open(p, encoding="utf-8") as f:
            cfg = _deep_merge(cfg, json.load(f))
    else:
        print(f"[config] 未找到 {p}，使用内置默认配置")

    # 环境变量覆盖（评测环境常用）
    env = os.environ
    if env.get("ZHIXUN_DATA_SOURCE"):
        cfg["data"]["source"] = env["ZHIXUN_DATA_SOURCE"]
    if env.get("ZHIXUN_DATA_DIR"):
        cfg["data"]["data_dir"] = env["ZHIXUN_DATA_DIR"]
    if env.get("SWANLAB_MODE", "").lower() == "disabled":
        cfg["logging"]["swanlab"] = False
    if env.get("ZHIXUN_RUNS_DIR"):
        cfg["logging"]["out_dir"] = env["ZHIXUN_RUNS_DIR"]
    return cfg
