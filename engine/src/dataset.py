"""数据流接入层：合成数据与官方数据包共用同一接口。

管线（src/pipeline.py）只依赖三个成员，任何新数据源实现这三个成员即可无缝接入：

    stream.name          -> str                      数据流名称，写入实验记录
    stream.drift_points  -> list[int]                真实漂移点（官方数据未知，返回空）
    stream.batches(bs)   -> Iterator[(offset, X, y)] 按批迭代，X 为二维特征、y 为一维标签

设计意图：官方公开集（训练集/验证集）11 月初随评测平台开放后，
只需把 config.json 的 data.source 改成 "official"，管线代码零改动。
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterator

import numpy as np

from .data import SyntheticDriftStream

# 官方数据包中可能的文件名（不含扩展名），按优先级查找
_SPLIT_ALIASES = ("val", "validation", "dev", "test", "train")
_SUPPORTED_EXT = (".npz", ".npy", ".csv")
_TARGET_ALIASES = ("y", "target", "label", "Y", "Target", "Label")


# ------------------------- 合成数据 -------------------------

class SyntheticStream:
    """合成漂移数据流适配器（保持原 SyntheticDriftStream 的批迭代语义）。"""

    def __init__(self, drift_type: str = "sudden", **kwargs):
        self._stream = SyntheticDriftStream(drift_type=drift_type, **kwargs)
        self.name = f"synthetic-{drift_type}"
        self.drift_points = list(self._stream.drift_points)

    def batches(self, batch_size: int = 200) -> Iterator[tuple]:
        return self._stream.batches(batch_size)


# ------------------------- 官方数据包 -------------------------

class OfficialStreamLoader:
    """官方数据包加载器。

    支持三种文件格式，按 val → validation → dev → test → train 的顺序自动查找：

    - .npz  键名 X / y（不区分大小写）；无明确键时取前两个数组
    - .npy  末列视为标签，其余列视为特征
    - .csv  表头中名为 y/target/label 的列作为标签，找不到则取最后一列

    说明：只加载带标签的公开集。隐藏测试流由评测平台内部使用，本地不可见。
    """

    def __init__(self, data_dir: str | Path, split: str = "val",
                 target_col: str = "y"):
        self.data_dir = Path(data_dir)
        self.split = split
        self.target_col = target_col
        self.name = f"official-{split}"
        self.drift_points: list[int] = []
        self.X, self.y = self._load()

    # ---- 查找与解析 ----

    def _find_file(self) -> Path:
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"官方数据目录不存在：{self.data_dir}\n"
                f"请把官方数据包解压到该目录，或用 --data-dir 指定实际路径。"
            )
        # 1) 精确匹配指定 split
        for alias in (self.split, *_SPLIT_ALIASES):
            for ext in _SUPPORTED_EXT:
                p = self.data_dir / f"{alias}{ext}"
                if p.exists():
                    return p
        # 2) 目录下任意受支持文件（只有一个时）
        files = [p for p in self.data_dir.iterdir()
                 if p.suffix.lower() in _SUPPORTED_EXT]
        if len(files) == 1:
            return files[0]
        if not files:
            raise FileNotFoundError(
                f"{self.data_dir} 下没有找到数据文件（支持 {'/'.join(_SUPPORTED_EXT)}）"
            )
        raise FileNotFoundError(
            f"{self.data_dir} 下有多个数据文件，无法自动选择："
            f"{[p.name for p in files]}；请用 --split 指定。"
        )

    def _load(self) -> tuple[np.ndarray, np.ndarray]:
        f = self._find_file()
        ext = f.suffix.lower()
        if ext == ".npz":
            X, y = self._load_npz(f)
        elif ext == ".npy":
            arr = np.load(f)
            X, y = arr[:, :-1], arr[:, -1]
        elif ext == ".csv":
            X, y = self._load_csv(f)
        else:
            raise ValueError(f"不支持的文件格式：{ext}")
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        if X.ndim != 2 or len(X) != len(y):
            raise ValueError(f"数据形状不一致：X{X.shape} y{y.shape}")
        print(f"[dataset] 已加载官方数据 {f.name}：{X.shape[0]} 样本 / {X.shape[1]} 特征")
        return X, y

    @staticmethod
    def _load_npz(f: Path) -> tuple[np.ndarray, np.ndarray]:
        d = np.load(f, allow_pickle=False)
        keys = list(d.keys())
        lower = {k.lower(): k for k in keys}
        if "x" in lower and ("y" in lower or "target" in lower):
            yk = lower.get("y") or lower.get("target")
            return d[lower["x"]], d[yk]
        if len(keys) < 2:
            raise ValueError(f"{f.name} 只包含 {keys}，无法拆分为特征与标签")
        return d[keys[0]], d[keys[1]]

    def _load_csv(self, f: Path) -> tuple[np.ndarray, np.ndarray]:
        with open(f, newline="", encoding="utf-8-sig") as fp:
            reader = csv.reader(fp)
            header = next(reader)
            rows = [r for r in reader if r]
        cols = [c.strip() for c in header]
        idx = None
        for name in (self.target_col, *_TARGET_ALIASES):
            if name in cols:
                idx = cols.index(name)
                break
        if idx is None:
            idx = len(cols) - 1
        X = [[float(v) for j, v in enumerate(r) if j != idx] for r in rows]
        y = [float(r[idx]) for r in rows]
        return np.asarray(X), np.asarray(y)

    def batches(self, batch_size: int = 200) -> Iterator[tuple]:
        for i in range(0, len(self.y), batch_size):
            yield i, self.X[i:i + batch_size], self.y[i:i + batch_size]


# ------------------------- 工厂函数 -------------------------

def load_stream(source: str = "synthetic", **kwargs):
    """按配置构造数据流。source: "synthetic" | "official" """
    source = (source or "synthetic").lower()
    if source == "synthetic":
        allowed = {k: v for k, v in kwargs.items()
                   if k in ("drift_type", "n_samples", "n_features",
                            "drift_points", "mean_shift", "noise", "seed")}
        return SyntheticStream(**allowed)
    if source == "official":
        return OfficialStreamLoader(
            data_dir=kwargs.get("data_dir", "data/official"),
            split=kwargs.get("split", "val"),
            target_col=kwargs.get("target_col", "y"),
        )
    raise ValueError(f"未知数据源：{source}（可选 synthetic / official）")
