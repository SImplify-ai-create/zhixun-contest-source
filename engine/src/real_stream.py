"""真实公开数据集的数据流适配层。

与 ``src/dataset.py`` 的 ``SyntheticStream`` / ``OfficialStreamLoader`` 保持同一协议：

    stream.name          -> str
    stream.drift_points  -> list[int]
    stream.batches(bs)   -> Iterator[(offset, X, y)]

因此 ``SelfEvolutionPipeline.run(stream, ...)`` 无需任何改动即可消费真实数据。

当前实现：NASA C-MAPSS（涡扇发动机剩余寿命回归）。
设计取舍见 ``real_data/真实数据迁移规划.md`` §3。
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np

# C-MAPSS 无表头的 26 列
_OP_COLS = slice(2, 5)      # op1..op3 三个操作设定
_SENSOR_COLS = slice(5, 26)  # s1..s21
_RUL_CAP = 125              # 工业惯例：RUL 截断上限，避免早期寿命段标签无意义
_ROLL_WIN = 5               # 滚动均值窗口（实测 30 会引入噪声，末段 RMSE 反而 52→146）


class CMAPSSStream:
    """C-MAPSS 单子集（FD001~FD004）的按"代"推进数据流。

    批次构造（关键设计，全文见规划文档 §3.2）：

    - 把所有 train 单元按 ``cycle`` 对齐成**共同时钟**：第 e 代 = 所有寿命 ≥ e 的
      单元在第 e 个循环的读数。这样"批"的推进方向 = 全机队退化程度加深，
      时间不可逆，且不会出现"单元重启导致标签跳回满寿命"的伪漂移。
    - 目标 ``y = min(life - cycle, 125)``，即剩余寿命。
    - 特征 = 24 列原始量（3 设定 + 21 传感器）+ 24 列 5 代滚动均值 = 48 维。
    - 标准化统计量**只在初始训练窗口上拟合**，之后冻結；这样后段协变量漂移
      不会被全局标准化抹平（否则漂移检测必然失效）。

    诚实声明：C-MAPSS 没有人工注入的概念漂移，本流的"漂移"来自
    ① 退化进度推进（协变量漂移）② FD002/FD004 的 6 种操作条件切换。
    这两类都是真实工业退化现象，但不是"P(Y|X) 突变"意义的概念漂移。
    """

    def __init__(self, data_dir: str | Path, subset: str = "FD001",
                 max_units: int | None = None, strict_era: bool = True,
                 reserve_units_every: int = 0):
        self.data_dir = Path(data_dir)
        self.subset = subset.upper()
        self.name = f"cmapss-{self.subset.lower()}"
        self.drift_points: list[int] = []   # 无人工注入漂移点
        self._max_units = max_units
        self._strict_era = strict_era
        # A3 抗遗忘回测：每隔 N 个单元抽走 1 个，整条寿命都不进流，
        # 抽走的样本存 reserved_* 作"从未参与任何训练"的回测集。
        # 0/1 = 关闭（默认），主流数据与逐批结果逐位不变。
        self._reserve_every = int(reserve_units_every)
        self.reserved_X: np.ndarray | None = None
        self.reserved_y: np.ndarray | None = None
        self.reserved_unit: np.ndarray | None = None
        self.reserved_era: np.ndarray | None = None
        self.X, self.y, self.unit_ids, self.cycles = self._build()

    # ------------------------- 数据装配 -------------------------

    def _read_train(self) -> np.ndarray:
        f = self.data_dir / f"train_{self.subset}.txt"
        if not f.exists():
            raise FileNotFoundError(
                f"找不到 {f}；请先按 real_data/数据清单.md §2 解压 C-MAPSS 到 {self.data_dir}")
        return np.loadtxt(f)

    def _rows(self, units, by_unit, lives, max_era):
        """按"代"铺样本（代优先、单元升序），max_era=None 表示各用各自寿命。"""
        era_end = (max(lives[u] for u in units) if max_era is None
                   else max_era)
        feat_raw, feat_roll, rows_y, rows_u, rows_c = [], [], [], [], []
        for era in range(1, era_end + 1):
            for u in units:
                v = by_unit[u]
                if len(v) < era:
                    continue
                row = v[era - 1]
                start = max(0, era - _ROLL_WIN)
                win = v[start:era]
                raw = np.concatenate([row[_OP_COLS], row[_SENSOR_COLS]])
                roll = np.concatenate([
                    win[:, _OP_COLS].mean(axis=0),
                    win[:, _SENSOR_COLS].mean(axis=0),
                ])
                feat_raw.append(raw)
                feat_roll.append(roll)
                rows_y.append(min(lives[u] - era, _RUL_CAP))
                rows_u.append(u)
                rows_c.append(era)
        X = np.hstack([np.asarray(feat_raw), np.asarray(feat_roll)])
        y = np.asarray(rows_y, dtype=np.float64)
        return X, y, np.asarray(rows_u), np.asarray(rows_c)

    def _build(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        arr = self._read_train()
        units = np.unique(arr[:, 0]).astype(int)
        if self._max_units:
            units = units[: self._max_units]

        # 逐单元算寿命，并按 cycle 建立"代"索引
        by_unit = {u: arr[arr[:, 0] == u] for u in units}
        lives = {u: len(v) for u, v in by_unit.items()}
        # 预先把每单元按 cycle 排好（原始文件已是升序，这里只做保险）
        for u in units:
            by_unit[u] = by_unit[u][np.argsort(by_unit[u][:, 1])]

        # A3 抗遗忘：按步长抽走一部分单元，整条寿命都不进流
        if self._reserve_every > 1:
            keep = [u for i, u in enumerate(units) if i % self._reserve_every != 0]
            held = [u for i, u in enumerate(units) if i % self._reserve_every == 0]
        else:
            keep, held = list(units), []
        units = np.asarray(keep, dtype=int)

        # 批内机队构成必须稳定：把"代"截断到最短寿命，保证每一代都有全部单元参与。
        # 否则末段只剩极少数长寿命单元，RMSE 被尾部外推主导，批与批之间不可比。
        max_era = (min(lives[u] for u in units) if self._strict_era
                   else max(lives[u] for u in units))

        X, y, u_ids, c_ids = self._rows(units, by_unit, lives, max_era)
        if held:
            # 回测集保留整条寿命，调用方再按"代"截取早期窗口
            (self.reserved_X, self.reserved_y,
             self.reserved_unit, self.reserved_era) = self._rows(
                np.asarray(held, dtype=int), by_unit, lives, None)
        return X, y, u_ids, c_ids

    # ------------------------- 流协议 -------------------------

    def batches(self, batch_size: int = 200) -> Iterator[tuple]:
        X, y = self.X, self.y
        for i in range(0, len(y), batch_size):
            yield i, X[i:i + batch_size], y[i:i + batch_size]


class StandardizedStream:
    """用**前 N 行**拟合 z-score，之后全程冻结的包装器。

    存在的理由：全局标准化会把后段漂移抹平，让"漂移检测"变成假命题。
    只允许用部署初期可见的统计量，是线上真实约束。
    """

    def __init__(self, inner, fit_rows: int):
        self._inner = inner
        self.name = inner.name
        self.drift_points = inner.drift_points
        X = inner.X
        head = X[: min(fit_rows, len(X))]
        self._mu = head.mean(axis=0)
        sd = head.std(axis=0)
        self._sd = np.where(sd < 1e-9, 1.0, sd)
        self.X = (X - self._mu) / self._sd
        self.y = inner.y
        # A3 回测集必须用同一套冻结标准化，否则量纲与模型输入不一致
        if getattr(inner, "reserved_X", None) is not None:
            self.reserved_X = (inner.reserved_X - self._mu) / self._sd
            self.reserved_y = inner.reserved_y
            self.reserved_era = getattr(inner, "reserved_era", None)
            self.reserved_unit = getattr(inner, "reserved_unit", None)
            self.reserved_pos = getattr(inner, "reserved_pos", None)
        # 供 A3 回测脚本统计"训练流单元数 / 回测单元数"
        self.unit_ids = getattr(inner, "unit_ids", None)

    def batches(self, batch_size: int = 200) -> Iterator[tuple]:
        for i in range(0, len(self.y), batch_size):
            yield i, self.X[i:i + batch_size], self.y[i:i + batch_size]


def load_cmapss(data_dir: str | Path, subset: str = "FD001",
                init_batches: int = 3, batch_size: int = 200,
                max_units: int | None = None,
                strict_era: bool = True,
                reserve_units_every: int = 0) -> CMAPSSStream:
    """构造已冻结标准化的 C-MAPSS 流（供管线直接消费）。"""
    raw = CMAPSSStream(data_dir, subset=subset, max_units=max_units,
                       strict_era=strict_era,
                       reserve_units_every=reserve_units_every)
    return StandardizedStream(raw, fit_rows=init_batches * batch_size)


# ============================ UCI 气体传感器阵列漂移 ============================

_GAS_DIM = 128  # 每个样本 128 个传感器响应（LibSVM 稀疏格式 1:v1 ... 128:v128）
_GAS_CLASSES = {1: "Ethanol", 2: "Ethylene", 3: "Ammonia",
                4: "Acetaldehyde", 5: "Acetone", 6: "Toluene"}


class GasStream:
    """UCI 气体传感器阵列漂移（6 类 / 128 维 / 10 批）。

    批次构造（关键设计，见规划文档 §3.3）：

    - **一个 batch 文件 = 一批**，不做任何重切。UCI 的 10 个 batch 是 10 次
      不同时间段的采集（传感器老化 + 环境变化），天然就是"时间不可逆"的
      漂移序列；按样本数切碎反而会破坏这一语义。
    - 因此批大小**不固定**（batch4 只有 161 条，batch7 有 3613 条）。管线的
      ``batch_size`` 参数在这里只用于"重训窗口取多少样本"，不是物理批大小。
    - 标准化统计量只用**首批**拟合后冻结，理由同 C-MAPSS：全局标准化会把
      传感器漂移抹平，检测器变成假命题。
    - 标签保持原始 1~6（不转 0-based），避免"argmax 列号当类别号"那类错误。

    诚实声明：batch3/4/5 本身不含类别 6，macro-F1 只在该批真实出现的类别上
    平均（见 task_adapter.ClassificationAdapter）；否则会把数据缺口记成模型错误。
    """

    def __init__(self, data_dir: str | Path, fit_batches: int = 1,
                 standardize: bool = True, reserve_batch1_frac: float = 0.0):
        import re

        self.data_dir = Path(data_dir)
        self.name = "gas-uci-drift"
        self.drift_points: list[int] = []  # 漂移由采集时间自然产生，无人工注入点
        files = sorted(
            (p for p in self.data_dir.glob("batch*.dat")),
            key=lambda p: int(re.search(r"batch(\d+)", p.name).group(1)))
        if not files:
            raise FileNotFoundError(
                f"{self.data_dir} 下没有 batch*.dat；"
                "请先按 real_data/数据清单.md §1 解压 UCI 气体数据集")

        Xs, ys = [], []
        for f in files:
            X, y = self._read_libsvm(f)
            Xs.append(X)
            ys.append(y)
        self.batch_names = [f.stem for f in files]

        # A3 抗遗忘：从漂移前稳定批（batch1）按固定步长抽走一部分样本做回测集。
        # 抽走的样本不进数据流；标准化统计量也只用留下的样本拟合。
        self.reserved_X: np.ndarray | None = None
        self.reserved_y: np.ndarray | None = None
        raw_held = raw_held_y = None
        if reserve_batch1_frac > 0:
            if not 0 < reserve_batch1_frac < 0.5:
                raise ValueError("reserve_batch1_frac 须在 (0, 0.5) 内")
            stride = max(2, int(round(1.0 / reserve_batch1_frac)))
            idx = np.arange(len(ys[0]))
            held, keep = idx[idx % stride == 0], idx[idx % stride != 0]
            raw_held, raw_held_y = Xs[0][held], ys[0][held]
            Xs[0], ys[0] = Xs[0][keep], ys[0][keep]

        if standardize:
            head = np.vstack(Xs[:fit_batches])
            self.mu = head.mean(axis=0)
            sd = head.std(axis=0)
            self.sd = np.where(sd < 1e-9, 1.0, sd)
            Xs = [(X - self.mu) / self.sd for X in Xs]
            if raw_held is not None:
                raw_held = (raw_held - self.mu) / self.sd
        if raw_held is not None:
            self.reserved_X, self.reserved_y = raw_held, raw_held_y

        self.X_list, self.y_list = Xs, ys
        self.batch_sizes = [len(y) for y in ys]
        self.X = np.vstack(Xs)
        self.y = np.concatenate(ys)

    @staticmethod
    def _read_libsvm(path: Path) -> tuple[np.ndarray, np.ndarray]:
        """解析 LibSVM 稀疏行：``label 1:v1 2:v2 ... 128:v128``。"""
        rows = []
        labels = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                t = line.split()
                if not t:
                    continue
                labels.append(int(float(t[0])))
                row = np.zeros(_GAS_DIM, dtype=np.float64)
                for kv in t[1:]:
                    k, _, v = kv.partition(":")
                    row[int(k) - 1] = float(v)
                rows.append(row)
        return np.asarray(rows), np.asarray(labels, dtype=np.int64)

    def batches(self, batch_size: int = 200) -> Iterator[tuple]:
        off = 0
        for X, y in zip(self.X_list, self.y_list):
            yield off, X, y
            off += len(y)


def load_gas(data_dir: str | Path, fit_batches: int = 1,
             reserve_batch1_frac: float = 0.0) -> GasStream:
    """构造已冻结标准化的气体传感器流（供管线直接消费）。"""
    return GasStream(data_dir, fit_batches=fit_batches,
                     reserve_batch1_frac=reserve_batch1_frac)


# ======================= XJTU-SY 轴承全寿命振动 =======================
#
# 官方口径（《XJTU-SY 滚动轴承加速寿命试验数据集解读》，机械工程学报 2019, 55(16)：
# 表2 试验工况、表3 数据集信息一览表、§1.2 数据采集、§3.2 失效阈值）：
#   - 采样频率 25.6 kHz，采样间隔 1 min，每次采样 1.28 s → 每条 32768 点 × 2 通道
#   - 每通道一个 csv；第一列水平、第二列竖直；文件按时间命名 1.csv … N.csv
#   - 三工况：1=2100 r/min(35 Hz)/12 kN；2=2250 r/min(37.5 Hz)/11 kN；3=2400 r/min(40 Hz)/10 kN
#   - **没有独立的失效时间标注文件**：试验在最大幅值超过 10×Ah（Ah=正常阶段最大幅值）
#     时立刻终止，因此**最后一个 csv 就是失效时刻**，N 即全寿命采集次数。
#     标签只能由 N 反推（详见 BearingStream 的标签说明）。

_XJTU_FS = 25600          # 采样频率 Hz（官方 25.6 kHz）
_XJTU_POINTS = 32768      # 每次采样点数 = 1.28 s × 25.6 kHz
_XJTU_INTERVAL_MIN = 1.0  # 采样间隔（分钟），也是"文件序号 → 运行时长"的换算系数

_XJTU_CONDITIONS = {
    1: {"rpm": 2100, "load_kN": 12, "bearings": ("1_1", "1_2", "1_3", "1_4", "1_5")},
    2: {"rpm": 2250, "load_kN": 11, "bearings": ("2_1", "2_2", "2_3", "2_4", "2_5")},
    3: {"rpm": 2400, "load_kN": 10, "bearings": ("3_1", "3_2", "3_3", "3_4", "3_5")},
}

# 官方表3：样本总数 / 实际寿命 / 失效位置。只用于解压后的完整性校验，不参与建模。
_XJTU_BEARINGS = {
    "1_1": {"files": 123,  "life": "2 h 3 min",  "fault": "外圈"},
    "1_2": {"files": 161,  "life": "2 h 41 min", "fault": "外圈"},
    "1_3": {"files": 158,  "life": "2 h 38 min", "fault": "外圈"},
    "1_4": {"files": 122,  "life": "2 h 2 min",  "fault": "保持架"},
    "1_5": {"files": 52,   "life": "52 min",     "fault": "内圈、外圈"},
    "2_1": {"files": 491,  "life": "8 h 11 min", "fault": "内圈"},
    "2_2": {"files": 161,  "life": "2 h 41 min", "fault": "外圈"},
    "2_3": {"files": 533,  "life": "8 h 53 min", "fault": "保持架"},
    "2_4": {"files": 42,   "life": "42 min",     "fault": "外圈"},
    "2_5": {"files": 339,  "life": "5 h 39 min", "fault": "外圈"},
    "3_1": {"files": 2538, "life": "42 h 18 min", "fault": "外圈"},
    "3_2": {"files": 2496, "life": "41 h 36 min", "fault": "内圈、滚动体、保持架、外圈"},
    "3_3": {"files": 371,  "life": "6 h 11 min",  "fault": "内圈"},
    "3_4": {"files": 1515, "life": "25 h 15 min", "fault": "内圈"},
    "3_5": {"files": 114,  "life": "1 h 54 min",  "fault": "外圈"},
}

# 默认特征提取配置（外置可配置，见 BearingStream 参数 feature_cfg）
XJTU_FEATURE_CFG = {
    "channels": (0, 1),        # 0=水平（载荷方向，退化信息更多），1=竖直
    "high_band_hz": 5000.0,    # 高频能量占比分界（轴承故障冲击集中在 5 kHz 以上）
    "env_band_hz": 2.0,        # 包络谱取峰容差（覆盖 ±1% 转速抖动）
    "env_harmonics": 3,        # 故障特征频率取前 3 阶谐波的最大幅值
    "roll_win": 5,             # 特征滑动平均窗口（0=关闭）；单次 1.28 s 采样抖动大
}

_TIME_FEATS = ("mean", "std", "rms", "peak", "p2p", "skew", "kurt",
               "crest", "impulse", "clearance", "shape")
_FREQ_FEATS = ("spec_centroid", "spec_entropy", "dom_freq",
               "high_band_ratio", "spec_kurt")
_ENV_FEATS = ("env_peak_ratio", "env_entropy", "bpfo_ratio", "bpfi_ratio")


def bearing_fault_orders():
    """由官方轴承几何（表1：8 滚珠、滚珠 7.92 mm、中径 34.55 mm、接触角 0°）
    推外圈/内圈故障特征频率相对转频的阶次。

    BPFO = n/2·(1-d/D) = 3.083，BPFI = n/2·(1+d/D) = 4.917。
    与官方解读 §2.2 给出的外圈故障频率 107.91 Hz（转频 35 Hz）一致：35×3.083=107.9。
    """
    n_balls, ball_d, pitch_d, contact = 8, 7.92, 34.55, 0.0
    r = ball_d / pitch_d * np.cos(np.radians(contact))
    return n_balls / 2.0 * (1.0 - r), n_balls / 2.0 * (1.0 + r)


def _band_peak(amp: np.ndarray, freqs: np.ndarray, f0: float, band: float) -> float:
    """包络谱在 f0±band 内的最大幅值（f0 超出频谱范围时返回该范围边界值 0）。"""
    lo = np.searchsorted(freqs, f0 - band)
    hi = np.searchsorted(freqs, f0 + band, side="right")
    if hi <= lo:
        return 0.0
    return float(amp[lo:hi].max())


def _time_feats(v: np.ndarray) -> list[float]:
    mu = float(v.mean())
    sd = float(v.std())
    absv = np.abs(v)
    rms = float(np.sqrt(np.mean(v * v)))
    peak = float(absv.max())
    mean_abs = float(absv.mean()) + 1e-12
    sd_safe = max(sd, 1e-12)
    z = (v - mu) / sd_safe
    return [
        mu, sd, rms, peak, float(v.max() - v.min()),
        float(np.mean(z ** 3)), float(np.mean(z ** 4)),
        peak / (rms + 1e-12),                       # 峰值因子
        peak / mean_abs,                            # 脉冲因子
        peak / (float(np.sqrt(absv).mean()) ** 2 + 1e-12),  # 裕度因子
        rms / mean_abs,                              # 波形因子
    ]


def _freq_feats(v: np.ndarray, fs: float, cfg: dict) -> list[float]:
    n = len(v)
    w = np.hanning(n)
    amp = np.abs(np.fft.rfft(v * w)) / max(float(w.sum()), 1e-12)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    psd = amp ** 2
    psd[0] = 0.0                                    # 去直流（传感器零漂）
    total = float(psd.sum()) + 1e-12
    p = psd / total
    nyq = fs / 2.0
    centroid = float((freqs * p).sum()) / nyq
    entropy = float(-(p * np.log(p + 1e-20)).sum()) / np.log(len(p))
    dom = float(freqs[1:][int(np.argmax(psd[1:]))]) / nyq
    high = float(psd[freqs >= cfg["high_band_hz"]].sum()) / total
    # 谱峭度（近似）：幅度谱的归一化 4 阶矩。**不是** STFT-SK 的严格定义，
    # 这里只需要一个"冲击成分是否集中"的标量，取 log1p 压缩量纲。
    a1 = amp[1:]
    am, asd = float(a1.mean()), max(float(a1.std()), 1e-12)
    sk = float(np.mean(((a1 - am) / asd) ** 4))
    return [centroid, entropy, dom, high, float(np.log1p(sk))]


def _env_feats(v: np.ndarray, fs: float, shaft_hz: float, cfg: dict) -> list[float]:
    from scipy.signal import hilbert

    env = np.abs(hilbert(v))
    env = env - env.mean()
    n = len(env)
    w = np.hanning(n)
    amp = np.abs(np.fft.rfft(env * w)) / max(float(w.sum()), 1e-12)
    freqs = np.fft.rfftfreq(n, 1.0 / fs)
    amp[0] = 0.0
    a1 = amp[1:]
    mean_a = float(a1.mean()) + 1e-12
    peak_ratio = float(a1.max()) / mean_a
    p = a1 ** 2
    p = p / (float(p.sum()) + 1e-12)
    entropy = float(-(p * np.log(p + 1e-20)).sum()) / np.log(len(p))
    bpfo_k, bpfi_k = bearing_fault_orders()
    band = float(cfg["env_band_hz"])
    bpfo = max(_band_peak(amp, freqs, k * bpfo_k * shaft_hz, band)
               for k in range(1, int(cfg["env_harmonics"]) + 1)) / mean_a
    bpfi = max(_band_peak(amp, freqs, k * bpfi_k * shaft_hz, band)
               for k in range(1, int(cfg["env_harmonics"]) + 1)) / mean_a
    return [peak_ratio, entropy, bpfo, bpfi]


def read_signal_csv(path: Path) -> np.ndarray:
    """读一个官方 csv（32768 行 × 2 列）。

    官方文件首行是通道名表头 ``Horizontal_vibration_signals,Vertical_vibration_signals``，
    第 2 行起才是数值；按首行首个字段能否转 float 判断是否跳行，缺表头的变体也兼容。
    """
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        head = f.readline()
    try:
        float(head.split(",")[0].strip())
        skip = 0
    except ValueError:
        skip = 1
    x = np.loadtxt(path, delimiter=",", skiprows=skip)
    if x.ndim == 1:                     # 单行/单列文件的兜底（理论上不应出现）
        x = x.reshape(-1, 1)
    return x


def _extract_one(path: Path, fs: float, shaft_hz: float, cfg: dict) -> np.ndarray:
    x = read_signal_csv(path)
    out: list[float] = []
    for c in cfg["channels"]:
        v = np.asarray(x[:, c], dtype=np.float64)
        out += _time_feats(v)
        out += _freq_feats(v, fs, cfg)
        out += _env_feats(v, fs, shaft_hz, cfg)
    return np.asarray(out, dtype=np.float64)


def _bearing_feature_names(cfg: dict) -> list[str]:
    names: list[str] = []
    for c in cfg["channels"]:
        tag = "h" if c == 0 else "v"      # h=水平 v=竖直
        names += [f"{tag}_{n}" for n in _TIME_FEATS + _FREQ_FEATS + _ENV_FEATS]
    if int(cfg.get("roll_win", 0)) > 0:
        names = names + [f"{n}_roll" for n in names]
    return names


def _resolve_bearing_dir(root: Path, bearing: str) -> Path:
    """容忍官方的几种目录写法（Bearing1_1 / Bearing1-1 / Bearing 1_1），
    并允许 root 比数据高一层（解压后常见是 .../Data/XJTU-SY_Bearing_Datasets/）。"""
    variants = [f"Bearing{bearing}", f"Bearing{bearing.replace('_', '-')}",
                f"Bearing {bearing}", f"Bearing {bearing.replace('_', '-')}"]
    cands: list[Path] = []
    for v in variants:
        cands += [root / v, *(root.glob(f"*/{v}")), *(root.glob(f"*/*/{v}"))]
    for c in cands:
        if c.is_dir() and any(c.glob("*.csv")):
            return c
    raise FileNotFoundError(
        f"在 {root} 下找不到 Bearing{bearing} 的 csv。"
        "若数据仍是分卷 rar，请先解压到 real_data\\xjtu_sy\\Bearing{}_x\\ 。"
        .format(bearing))


def xjtu_available_bearings(root: str | Path) -> list[str]:
    """列出本地实际可读的 Bearing 编号（用于解压后的完整性核对）。"""
    root = Path(root)
    found = []
    for b in _XJTU_BEARINGS:
        try:
            _resolve_bearing_dir(root, b)
            found.append(b)
        except FileNotFoundError:
            continue
    return found


class BearingStream:
    """XJTU-SY 单个轴承的全寿命振动流（回归任务）。

    批次构造（与气体/C-MAPSS 的差异，见规划文档 §2.3）：
    - **一行 = 一个 csv（一次 1.28 s 采样）的特征向量**；
    - **一批 = 连续 ``batch_size`` 个 csv**（用户要求的"固定数量连续采集文件"）；
      因此一个轴承的总批数 = ceil(N / batch_size)，N 由官方表3 决定（如 3_1 = 2538）。
    - 特征 = 2 通道 × (11 时域 + 5 频域 + 4 包络谱) = 40 维，再拼 5 点滑动平均 = 80 维。
    - 标准化统计量只用**最初若干批**拟合并冻结（``load_bearing`` 用 StandardizedStream），
      理由同 C-MAPSS / 气体：全局标准化会把后段退化漂移抹平，检测器变成假命题。

    标签口径（**本数据集最大的风险点，必须写清楚**）：
    官方只提供全寿命振动，没有逐点标签；试验在"最大幅值 > 10×Ah"时立即终止，
    所以**最后一个 csv 即失效时刻**，N 就是全寿命采集次数。据此把标签定义为
    剩余寿命（与 C-MAPSS 的 RUL 同源），并给两种刻度：

    - ``label_mode="rul_ratio"``（默认）：``y = (N - i) / (N - 1)``，i 从 0 起，
      首条 = 1.0、末条 = 0.0。**用比例而不是绝对分钟**，是因为 15 个轴承的实际寿命
      差 60 倍（42 min ~ 42 h18min），绝对 RMSE 会被长寿命轴承主导、跨轴承不可平均。
    - ``label_mode="rul_min"``：``y = (N - 1 - i) × 1 min``，绝对剩余寿命（分钟）。

    三条必须在对外材料中声明的局限：
    ① 标签需要知道 N（失效时刻）才能构造 → **只有离线回放可用，线上不可得**；
       这不是特征泄漏（X 里没有任何未来信息），但属于"事后口径"，不能对外说成在线预测。
    ② 单轴承时序切分下，模型拟合的是一条退化轨迹本身的形状，**不是跨个体泛化**；
       验证跨轴承泛化要留一轴承，本轮不做（样本只有 15 条轨迹）。
    ③ 1_5 / 2_4 / 3_2 等属于"突然失效型"：失效前信号无可见前兆，
       任何基于振动统计量的模型都无法在末尾几批预测，RMSE 必然集中在尾部；
       这是数据性质，不是方法缺陷，但会造成"末段 RMSE 难看"，需在结果表里注明。
    """

    def __init__(self, data_dir: str | Path, bearing: str = "1_1",
                 label_mode: str = "rul_ratio",
                 feature_cfg: dict | None = None,
                 limit_files: int | None = None,
                 use_cache: bool = True,
                 reserve_file_stride: int = 0):
        if bearing not in _XJTU_BEARINGS:
            raise ValueError(f"未知轴承编号 {bearing!r}；可选：{sorted(_XJTU_BEARINGS)}")
        if label_mode not in ("rul_ratio", "rul_min"):
            raise ValueError(f"label_mode 只支持 rul_ratio / rul_min，收到 {label_mode!r}")

        self.data_dir = Path(data_dir)
        self.bearing = bearing
        self.label_mode = label_mode
        self.condition = int(bearing.split("_")[0])
        cond = _XJTU_CONDITIONS[self.condition]
        self.shaft_hz = cond["rpm"] / 60.0          # 35 / 37.5 / 40 Hz
        self.load_kN = cond["load_kN"]
        self.name = f"xjtu-sy-{bearing}"
        self.drift_points: list[int] = []           # 退化自然推进，无人工注入点
        self.official_files = _XJTU_BEARINGS[bearing]["files"]
        self.feature_cfg = dict(XJTU_FEATURE_CFG)
        if feature_cfg:
            self.feature_cfg.update(feature_cfg)
        self.feature_names = _bearing_feature_names(self.feature_cfg)

        bdir = _resolve_bearing_dir(self.data_dir, bearing)
        files = sorted(bdir.glob("*.csv"), key=lambda p: int(p.stem))
        if limit_files:
            files = files[: int(limit_files)]
        self.n_files = len(files)
        self.bearing_dir = bdir

        X_raw = self._load_or_extract(files, use_cache=use_cache)
        X = self._append_roll(X_raw)
        y_all = self._labels(self.n_files, label_mode)

        # A3 抗遗忘：按固定步长抽走一部分 csv（整行特征），这些样本不进数据流。
        # 滚动均值特征在全序列上先算好再切，保证保留样本的特征与主流口径一致；
        # 标签始终按**原始**文件序号相对真实失效时刻计算，抽走不影响刻度。
        self.reserved_X: np.ndarray | None = None
        self.reserved_y: np.ndarray | None = None
        self.reserved_pos: np.ndarray | None = None
        if int(reserve_file_stride) > 1:
            idx = np.arange(self.n_files)
            held = idx[idx % int(reserve_file_stride) == 0]
            keep = idx[idx % int(reserve_file_stride) != 0]
            self.reserved_X, self.reserved_y = X[held], y_all[held]
            self.reserved_pos = held
            X, y_all = X[keep], y_all[keep]

        self.X = X
        self.y = y_all
        self.y_unit = "life_ratio" if label_mode == "rul_ratio" else "minutes"

    # ------------------------- 特征 -------------------------

    def _cache_path(self, files: list[Path]) -> Path:
        import hashlib

        h = hashlib.md5()
        h.update(repr(sorted(self.feature_cfg.items())).encode())
        head = f"{self.bearing}|{self.shaft_hz}|{len(files)}|{files[0].name}|{files[-1].name}"
        h.update(head.encode())
        h.update(str(files[0].stat().st_size).encode())
        h.update(str(files[-1].stat().st_size).encode())
        cache_dir = Path(__file__).resolve().parent.parent / "runs" / "_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"bearing_{self.bearing}_{h.hexdigest()[:12]}.npz"

    def _load_or_extract(self, files: list[Path], use_cache: bool) -> np.ndarray:
        if use_cache and files:
            cp = self._cache_path(files)
            if cp.exists():
                return np.load(cp)["X"]
        rows = []
        for k, f in enumerate(files):
            rows.append(_extract_one(f, _XJTU_FS, self.shaft_hz, self.feature_cfg))
            if (k + 1) % 200 == 0:
                print(f"  [bearing {self.bearing}] 特征提取 {k + 1}/{len(files)}")
        X = np.asarray(rows, dtype=np.float64)
        if use_cache and files:
            np.savez_compressed(self._cache_path(files), X=X)
        return X

    def _append_roll(self, X: np.ndarray) -> np.ndarray:
        win = int(self.feature_cfg.get("roll_win", 0))
        if win <= 1 or not len(X):
            return X
        cs = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), X]), axis=0)
        idx = np.arange(len(X))
        lo = np.maximum(0, idx - win + 1)
        roll = (cs[idx + 1] - cs[lo]) / (idx + 1 - lo)[:, None]
        return np.hstack([X, roll])

    @staticmethod
    def _labels(n: int, mode: str) -> np.ndarray:
        i = np.arange(n)
        rem = (n - 1 - i).astype(np.float64)      # 末端=0，即失效
        if mode == "rul_ratio":
            return rem / max(n - 1, 1)
        return rem * _XJTU_INTERVAL_MIN

    # ------------------------- 流协议 -------------------------

    def batches(self, batch_size: int = 50) -> Iterator[tuple]:
        for i in range(0, len(self.y), batch_size):
            yield i, self.X[i:i + batch_size], self.y[i:i + batch_size]
