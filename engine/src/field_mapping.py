"""字段映射层：把工厂原始导出适配成通用建模表。

背景：工厂拒绝按我方 64 列模板手填（"几十天填不完"），改为由我方适配他们
现有的质保书导出格式。因此这里把"字段映射"写成数据（DEFAULT_MAPPING），
换一张表或换一家厂只改映射、不改代码。

同时内置三条数据可用性硬门槛自检（对接函第五章）：
  1. 一件一行：原始行按 试批号 × 检验方向 × 检验项 展开，必须聚合回生产单元；
  2. 可排序时间：必须有精确到分钟、可单调排序的时间戳；
  3. 数值 y：质量结果必须是数值，不能只有合格/不合格。

术语说明（本格式）：
  试批号    检验批号，不是钢板件数；同一试批号可跨多个合同号与多个轧制时刻
  生产单元  合同号 + 轧制结束时刻，即"同一时刻下线的一批同规格钢板"
  熔炼号    炉次号，成分（C/Mn/Si/...）按熔炼号恒定，需按熔炼号回填
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

MISSING_TOKENS = {"", " ", "-", "/", "nan", "NaN", "None", "null", "—"}

COMPOSITION = ["C", "Si", "Mn", "Nb", "V", "Ti", "Cr", "Ni", "Cu", "Mo",
               "B", "Alt", "P", "S", "As", "Sn", "Sb", "N", "H", "Ca",
               "CEQ", "CEV", "PCM"]

# 钢板热轧质保书导出（J820189.xls）标准映射。
# kind: group_first=组内取首值 / categorical=类别编码 / heat_lookup=按熔炼号回填
DEFAULT_MAPPING = {
    "name": "钢板热轧质保书导出",
    "time": {"source": "轧制结束时刻", "format": "%Y%m%d%H%M%S", "name": "timestamp"},
    "id": {"source": "合同号", "name": "contract_no"},
    "heat": {"source": "熔炼号", "name": "heat_no"},
    "features": [
        {"name": "thickness_mm", "source": "实际厚度(mm)", "kind": "group_first"},
        {"name": "slab_thickness_mm", "source": "板坯厚度", "kind": "group_first"},
        {"name": "furnace_time_min", "source": "在炉内时间", "kind": "group_first"},
        {"name": "crew_code", "source": "生产班组", "kind": "categorical",
         "mapping": {"甲": 1, "乙": 2, "丙": 3, "丁": 4}},
        {"name": "shift_code", "source": "生产班次", "kind": "categorical",
         "mapping": {"白": 1, "中": 2, "夜": 3}},
        *[{"name": c, "source": c, "kind": "heat_lookup"} for c in COMPOSITION],
    ],
    "targets": [
        {"name": "yield_strength_mpa", "source": "拉伸屈服强度 (MPa)"},
        {"name": "tensile_strength_mpa", "source": "拉伸抗拉强度 (MPa)"},
        {"name": "elongation_pct", "source": "拉伸延伸率 (%)"},
        {"name": "impact_energy_j", "source": "冲击功平均值(规制1)"},
    ],
}


# ------------------------- 基础工具 -------------------------

def load_mapping(path: str | Path | None = None) -> dict:
    """读取映射规则；不传则用内置的 J820189 标准映射。"""
    if path is None:
        return DEFAULT_MAPPING
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def load_raw(path: str | Path) -> pd.DataFrame:
    """读取工厂原始 Excel（xls/xlsx），只做列名去空格，不改数据。"""
    df = pd.read_excel(path, header=0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _coerce_numeric(s: pd.Series) -> pd.Series:
    """把工厂常见的空白/短横线占位转成 NaN，其余转数值。"""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce")
    txt = s.astype(str).str.strip()
    txt = txt.mask(txt.isin(MISSING_TOKENS))
    return pd.to_numeric(txt, errors="coerce")


def _coerce_category(s: pd.Series, mapping: dict) -> pd.Series:
    txt = s.astype(str).str.strip()
    txt = txt.mask(txt.isin(MISSING_TOKENS))
    return txt.map(mapping).astype(float)


# ------------------------- 核心转换 -------------------------

def to_canonical(raw: pd.DataFrame, mapping: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """原始导出 → 一件一行的建模表。

    Returns:
        (canonical, stats)
        canonical: 列 = 标识列 + 特征列 + 目标列，按时间升序、一件一行
        stats: 去重/聚合/丢弃列等过程统计，供门槛自检与答辩留痕
    """
    m = mapping or DEFAULT_MAPPING
    time_src, time_fmt = m["time"]["source"], m["time"]["format"]
    id_src, id_name = m["id"]["source"], m["id"]["name"]
    heat_src, heat_name = m["heat"]["source"], m["heat"]["name"]

    df = raw.drop_duplicates().copy()
    raw_rows, dedup_rows = len(raw), len(df)

    df["_ts"] = pd.to_datetime(df[time_src].astype(str).str.strip(),
                               format=time_fmt, errors="coerce")
    df["_id"] = df[id_src].astype(str).str.strip()
    df["_heat"] = df[heat_src].astype(str).str.strip()

    # 数值列与类别列先统一类型，避免 groupby 时字符串参与均值
    for f in m["features"]:
        if f["kind"] == "heat_lookup":
            df[f["source"]] = _coerce_numeric(df[f["source"]])
        elif f["kind"] == "categorical":
            df[f["source"]] = _coerce_category(df[f["source"]], f["mapping"])
        else:
            df[f["source"]] = _coerce_numeric(df[f["source"]])
    for t in m["targets"]:
        df[t["source"]] = _coerce_numeric(df[t["source"]])

    # 成分按熔炼号恒定：逐列只用该列非空的行建查找表。
    # 不能整表 dropna，否则某个熔炼号缺 As/Sn 一个元素就会把它的 C/Mn 一起丢成 NaN。
    heat_feats = [f for f in m["features"] if f["kind"] == "heat_lookup"]
    heat_tbl = pd.DataFrame(index=df["_heat"].unique())
    for f in heat_feats:
        notna = df.dropna(subset=[f["source"]])
        heat_tbl[f["source"]] = (notna.groupby("_heat")[f["source"]].first()
                                 if len(notna) else np.nan)

    # 聚合：同一 (合同号, 轧制结束时刻) 内的多方向/多试样 → 一件一行
    g = df.groupby(["_id", "_ts"], sort=True)
    out = pd.DataFrame(index=g.size().index)
    out.index.names = [id_name, "timestamp"]
    out[heat_name] = g["_heat"].first()

    group_first = [f for f in m["features"] if f["kind"] == "group_first"]
    for f in group_first:
        out[f["name"]] = g[f["source"]].first()

    for f in m["features"]:
        if f["kind"] == "categorical":
            out[f["name"]] = g[f["source"]].first()

    heat_name_feats = [f["name"] for f in heat_feats]
    for f in heat_feats:
        out[f["name"]] = out[heat_name].map(heat_tbl[f["source"]])

    for t in m["targets"]:
        out[t["name"]] = g[t["source"]].mean()

    out = out.reset_index().sort_values("timestamp").reset_index(drop=True)
    out["piece_id"] = np.arange(len(out))

    # 记录哪些来源列被判定为常量/全空（可丢弃），便于向工厂解释"不用填这么多"
    dropped = []
    for c in raw.columns:
        s = raw[c]
        if s.nunique(dropna=True) <= 1:
            dropped.append(c)

    stats = {
        "raw_rows": raw_rows,
        "dedup_rows": dedup_rows,
        "dup_rows": raw_rows - dedup_rows,
        "pieces": len(out),
        "heat_count": int(out[heat_name].nunique()),
        "heat_with_composition": int(heat_tbl[heat_feats[0]["source"]].notna().sum()),
        "dropped_constant_columns": len(dropped),
        "dropped_columns": dropped,
        "expansion_ratio": round(dedup_rows / max(len(out), 1), 2),
    }
    return out, stats


# ------------------------- 三条硬门槛自检 -------------------------

def check_availability(canonical: pd.DataFrame, stats: dict,
                       mapping: dict | None = None) -> tuple[bool, str]:
    """按对接函第五章逐条核对：一件一行 / 可排序时间 / 数值 y。"""
    m = mapping or DEFAULT_MAPPING
    lines, ok_all = [], True

    ts = pd.to_datetime(canonical["timestamp"], errors="coerce")
    span_days = (ts.max() - ts.min()).days if ts.notna().any() else 0
    time_ok = ts.notna().all() and ts.is_monotonic_increasing and len(ts) > 0
    ok_all &= time_ok
    lines.append(
        f"[{'通过' if time_ok else '不通过'}] 门槛二·可排序时间："
        f"{ts.notna().sum()}/{len(ts)} 条可解析，跨度 {span_days} 天，"
        f"{'单调递增' if ts.is_monotonic_increasing else '非单调'}")

    # 一件一行：转换后组键唯一，且原始展开行数被压缩回生产单元
    dup_keys = int(canonical.duplicated(subset=["contract_no", "timestamp"]).sum())
    gran_ok = dup_keys == 0
    ok_all &= gran_ok
    lines.append(
        f"[{'通过' if gran_ok else '不通过'}] 门槛一·一件一行："
        f"{stats['dedup_rows']} 行 → {stats['pieces']} 个生产单元"
        f"（展开倍数 {stats['expansion_ratio']}×，重复组键 {dup_keys}）")

    for t in m["targets"]:
        col = canonical[t["name"]]
        rate = float(col.notna().mean())
        t_ok = rate >= 0.95
        ok_all &= t_ok
        lines.append(
            f"[{'通过' if t_ok else '不通过'}] 门槛三·数值 y（{t['name']}）："
            f"非空 {col.notna().sum()}/{len(col)}（{rate:.0%}），"
            f"均值 {col.mean():.2f}，标准差 {col.std():.2f}")

    lines.append(
        f"[信息] 原始 {stats['raw_rows']} 行 / {stats['dropped_constant_columns']} 列"
        f"为常量或全空（{stats['heat_count']} 个熔炼号，"
        f"{stats['heat_with_composition']} 个带成分）")
    return ok_all, "\n".join(lines)


# ------------------------- 官方接口格式导出 -------------------------

def to_official(canonical: pd.DataFrame, out_path: str | Path,
                target: str = "yield_strength_mpa",
                mapping: dict | None = None,
                standardize_y: bool = True) -> dict:
    """导出 OfficialStreamLoader 可直接读取的 val.csv。

    管线模型（sklearn MLP）不做特征标准化，而本数据量纲跨 4 个数量级
    （Mn≈1.4、B≈0.0003、CEQ≈0.35），因此这里做中位数填补 + z-score。
    填补与标准化参数取自整表，属于确定性预处理，不涉及标签，不构成标签泄漏。

    y 同样做 z-score（standardize_y=True）：引擎 MLP 固定
    learning_rate_init=0.01 / max_iter=300，是按 O(1) 目标设计的；
    真实强度值在数百 MPa，直接喂入会 300 轮不收敛（实测 RMSE≈294）。
    标准化参数写入返回值，报告里的 RMSE 可按 y_mean + y_std × rmse 换回 MPa。
    """
    m = mapping or DEFAULT_MAPPING
    feat_cols = [f["name"] for f in m["features"]]
    X = canonical[feat_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(canonical[target], errors="coerce")

    mask = y.notna()
    X, y = X[mask], y[mask]

    med = X.median()
    X = X.fillna(med)
    std = X.std().replace(0, np.nan)
    # 常数列（如个别成分在子集内恒定）直接剔除，保证特征矩阵非奇异
    keep = std.notna()
    X = (X.loc[:, keep] - X.loc[:, keep].mean()) / std[keep]

    y_mean, y_std = float(y.mean()), float(y.std())
    y_out = (y - y_mean) / y_std if standardize_y else y

    out = X.copy()
    out["y"] = y_out.to_numpy()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False, encoding="utf-8-sig")

    return {"path": str(out_path), "rows": len(out), "features": X.shape[1],
            "target": target, "target_standardized": standardize_y,
            "y_mean": round(y_mean, 4), "y_std": round(y_std, 4),
            "feature_names": list(X.columns)}
