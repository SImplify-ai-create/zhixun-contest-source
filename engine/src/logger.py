"""实验记录：优先写入 SwanLab 云端（赛题指定平台），不可用时降级为本地 CSV。

赛题要求"全部训练、微调、调整与评估实验须按《实验记录规范》在云端实验记录平台
完整记录"——SwanLab 校赛阶段即可免费注册使用。

启用云端的方式（任选其一）：
    1. swanlab login                        （一次登录，凭据本地保存）
    2. 设置环境变量 SWANLAB_API_KEY
    3. config.json 里保持 logging.swanlab = true

强制只用本地 CSV：config.json 里设 logging.swanlab = false，
或设置环境变量 SWANLAB_MODE=disabled（评测环境的离线场景常用）。
"""
from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path

# 官方《实验记录规范》字段映射：官方列名 -> 本系统列名（或 {"const": 值} / {"eval": 表达式}）
# None 表示使用默认值：列名按原样输出（官方格式下发后只需在此填映射，不动代码逻辑）
OFFICIAL_COLUMN_MAP: dict[str, str | dict] | None = None
# {"const": 值}      固定常量（所有行同值，如 run_id、阶段、方法名）
# {"eval": "表达式"}  可引用 row（单行 dict）和 metrics（最后一行），如 "row.get('eval_rmse')"
# 映射缺列时保留原列名；多余列不删除（规范字段未公开前，先保证可重映射）


class ExperimentLogger:
    def __init__(self, project="zhixun-engine", experiment_name=None,
                 out_dir="runs", use_swanlab=True, config=None):
        self.rows = []
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%m%d-%H%M%S")
        self.run_name = experiment_name or "run"
        self.csv_path = out / f"{self.run_name}-{stamp}.csv"
        self.summary_path = out / f"{self.run_name}-{stamp}-summary.json"
        self.run_url = None
        self._swan = None

        if not use_swanlab:
            print("[logger] 已按配置关闭 SwanLab，仅写本地 CSV")
            return
        # 先检测登录状态：未登录且无 API Key 时不调用 swanlab.init，
        # 否则 0.10.x 会弹交互式菜单导致非交互环境下进程退出。
        if not self._swanlab_logged_in():
            print("[logger] SwanLab 未登录（无 ~/.swanlab/.netrc 且 SWANLAB_API_KEY 未设置），"
                  "降级为本地 CSV。登录后自动启用云端记录。")
            return
        try:
            import swanlab
            mode = os.environ.get("SWANLAB_MODE") or "cloud"
            swanlab.init(project=project, experiment_name=self.run_name,
                         config=config or {}, mode=mode)
            self._swan = swanlab
            self.run_url = self._resolve_url(swanlab)
            print(f"[logger] 已连接 SwanLab 云端实验记录：{self.run_url or '(URL 未知)'}")
        except Exception as e:
            print(f"[logger] SwanLab 不可用，降级为本地 CSV（{type(e).__name__}: {e}）")

    @staticmethod
    def _swanlab_logged_in() -> bool:
        """检测 SwanLab 是否已登录（有 netrc 凭据或 SWANLAB_API_KEY 环境变量）。"""
        if os.environ.get("SWANLAB_API_KEY"):
            return True
        try:
            from swanlab.sdk.cmd.utils import get_nrc_path
            from swanlab.sdk.internal.pkg import nrc
            return nrc.read(get_nrc_path("root")) is not None
        except Exception:
            return False

    @staticmethod
    def _resolve_url(swanlab) -> str | None:
        """尽量取到本次实验的可分享链接（不同版本 API 名称不一，失败则返回 None）。"""
        try:
            run = swanlab.get_run()
            for attr in ("url", "cloud_url"):
                if hasattr(run, attr):
                    return getattr(run, attr)
        except Exception:
            pass
        return None

    # ---- 记录 ----

    def log(self, metrics: dict, step: int | None = None):
        self.rows.append({"ts": time.strftime("%H:%M:%S"), **metrics})
        if self._swan:
            flat = {k: v for k, v in metrics.items()
                    if isinstance(v, (int, float, bool))}
            if not flat:
                return
            try:
                self._swan.log(flat, step=step) if step is not None \
                    else self._swan.log(flat)
            except TypeError:
                self._swan.log(flat)
            except Exception:
                pass

    # ---- 官方格式映射（实验记录规范字段，下发后只需改 OFFICIAL_COLUMN_MAP） ----

    def _apply_official_map(self, row: dict) -> dict:
        """按 OFFICIAL_COLUMN_MAP 重命名/重算列名。返回新 dict，不改原 row。"""
        if OFFICIAL_COLUMN_MAP is None:
            return row
        out = {}
        for official, src in OFFICIAL_COLUMN_MAP.items():
            if isinstance(src, str) and src in row:
                out[official] = row[src]
            elif isinstance(src, dict) and "const" in src:
                out[official] = src["const"]
            elif isinstance(src, dict) and "eval" in src:
                out[official] = eval(src["eval"], {"row": row, "metrics": self.rows[-1] if self.rows else {}}, {})
            else:
                out[official] = ""  # 映射指向的列不存在，留空不报错
        # 未映射的原始列追加在后，保证信息不丢
        for k, v in row.items():
            if k not in out:
                out[k] = v
        return out

    def close(self):
        rows = [self._apply_official_map(r) for r in self.rows]
        keys = sorted({k for r in rows for k in r})
        with open(self.csv_path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        with open(self.summary_path, "w", encoding="utf-8") as f:
            last_metric = next(
                (r for r in reversed(self.rows)
                 if r.get("eval_rmse") not in (None, "")
                 or r.get("eval_loss") not in (None, "")),
                self.rows[-1] if self.rows else {})
            compute = next((r for r in self.rows
                            if r.get("event") == "compute_summary"), {})
            json.dump({
                "run_name": self.run_name,
                "swanlab_url": self.run_url,
                "csv": str(self.csv_path),
                "n_rows": len(self.rows),
                "last": last_metric,
                "compute": compute,
            }, f, ensure_ascii=False, indent=2)
        print(f"[logger] 实验记录已保存：{self.csv_path}")
        if self.run_url:
            print(f"[logger] 云端记录链接：{self.run_url}")
        if self._swan:
            try:
                self._swan.finish()
            except Exception:
                pass
