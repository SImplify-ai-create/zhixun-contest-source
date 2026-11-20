"""规则提案层：把历史证据整理成摘要，交给提案方写成可审计的数值规则。

设计约束：
  - 提案与验收严格分离。本模块只产出规则 JSON 与来源元数据，不做任何"好不好"的判断；
    是否采纳完全由 scripts/verify_rules.py 的数字闸门决定。
  - 提案方可以是 LLM（PyTRIO / 现有 LLMController 同款的 OpenAI 兼容接口），
    也可以是确定性离线启发式（在没有可用模型时仍能跑完整闭环）。
  - 记录 provider / model / tokens_in / tokens_out / latency / attempts / raw_sha1；
    凭据只从本机 profile 或环境变量读取，绝不写入任何产物。
  - 复用既有工具：决策点证据来自 pytrio_20261111/replay_decision.py 的 build_cases
    （按批从 real_stream 现算，不拼事件行 CSV）；标定表来自
    runs/action_calibration.json，用 src/decision.py 的 CostSensitiveDetector 查代价。
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import time
from datetime import datetime
from pathlib import Path

from .decision import CostSensitiveDetector
from .drift import DriftDetector
from .rule_program import (FEATURES, OPS, TIERS, compile_program,
                           validate_program)

REPO = Path(__file__).resolve().parent.parent
WORKSPACE = REPO.parent.parent           # 工作区根目录
PYTRIO_DIR = WORKSPACE / "pytrio_20261111"
REAL_DATA = WORKSPACE / "real_data"
DEFAULT_CALIB = REPO / "runs" / "action_calibration.json"

GATE_EVENTS = ("static_anchor_reject", "accept_invalid", "gradual_retrain_reject",
               "holdout_reject", "controller_decision", "controller_fallback",
               "apd_suppress", "static_baseline_deficit")


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _read_csv_rows(path: Path):
    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                yield row
    except Exception:
        return


# ----------------------------------------------------------- 历史实验摘要

def load_reference_curves(runs_dir: Path) -> dict:
    """读取现有真实数据参考曲线（最后一次生成的 runs/real_*.csv）。"""
    out = {}
    specs = {
        "gas": (runs_dir / "real_gas_uci.csv", "macro_f1", "C_selfevo(or)"),
        "cmapss-fd001": (runs_dir / "real_cmapss_fd001.csv", "rmse", "C_selfevo(or)"),
        "cmapss-fd002": (runs_dir / "real_cmapss_fd002.csv", "rmse", "C_selfevo(or)"),
    }
    for name, (path, metric, arm) in specs.items():
        if not path.exists():
            continue
        vals, alarms = [], []
        for row in _read_csv_rows(path):
            if row.get("arm") != arm:
                continue
            try:
                vals.append(float(row[metric]))
            except (TypeError, ValueError):
                continue
            alarms.append(str(row.get("alarm", "")).lower() == "true")
        if vals:
            k = 3 if metric == "macro_f1" else 5
            out[name] = {"metric": metric, "batches": len(vals),
                         "tail": round(float(sum(vals[-k:]) / min(k, len(vals))), 6),
                         "min": round(min(vals), 6), "max": round(max(vals), 6),
                         "alarm_rate": round(float(sum(alarms) / len(alarms)), 4),
                         "source": str(path.relative_to(REPO)).replace("\\", "/")}
    return out


def scan_gate_logs(runs_dir: Path, max_files: int = 400) -> dict:
    """扫描历史实验记录中的闸门拒绝日志（事件行）。"""
    files = sorted(glob.glob(str(runs_dir / "*.csv")),
                   key=os.path.getmtime, reverse=True)[:max_files]
    events, reasons = {}, {}
    for fp in files:
        for row in _read_csv_rows(Path(fp)):
            ev = (row.get("event") or "").strip()
            if not ev:
                continue
            events[ev] = events.get(ev, 0) + 1
            reason = (row.get("reason") or "").strip()
            if reason:
                reasons[f"{ev}:{reason}"] = reasons.get(f"{ev}:{reason}", 0) + 1
    keep = {k: v for k, v in events.items() if k in GATE_EVENTS}
    return {"files_scanned": len(files),
            "events": dict(sorted(keep.items(), key=lambda kv: -kv[1])),
            "reasons": dict(sorted(reasons.items(), key=lambda kv: -kv[1])[:20]),
            "note": "只统计 event/reason 字段；历史 CSV 存在跨行拆散，故仅用于计数。"}


# ----------------------------------------------------------- 决策点证据

def _replay_args():
    class _Args:
        gas_dir = str(REAL_DATA / "gas_sensor" / "Dataset")
        cmapss_dir = str(REAL_DATA / "cmapss")
        gas_batch_size = 600
        fd001_batches = 12
        fd001_start = 28
        label_budget = 4000
        max_retrains = 6
        retrain_cost = 600
        residual_factor = 3.0
    return _Args()


def build_decision_cases(streams=("gas", "cmapss-fd001")) -> dict:
    """复用 replay_decision.build_cases；证据按批从 real_stream 现算。"""
    import sys
    if str(PYTRIO_DIR) not in sys.path:
        sys.path.insert(0, str(PYTRIO_DIR))
    try:
        import replay_decision as rd
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"无法导入 replay_decision：{exc}") from exc
    args = _replay_args()
    cases = {}
    for kind in streams:
        cs, _ctx = rd.build_cases(kind, args)
        cases[kind] = cs
    return cases


def _evidence_vector(ev: dict) -> list[float]:
    return [float(ev["residual_ratio"]) / 3.0,
            min(float(ev["mean_psi"]) / 0.25, 2.6),
            float(ev["mmd_z"]) / 3.0]


def calibrated_actions(cases: dict, calib_path: Path = DEFAULT_CALIB) -> dict:
    """用既有动作标定表 + CostSensitiveDetector 查每个决策点的代价最优动作。

    标定表来自历史合成流（内部夹具派生），只作为提案输入，不作为真实数据结论。
    """
    if not Path(calib_path).exists():
        return {"available": False, "reason": f"缺少标定表 {calib_path}"}
    det = CostSensitiveDetector(calib_path=str(calib_path), cost_label=1e-4, knn_k=3)
    det.psi_drift_threshold = DriftDetector().psi_drift_threshold
    rows, wins, unavailable = [], {"observe": 0, "incremental": 0, "full": 0}, 0
    for kind, cs in cases.items():
        for c in cs:
            ev = c["evidence"]
            x = _evidence_vector(ev)
            import numpy as np
            xv = np.asarray(x, dtype=float)
            costs = {}
            for action in ("observe", "incremental", "full"):
                r = det._knn_predict(action, xv)  # noqa: SLF001 - 复用既有查表
                if r is not None:
                    costs[action] = r[0] + det.cost_label * r[1]
            if not costs:
                unavailable += 1
                continue
            best = min(costs, key=costs.get)
            wins[best] += 1
            rows.append({"stream": kind, "batch": c["batch"],
                         "batch_index": c["batch_index"],
                         "rule_tier": c["rule_tier"], "calib_action": best,
                         "costs": {k: round(v, 5) for k, v in costs.items()},
                         "evidence": {f: ev.get(f) for f in FEATURES if f in ev}})
    disagree = [r for r in rows if r["calib_action"] != r["rule_tier"]]
    return {"available": True, "calib_path": str(Path(calib_path).relative_to(REPO)),
            "rows": rows, "wins": wins, "unavailable": unavailable,
            "disagreements": disagree[:8],
            "note": "标定表为历史合成流产物，仅内部追溯，不作为真实数据结论。"}


def build_digest(cases: dict, calib: dict, runs_dir: Path) -> dict:
    tier_dist, by_stream = {}, {}
    ranges = {f: [None, None] for f in FEATURES}
    for kind, cs in cases.items():
        tiers = {}
        for c in cs:
            ev = c["evidence"]
            tiers[c["rule_tier"]] = tiers.get(c["rule_tier"], 0) + 1
            tier_dist[c["rule_tier"]] = tier_dist.get(c["rule_tier"], 0) + 1
            for f in FEATURES:
                v = ev.get(f)
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                lo, hi = ranges[f]
                ranges[f] = [v if lo is None else min(lo, v),
                             v if hi is None else max(hi, v)]
        by_stream[kind] = {"n": len(cs), "rule_tiers": tiers,
                           "alarms": sum(1 for c in cs if c["evidence"].get("structural_alarm")),
                           "residual_alarms": sum(1 for c in cs
                                                  if c["evidence"].get("residual_ratio", 0) > 3.0)}
    digest = {
        "streams": by_stream,
        "tier_dist": tier_dist,
        "ranges": {k: v for k, v in ranges.items() if v[0] is not None},
        "calibration": {"available": calib.get("available"),
                        "wins": calib.get("wins"),
                        "disagreements": calib.get("disagreements", [])[:6]},
        "gate_logs": scan_gate_logs(runs_dir),
        "reference_curves": load_reference_curves(runs_dir),
    }
    digest["digest_sha1"] = _sha1(json.dumps(digest, ensure_ascii=False,
                                             sort_keys=True, default=str))
    return digest


# ----------------------------------------------------------- 提示词

def build_prompt(digest: dict) -> str:
    schema = {
        "program_id": "p-<短标识>",
        "default_tier": "follow_severity",
        "rules": [
            {"id": "r1",
             "if": {"all": [{"feature": "residual_ratio", "op": ">=", "value": 4.0},
                            {"feature": "labels_left", "op": ">=", "value": 1200}]},
             "then": {"tier": "full"}},
            {"id": "r2",
             "if": {"all": [{"feature": "mmd_z", "op": ">=", "value": 3.0},
                            {"feature": "residual_ratio", "op": "<", "value": 2.0}]},
             "then": {"tier": "observe"}},
        ],
    }
    return f"""你是持续学习系统的策略提案器。根据下面的历史证据摘要，写一套数值 if-then 规则，
用来把漂移证据映射成动作档位，替代原来"按严重度阈值直接分档"的规则。

可用的输入特征（只能用这些，数值型，单位已归一）：
{", ".join(FEATURES)}

可选动作档位： observe / incremental / full
可用算子：{", ".join(OPS)}
条件树深度最多 2 层：根节点是 all 或 any，叶子是单特征比较。
规则按数组顺序匹配，首条命中生效；default_tier 写 "follow_severity" 表示沿用原分档。

只输出一行 JSON，不要解释。schema 示例：
{json.dumps(schema, ensure_ascii=False)}

证据摘要（JSON）：
{json.dumps(digest, ensure_ascii=False, sort_keys=True)[:7000]}
"""


def parse_program(text: str) -> tuple[dict | None, str | None]:
    """从模型输出里提取第一个合法规则程序；返回 (program, error)。"""
    start = text.find("{")
    while start >= 0:
        depth, end = 0, -1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i
                    break
        if end > start:
            try:
                obj = json.loads(text[start:end + 1])
            except Exception:
                obj = None
            if isinstance(obj, dict) and "rules" in obj:
                errs = validate_program(obj)
                if errs:
                    return None, "规则校验失败：" + "; ".join(errs)
                return obj, None
        start = text.find("{", start + 1)
    return None, "未找到含 rules 的 JSON"


def _normalize_program(obj: dict, source: dict) -> dict:
    obj = dict(obj)
    obj.setdefault("default_tier", "follow_severity")
    obj["source"] = source
    body = {k: obj[k] for k in ("rules", "default_tier")}
    digest = _sha1(json.dumps(body, ensure_ascii=False, sort_keys=True))
    obj["program_id"] = obj.get("program_id") or f"p-{digest[:12]}"
    return obj


# ----------------------------------------------------------- 提案方

class OfflineProposer:
    """确定性启发式提案：不联网、不耗 token，用于跑通闭环与做对照。"""

    provider = "heuristic"
    model = None

    def __init__(self, digest: dict):
        self.digest = digest

    def propose(self) -> list[dict]:
        t0 = time.perf_counter()
        cands = [
            # 透传对照：空规则 + 沿用严重度分档，必须与固定阈值基线逐位一致
            ("passthrough_control", []),
            ("perf_soft_observe", [
                {"id": "r1", "if": {"all": [
                    {"feature": "residual_ratio", "op": "<", "value": 2.0},
                    {"feature": "mmd_z", "op": ">=", "value": 3.0}]},
                 "then": {"tier": "observe"}}]),
            ("deep_drift_full", [
                {"id": "r1", "if": {"all": [
                    {"feature": "residual_ratio", "op": ">=", "value": 4.0},
                    {"feature": "labels_left", "op": ">=", "value": 1200}]},
                 "then": {"tier": "full"}},
                {"id": "r2", "if": {"all": [
                    {"feature": "residual_ratio", "op": ">=", "value": 4.0},
                    {"feature": "labels_left", "op": "<", "value": 1200}]},
                 "then": {"tier": "incremental"}}]),
            ("budget_strict", [
                {"id": "r1", "if": {"feature": "labels_left", "op": "<", "value": 1200},
                 "then": {"tier": "incremental"}}]),
            ("late_slot_incremental", [
                {"id": "r1", "if": {"all": [
                    {"feature": "batch_index", "op": ">=", "value": 25},
                    {"feature": "retrains_left", "op": "<=", "value": 2}]},
                 "then": {"tier": "incremental"}}]),
            ("low_severity_observe", [
                {"id": "r1", "if": {"feature": "severity", "op": "<", "value": 1.5},
                 "then": {"tier": "observe"}}]),
        ]
        latency = round(time.perf_counter() - t0, 3)
        out = []
        for name, rules in cands:
            src = {"kind": "offline", "provider": self.provider, "model": None,
                   "tokens_in": 0, "tokens_out": 0, "latency_s": latency,
                   "attempts": 0, "strategy": name,
                   "digest_sha1": self.digest["digest_sha1"]}
            out.append(_normalize_program({"rules": rules,
                                           "default_tier": "follow_severity"},
                                          src))
        return out


class PytrioProposer:
    """PyTRIO 生成式提案：凭据由本机 `trio login` profile 提供，不经本模块传递。"""

    provider = "pytrio"

    def __init__(self, digest: dict, model: str = "Qwen/Qwen3.5-4B",
                 max_tokens: int = 900, seed: int = 42):
        self.digest, self.model = digest, model
        self.max_tokens, self.seed = max_tokens, seed

    def propose(self) -> list[dict]:
        prompt = build_prompt(self.digest)
        base = {"kind": "llm", "provider": self.provider, "model": self.model,
                "digest_sha1": self.digest["digest_sha1"], "attempts": 0}
        t0 = time.time()
        try:
            import pytrio as trio
            client = trio.ServiceClient()
            sampler = client.create_sampling_client(base_model=self.model)
            tokenizer = sampler.get_tokenizer()
            try:
                text_in = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
            except TypeError:
                text_in = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False,
                    add_generation_prompt=True)
            tokens = tokenizer.encode(text_in, add_special_tokens=False)
            res = sampler.sample(
                prompt=trio.ModelInput.from_ints(tokens),
                sampling_params=trio.SamplingParams(
                    max_tokens=self.max_tokens, temperature=0.0, seed=self.seed),
                num_samples=1).result()
            text = res.sequences[0].text
            base.update({"attempts": 1, "latency_s": round(time.time() - t0, 2),
                         "tokens_in": int(getattr(res, "input_tokens", len(tokens))),
                         "tokens_out": int(getattr(res, "output_tokens", 0)),
                         "token_source": "SDK input_tokens/output_tokens"})
            raw_sha = _sha1(text)
            program, err = parse_program(text)
            base["raw_sha1"] = raw_sha
            base["raw_excerpt"] = text[:1500]
            if program is None:
                base["error"] = err
                return [{"program_id": None, "rules": [], "default_tier": None,
                         "source": base, "invalid": True}]
            return [_normalize_program(program, base)]
        except Exception as exc:  # noqa: BLE001 - 提案失败必须可审计、可继续
            base.update({"attempts": 1, "latency_s": round(time.time() - t0, 2),
                         "tokens_in": None, "tokens_out": None,
                         "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            return [{"program_id": None, "rules": [], "default_tier": None,
                     "source": base, "invalid": True}]


class OpenAICompatProposer:
    """复用现有 LLMController 同款环境变量（ZHIXUN_LLM_*）的兼容接口提案方。"""

    provider = "openai-compat"

    def __init__(self, digest: dict, timeout: float = 60.0):
        self.digest = digest
        self.timeout = timeout
        self.base_url = (os.getenv("ZHIXUN_LLM_BASE_URL", "") or "").rstrip("/")
        self.api_key = os.getenv("ZHIXUN_LLM_API_KEY", "")
        self.model = os.getenv("ZHIXUN_LLM_MODEL", "")

    def propose(self) -> list[dict]:
        base = {"kind": "llm", "provider": self.provider,
                "model": self.model or None,
                "digest_sha1": self.digest["digest_sha1"], "attempts": 0}
        if not (self.base_url and self.api_key and self.model):
            base["error"] = "未配置 ZHIXUN_LLM_BASE_URL / ZHIXUN_LLM_API_KEY / ZHIXUN_LLM_MODEL"
            return [{"program_id": None, "rules": [], "default_tier": None,
                     "source": base, "invalid": True}]
        import urllib.request
        prompt = build_prompt(self.digest)
        body = json.dumps({"model": self.model, "temperature": 0,
                           "messages": [{"role": "user", "content": prompt}]}
                          ).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        t0 = time.time()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"]
            usage = payload.get("usage") or {}
            base.update({"attempts": 1, "latency_s": round(time.time() - t0, 2),
                         "tokens_in": usage.get("prompt_tokens"),
                         "tokens_out": usage.get("completion_tokens"),
                         "token_source": "response.usage"})
            program, err = parse_program(text)
            base["raw_sha1"] = _sha1(text)
            base["raw_excerpt"] = text[:1500]
            if program is None:
                base["error"] = err
                return [{"program_id": None, "rules": [], "default_tier": None,
                         "source": base, "invalid": True}]
            return [_normalize_program(program, base)]
        except Exception as exc:  # noqa: BLE001
            base.update({"attempts": 1, "latency_s": round(time.time() - t0, 2),
                         "error": f"{type(exc).__name__}: {str(exc)[:300]}"})
            return [{"program_id": None, "rules": [], "default_tier": None,
                     "source": base, "invalid": True}]


def make_proposer(provider: str, digest: dict):
    p = provider.lower()
    if p == "offline":
        return OfflineProposer(digest)
    if p == "pytrio":
        return PytrioProposer(digest)
    if p in ("deepseek", "openai", "openai-compat"):
        return OpenAICompatProposer(digest)
    raise SystemExit(f"未知提案方 {provider!r}（offline/pytrio/deepseek）")


def collect_inputs(runs_dir: Path, streams=("gas", "cmapss-fd001"),
                   calib_path: Path = DEFAULT_CALIB) -> tuple[dict, dict]:
    cases = build_decision_cases(streams)
    calib = calibrated_actions(cases, calib_path)
    return cases, build_digest(cases, calib, runs_dir)


def save_proposals(proposals: list[dict], out_dir: Path, provider: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"proposals-{provider}-{stamp}.json"
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "provider": provider,
        "n_proposals": len(proposals),
        "n_valid": sum(1 for p in proposals if not p.get("invalid")),
        "proposals": proposals,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    return path


def validate_and_compile(program: dict):
    """供验收脚本调用：把 JSON 程序编译成控制器；非法则抛错。"""
    return compile_program(program)


__all__ = ["collect_inputs", "make_proposer", "save_proposals",
           "build_prompt", "parse_program", "load_reference_curves",
           "scan_gate_logs", "build_decision_cases", "calibrated_actions",
           "validate_and_compile", "TIERS"]
