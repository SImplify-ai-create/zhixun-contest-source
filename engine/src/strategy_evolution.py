"""策略层自进化链（五段，默认不启用）。

本模块把两条互不相通的候选源（pef 确定性枚举 / LLM 规则提案）接到同一条链上：
候选生成 → 预注册闸门 → 跨运行验收 → 采纳账本 → 生效版本指针。

边界（`tests/test_strategy_evolution.py` 会锁定）：
1. **默认不启用**：本模块不被 `pipeline` / `advanced` / 任何生产入口导入；
   只有显式运行 `scripts/strategy_evolution_dryrun.py` 才会走到这里。
2. **不改运行结果**：链只产出候选记录与账本行，不写 `active_version`、
   不改 `config.json`、不切换任何开关。`read_version_pointer` 是只读的。
3. **不新增外部消耗**：不调用 LLM、不训练模型、不读官方数据；LLM 候选只从
   已落盘的 `docs/llm_rule_ledger.jsonl` 读取（历史提案，0 采纳）。
4. **判定纯函数**：`evaluate_acceptance` 只读传入的 cells / sample / 阈值，
   不读时间、随机数、文件或模型自评。同一份输入必然给出同一结论。
5. **判据预注册**：`PREREGISTERED_GATES` 的每条判据都带 `doc_ref`（文件 + 行号）
   与 `doc_anchor`（该行区里必须出现的原文）。`check_doc_refs` 会真的去盘上核对；
   核不上就拒绝，不接受"看起来合理"。

措辞约定：链条做的是"按预注册判据放行/拒绝并留痕"，不是"自动激活能力"。
即使结论为 `accept`，也只表示"证据满足预注册判据"，本模块没有任何生效写路径。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parent.parent          # zhixun-engine
WORKSPACE = REPO.parent                                # 工作区根目录
DEFAULT_PEF_DIR = WORKSPACE / "2026-11-14" / "policy-evolution-framework"
DEFAULT_LLM_LEDGER = REPO / "docs" / "llm_rule_ledger.jsonl"
DEFAULT_ADOPTION_LEDGER = REPO / "runs" / "strategy_evolution" / "adoption_ledger.jsonl"

# 链默认不启用：没有任何调用方，也没有任何"自动生效"的入口。
ENABLED_BY_DEFAULT = False
CHAIN_MOVES_POINTER = False

SOURCE_DETERMINISTIC = "deterministic_enum"
SOURCE_LLM = "llm_proposal"

# 验收结论（四条互斥；不足样本单独一条，绝不与通过共用路径）
ACCEPT = "accept"
REJECT = "reject"
INSUFFICIENT = "insufficient_evidence"
REJECT_PREREG = "reject_preregistration"
VERDICTS = (ACCEPT, REJECT, INSUFFICIENT, REJECT_PREREG)

# 统一验收口径（与 pef 预注册一致，测试会核对两处常量没有漂移）
PROTOCOL_ID = "X-RUN-SIGN-1"
MIN_DECISIONS = 200
MIN_STREAMS = 3
MIN_REPEATS = 2
MAX_WORSE_PCT = 0.10
MIN_NON_WORSE_RATIO = 0.85


# ---------------------------------------------------------------- 预注册闸门

@dataclass(frozen=True)
class GateSpec:
    """一条预注册判据：判据正文 + 可核对的文档位置。"""

    gate: str
    criterion: str
    doc_refs: tuple[str, ...]
    doc_anchors: tuple[str, ...]


# 相对 WORKSPACE 的路径；行号必须能在当前盘上文件里核到锚点原文。
PREREGISTERED_GATES = (
    GateSpec(
        gate="P1_candidate_space",
        criterion=("候选必须来自预注册空间：确定性枚举为 trigger_confirm×"
                   "trigger_cooldown×periodic_period×reserve 的全量网格；"
                   "LLM 规则必须通过特征白名单、深度<=2、档位合法校验。"),
        doc_refs=("2026-11-14/policy-evolution-framework/pef/learning.py#L137-L140",
                  "zhixun-engine/src/rule_program.py#L51-L80"),
        doc_anchors=("def rule_space", "def validate_program"),
    ),
    GateSpec(
        gate="P2_identifier",
        criterion=("候选标识非空且在本轮唯一；可由规则正文复算时必须一致，"
                   "不可复算时保留 derived_id 供审计，不擅自改号。"),
        doc_refs=("zhixun-engine/src/rule_proposal.py#L282-L289",),
        doc_anchors=("p-{digest[:12]}",),
    ),
    GateSpec(
        gate="P3_acceptance_protocol",
        criterion=("跨运行符号一致性协议已预注册：非劣比例>=85%、单格劣化<=10%、"
                   "每流重复>=2；样本不足只允许出 insufficient_evidence。"),
        doc_refs=("2026-11-14/policy-evolution-framework/pef/learning.py#L240-L271",),
        doc_anchors=("def cross_run_accept",),
    ),
)


def parse_doc_ref(ref: str) -> tuple[str, int, int]:
    """把 `path#L10-L20` 解析成 (相对路径, 起始行, 结束行)。"""
    path_part, _, line_part = ref.partition("#L")
    if not path_part or not line_part:
        raise ValueError(f"非法 doc_ref: {ref!r}")
    start_s, _, end_s = line_part.partition("-L")
    start = int(start_s)
    end = int(end_s) if end_s else start
    if end < start:
        raise ValueError(f"非法行号区间: {ref!r}")
    return path_part, start, end


def check_doc_refs(root: Path = WORKSPACE) -> list[dict]:
    """真的去盘上核对每条判据引用的文件、行号与锚点原文。"""
    rows = []
    for spec in PREREGISTERED_GATES:
        for ref, anchor in zip(spec.doc_refs, spec.doc_anchors):
            path_rel, start, end = parse_doc_ref(ref)
            path = Path(root) / path_rel
            exists = path.exists()
            lines_ok = False
            anchor_found = False
            if exists:
                lines = path.read_text(encoding="utf-8").splitlines()
                lines_ok = 1 <= start <= end <= len(lines)
                if lines_ok:
                    block = "\n".join(lines[start - 1:end])
                    anchor_found = anchor in block
            rows.append({"gate": spec.gate, "doc_ref": ref, "anchor": anchor,
                         "exists": exists, "lines_ok": lines_ok,
                         "anchor_found": anchor_found,
                         "ok": exists and lines_ok and anchor_found})
    return rows


# ---------------------------------------------------------------- 候选

@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    source: str
    body: Mapping[str, Any]
    provenance: Mapping[str, Any]
    proxy_score: float | None = None


def canonical_body(body: Mapping[str, Any]) -> dict:
    """规则正文的规范化视图（与 `rule_proposal._normalize_program` 同口径）。"""
    return {"rules": body.get("rules", []),
            "default_tier": body.get("default_tier")}


def derived_program_id(body: Mapping[str, Any]) -> str:
    payload = json.dumps(canonical_body(body), ensure_ascii=False, sort_keys=True)
    return f"p-{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:12]}"


def _load_pef_learning(pef_dir: Path):
    """只读加载 pef/learning.py（仅 stdlib 顶层导入，无副作用）。"""
    path = Path(pef_dir) / "pef" / "learning.py"
    if not path.exists():
        raise FileNotFoundError(f"找不到离线学习层：{path}")
    spec = importlib.util.spec_from_file_location("_pef_learning_readonly", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclass 解析注解时需要模块已登记在 sys.modules；仅登记，不写盘。
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _assert_thresholds(learning_module) -> None:
    """预注册阈值漂移检查：pef 改了门槛，本链必须显式报错而不是静默跟随。"""
    observed = (learning_module.MIN_DECISIONS, learning_module.MIN_STREAMS,
                learning_module.MIN_REPEATS)
    expected = (MIN_DECISIONS, MIN_STREAMS, MIN_REPEATS)
    if observed != expected:
        raise RuntimeError(
            f"pef 启动门槛已变为 {observed}，与预注册 {expected} 不一致；"
            "请先更新预注册判据再运行本链。")


def deterministic_candidates(pef_dir: Path = DEFAULT_PEF_DIR
                             ) -> tuple[list[Candidate], dict]:
    """第一段之一：确定性枚举（沿用 pef 的候选空间与启动门槛）。"""
    pef_dir = Path(pef_dir)
    learning = _load_pef_learning(pef_dir)
    _assert_thresholds(learning)
    runs_dir = pef_dir / "outputs" / "runs"
    corpus = learning.build_corpus(runs_dir)
    stats = learning.corpus_stats(corpus)
    switch = learning.switch_state(stats)
    meta = {"pef_dir": str(pef_dir), "runs_glob": "outputs/runs/*/s*/ledger.jsonl",
            "corpus": stats, "switch": switch, "n_scored": 0}
    if not switch["enabled"]:
        # 启动门槛不满足 → 不进候选空间，原因写进 meta，不产规则。
        meta["note"] = switch["reason"]
        return [], meta

    scored = learning.search(corpus, top_k=len(learning.rule_space()))
    meta["n_scored"] = len(scored)
    cands = []
    for item in scored:
        cand = item["candidate"]
        cands.append(Candidate(
            candidate_id=cand["version"],
            source=SOURCE_DETERMINISTIC,
            body={"params": {k: cand[k] for k in
                             ("trigger_confirm", "trigger_cooldown",
                              "periodic_period", "reserve")}},
            provenance={
                "candidate_space": "pef/learning.py#L137-L140",
                "corpus": (f"outputs/runs/*/s*/ledger.jsonl"
                           f"（{stats['n_decisions']} 决策 / "
                           f"{stats['n_streams']} 流 / 每流 {stats['min_repeats']} 次）"),
                "proxy_target": "pef/learning.py#L197-L209（仅排序用，不是性能结论）",
            },
            proxy_score=float(item["proxy"]["score"]),
        ))
    # 代理分相同的候选按标识排序，保证同输入同顺序。
    cands.sort(key=lambda c: (-(c.proxy_score or 0.0), c.candidate_id))
    meta["note"] = switch["reason"]
    return cands, meta


def llm_candidates(ledger_path: Path = DEFAULT_LLM_LEDGER
                   ) -> tuple[list[Candidate], dict]:
    """第一段之二：LLM 提案（只读历史账本，不发起新调用）。"""
    path = Path(ledger_path)
    if not path.exists():
        raise FileNotFoundError(f"找不到 LLM 规则账本：{path}")
    meta = {"ledger": str(path), "n_lines": 0, "n_unique": 0,
            "n_without_id": 0, "note": "只读已落盘提案，不新增模型调用"}
    merged: dict[str, tuple[int, dict]] = {}
    n_lines = 0
    n_without_id = 0
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        n_lines += 1
        entry = json.loads(raw)
        pid = entry.get("program_id")
        if not pid:
            n_without_id += 1
            continue
        merged[pid] = (line_no, entry)   # 同 id 后写覆盖：与 verify_rules.load_ledger 同口径
    meta.update({"n_lines": n_lines, "n_unique": len(merged),
                 "n_without_id": n_without_id})

    cands = []
    for pid in sorted(merged):
        line_no, entry = merged[pid]
        rule = entry.get("rule") or {}
        verification = entry.get("verification") or {}
        rejected_by = [f"{g.get('gate')}@{g.get('stream')}"
                       for g in entry.get("rejected_by") or []]
        src = entry.get("source") or {}
        cands.append(Candidate(
            candidate_id=pid,
            source=SOURCE_LLM,
            body={"rules": rule.get("rules", []),
                  "default_tier": rule.get("default_tier")},
            provenance={
                "ledger": f"zhixun-engine/docs/llm_rule_ledger.jsonl#L{line_no}",
                "provider": src.get("provider"),
                "model": src.get("model"),
                "recorded_decision": entry.get("decision"),
                "recorded_rejected_by": rejected_by,
                "verification_streams": list(verification.get("streams") or []),
                "verification_runs_per_stream": 1,
            },
        ))
    return cands, meta


# ---------------------------------------------------------------- 第二段：预注册闸门

def _deterministic_in_space(cand: Candidate) -> tuple[bool, str]:
    params = cand.body.get("params") if isinstance(cand.body, Mapping) else None
    if not isinstance(params, Mapping):
        return False, "缺少 params"
    space = {"trigger_confirm": (1, 2, 3), "trigger_cooldown": (0, 2, 3, 5),
             "periodic_period": (4, 6, 8, 10), "reserve": (True, False)}
    bad = [k for k, allowed in space.items() if params.get(k) not in allowed]
    if bad:
        return False, f"越出预注册网格：{bad}"
    version = (f"v1.2-learned-c{params['trigger_confirm']}"
               f"cd{params['trigger_cooldown']}p{params['periodic_period']}"
               f"r{int(params['reserve'])}")
    if version != cand.candidate_id:
        return False, f"标识与参数不一致：{cand.candidate_id} != {version}"
    return True, version


def _llm_body_valid(cand: Candidate) -> tuple[bool, str]:
    from .rule_program import validate_program
    errs = validate_program(dict(cand.body))
    return (not errs), ("合法" if not errs else "; ".join(errs))


def preregistration_rows(candidates: Sequence[Candidate]) -> list[dict]:
    """对每个候选跑三条预注册判据；任一不过 → 拒绝（不进验收）。"""
    id_counts = Counter(c.candidate_id for c in candidates)
    rows = []
    for cand in candidates:
        gates = []
        if cand.source == SOURCE_DETERMINISTIC:
            ok, observed = _deterministic_in_space(cand)
        elif cand.source == SOURCE_LLM:
            ok, observed = _llm_body_valid(cand)
        else:
            ok, observed = False, f"未知来源 {cand.source!r}"
        gates.append({"gate": "P1_candidate_space", "passed": bool(ok),
                      "observed": observed,
                      "doc_ref": PREREGISTERED_GATES[0].doc_refs[0]})

        duplicate = id_counts[cand.candidate_id] > 1
        derived = derived_program_id(cand.body) if cand.source == SOURCE_LLM else None
        id_ok = bool(cand.candidate_id) and not duplicate
        gates.append({"gate": "P2_identifier", "passed": id_ok,
                      "observed": (f"id={cand.candidate_id}; duplicate={duplicate}; "
                                   f"derived_id={derived}; "
                                   f"recomputed_match={(derived == cand.candidate_id)}"),
                      "doc_ref": PREREGISTERED_GATES[1].doc_refs[0]})

        protocol_ok = PROTOCOL_ID == "X-RUN-SIGN-1"
        gates.append({"gate": "P3_acceptance_protocol", "passed": protocol_ok,
                      "observed": (f"protocol={PROTOCOL_ID}; non_worse>="
                                   f"{MIN_NON_WORSE_RATIO:.0%}; max_worse<="
                                   f"{MAX_WORSE_PCT:.0%}; min_repeats={MIN_REPEATS}"),
                      "doc_ref": PREREGISTERED_GATES[2].doc_refs[0]})

        failed = [g["gate"] for g in gates if not g["passed"]]
        rows.append({"candidate_id": cand.candidate_id, "source": cand.source,
                     "passed": not failed, "failed": failed, "gates": gates})
    return rows


# ---------------------------------------------------------------- 第三段：跨运行验收

def evaluate_acceptance(cells: Sequence[Mapping[str, Any]],
                        sample: Mapping[str, Any], *,
                        min_decisions: int = MIN_DECISIONS,
                        min_streams: int = MIN_STREAMS,
                        min_repeats: int = MIN_REPEATS,
                        max_worse_pct: float = MAX_WORSE_PCT,
                        min_non_worse_ratio: float = MIN_NON_WORSE_RATIO) -> dict:
    """跨运行符号一致性验收（纯函数）。

    三条互斥出口，顺序固定：
    1. 样本不足 → `insufficient_evidence`（`passed=None`，不进入通过判定）；
    2. 该候选没有跨运行实测单元 → `insufficient_evidence`（同样不进通过判定）；
    3. 有完整样本才计算非劣比例与单格上限 → `accept` / `reject`。
    """
    missing = []
    n_dec = sample.get("n_decisions")
    n_streams = sample.get("n_streams")
    n_repeats = sample.get("n_repeats")
    if not isinstance(n_dec, int) or n_dec < min_decisions:
        missing.append(f"决策样本 {n_dec} < {min_decisions}")
    if not isinstance(n_streams, int) or n_streams < min_streams:
        missing.append(f"流数 {n_streams} < {min_streams}")
    if not isinstance(n_repeats, int) or n_repeats < min_repeats:
        missing.append(f"每流重复 {n_repeats} < {min_repeats}")
    if missing:
        return {"verdict": INSUFFICIENT, "passed": None,
                "reason": "样本不足：" + "；".join(missing),
                "sample": dict(sample), "measured": None}

    if not cells:
        return {"verdict": INSUFFICIENT, "passed": None,
                "reason": "该候选没有跨运行实测单元（只有排序代理分）",
                "sample": dict(sample), "measured": None}

    rows = []
    non_worse = 0
    for cell in cells:
        incumbent = float(cell["incumbent"])
        candidate = float(cell["candidate"])
        lower_is_better = bool(cell.get("lower_is_better", True))
        # diff > 0 表示候选更差；方向由单元格显式声明，不靠猜。
        diff = (candidate - incumbent) if lower_is_better else (incumbent - candidate)
        ok = diff <= 0.0
        pct = (diff / abs(incumbent)) if incumbent else float("inf")
        non_worse += int(ok)
        rows.append({"stream": cell.get("stream"), "seed": cell.get("seed"),
                     "incumbent": incumbent, "candidate": candidate,
                     "lower_is_better": lower_is_better,
                     "diff": round(diff, 6), "pct": round(pct, 6),
                     "non_worse": ok, "worse_beyond_cap": pct > max_worse_pct})

    need = math.ceil(min_non_worse_ratio * len(rows))
    worse = [r for r in rows if r["worse_beyond_cap"]]
    measured = {"n_cells": len(rows), "non_worse": non_worse, "required": need,
                "max_worse_pct": max_worse_pct, "cells": rows}
    if worse:
        detail = "，".join(f"{r['stream']} s{r['seed']} {r['pct']:+.1%}" for r in worse)
        return {"verdict": REJECT, "passed": False,
                "reason": f"存在单格劣化超过上限 {max_worse_pct:.0%}：{detail}",
                "sample": dict(sample), "measured": measured}
    if non_worse < need:
        return {"verdict": REJECT, "passed": False,
                "reason": f"非劣 {non_worse}/{len(rows)} < 要求 {need}",
                "sample": dict(sample), "measured": measured}
    return {"verdict": ACCEPT, "passed": True,
            "reason": f"符号一致性通过：非劣 {non_worse}/{len(rows)} >= {need}",
            "sample": dict(sample), "measured": measured}


def recorded_cells(learning_report: Mapping[str, Any],
                   candidate_id: str) -> list[dict]:
    """只取"该候选"在盘上已有的跨运行实测单元；没有就返回空。"""
    chosen = learning_report.get("chosen_candidate") or {}
    if chosen.get("version") != candidate_id:
        return []
    acceptance = learning_report.get("acceptance") or {}
    cells = []
    for cell in acceptance.get("cells") or []:
        # GAS 是分类（macro-F1，越高越好）；其余为 RMSE（越低越好）。
        cells.append({**cell, "lower_is_better": cell.get("stream") != "GAS"})
    return cells


# ---------------------------------------------------------------- 第四段：采纳账本

def _next_round(path: Path) -> int:
    path = Path(path)
    if not path.exists():
        return 1
    rounds = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.strip():
            rounds.append(int(json.loads(raw).get("round", 0)))
    return (max(rounds) + 1) if rounds else 1


def _evidence_ref(cand: Candidate, cells: Sequence[Mapping[str, Any]]) -> str:
    if cand.source == SOURCE_DETERMINISTIC:
        base = ("2026-11-14/policy-evolution-framework/pef/learning.py#L137-L140；"
                "outputs/runs/*/s*/ledger.jsonl（1443 决策 / 8 流 / 每流 3 次）")
        if cells:
            worst, worst_pct = max(((c, float(c.get("pct") or 0.0)) for c in cells),
                                   key=lambda item: item[1])
            detail = (f"{len(cells)} 格，最大劣化 {worst['stream']} s{worst['seed']} "
                      f"{worst_pct:+.1%}" if worst_pct > 0
                      else f"{len(cells)} 格，无劣化格")
            return (base + f"；outputs/learning/learning_report.json acceptance.cells"
                    f"（{detail}）")
        return base + "；无该候选的跨运行实测（仅代理分）"
    return (f"{cand.provenance.get('ledger')}"
            f"（provider={cand.provenance.get('provider')}；"
            f"recorded={cand.provenance.get('recorded_decision')}；"
            f"rejected_by={','.join(cand.provenance.get('recorded_rejected_by') or []) or '-'}）")


def build_adoption_rows(candidates: Sequence[Candidate],
                        prereg: Sequence[Mapping[str, Any]],
                        *, round_no: int, pointer: Mapping[str, Any],
                        samples: Mapping[str, Mapping[str, Any]],
                        cells_by_id: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict]:
    """每候选一行，字段固定、可比对；不含时间戳（同输入同字节）。"""
    prereg_by_id = {row["candidate_id"]: row for row in prereg}
    rows = []
    for cand in candidates:
        gate_row = prereg_by_id[cand.candidate_id]
        sample = samples[cand.source]
        if gate_row["passed"]:
            decision = evaluate_acceptance(cells_by_id.get(cand.candidate_id, []),
                                           sample)
            verdict, reason = decision["verdict"], decision["reason"]
            measured = decision["measured"]
        else:
            verdict = REJECT_PREREG
            reason = "预注册闸门未通过：" + ", ".join(gate_row["failed"])
            measured = None
        rows.append({
            "round": int(round_no),
            "candidate_id": cand.candidate_id,
            "source": cand.source,
            "criterion": (f"{PROTOCOL_ID}: 非劣>={MIN_NON_WORSE_RATIO:.0%} 且 "
                          f"单格劣化<={MAX_WORSE_PCT:.0%} 且 "
                          f"样本>={MIN_DECISIONS}/{MIN_STREAMS}流/{MIN_REPEATS}次"),
            "verdict": verdict,
            "passed": None if measured is None else bool(measured),
            "reason": reason,
            "evidence_ref": _evidence_ref(cand, cells_by_id.get(cand.candidate_id, [])),
            "preregistration": {"passed": gate_row["passed"],
                                "failed": gate_row["failed"],
                                "gates": gate_row["gates"]},
            "sample": dict(sample),
            "measured": measured,
            "proxy_score": cand.proxy_score,
            "recorded_decision": cand.provenance.get("recorded_decision"),
            "version_pointer": {"active_version": pointer.get("active_version"),
                                "chain_moves_pointer": CHAIN_MOVES_POINTER,
                                "sha256": pointer.get("sha256")},
        })
    return rows


def append_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return path


# ---------------------------------------------------------------- 第五段：生效版本指针

def read_version_pointer(pef_dir: Path = DEFAULT_PEF_DIR) -> dict:
    """只读快照：链没有任何写回路径。"""
    registry_path = Path(pef_dir) / "outputs" / "registry.json"
    if not registry_path.exists():
        return {"available": False, "source": str(registry_path),
                "active_version": None, "sha256": None}
    raw = registry_path.read_bytes()
    data = json.loads(raw.decode("utf-8"))
    config_path = REPO / "config.json"
    strategy = None
    if config_path.exists():
        strategy = (json.loads(config_path.read_text(encoding="utf-8"))
                    .get("decision") or {}).get("strategy")
    return {"available": True,
            "source": "2026-11-14/policy-evolution-framework/outputs/registry.json",
            "active_version": data.get("active_version"),
            "rollback_to": data.get("rollback_to"),
            "decision_strategy": strategy,
            "sha256": hashlib.sha256(raw).hexdigest()}


def pointer_unchanged(before: Mapping[str, Any], after: Mapping[str, Any]) -> bool:
    return before.get("sha256") == after.get("sha256")


# ---------------------------------------------------------------- 编排

def run_chain(*, pef_dir: Path = DEFAULT_PEF_DIR,
              llm_ledger: Path = DEFAULT_LLM_LEDGER,
              round_no: int | None = None,
              adoption_ledger: Path = DEFAULT_ADOPTION_LEDGER) -> dict:
    """跑完整五段；返回行、统计与核对结果，不写任何生效开关。"""
    doc_check = check_doc_refs()
    if not all(row["ok"] for row in doc_check):
        bad = [row for row in doc_check if not row["ok"]]
        raise RuntimeError(f"预注册判据无法核对到盘上原文：{bad}")

    det_cands, det_meta = deterministic_candidates(pef_dir)
    llm_cands, llm_meta = llm_candidates(llm_ledger)
    candidates = list(det_cands) + list(llm_cands)

    prereg = preregistration_rows(candidates)
    learning_report_path = Path(pef_dir) / "outputs" / "learning" / "learning_report.json"
    learning_report = (json.loads(learning_report_path.read_text(encoding="utf-8"))
                       if learning_report_path.exists() else {})
    cells_by_id = {}
    for cand in det_cands:
        cells = recorded_cells(learning_report, cand.candidate_id)
        if cells:
            cells_by_id[cand.candidate_id] = cells
    for cand in llm_cands:
        prov = cand.provenance
        cells_by_id[cand.candidate_id] = []   # 账本只有 1 次/流，不构成跨运行单元

    samples = {
        SOURCE_DETERMINISTIC: {
            "n_decisions": det_meta["corpus"]["n_decisions"],
            "n_streams": det_meta["corpus"]["n_streams"],
            "n_repeats": det_meta["corpus"]["min_repeats"],
            "basis": "pef outputs/runs 账本（真实离线跑批）",
        },
        SOURCE_LLM: {
            "n_decisions": None,
            "n_streams": len(llm_cands[0].provenance.get("verification_streams") or [])
            if llm_cands else 0,
            "n_repeats": 1,
            "basis": "docs/llm_rule_ledger.jsonl 每条流仅 1 次验收（无重复种子）",
        },
    }

    pointer_before = read_version_pointer(pef_dir)
    rnd = _next_round(adoption_ledger) if round_no is None else int(round_no)
    rows = build_adoption_rows(candidates, prereg, round_no=rnd,
                               pointer=pointer_before, samples=samples,
                               cells_by_id=cells_by_id)
    pointer_after = read_version_pointer(pef_dir)
    unchanged = pointer_unchanged(pointer_before, pointer_after)

    counts = Counter(row["verdict"] for row in rows)
    by_source = {}
    for source in (SOURCE_DETERMINISTIC, SOURCE_LLM):
        subset = [r for r in rows if r["source"] == source]
        by_source[source] = {"n": len(subset),
                             "verdicts": dict(Counter(r["verdict"] for r in subset))}
    return {"rows": rows, "round": rnd, "counts": dict(counts),
            "by_source": by_source, "doc_check": doc_check,
            "deterministic_meta": det_meta, "llm_meta": llm_meta,
            "pointer": pointer_before, "pointer_after": pointer_after,
            "pointer_unchanged": unchanged,
            "preregistration_counts": dict(Counter(
                "pass" if r["passed"] else "reject" for r in prereg))}


__all__ = [
    "ACCEPT", "REJECT", "INSUFFICIENT", "REJECT_PREREG", "VERDICTS",
    "SOURCE_DETERMINISTIC", "SOURCE_LLM", "PROTOCOL_ID",
    "MIN_DECISIONS", "MIN_STREAMS", "MIN_REPEATS",
    "PREREGISTERED_GATES", "GateSpec", "Candidate",
    "parse_doc_ref", "check_doc_refs", "canonical_body", "derived_program_id",
    "deterministic_candidates", "llm_candidates", "preregistration_rows",
    "evaluate_acceptance", "recorded_cells", "build_adoption_rows",
    "append_rows", "read_version_pointer", "pointer_unchanged", "run_chain",
]
