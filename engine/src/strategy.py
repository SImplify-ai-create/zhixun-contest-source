"""策略层：把"漂移证据"映射成"动作决策"。

前两版把决策写死在 classify 里（严重度阈值 → observe/incremental/full），
答辩追问"换个数据流还成立吗"时只能回答"阈值可标定"，说服力有限。
这里把决策拆成独立、可替换的控制器：

  RuleController  —— 默认。保留阈值直通，只在预算紧张时做降级，
                     起"兜底护栏"作用（现场不允许决策层把预算花光）。
  LLMController   —— 可选。把结构化证据（严重度/残差倍数/PSI/MMD z/预算）
                     交给大模型给出动作 + 一句理由；任何异常（无 key、超时、
                     返回不可解析、越权档位）都自动回退 RuleController，
                     主流程绝不因决策层失败而中断。

证据 → 决策的证据链必须可审计：两条控制器都只输出 {tier, rationale}，
最终动作仍由 pipeline 记录到 CSV/SwanLab。
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request

ALLOWED_TIERS = {"observe", "incremental", "full"}


class RuleController:
    name = "rule"

    def decide(self, evidence: dict) -> dict:
        tier = evidence.get("tier", "observe")
        labels_left = int(evidence.get("labels_left", 10 ** 9))
        # 预算护栏：剩余标注不足一次全量重训（≈600 条）时降级为增量，
        # 避免"最后一次机会被全量重训吃掉、后续漂移无钱可修"。
        if tier == "full" and labels_left < 600:
            return {"tier": "incremental",
                    "rationale": f"剩余标注 {labels_left} 条不足以支撑全量重训，降级增量"}
        return {"tier": tier, "rationale": "规则直通（严重度分档）"}


class LLMController:
    """OpenAI 兼容接口的决策控制器；未配置或失败时自动回退规则控制器。"""

    name = "llm"

    def __init__(self, fallback: RuleController | None = None,
                 base_url: str | None = None, api_key: str | None = None,
                 model: str | None = None, timeout: float = 5.0):
        self.fallback = fallback or RuleController()
        self.base_url = (base_url or os.getenv("ZHIXUN_LLM_BASE_URL", "")).rstrip("/")
        self.api_key = api_key or os.getenv("ZHIXUN_LLM_API_KEY", "")
        self.model = model or os.getenv("ZHIXUN_LLM_MODEL", "")
        self.timeout = timeout

    def decide(self, evidence: dict) -> dict:
        if not (self.base_url and self.api_key and self.model):
            return self.fallback.decide(evidence)
        prompt = (
            "你是持续学习系统的动作决策器。根据证据在 observe/incremental/full "
            "三档中选一档：observe=不重训；incremental=增量微调；full=清洗后全量重训。"
            "残差显著恶化才允许 full；输入偏移为主时优先 incremental；预算不足时降级。"
            "只输出 JSON：{\"tier\": \"...\", \"rationale\": \"...\"}。\n证据："
            + json.dumps(evidence, ensure_ascii=False)
        )
        body = json.dumps({
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            text = payload["choices"][0]["message"]["content"]
            start, end = text.find("{"), text.rfind("}")
            dec = json.loads(text[start:end + 1])
            tier = dec.get("tier")
            if tier not in ALLOWED_TIERS:
                raise ValueError(f"越权档位: {tier}")
            return {"tier": tier,
                    "rationale": str(dec.get("rationale", ""))[:200]}
        except Exception:  # noqa: BLE001 - 决策层失败必须回退，主流程不受影响
            return self.fallback.decide(evidence)


def build_controller(name: str) -> RuleController | LLMController:
    # 交付镜像（离线评测）必须不存在任何可被触发的外部模型调用路径：
    # ZHIXUN_OFFLINE=1 时 llm 控制器根本不会被构造，直接降级为规则控制器。
    if str(name).lower() == "llm" and os.getenv("ZHIXUN_OFFLINE") == "1":
        print("[offline] strategy=llm disabled in offline image; "
              "falling back to rule controller (no external calls)",
              file=sys.stderr)
        return RuleController()
    return LLMController() if str(name).lower() == "llm" else RuleController()
