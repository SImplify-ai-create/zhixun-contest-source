# -*- coding: utf-8 -*-
"""v6 纯函数：alpha_grid 的 configuration 序列化与四级砍配。

结构见 03-alpha_grid-序列化.md；键名为占位，实现侧定名后回填。
砍配纪律：逐级执行、每级留显式标记、第四级之后只许报错。
"""
import json

DEFAULT_BUDGET = 8192


class GridTooLargeError(Exception):
    """四级砍配后仍超预算：结构问题，拒绝写入，不允许静默再砍。"""


def _dumps(obj):
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _size(obj):
    return len(_dumps(obj).encode("utf-8"))


def _round_scores(scores, prec):
    return [round(float(s), prec) for s in scores]


def serialize_alpha_grid(grid, budget=DEFAULT_BUDGET):
    """grid: {"slice","metric","alphas","margin","pairs":[{id,a,b,scores,sel,
    scores_raw?, endpoints_rel?}]}。

    双列口径（05 号文档）：`scores` = as-submitted（决策列），
    `scores_raw` = 统一未校准（诊断列）。诊断列必须逐候选落账本——
    它是 schema 的一部分，不是叙述；砍配时两列同生同灭，不允许只留一列。

    级联纪律（自适应）：level 0=全量；1=分数降精度；2=未选中对只留最优；
    3=整对丢弃。**只采用严格变小的级**——审计标记（"tl"/"k"）有字节成本，
    若某级算上标记后不比上一级小，它永远买不起信息（上一级放不下时它更
    放不下），直接跳过；"tl" 记录实际采用的级，"k" 记录决策列最优下标。
    仍超 → GridTooLargeError。
    """
    base = {
        "slice": grid["slice"], "metric": grid["metric"],
        "alphas": [float(a) for a in grid["alphas"]],
        "margin": round(float(grid.get("margin") or 0.0), 6),
        "pairs": [dict(p) for p in grid["pairs"]],
    }
    n_alpha = len(base["alphas"])
    for pair in base["pairs"]:
        pair["scores"] = _round_scores(pair["scores"], 4)
        if len(pair["scores"]) != n_alpha:
            raise ValueError(
                "pair %s 的 scores 长度 %d 与 alphas 长度 %d 不符"
                % (pair.get("id"), len(pair["scores"]), n_alpha))
        raw = pair.get("scores_raw")
        if raw is not None:
            if len(raw) != n_alpha:
                raise ValueError(
                    "pair %s 的 scores_raw 长度 %d 与 alphas 长度 %d 不符"
                    % (pair.get("id"), len(raw), n_alpha))
            pair["scores_raw"] = _round_scores(raw, 4)

    def _build(level):
        """构造第 level 级形态（不改动 base）。返回新对象。"""
        import copy
        obj = copy.deepcopy(base)
        if level >= 1:
            for pair in obj["pairs"]:
                pair["scores"] = _round_scores(pair["scores"], 2)
                if "scores_raw" in pair:
                    pair["scores_raw"] = _round_scores(pair["scores_raw"], 2)
            obj["prec"] = 2
        if level >= 2:
            def _sel_score(pair):
                idx = obj["alphas"].index(float(pair["sel"]))
                return pair["scores"][idx]
            winner = max(range(len(obj["pairs"])),
                         key=lambda i: _sel_score(obj["pairs"][i]))
            for i, pair in enumerate(obj["pairs"]):
                if i == winner:
                    continue
                best = max(range(len(pair["scores"])),
                           key=lambda j: pair["scores"][j])
                pair["scores"] = [pair["scores"][best]]
                if "scores_raw" in pair:
                    pair["scores_raw"] = [pair["scores_raw"][best]]
                pair["t"] = 1
                pair["ba"] = obj["alphas"][best]
                pair["k"] = best
        if level >= 3:
            order = sorted(range(len(obj["pairs"])),
                           key=lambda i: obj["pairs"][i].get("endpoints_rel", i))
            dropped = []
            # 第 3 级在固定预算外先丢到能丢的最小形态，由外层判定是否够小
            while obj["pairs"] and _size(obj) > budget:
                victim = order.pop(0)
                dropped.append(obj["pairs"][victim]["id"])
                obj["pairs"].pop(victim)
                order = [i - 1 if i > victim else i for i in order]
            obj["dropped"] = dropped
        if level >= 1:
            obj["tl"] = level
        return obj

    prev_size = None
    for level in (0, 1, 2, 3):
        obj = _build(level)
        size = _size(obj)
        if prev_size is not None and size >= prev_size:
            continue  # 不变小的级买不起信息，跳过
        if size <= budget:
            return _dumps(obj), level
        prev_size = size
    raise GridTooLargeError(
        "级联后仍 %d B 超预算 %d B，拒绝写入" % (_size(_build(3)), budget))
