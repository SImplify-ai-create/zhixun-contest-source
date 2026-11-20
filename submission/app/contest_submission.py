#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v2 P0 选手模块（对应 `/app/contest_submission.py`）。

P0 只交两件：`describe` 与 `predict`。`train` 留接口不实现。

设计边界（与本轮任务书一致）：
- 导入无副作用：不读 `/data`、不联网、不扫 `/app` 找权重；numpy 也在真正需要时才导入。
- `model_dir` 是 `/model` 根目录，检查点路径来自 `CONTEST_ARTIFACT_PATH`，不是目录。
- predict 逐行调用；只允许 causal（因果）动作，状态只向前累积，绝不回看未来行。
- 输出形状的 task 优先级：**上下文 > 行 > 产物**。上下文与行都由平台生成、被 SDK 绑死；
  产物是我们自己挂的，只在两者都缺失时才作为最后退路（stderr 标注）。上下文与产物冲突
  不再整批 fail-fast：该行按声明默认输出，其余行继续处理。
- 批级检测器（KL / 马氏 / BBSD）只在 `train_stage` 里跑，本模块不导入 `detectors`。
- **载入期常量预计算**：feature_order 在装载时钉成 tuple，动作门禁/参数校验/实现选择一次
  算好并捕获成 `step`，coef 转连续 float 数组；每行只做「取数 → 动作 → 一次矩阵乘」，
  不重复排序、不求均值、不再读文件。检查点未给 feature_order 时只在首行排序一次并钉住。
- **总输出 + 声明默认**：每一行都必须吐一行、形状必须对；处理不了的行给预注册的确定性兜底，
  触发原因写 stderr。不 fail-fast（平台缺一行就整批作废），也不伪装成正常推断。
  兜底与"不猜"的界线见 `P0设计说明.md` 第一节。
- 分类：label 非空且不重复、score 有限且落在 [0,1]、最高分唯一（并列按 label 字典序 tie-break）。
- 回归：value 有限，负值 clip 到 0。

检查点契约（npz，allow_pickle=False）：
    task                  标量字符串，"classification" 或 "regression"（必填）
    coef                  回归 (n_features,)，分类 (n_classes, n_features)（必填）
    intercept             回归标量或分类 (n_classes,)（可选，缺省 0）
    classes               分类标签，字符串数组（分类必填）
    feature_order         特征名顺序，字符串数组（可选；缺省时按 inputs 键名字典序取，并要求维度一致）
    prior_majority_class  分类兜底用的先验多数类（可选；缺省取 classes[0] 并在 stderr 标注未预注册）
    prior_mean            回归兜底用的先验均值（可选；缺省 0.0 并在 stderr 标注未预注册）
    action                特征动作名（可选；缺省 identity）。predict 侧只允许 causal：
                          identity / streaming_ema_norm / per_cond_seq；
                          transductive 动作（per_cond_mono_norm / batch_zscore）显式拒绝。
    action_params         动作参数，JSON 字符串（可选；如 {"alpha":0.5,"eps":1e-8}）

关于 `tasks` 声明：`describe` 报本镜像支持的两类 task。候选在 SDK 里不申报 task，
候选是每轮每数据集一个，因此单个产物的 task 由它自己声明；predict 侧按
`CONTEST_TASK` / `CONTEST_PREDICTION_CONTEXT` 泛型分派，不靠候选申报。
"""

import sys

sys.dont_write_bytecode = True

import json  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:  # 同一目录的 actions / train_stage 才可被导入
    sys.path.insert(0, _HERE)

import actions  # noqa: E402

PROTOCOL = "contest.harness.workspace.v2"
MODEL_ACCESS = "host_gateway_rpc"
TASKS = ("classification", "regression")

# 官方基线检查点的键：w1/w2 是 MLP 权重，mean/scale 是输入侧标准化，metadata 里带
# task/paths/classes/target_mean/target_scale。v2 自有格式是 task+coef(+classes)。
OFFICIAL_KEYS = ("w1", "b1", "w2", "b2", "mean", "scale")
OFFICIAL_FORBIDDEN = {"truerul", "normalizedrul", "remainingusefullife", "rul", "totalsnapshots"}

# 输入特征的量级闸门。官方前向会先做 (x-mean)/scale 再 clip 到 ±10σ，1e30 这类
# 越界值会被 clip 无声吞掉、照常出分；因此必须在 clip 之前判越界，走声明默认。
# 1e12 远高于三套公开数据的真实量级，只拦"明显不可能"的脏值。
FEATURE_ABS_LIMIT = 1e12

_CACHE = {}
_STREAM_STATE = {}

# 行路径计数器（只增不减，供本地断言/事后对账；不参与平台协议）。
# 目标：整个 predict 进程里 np.load 恰好 1 次、特征排序恰好 0~1 次、
# 均值/归一化常量在装载期算好，每行只剩一次矩阵乘。
_STATS = {"loads": 0, "rows": 0, "sorts": 0}


class ContractError(Exception):
    """检查点或样本不满足本模块声明的契约。

    predict 内部分成两级：
    - 批级（检查点读不出等产物不可用）：向上抛，harness 转 exit 1。
    - 行级（缺字段、非数值、维度不符、按 上下文 > 行 > 产物 解析出的 task 与产物冲突）：
      不外抛，该行走声明默认，其余行继续。
    """


def _classes_contract():
    """从 `manifest.json` 的 `classes_contract` 读死规则与实际标签（describe 的唯一来源）。

    读不到（镜像未带 manifest、JSON 损坏）就返回 None，describe 仍可用；绝不在
    describe 里读 `/data` 或检查点。
    """
    path = os.environ.get("CONTEST_MANIFEST") or os.path.join(
        os.path.dirname(_HERE), "manifest.json")
    try:
        with open(path, encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, ValueError):
        return None
    block = manifest.get("classes_contract") if isinstance(manifest, dict) else None
    if not isinstance(block, dict) or not isinstance(block.get("labels"), list):
        return None
    return block


def describe():
    info = {"protocol": PROTOCOL, "model_access": MODEL_ACCESS, "tasks": list(TASKS)}
    classes = _classes_contract()
    if classes is not None:
        info["classes"] = classes
    return info


def train(rnd):
    """一轮训练由 `train_round.run(rnd)` 完成，顺序与硬约束都写在那里。

    本轮（P2）固定遵守：
    1. 父检查点走 `rnd.parent.path(...)` 并自行复核 sha256，不扫 `/app`、不改 `/previous`。
    2. 每轮真的调一次 `rnd.model()` 并承担提案职责；`max_tokens` 显式传，不重复注入 `rnd.prompt`。
    3. `receipt.text` 为空（或不可解析）时退回"续训父检查点"，并记 stderr。
    4. `finish` 的 `model_call_ids` 来自 `rnd.call_ids`，为空直接整轮失败。
    5. 第 1 轮不再强制父候选：提案直接生效并真训；`reason` 只引平台返回的数字。
    """
    import train_round  # 延迟导入：describe / predict 不加载训练侧的检测器

    return train_round.run(rnd)


def _note(msg):
    sys.stderr.write("[submission] %s\n" % msg)
    sys.stderr.flush()


# ---------------------------------------------------------------- 检查点装载

class _Head(object):
    """装载期算好一切逐行常量：feature_order、动作参数、动作实现、系数矩阵。

    行路径只允许读这些字段；不得再排序、求均值或读文件。
    """

    __slots__ = ("task", "coef", "intercept", "classes", "feature_order", "order",
                 "order_source", "n_features", "prior_majority_class", "prior_mean",
                 "action", "action_params", "act", "format", "official")

    def __init__(self, task, coef, intercept, classes, feature_order, order,
                 order_source, n_features, prior_majority_class, prior_mean,
                 action, action_params, act, format="v2", official=None):
        self.task = task
        self.coef = coef
        self.intercept = intercept
        self.classes = classes
        self.feature_order = feature_order
        self.order = order
        self.order_source = order_source
        self.n_features = n_features
        self.prior_majority_class = prior_majority_class
        self.prior_mean = prior_mean
        self.action = action
        self.action_params = action_params
        self.act = act
        self.format = format
        self.official = official


def _artifact_path():
    path = os.environ.get("CONTEST_ARTIFACT_PATH")
    if not path:
        raise ContractError("未设置 CONTEST_ARTIFACT_PATH，无法定位检查点")
    if not os.path.isfile(path):
        raise ContractError("CONTEST_ARTIFACT_PATH 指向的文件不存在：%s" % path)
    return path


def _as_str(value, name):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str):
        raise ContractError("%s 必须是字符串，实际为 %r" % (name, value))
    return str(value)


def _as_str_list(array, name):
    return [_as_str(v, "%s[%d]" % (name, i)) for i, v in enumerate(list(array))]


def _fallback_classes(n_classes):
    """官方检查点缺 metadata.classes 时的确定性兜底：从 `/data/train.jsonl` 的标签集合
    取**字典序**（sorted），与行序无关（不是首次出现序）。只用标签集合，不读 answer 之外的字段。

    这是最后手段：官方训练器按配置里的 classes 顺序排 w2 的列，字典序不保证与列序一致。
    因此正常路径必须依赖 metadata.classes；这里触发时会在 stderr 明确标注可能错位。
    """
    base = os.environ.get("CONTEST_DATA_DIR") or "/data"
    path = os.path.join(base, "train.jsonl")
    seen = set()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                answer = row.get("answer") if isinstance(row, dict) else None
                if isinstance(answer, str) and answer:
                    seen.add(answer)
    except (OSError, ValueError) as exc:
        raise ContractError("官方检查点缺 metadata.classes，且无法从 %s 读标签：%s"
                            % (path, exc))
    if len(seen) != n_classes:
        raise ContractError("官方检查点缺 metadata.classes，%s 标签数 %d 与 w2 列数 %d 不一致"
                            % (path, len(seen), n_classes))
    classes = sorted(seen)
    _note("官方检查点缺 metadata.classes，按 %s 标签字典序兜底：%r（可能与 w2 列序错位）"
          % (path, classes))
    return classes


def _flatten_official(inputs, expected_paths=None):
    """按官方 `data.flatten` 的规则展开嵌套 inputs：对象键字典序、数组保持顺序、
    只接受有限数值（bool 不算数值），并生成与官方一致的路径串。

    官方 predict 会逐行比对 `paths == metadata.paths`，对不上直接报错；这里照搬，
    否则特征会按位置错位喂进 w1，分数莫名偏低还会被误读成模型不好。
    """
    import numpy as np

    paths, values = [], []

    def visit(value, path, depth):
        if depth > 16:
            raise ContractError("inputs 嵌套超过 16 层")
        if isinstance(value, dict):
            for key in sorted(value):
                if not isinstance(key, str):
                    raise ContractError("inputs 的键必须是字符串")
                norm = "".join(c for c in key.lower() if c.isalnum())
                if norm in OFFICIAL_FORBIDDEN:
                    raise ContractError("inputs 含目标派生字段 %r" % (key,))
                visit(value[key], path + [["key", key]], depth + 1)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, path + [["index", index]], depth + 1)
        elif isinstance(value, bool):
            raise ContractError("inputs 只接受有限数值，bool 不算数值")
        elif isinstance(value, (int, float)):
            number = float(value)
            if not math.isfinite(number):
                raise ContractError("inputs 含非有限数值")
            if abs(number) >= FEATURE_ABS_LIMIT:
                raise ContractError("inputs 含越界数值（|x|=%.6g ≥ %g），在 clip 之前先判越界"
                                    % (number, FEATURE_ABS_LIMIT))
            paths.append(json.dumps(path, ensure_ascii=False, separators=(",", ":")))
            values.append(number)
        else:
            raise ContractError("inputs 只接受有限数值，实际为 %r" % (value,))

    visit(inputs, [], 0)
    if not values or len(values) > 4096:
        raise ContractError("inputs 展开后应有 1..4096 个数值，实际 %d" % len(values))
    if expected_paths is not None and paths != list(expected_paths):
        raise ContractError("inputs 的特征路径与检查点 metadata.paths 不一致（%d vs %d 项）"
                            % (len(paths), len(expected_paths)))
    return np.asarray(values, dtype=float)


def _load_official(archive, key):
    """官方格式：mean/scale 标准化 → tanh(x@w1+b1) → h@w2+b2。"""
    import numpy as np

    w1 = np.asarray(archive["w1"], dtype=float)
    b1 = np.asarray(archive["b1"], dtype=float).reshape(-1)
    w2 = np.asarray(archive["w2"], dtype=float)
    b2 = np.asarray(archive["b2"], dtype=float).reshape(-1)
    mean = np.asarray(archive["mean"], dtype=float).reshape(-1)
    scale = np.asarray(archive["scale"], dtype=float).reshape(-1)

    meta = {}
    if "metadata" in archive.files:
        raw = _as_str(archive["metadata"], "metadata")
        try:
            meta = json.loads(raw)
        except ValueError as exc:
            raise ContractError("官方检查点 metadata 不是合法 JSON：%s" % exc)
        if not isinstance(meta, dict):
            raise ContractError("官方检查点 metadata 必须是 JSON 对象")

    if w1.ndim != 2 or w2.ndim != 2:
        raise ContractError("官方 w1/w2 必须是二维矩阵，实际 %r/%r" % (w1.shape, w2.shape))
    if w1.shape[1] != w2.shape[0]:
        raise ContractError("官方 w1 输出维 %d 与 w2 输入维 %d 不一致"
                            % (w1.shape[1], w2.shape[0]))
    if b1.shape != (w1.shape[1],):
        raise ContractError("官方 b1 形状应为 (%d,)，实际 %r" % (w1.shape[1], b1.shape))
    if b2.shape != (w2.shape[1],):
        raise ContractError("官方 b2 形状应为 (%d,)，实际 %r" % (w2.shape[1], b2.shape))
    n_features = int(w1.shape[0])
    if mean.shape != (n_features,) or scale.shape != (n_features,):
        raise ContractError("官方 mean/scale 维度应为 (%d,)，实际 %r/%r"
                            % (n_features, mean.shape, scale.shape))
    arrays = {"w1": w1, "b1": b1, "w2": w2, "b2": b2, "mean": mean, "scale": scale}
    for name, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise ContractError("官方 %s 含非有限值" % name)
    if np.any(scale == 0.0):
        raise ContractError("官方 scale 含 0")

    task = meta.get("task")
    if task is None:
        task = "classification" if int(w2.shape[1]) > 1 else "regression"
    if task not in TASKS:
        raise ContractError("官方检查点 task=%r 不是 %s" % (task, "/".join(TASKS)))

    classes = None
    classes_source = None
    prior_class = None
    if task == "classification":
        declared = meta.get("classes")
        if isinstance(declared, list) and len(declared) == int(w2.shape[1]):
            classes = [str(item) for item in declared]
            classes_source = "metadata.classes"
        else:
            classes = _fallback_classes(int(w2.shape[1]))
            classes_source = "train.jsonl.sorted"
        if len(set(classes)) != len(classes):
            raise ContractError("官方 classes 有重复：%r" % (classes,))
        prior_class = classes[0]
    else:
        if int(w2.shape[1]) != 1:
            raise ContractError("官方回归 w2 输出维应为 1，实际 %d" % int(w2.shape[1]))

    target_mean = float(meta.get("target_mean", 0.0))
    target_scale = float(meta.get("target_scale", 1.0))
    if task == "regression" and (not math.isfinite(target_mean)
                                or not math.isfinite(target_scale) or target_scale == 0.0):
        raise ContractError("官方回归 target_mean/target_scale 非法")

    # 官方 predict 逐行要求 flatten 出来的路径串与 metadata.paths 完全相同。
    paths = meta.get("paths")
    if paths is not None:
        if (not isinstance(paths, list) or len(paths) != n_features
                or any(not isinstance(item, str) for item in paths)):
            raise ContractError("官方 metadata.paths 应为 %d 个字符串，实际 %r"
                                % (n_features, paths))
        paths = list(paths)

    try:
        prepared = actions.prepare("identity", {})
    except actions.ActionError as exc:
        raise ContractError(str(exc))

    official = {
        "w1": np.ascontiguousarray(w1, dtype=float),
        "b1": np.ascontiguousarray(b1, dtype=float),
        "w2": np.ascontiguousarray(w2, dtype=float),
        "b2": np.ascontiguousarray(b2, dtype=float),
        "mean": np.ascontiguousarray(mean, dtype=float),
        "scale": np.ascontiguousarray(scale, dtype=float),
        "target_mean": target_mean,
        "target_scale": target_scale,
        "hidden": int(w1.shape[1]),
        "paths": paths,
    }
    head = _Head(task, None, None, classes, None, None, "official.paths", n_features,
                 prior_class, (target_mean if task == "regression" else None),
                 "identity", {}, prepared, format="official", official=official)
    _CACHE[key] = head
    _STATS["loads"] += 1
    _note("按官方格式装载检查点：task=%s hidden=%d classes_source=%s"
          % (task, head.official["hidden"], classes_source))
    return head


def _load_head(model_dir):
    """按 CONTEST_ARTIFACT_PATH 装载检查点，进程内只装一次。"""
    key = os.path.abspath(model_dir or "/model")
    if key in _CACHE:
        return _CACHE[key]

    import numpy as np

    path = _artifact_path()
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ContractError("检查点无法按 npz 读取（%s）：%s" % (path, exc))
    with archive:
        if "task" not in archive.files:
            if all(name in archive.files for name in OFFICIAL_KEYS):
                return _load_official(archive, key)
            raise ContractError("检查点既缺 task，也不是官方格式（缺 %s）"
                                % "/".join(name for name in OFFICIAL_KEYS
                                           if name not in archive.files))
        task = _as_str(archive["task"], "task")
        if task not in TASKS:
            raise ContractError("检查点 task=%r 不是 %s" % (task, "/".join(TASKS)))
        if "coef" not in archive.files:
            raise ContractError("检查点缺 coef 字段")
        coef = np.asarray(archive["coef"], dtype=float)
        if not np.all(np.isfinite(coef)):
            raise ContractError("coef 含非有限值")
        intercept = np.asarray(archive["intercept"], dtype=float) if "intercept" in archive.files else None
        classes = _as_str_list(archive["classes"], "classes") if "classes" in archive.files else None
        feature_order = (_as_str_list(archive["feature_order"], "feature_order")
                         if "feature_order" in archive.files else None)
        prior_class = (_as_str(archive["prior_majority_class"], "prior_majority_class")
                       if "prior_majority_class" in archive.files else None)
        prior_mean = None
        if "prior_mean" in archive.files:
            prior_mean = float(np.asarray(archive["prior_mean"], dtype=float).reshape(-1)[0])
            if not math.isfinite(prior_mean):
                raise ContractError("prior_mean 不是有限值")
        action = _as_str(archive["action"], "action") if "action" in archive.files else "identity"
        action_params = {}
        if "action_params" in archive.files:
            raw = _as_str(archive["action_params"], "action_params")
            try:
                action_params = json.loads(raw)
            except ValueError as exc:
                raise ContractError("action_params 不是合法 JSON：%s" % exc)
            if not isinstance(action_params, dict):
                raise ContractError("action_params 必须是 JSON 对象")

    # 动作门禁与参数校验是批级判断，只在装载时做一次：transductive 动作在 predict 侧显式拒绝，
    # 不按行兜底。返回的 step 已捕获校验后的参数，行路径不再解析参数或查表。
    try:
        prepared = actions.prepare(action, action_params)
    except actions.ActionError as exc:
        raise ContractError(str(exc))

    if task == "classification":
        if not classes:
            raise ContractError("分类检查点必须提供 classes")
        if len(set(classes)) != len(classes):
            raise ContractError("classes 有重复：%r" % (classes,))
        if coef.ndim != 2 or coef.shape[0] != len(classes):
            raise ContractError("分类 coef 形状应为 (%d, n_features)，实际为 %r"
                                % (len(classes), coef.shape))
        if intercept is None:
            intercept = np.zeros(len(classes), dtype=float)
        elif intercept.shape != (len(classes),):
            raise ContractError("分类 intercept 形状应为 (%d,)，实际为 %r" % (len(classes), intercept.shape))
        if prior_class is not None and prior_class not in classes:
            raise ContractError("prior_majority_class=%r 不在 classes 里" % prior_class)
    else:
        if coef.ndim != 1:
            raise ContractError("回归 coef 形状应为 (n_features,)，实际为 %r" % (coef.shape,))
        if intercept is None:
            intercept = 0.0
        else:
            flat = np.asarray(intercept, dtype=float).reshape(-1)
            if flat.size != 1:
                raise ContractError("回归 intercept 必须是标量")
            intercept = float(flat[0])

    if feature_order is not None:
        if len(feature_order) != coef.shape[-1]:
            raise ContractError("feature_order 长度 %d 与 coef 特征维 %d 不一致"
                                % (len(feature_order), coef.shape[-1]))

    # 装载期常量：特征顺序先钉成 tuple；缺 feature_order 时推迟到首行（只排序一次），
    # 之后行路径只做成员检查。coef 转成连续 float 数组，行路径只做一次矩阵乘。
    order = tuple(feature_order) if feature_order is not None else None
    coef = np.ascontiguousarray(coef, dtype=float)
    n_features = int(coef.shape[-1])

    head = _Head(task, coef, intercept, classes, feature_order, order,
                 "feature_order" if order is not None else "first_row", n_features,
                 prior_class, prior_mean, action, action_params, prepared)
    _CACHE[key] = head
    _STATS["loads"] += 1
    if task == "classification" and prior_class is None:
        _note("检查点未预注册 prior_majority_class，兜底将使用 classes[0]=%r" % head.classes[0])
    if task == "regression" and prior_mean is None:
        _note("检查点未预注册 prior_mean，兜底将使用 0.0")
    return head


# ---------------------------------------------------------------- 前向

def _vector(inputs, head):
    """按装载期钉好的特征顺序取值。行路径不做排序（除非检查点没给 feature_order，
    此时只在首行排序一次并把顺序钉在 head 上）。"""
    import numpy as np

    if not isinstance(inputs, dict):
        raise ContractError("inputs 不是对象")
    order = head.order
    if order is None:
        _STATS["sorts"] += 1
        order = tuple(sorted(inputs.keys()))
        head.order = order
    missing = [name for name in order if name not in inputs]
    if missing:
        raise ContractError("样本缺特征：%r" % (missing,))
    values = []
    for name in order:
        raw = inputs[name]
        if isinstance(raw, bool):
            raw = float(raw)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            raise ContractError("特征 %r 不是数值：%r" % (name, raw))
        if not math.isfinite(val):
            raise ContractError("特征 %r 不是有限值：%r" % (name, raw))
        if abs(val) >= FEATURE_ABS_LIMIT:
            raise ContractError("特征 %r 越界（|x|=%.6g ≥ %g），在进入模型前先判越界"
                                % (name, val, FEATURE_ABS_LIMIT))
        values.append(val)
    return np.asarray(values, dtype=float)


def _softmax(logits):
    peak = float(max(logits))
    exp = [math.exp(float(v) - peak) for v in logits]
    total = sum(exp)
    if not math.isfinite(total) or total <= 0.0:
        raise ContractError("softmax 归一化失败：total=%r" % total)
    return [v / total for v in exp]


def _enforce_unique_max(scores, labels):
    """分类最高分必须唯一。并列时按 label 字典序取第一个，其余压到更低分。"""
    for s in scores:
        if not math.isfinite(s):
            raise ContractError("score 非有限：%r" % s)
        if s < 0.0 or s > 1.0:
            raise ContractError("score 越界 [0,1]：%r" % s)

    for _ in range(8):
        top = max(scores)
        tied = [i for i, s in enumerate(scores) if s == top]
        if len(tied) == 1:
            return scores
        tied.sort(key=lambda i: labels[i])
        winner = tied[0]
        if top > 0.0:
            step = max(top * 1e-9, 1e-300)
            for k, idx in enumerate(tied[1:], start=1):
                scores[idx] = max(0.0, top - step * k)
        else:
            # 全部为 0 的退化行：给胜者一个可区分的最小正分，其余保持 0。
            scores[winner] = 1e-12
            for idx in tied[1:]:
                scores[idx] = 0.0
    remaining = [i for i, s in enumerate(scores) if s == max(scores)]
    if len(remaining) != 1:
        raise ContractError("无法打破并列最高分：%r" % (scores,))
    return scores


def _assert_payload(head, payload):
    """出站前再断言一遍形状与数值，不依赖上游调用顺序。"""
    if head.task == "regression":
        if not isinstance(payload, dict) or set(payload.keys()) != {"value"}:
            raise ContractError("回归输出形状不对：%r" % (payload,))
        value = payload["value"]
        if not isinstance(value, float) or not math.isfinite(value) or value < 0.0:
            raise ContractError("回归输出越界：%r" % (value,))
        return payload

    if not isinstance(payload, list) or not payload:
        raise ContractError("分类输出必须是非空列表")
    labels = []
    for item in payload:
        if not isinstance(item, dict) or set(item.keys()) != {"label", "score"}:
            raise ContractError("分类条目形状不对：%r" % (item,))
        label, score = item["label"], item["score"]
        if not isinstance(label, str) or not label:
            raise ContractError("label 必须是非空字符串：%r" % (label,))
        if not isinstance(score, float) or not math.isfinite(score) or score < 0.0 or score > 1.0:
            raise ContractError("score 越界 [0,1]：%r" % (score,))
        labels.append(label)
    if len(set(labels)) != len(labels):
        raise ContractError("label 重复：%r" % (labels,))
    scores = [item["score"] for item in payload]
    if scores.count(max(scores)) != 1:
        raise ContractError("最高分不唯一：%r" % (scores,))
    return payload


def _declared_default(head):
    """预注册的确定性兜底：分类回先验多数类（score=1），回归回先验均值（clip 到 ≥0）。"""
    if head.task == "classification":
        prior = head.prior_majority_class or head.classes[0]
        return [{"label": label, "score": 1.0 if label == prior else 0.0}
                for label in head.classes]
    prior = head.prior_mean if head.prior_mean is not None else 0.0
    return {"value": max(0.0, float(prior))}


def _resolve_task(sample, head):
    """task 优先级：上下文 > 行 > 产物。

    上下文和行都由平台生成，产物是我们自己挂的，最下游的不该当权威。
    返回 (task, source)。
    """
    ctx_task = (sample.context or {}).get("task") if sample.context else None
    if ctx_task:
        return ctx_task, "context"
    if sample.task:
        return sample.task, "row"
    return head.task, "artifact"


def _infer_official(head, sample):
    """官方格式前向：mean/scale 在输入侧，tanh 在单隐层，输出层再接 target_scale/mean。"""
    import numpy as np

    task, source = _resolve_task(sample, head)
    if source == "artifact":
        _note("本行上下文与行内 task 都缺失，按产物 task=%s 处理" % head.task)
    if task != head.task:
        raise ContractError("task 优先级取到 %s（来源 %s），与产物 task=%s 不符"
                            % (task, source, head.task))

    _STATS["rows"] += 1
    official = head.official
    vector = _flatten_official(sample.inputs, official.get("paths"))
    if vector.size != head.n_features:
        raise ContractError("样本特征维 %d 与检查点期望 %d 不一致"
                            % (vector.size, head.n_features))

    normalized = np.clip((vector - official["mean"]) / official["scale"], -10.0, 10.0)
    hidden = np.tanh(normalized @ official["w1"] + official["b1"])
    output = hidden @ official["w2"] + official["b2"]

    if head.task == "regression":
        value = float(output[0]) * official["target_scale"] + official["target_mean"]
        if not math.isfinite(value):
            raise ContractError("回归输出非有限：%r" % value)
        return {"value": max(0.0, value)}, None

    scores = _enforce_unique_max(_softmax([float(v) for v in output]), head.classes)
    payload = [{"label": label, "score": float(scores[i])}
               for i, label in enumerate(head.classes)]
    return payload, None


def _infer(head, sample, state):
    if sample.invalid:
        raise ContractError(sample.invalid)

    if head.format == "official":
        return _infer_official(head, sample)

    task, source = _resolve_task(sample, head)
    if source == "artifact":
        _note("本行上下文与行内 task 都缺失，按产物 task=%s 处理" % head.task)
    if task != head.task:
        # 优先级已经定了形状，产物产不出这个形状就是它服务不了这一行，不能硬套。
        raise ContractError("task 优先级取到 %s（来源 %s），与产物 task=%s 不符"
                            % (task, source, head.task))

    _STATS["rows"] += 1
    vector = _vector(sample.inputs, head)
    if vector.size != head.n_features:
        raise ContractError("样本特征维 %d 与检查点期望 %d 不一致"
                            % (vector.size, head.n_features))

    try:
        vector, pending = head.act["step"](vector, state, sample.dataset_key)
    except actions.ActionError as exc:
        raise ContractError("动作 %s 失败：%s" % (head.action, exc))

    if head.task == "regression":
        value = float(vector @ head.coef) + head.intercept
        if not math.isfinite(value):
            raise ContractError("回归输出非有限：%r" % value)
        return {"value": max(0.0, value)}, pending

    logits = head.coef @ vector + head.intercept
    scores = _enforce_unique_max(_softmax(logits), head.classes)
    payload = [{"label": label, "score": float(scores[i])}
               for i, label in enumerate(head.classes)]
    return payload, pending


def predict(sample, model_dir):
    """逐行预测。行级问题一律走声明默认，绝不吞行；契约冲突也按行走默认。"""
    head = _load_head(model_dir)

    try:
        payload, pending = _infer(head, sample, _STREAM_STATE)
        payload = _assert_payload(head, payload)
    except ContractError as exc:
        where = sample.lineno if getattr(sample, "lineno", None) is not None else "?"
        _note("第 %s 行走声明默认（产物 task=%s）：%s" % (where, head.task, exc))
        # 兜底分支不得更新任何流式状态：坏行不能把后续行的归一化带偏。
        # 因此这里只读 head 与 sample，不碰 _STREAM_STATE。
        return _assert_payload(head, _declared_default(head))

    # 只有本行成功出站之后才提交新状态；上面的兜底路径永远走不到这里。
    if pending is not None:
        _STREAM_STATE.clear()
        _STREAM_STATE.update(pending)
    return payload


def _debug_state():
    """供本地假宿主检查缓存状态，不参与平台协议。"""
    return {
        "cached_artifacts": sorted(_CACHE.keys()),
        "stats": dict(_STATS),
        "precomputed": {
            key: {"task": head.task, "order_source": head.order_source,
                  "n_features": head.n_features, "action": head.action,
                  "action_kind": head.act["kind"], "action_params": dict(head.act["params"])}
            for key, head in _CACHE.items()
        },
    }


if __name__ == "__main__":
    sys.stderr.write(json.dumps(describe(), ensure_ascii=False) + "\n")
