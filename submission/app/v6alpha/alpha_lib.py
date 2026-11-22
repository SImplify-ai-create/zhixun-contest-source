# -*- coding: utf-8 -*-
"""v6 纯函数库：checkpoint 权重空间插值（WiSE-FT/权重平均的公共操作）。

最强假设：本模块独立成立，不依赖主仓已有任何 α/soup 逻辑。

纪律（任务书 §2.3 + ⑥ 修补 + 契约复核四项，逐条对应）：
- 只插模型参数 (w1,b1,w2,b2)；
- mean/scale 与 metadata 里的 target_mean/target_scale **不参与插值**：
  同血缘两端必然逐位一致（续训沿用检查点状态），不一致即血缘异常，抛错；
- calibration / bias 一律弹出——主仓重训时本就 `meta.pop("calibration")`，
  换权重后旧校准失效；插值产物带一份来路不明的校准，predict 会静默改分
  （契约复核实测：α=1、数组逐位相同，预测仍被 a=1.0255 放大 2.55%）；
- Adam 一阶/二阶矩一律排除；steps/epochs 保留（取两端较大者），
  metadata 写 resume_policy——插值产物若被平台选为下轮父产物，
  续训装载必须按"优化器状态缺失 = fresh Adam + 账本记录"处理；
- 插值产物在 metadata 里声明端点 sha 与 α（留痕）；
- dtype/shape 与端点逐位一致，不一致即报错，不悄悄转换。
"""
import hashlib
import json
import os

import numpy as np

MODEL_KEYS = ("w1", "b1", "w2", "b2")
NORM_KEYS = ("mean", "scale")
OPTIMIZER_PREFIXES = ("adam_first_", "adam_second_")
PROGRESS_KEYS = ("steps", "epochs", "step", "epoch")
METADATA_KEY = "metadata"
# metadata 里两端必须逐值一致的字段（不一致 = 不同血缘，拒绝插值）
IDENTITY_FIELDS = ("task", "paths", "classes", "hidden", "format", "device")
# metadata 里必须相等的数值字段（同血缘归一化状态），不等即抛
EQUAL_SCALARS = ("target_mean", "target_scale")
# 换权重即失效、一律弹出的字段
DROP_FIELDS = ("calibration", "bias")

ARTIFACT_CAP = 64 * 1024 * 1024  # 64 MiB 契约上限


class InterpolationError(Exception):
    """端点不满足插值条件；调用方据此放弃该候选对，不静默降级。"""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_checkpoint(path):
    """读 checkpoint.npz，返回 {"arrays": {...}, "metadata": dict}。"""
    z = np.load(path, allow_pickle=False)
    arrays = {k: z[k] for k in z.files if k != METADATA_KEY}
    meta = {}
    if METADATA_KEY in z.files:
        meta = json.loads(str(z[METADATA_KEY].item()))
    return {"arrays": arrays, "metadata": meta}


def _is_optimizer_key(key):
    return any(key.startswith(p) for p in OPTIMIZER_PREFIXES)


def _check_endpoint(ckpt, tag):
    missing = [k for k in MODEL_KEYS + NORM_KEYS if k not in ckpt["arrays"]]
    if missing:
        raise InterpolationError("%s 缺数组：%s" % (tag, ",".join(missing)))


def interpolate_checkpoints(path_a, path_b, alpha):
    """W(α) = (1−α)·W_a + α·W_b。返回 {"arrays", "metadata"}，不落盘。

    数组一律在 float64 里混合后 cast 回端点 dtype——α=0/1 时与端点逐位相同
    （float32→float64→float32 往返精确）。
    """
    if not 0.0 <= float(alpha) <= 1.0:
        raise InterpolationError("alpha 必须在 [0,1]：%r" % (alpha,))
    a, b = load_checkpoint(path_a), load_checkpoint(path_b)
    _check_endpoint(a, "端点a")
    _check_endpoint(b, "端点b")

    keys = set(a["arrays"]) | set(b["arrays"])
    out_arrays = {}
    for key in sorted(keys):
        if _is_optimizer_key(key) or key in PROGRESS_KEYS:
            continue  # 优化器状态与计数数组：排除，不进入插值产物
        if key in MODEL_KEYS:
            va, vb = a["arrays"].get(key), b["arrays"].get(key)
            if va is None or vb is None:
                raise InterpolationError("数组 %s 只在一端存在" % key)
            if va.shape != vb.shape or va.dtype != vb.dtype:
                raise InterpolationError(
                    "数组 %s 端点不一致：%s/%s vs %s/%s"
                    % (key, va.shape, va.dtype, vb.shape, vb.dtype))
            mixed = ((1.0 - alpha) * va.astype(np.float64)
                     + alpha * vb.astype(np.float64))
            out_arrays[key] = mixed.astype(va.dtype)
        else:
            # mean/scale 与未知数组：两端逐位一致才携带。
            # 同血缘续训沿用归一化状态，不一致 = 血缘异常；
            # 在这里插值等于静默做归一化加权平均（契约复核第三条）。
            va, vb = a["arrays"].get(key), b["arrays"].get(key)
            if va is None or vb is None or va.shape != vb.shape \
                    or va.dtype != vb.dtype or not np.array_equal(va, vb):
                raise InterpolationError(
                    "数组 %s 两端不一致（血缘异常），拒绝插值" % key)
            out_arrays[key] = va

    meta = _merge_metadata(a["metadata"], b["metadata"], alpha,
                           sha256_file(path_a), sha256_file(path_b))
    return {"arrays": out_arrays, "metadata": meta}


def _merge_metadata(ma, mb, alpha, sha_a, sha_b):
    for field in IDENTITY_FIELDS:
        if ma.get(field) != mb.get(field):
            raise InterpolationError(
                "metadata.%s 两端不一致（血缘不同），拒绝插值" % field)
    for key in EQUAL_SCALARS:
        if ma.get(key) != mb.get(key):
            raise InterpolationError(
                "metadata.%s 两端不一致（归一化状态分叉），拒绝插值" % key)
    out = dict(ma)
    for key in DROP_FIELDS:
        out.pop(key, None)  # 换权重后旧校准/旧偏置失效，与主仓重训口径一致
    # steps/epochs 保留且取大者：产物可能被平台选为下轮父产物，
    # 计数倒退比"优化器状态缺失"更难解释。
    for key in PROGRESS_KEYS:
        va, vb = ma.get(key), mb.get(key)
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
            out[key] = max(va, vb)
    out["interpolation"] = {
        "a_sha256": sha_a, "b_sha256": sha_b, "alpha": float(alpha),
        "optimizer_state": "dropped",
        "resume_policy": "fresh_adam_on_missing_state",
        "usage": "predict_and_submit_only",
        # steps/epochs 是续训排程锚点（取两端较大者），不是训练步数；
        # 产物本体是权重空间插值，未经训练。不写清就是新一条账本矛盾。
        "steps_semantics": "schedule_anchor_not_trained_steps",
    }
    return out


def write_checkpoint(ckpt, path, cap=ARTIFACT_CAP):
    """np.savez 不压缩（与主仓三处写点一致）；tmp+replace 原子写——比主仓严格
    （主仓只有 bias_family 原子写，train_mlp/train_round 就地直写）。"""
    payload = dict(ckpt["arrays"])
    payload[METADATA_KEY] = np.array(
        json.dumps(ckpt["metadata"], ensure_ascii=False, sort_keys=True))
    tmp = path + ".tmp-%d" % os.getpid()
    try:
        with open(tmp, "wb") as fh:  # 文件对象写入，避免 savez 自动追加 .npz
            np.savez(fh, **payload)
        size = os.path.getsize(tmp)
        if size > cap:
            raise InterpolationError(
                "产物 %d B 超契约上限 %d B" % (size, cap))
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return os.path.getsize(path)


def assert_artifact_size(path, cap=ARTIFACT_CAP):
    size = os.path.getsize(path)
    if size > cap:
        raise InterpolationError("产物 %d B 超契约上限 %d B" % (size, cap))
    return size


def lambda_init(eta, steps, r_star):
    """L2-SP 初值：常梯度近似 λ = 2(1−r*)/(η·(T+1))。

    r_star = d_target/d₀；r_star ≥ 1（无需阻尼）返回 0.0。
    仅作二分起点，实测 rel_l2 才是判据（见 02-lambda-标定.md）。
    """
    eta, steps, r_star = float(eta), float(steps), float(r_star)
    if eta <= 0 or steps <= 0:
        raise InterpolationError("eta/steps 必须为正：%r/%r" % (eta, steps))
    if r_star >= 1.0:
        return 0.0
    return 2.0 * (1.0 - r_star) / (eta * (steps + 1.0))


def damping_ratio(eta, lam, steps):
    """常梯度近似下的位移衰减比 r(λ)=[1−(1−ηλ)^T]/(ηλT)，λ→0 时趋于 1。"""
    eta, lam, steps = float(eta), float(lam), float(steps)
    if lam <= 0.0:
        return 1.0
    x = eta * lam
    if x >= 1.0:
        return 1.0 / (x * steps)  # (1−x)^T ≤ 0 的退化情形按全阻尼处理
    return (1.0 - (1.0 - x) ** steps) / (x * steps)
