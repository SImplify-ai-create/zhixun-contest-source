# -*- coding: utf-8 -*-
"""v6 纯函数：L2-SP 软近端——惩罚项、梯度项与 λ 标定协议。

独立成立，不依赖主仓；训练循环通过 `measure_fn(lambda) -> rel_l2` 注入，
本模块不碰任何训练代码。纪律（任务书 §3 + 用户提醒）：
- λ=0 时惩罚与梯度恒为零——端点恒等，模块不工作时不留痕迹；
- d₀ 在 [0, 1.1]×目标内不触发（λ=0），它的价值是防尾部，
  不许为了让读数显得在工作而调低触发门槛；
- 标定全程留 trace，耗尽未收敛必须显式标记，不许静默取边界。
"""
from .alpha_lib import InterpolationError, damping_ratio, lambda_init

TRIGGER_TOL = 0.1   # d₀ ≤ (1+TOL)×目标 → 不触发
CONVERGE_TOL = 0.1  # rel_l2 ∈ [0.9, 1.1]×目标 → 收敛
MAX_ITER = 6
# rel_l2 ≤ 此值 = 训练端点与起点逐位相同：没有任何一轮跑赢父产物，
# 位移恒为 0。这不是"阻尼恰好"的读数，而是"锚把模型冻住"的退化点，
# 不能当成可收敛点参与选点（2026-11-22 修正）。
DEGENERATE_EPS = 1e-12


class CalibrationError(InterpolationError):
    """标定输入不合法；调用方据此放弃本轮 L2-SP，按 λ=0 走。"""


def l2sp_penalty(params, start, lam):
    """(λ/2)·‖W − W_start‖²，params/start 为 {name: ndarray}。λ=0 恒为 0。"""
    lam = float(lam)
    if lam == 0.0:
        return 0.0
    if lam < 0.0:
        raise CalibrationError("lambda 必须非负：%r" % (lam,))
    total = 0.0
    for name, value in params.items():
        diff = value - start[name]
        total += float((diff * diff).sum())
    return 0.5 * lam * total


def l2sp_gradient(params, start, lam):
    """λ·(W − W_start) 逐张量。λ=0 返回全零（端点恒等的可检验形式）。"""
    lam = float(lam)
    if lam < 0.0:
        raise CalibrationError("lambda 必须非负：%r" % (lam,))
    return {name: lam * (value - start[name])
            for name, value in params.items()}


def plan_lambda(d0, d_target, eta, steps, trigger_tol=TRIGGER_TOL):
    """判定是否触发并给初值与二分区间。返回决策字典（不执行测量）。"""
    d0, d_target = float(d0), float(d_target)
    if d_target <= 0:
        raise CalibrationError("d_target 必须为正：%r" % (d_target,))
    if d0 <= (1.0 + trigger_tol) * d_target:
        return {"triggered": False, "lambda": 0.0,
                "reason": "d0=%.4f 在 %.1f%% 带内（目标 %.4f），λ=0"
                          % (d0, trigger_tol * 100, d_target)}
    r_star = d_target / d0
    lam0 = lambda_init(eta, steps, r_star)
    return {"triggered": True, "r_star": r_star, "lambda_init": lam0,
            "bracket": (lam0 / 4.0, 4.0 * lam0)}


def calibrate_lambda(measure_fn, d_target, eta, steps,
                     trigger_tol=TRIGGER_TOL, converge_tol=CONVERGE_TOL,
                     max_iter=MAX_ITER):
    """标定协议：探针 → 判定 → 二分。measure_fn(lambda) -> rel_l2 由训练侧注入。

    返回 {"lambda", "triggered", "unconverged", "trace"}；
    trace 每项 {"lambda", "rel_l2"}，探针为第一条。
    """
    d0 = float(measure_fn(0.0))
    trace = [{"lambda": 0.0, "rel_l2": d0}]
    plan = plan_lambda(d0, d_target, eta, steps, trigger_tol)
    if not plan["triggered"]:
        return {"lambda": 0.0, "triggered": False, "unconverged": False,
                "d0": d0, "trace": trace, "reason": plan["reason"]}
    lo, hi = plan["bracket"]
    best = None
    probed = 0
    for _ in range(max_iter):
        mid = (lo * hi) ** 0.5  # λ 跨数量级，几何中点
        rel = float(measure_fn(mid))
        probed += 1
        trace.append({"lambda": mid, "rel_l2": rel})
        if rel <= DEGENERATE_EPS:
            # 退化点：锚把训练冻住（没有任何一轮被接受），位移恒 0。
            # 语义是"锚过强"，不是"阻尼刚好"；既不参与 best，也向更小 λ 收缩。
            hi = mid
            continue
        if abs(rel - d_target) <= converge_tol * d_target:
            return {"lambda": mid, "triggered": True, "unconverged": False,
                    "d0": d0, "trace": trace}
        if best is None or abs(rel - d_target) < abs(best[1] - d_target):
            best = (mid, rel)
        if rel > d_target:  # 阻尼不足
            lo = mid
        else:               # 阻尼过度
            hi = mid
    if best is None:
        # 所有探针都把训练冻住：目标带在"仍可训练"的区间里不可达。
        # 正确处置是关掉锚（λ=0，正则不介入），而不是退回最冻的那个 λ。
        return {"lambda": 0.0, "triggered": False, "unconverged": True,
                "degenerate": True, "d0": d0, "trace": trace,
                "reason": "全部 %d 次探针把训练冻住（rel_l2≤%.0e），"
                          "目标带不可达，回退 λ=0（L2-SP 不介入）"
                          % (probed, DEGENERATE_EPS)}
    # 耗尽：取最接近目标的**有效**实测点并显式标记，不静默收边
    lam_star = best[0]
    return {"lambda": lam_star, "triggered": True, "unconverged": True,
            "d0": d0, "trace": trace,
            "reason": "%d 次二分未进 %.0f%% 带，取最接近实测点"
                      % (max_iter, converge_tol * 100)}


def expected_rel_l2(d0, lam, eta, steps):
    """常梯度近似下的预期 rel_l2（仅供标定日志对照，不作判据）。"""
    return float(d0) * damping_ratio(eta, lam, steps)
