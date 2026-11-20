"""规则程序编译与执行单测（不依赖真实数据）。"""
import json

import pytest

from src.rule_program import (RuleProgramController, RuleProgramError,
                              compile_program, validate_program)


def _prog(rules, default="follow_severity"):
    return {"program_id": "t-1", "default_tier": default, "rules": rules}


def test_valid_program_compiles():
    p = _prog([{"id": "r1",
                "if": {"all": [
                    {"feature": "residual_ratio", "op": ">=", "value": 4.0},
                    {"feature": "labels_left", "op": ">=", "value": 1200}]},
                "then": {"tier": "full"}}])
    c = compile_program(p)
    tier, rid = c.evaluate({"residual_ratio": 5.0, "labels_left": 2000})
    assert (tier, rid) == ("full", "r1")


def test_depth_over_two_rejected():
    p = _prog([{"id": "r1",
                "if": {"all": [{"any": [
                    {"feature": "severity", "op": ">", "value": 1.0}]}]},
                "then": {"tier": "observe"}}])
    errs = validate_program(p)
    assert any("深度" in e for e in errs)


def test_unknown_feature_rejected():
    p = _prog([{"id": "r1",
                "if": {"feature": "not_a_feature", "op": ">", "value": 1},
                "then": {"tier": "observe"}}])
    assert any("白名单" in e for e in validate_program(p))


def test_bad_tier_and_op_rejected():
    p = _prog([{"id": "r1", "if": {"feature": "severity", "op": "~", "value": 1},
                "then": {"tier": "retrain"}}])
    errs = validate_program(p)
    assert any("op" in e for e in errs)
    assert any("tier" in e for e in errs)


def test_first_match_wins():
    p = _prog([
        {"id": "first", "if": {"feature": "severity", "op": ">", "value": 1.0},
         "then": {"tier": "observe"}},
        {"id": "second", "if": {"feature": "severity", "op": ">", "value": 0.0},
         "then": {"tier": "full"}},
    ])
    c = compile_program(p)
    assert c.evaluate({"severity": 2.0}) == ("observe", "first")


def test_default_follows_severity():
    c = compile_program(_prog([]))
    ctl = RuleProgramController(c)
    out = ctl.decide({"tier": "incremental", "labels_left": 4000})
    assert out["tier"] == "incremental"


def test_budget_guard_downgrades_full():
    c = compile_program(_prog([
        {"id": "r", "if": {"feature": "residual_ratio", "op": ">", "value": 1.0},
         "then": {"tier": "full"}}]))
    ctl = RuleProgramController(c)
    assert ctl.decide({"residual_ratio": 5.0, "labels_left": 300})["tier"] == "incremental"
    assert ctl.decide({"residual_ratio": 5.0, "labels_left": 1200})["tier"] == "full"


def test_missing_feature_is_conservative():
    c = compile_program(_prog([
        {"id": "r", "if": {"feature": "psi", "op": ">", "value": 2.0},
         "then": {"tier": "full"}}]))
    assert c.evaluate({"severity": 5.0}) == (None, None)


def test_any_branch_and_equality():
    c = compile_program(_prog([
        {"id": "r", "if": {"any": [
            {"feature": "retrains_left", "op": "==", "value": 0},
            {"feature": "task_alarm", "op": "==", "value": True}]},
         "then": {"tier": "observe"}}]))
    assert c.evaluate({"retrains_left": 0})[1] == "r"
    assert c.evaluate({"task_alarm": True})[1] == "r"
    assert c.evaluate({"retrains_left": 2, "task_alarm": False})[0] is None


def test_duplicate_ids_rejected():
    p = _prog([{"id": "dup", "if": {"feature": "severity", "op": ">", "value": 1},
                "then": {"tier": "observe"}},
               {"id": "dup", "if": {"feature": "severity", "op": ">", "value": 2},
                "then": {"tier": "full"}}])
    assert any("重复" in e for e in validate_program(p))


def test_compile_raises_on_invalid():
    with pytest.raises(RuleProgramError):
        compile_program({"rules": "not-a-list"})
