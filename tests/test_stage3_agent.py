"""Stage-3 agent loop tests (no API). Sim-based tests build the edge-slide env (~seconds each)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from imbench.vlm_eval.client import Ledger
from imbench.vlm_eval.stage3 import primitives as P
from imbench.vlm_eval.stage3.agent import EpisodeConfig, MockToolClient, build_user_content, run_episode, system_prompt
from imbench.vlm_eval.stage3.memory import FrameHistory, ReactMemory
from imbench.vlm_eval.stage3.plan import ORACLE_PLAN_TEXT, load_plan
from imbench.vlm_eval.stage3.tools import parse_call, tool_schemas

STAGE2 = Path("runs/gpt-6-astra/edge-slide/seed_0/sample_0/stage2.json")
SCRIPTED = Path("runs/stage3_scripted/results.json")


def _args(params, **kw):
    return json.dumps({"reasoning": "r", "description": "d", "active_subgoal_id": "S1",
                       "subgoal_status": "in_progress", "params": params, **kw})


# ── schemas ─────────────────────────────────────────────────────────────────

def _check_strict(schema):
    if schema.get("type") == "object":
        assert schema["additionalProperties"] is False
        assert set(schema["required"]) == set(schema["properties"])
        for v in schema["properties"].values():
            _check_strict(v)


def test_tool_schemas_strict():
    tools = tool_schemas()
    assert [t["name"] for t in tools] == ["MOVE_EEF", "CLOSE_GRIPPER", "OPEN_GRIPPER", "LIFT", "DESCEND",
                                         "ATOMIC_ACTION", "HOLD", "DONE"]
    for t in tools:
        assert t["strict"] is True
        _check_strict(t["parameters"])
        assert {"reasoning", "description", "active_subgoal_id", "subgoal_status", "params"} <= set(
            t["parameters"]["required"])


@pytest.mark.parametrize("name,args,ok", [
    ("MOVE_EEF", _args({"position_m": [0.1, 0, 0.9], "orientation_rpy_deg": None, "max_steps": 300}), True),
    ("MOVE_EEF", _args({"position_m": [0.1, 0, 0.9], "orientation_rpy_deg": [180, 0, 0], "max_steps": 300}), True),
    ("MOVE_EEF", _args({"position_m": [0.1, 0.9], "orientation_rpy_deg": None, "max_steps": 300}), False),
    ("ATOMIC_ACTION", _args({"action7": [0, 0, 0, 0, 0, 0, 2], "n_ticks": 5}), False),
    ("ATOMIC_ACTION", _args({"action7": [0, 0, 0.2, 0, 0, 0, -1], "n_ticks": 5}), True),
    ("HOLD", _args({"n_ticks": 0}), False),
    ("DONE", _args({}), True),
    ("DONE", _args({}, subgoal_status="finished"), False),
    ("TELEPORT", _args({}), False),
    ("LIFT", '{"reasoning": ', False),
    (None, None, False),
])
def test_parse_call(name, args, ok):
    call, err = parse_call(name, args)
    assert (call is not None) == ok, err


# ── memory windowing ────────────────────────────────────────────────────────

def test_memory_window():
    m = ReactMemory(H=3)
    st = {"eef_pos_m": [0, 0, 1], "eef_rpy_deg": [180, 0, 0], "gripper_opening_m": 0.08}
    for k in range(1, 6):
        m.add(k, st, "HOLD", {"n_ticks": 1}, {"ticks": 1}, 0.0)
    assert [e["step"] for e in m.entries] == [3, 4, 5]
    txt = m.render()
    assert "step 3" in txt and "step 2" not in txt
    h = FrameHistory(H=2)
    for k in range(1, 5):
        h.add(k, [("cam", f"f{k}.jpg")])
    assert [s for s, _ in h.batches] == [3, 4]


# ── plan injection ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not STAGE2.exists(), reason="needs the Stage-2 sample for seed 0")
def test_plan_stage2_injected():
    text, meta = load_plan("stage2", 0, 0)
    plan = json.loads(STAGE2.read_text())["plan"]
    for g in plan["subgoals"]:
        assert g["name"] in text and g["physical_outcome"] in text
    assert meta["plan_seed"] == 0 and "seed 0" in text
    st = {"eef_pos_m": [0, 0, 1]}
    parts, rec = build_user_content(1, 24, text, FrameHistory(12), st, {}, None, [], ReactMemory(12), 0)
    assert any(g["name"] in p.get("text", "") for p in parts for g in plan["subgoals"])


def test_plan_stage2_missing_seed_fails_loudly():
    with pytest.raises(FileNotFoundError):
        load_plan("stage2", 987, 0)


def test_plan_oracle_and_none():
    text, _ = load_plan("oracle", 0)
    assert ORACLE_PLAN_TEXT in text and "overhang" in text
    assert load_plan("none", 0)[0] is None
    assert "No plan is given" in system_prompt(24, False)
    assert "PLAN block" in system_prompt(24, True)


def test_system_prompt_has_measured_values():
    s = system_prompt(24, True)
    for frag in ("at least 15 cm above the table surface", "z >= 0.95 m", "0.80 m", "(180, 0, 0)", "EXTRINSIC X-Y-Z", "SIDE face", "0.003 m", "Pressing down"):
        assert frag in s, frag
    assert "{" not in s.replace("{n", "")  # every slot filled


# ── RPY / register (pure) ───────────────────────────────────────────────────

def test_rpy_round_trip():
    for rpy in ([180, 0, 0], [-90, 0, 0], [170, 15, -60]):
        back = P.quat_to_rpy_deg(P.rpy_deg_to_rot(rpy).as_quat())
        assert np.degrees((P.rpy_deg_to_rot(back) * P.rpy_deg_to_rot(rpy).inv()).magnitude()) < 1e-6


# ── full loop in sim (mock client) ──────────────────────────────────────────

def _run(calls, tmp_path, **kw):
    ledger = kw.pop("ledger", None)
    cfg = EpisodeConfig(seed=0, plan_text=None, plan_meta={"plan_source": "none"}, record=False, **kw)
    return run_episode(MockToolClient(calls, ledger), cfg, tmp_path, log=lambda *_: None)


@pytest.mark.skipif(not SCRIPTED.exists(), reason="needs runs/stage3_scripted/results.json (step 3)")
def test_mock_replay_of_scripted_sequence_succeeds(tmp_path):
    from imbench.vlm_eval.run_stage3 import scripted_calls
    s = _run(scripted_calls(0), tmp_path)
    assert s["success"] and s["exit_reason"] == "success"
    recs = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert all(r["valid"] for r in recs)
    # frame history and memory grow in later prompts
    p5 = json.loads((tmp_path / "step05_prompt.json").read_text())["content"]
    assert sum(1 for c in p5 if c.get("detail") == "low") == 4 * 4          # 4 history batches x 4 views
    assert any("step 4:" in c.get("text", "") for c in p5)


def test_failures_execute_hold_and_continue(tmp_path):
    calls = [{"malformed": True}, {"unknown": True}, {"timeout": True},
             {"name": "DONE", "args": {"params": {}}}]
    s = _run(calls, tmp_path)
    recs = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [r["executed"] for r in recs] == ["HOLD", "HOLD", "HOLD", "DONE"]
    assert "malformed" in recs[0]["error"] and "unknown tool" in recs[1]["error"] and "API failure" in recs[2]["error"]
    assert s["exit_reason"] == "done" and not s["success"]


def test_exit_max_steps(tmp_path):
    s = _run([{"name": "HOLD", "args": {"params": {"n_ticks": 5}}}] * 5, tmp_path, K=3)
    assert s["exit_reason"] == "max_steps" and s["steps"] == 3


def test_exit_tick_cap(tmp_path):
    s = _run([{"name": "HOLD", "args": {"params": {"n_ticks": 200}}}] * 5, tmp_path, tick_cap=250)
    assert s["exit_reason"] == "tick_cap" and s["ticks"] <= 250


def test_exit_budget(tmp_path):
    s = _run([{"name": "HOLD", "args": {"params": {"n_ticks": 5}}}] * 5, tmp_path, ledger=Ledger(budget_usd=0.0))
    assert s["exit_reason"] == "budget" and s["ticks"] == 0
