"""Offline tests for imbench.vlm_eval (no API calls, no MuJoCo)."""

from __future__ import annotations

import json

import pytest
from PIL import Image

from imbench.vlm_eval import mock, stage1, stage2
from imbench.vlm_eval.client import BudgetExceeded, Ledger, VLMClient, cost_usd
from imbench.vlm_eval.common import load_prompt
from imbench.vlm_eval.grade_edge_slide import load_rubric, mean_std, stage1_pass, stage2_pass
from imbench.vlm_eval.schemas import (
    ConstraintArtifact, Stage2Plan, audit_mean, check_ref, gate_stage1, gate_stage2,
)


@pytest.fixture
def bundle(tmp_path):
    (tmp_path / "frames").mkdir()
    frames = []
    for cam in ("exo_right_cam", "ego_cam"):
        for i in range(3):
            f = f"frames/{cam}_f{i}.jpg"
            Image.new("RGB", (8, 8)).save(tmp_path / f)
            frames.append({"camera": cam, "frame_idx": i, "t_s": i * 0.5, "file": f})
    md = {"task_description": "Lift the plate above the table.",
          "controller_interface": {"end_effector": "parallel jaw"},
          "cameras": {"exo_right_cam": "right", "ego_cam": "front"},
          "sensor_snapshot": [{"frame_idx": 0, "wrist_force_N": [0, 0, -5]}]}
    return {"frames": frames, "metadata": md}, tmp_path


# ── schemas / gate ───────────────────────────────────────────────────────

def test_schemas_strict_json_schema():
    from openai.lib._pydantic import to_strict_json_schema
    for m in (ConstraintArtifact, Stage2Plan):
        s = to_strict_json_schema(m)
        assert s["additionalProperties"] is False


def test_evidence_refs(bundle):
    b, _ = bundle
    assert check_ref("cam:exo_right_cam:f2", b)
    assert not check_ref("cam:exo_right_cam:f3", b)
    assert not check_ref("cam:sideview:f0", b)
    assert check_ref("meta:controller_interface.end_effector", b)
    assert check_ref("meta:sensor_snapshot.0.wrist_force_N", b)
    assert not check_ref("meta:sensor_snapshot.1.wrist_force_N", b)
    assert not check_ref("meta:object_pose", b)
    assert not check_ref("exo_right_cam frame 0", b)


def test_gate_stage1(bundle):
    b, _ = bundle
    art = mock.stage1_artifact()
    assert gate_stage1(art, b) == []
    bad = art.model_copy(deep=True)
    bad.constraints[0].evidence_refs = ["cam:nope:f0"]
    bad.episodic[0].frame_idx = 9
    errs = gate_stage1(bad, b)
    assert any("invalid evidence_ref" in e for e in errs) and any("unknown frame" in e for e in errs)
    empty = art.model_copy(update={"constraints": []})
    assert "no constraints" in gate_stage1(empty, b)


def test_gate_stage2_and_audit():
    plan = mock.stage2_plan(0.9)
    assert gate_stage2(plan, {"C1"}) == []
    assert gate_stage2(plan, {"C2"})          # cites unknown C1
    assert audit_mean(plan) == pytest.approx(0.9)
    plan.self_audit.F1_all_prerequisites_included.score = 7    # clipped to 1
    assert audit_mean(plan) == pytest.approx((1 + 0.9 + 0.9) / 3)


# ── stage loops (mock client) ────────────────────────────────────────────

def test_stage1_reruns_on_gate_failure(bundle):
    b, d = bundle
    calls = []

    def fn(tag, schema, n):
        calls.append(tag)
        art = mock.stage1_artifact()
        if n < 3:
            art.constraints[0].evidence_refs = ["bogus"]
        return art
    logs = []
    r = stage1.run(VLMClient(mock=True, mock_fn=fn), b, d, logs.append)
    assert r["passed"] and r["attempts"] == 3 and len(logs) == 3
    assert [l["gate_passed"] for l in logs] == [False, False, True]


def test_stage2_retry_keeps_best(bundle):
    b, d = bundle
    scores = [0.5, 0.7, 0.6, 0.65, 0.4, 0.3]
    fn = lambda tag, schema, n: mock.stage2_plan(scores[n - 1])
    logs = []
    r = stage2.run(VLMClient(mock=True, mock_fn=fn), mock.stage1_artifact(), b, d, logs.append)
    assert r["attempts"] == 6 and r["best_attempt"] == 2 and not r["accepted"]
    assert r["audit_mean"] == pytest.approx(0.7)


def test_stage2_stops_when_accepted(bundle):
    b, d = bundle
    r = stage2.run(VLMClient(mock=True), mock.stage1_artifact(), b, d, lambda x: None)
    assert r["attempts"] == 2 and r["accepted"]       # canned: 0.6 then 0.9
    # retry prompt carries the previous plan + audit
    assert "REVISION REQUEST" in json.dumps(r["prompts"][1])


def test_budget_stop(bundle):
    b, d = bundle
    c = VLMClient(mock=True, ledger=Ledger(budget_usd=1.0), mock_fn=lambda t, s, n: mock.stage2_plan(0.5))
    c.mock_cost_usd = 0.6
    with pytest.raises(BudgetExceeded):
        stage2.run(c, mock.stage1_artifact(), b, d, lambda x: None)
    assert c.ledger.calls == 2


def test_cost():
    assert cost_usd(1_000_000, 0) == pytest.approx(10.0)
    assert cost_usd(0, 1_000_000) == pytest.approx(50.0)


# ── ground truth must not leak into normal prompts ───────────────────────

@pytest.mark.parametrize("name", ["stage1_system.md", "stage2_system.md", "stage2_retry.md"])
def test_no_gt_leak(name):
    import re
    p = load_prompt(name).lower()
    for w in ("overhang", "lip", "edge", "slide", "slid", "slidi", "push", "plate", "thin"):
        assert not re.search(rf"\b{w}\w*", p), f"{name} contains ground-truth hint {w!r}"


def test_gt_artifact_valid():
    art = stage2.gt_artifact()
    assert {c.id for c in art.constraints} == {"C1", "C2", "C3"}


# ── grading ──────────────────────────────────────────────────────────────

def test_grading_logic():
    r = load_rubric()
    ok = {i["id"]: True for i in r["stage1"]["required"]} | {i["id"]: False for i in r["stage1"]["forbidden"]}
    assert stage1_pass(r, ok)
    assert not stage1_pass(r, ok | {"S1.R2": False})
    assert not stage1_pass(r, ok | {"S1.F3": True})
    ok2 = {i["id"]: True for i in r["stage2"]["criteria"]} | {i["id"]: False for i in r["stage2"]["forbidden"]}
    assert stage2_pass(r, ok2) and not stage2_pass(r, ok2 | {"S2.F1": True})
    m, sd = mean_std([1, 1, 1, 1, 0])
    assert m == pytest.approx(80.0) and sd == pytest.approx(44.72, abs=0.01)
