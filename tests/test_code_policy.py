"""Code-as-Policies mode (Stage 3, --mode code): sandbox, runner limits, schemas, and the mock loop in sim.
No API calls."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from imbench.vlm_eval.stage3 import code_policy as C


# ── fake executor (no sim) ──────────────────────────────────────────────────

class FakeEx:
    def __init__(self, stop_after_ticks=None):
        self.st = SimpleNamespace(ticks=0, success=False, terminated=False, done_called=False)
        self.calls = []
        self.stop_after = stop_after_ticks
        self.pos = np.array([0.0, 0.0, 1.0])
        self.rpy = np.array([180.0, 0.0, 0.0])
        self.width = 0.08

    @property
    def stopped(self):
        return self.st.success or self.st.terminated

    def force_world(self):
        return np.zeros(3)

    def tactile_max(self):
        return 0.0

    def run(self, name, params):
        self.calls.append((name, params))
        self.st.ticks += 10
        if name == "MOVE_EEF":
            self.pos = np.array(params["position_m"])
        if self.stop_after is not None and self.st.ticks >= self.stop_after:
            self.st.success = True
        return {"converged": True, "final_position_m": self.pos.tolist()}

    def done(self):
        self.st.done_called = True
        return {"done": True}


def runner(**kw):
    ex = kw.pop("ex", None) or FakeEx()
    return C.CodeRunner(ex, lambda: {"eef_pos_m": ex.pos.tolist()}, kw.pop("objects_fn", None), **kw), ex


# ── static check ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("code", [
    "import os", "from math import pi", "x = ().__class__", "open('f')", "eval('1')", "np.save('a', 1)",
    "a = np.zeros(3); a.tofile('x')", "try:\n    pass\nexcept:\n    pass",
    "try:\n    pass\nexcept BaseException:\n    pass", "def f():\n    global x", "getattr(np, 'x')",
    "class A:\n    pass", "_x = 1", "type(1)", "", "x = (", "np.lib.format",
])
def test_check_code_rejects(code):
    assert C.check_code(code) is not None


def test_check_code_accepts_normal_program():
    code = ("def f(a, b=2):\n    return a * b\n"
            "for i in range(3):\n    try:\n        r = move_eef([0.1, 0, 0.9], None, 300)\n"
            "    except Exception as e:\n        print(e)\n"
            "x = [abs(v) for v in np.array([1.0, -2.0])]\nresult = {'x': x, 'f': f(3)}\n")
    assert C.check_code(code) is None


# ── runner ─────────────────────────────────────────────────────────────────

def test_print_result_and_primitive_log():
    r, ex = runner()
    out = r.run("print('hi', 1)\nmove_eef([0.1, 0.0, 0.9])\nresult = {'p': eef_pos()}")
    assert out["status"] == "ok" and out["stdout"] == "hi 1\n" and out["primitive_calls"] == 1
    assert out["primitive_log"][0]["call"] == "move_eef"
    assert out["primitive_log"][0]["args"] == {"position_m": [0.1, 0.0, 0.9], "orientation_rpy_deg": None,
                                               "max_steps": 400}
    assert out["result_variable"] == {"p": [0.1, 0.0, 0.9]}
    assert ex.calls == [("MOVE_EEF", {"position_m": [0.1, 0.0, 0.9], "orientation_rpy_deg": None, "max_steps": 400})]


def test_namespace_persists_and_old_functions_see_new_globals():
    r, _ = runner()
    r.run("k = 2\ndef g():\n    print('g', k)\n    return k * 10")
    out = r.run("k = 5\nresult = g()")
    assert out["result_variable"] == 50 and out["stdout"] == "g 5\n"     # print goes to the CURRENT turn
    assert "g" in out["persistent_names"] and "k" in out["persistent_names"]
    out = r.run("print('x')")
    assert "result_variable" not in out                                   # `result` is per turn


def test_api_cannot_be_shadowed_across_turns():
    r, ex = runner()
    r.run("move_eef = 5")
    out = r.run("move_eef([0.0, 0.1, 0.9])")
    assert out["status"] == "ok" and len(ex.calls) == 1


def test_invalid_args_raise_valueerror_catchable_and_uncaught():
    r, ex = runner()
    out = r.run("try:\n    move_eef([0.1, 0.9])\nexcept ValueError as e:\n    print('caught', e)")
    assert out["status"] == "ok" and "caught move_eef(): invalid arguments" in out["stdout"] and not ex.calls
    out = r.run("hold(0)\nprint('never')")
    assert out["status"] == "error" and "ValueError" in out["error"] and "line 1" in out["error"]
    assert out["stdout"] == ""
    out = r.run("x = 1 / 0")
    assert out["status"] == "error" and "ZeroDivisionError" in out["error"]


def test_rejected_program_runs_nothing():
    r, ex = runner()
    out = r.run("import os\nmove_eef([0, 0, 1])")
    assert out["status"] == "rejected" and "import" in out["error"] and not ex.calls


def test_wall_clock_limit_not_catchable():
    r, _ = runner(wall_s=0.5)
    out = r.run("while True:\n    try:\n        x = 1\n    except Exception:\n        pass")
    assert out["status"] == "turn_limit" and "wall clock" in out["error"]


def test_primitive_and_tick_caps():
    r, ex = runner(max_primitives=3)
    out = r.run("while True:\n    try:\n        hold(10)\n    except Exception:\n        pass")
    assert out["status"] == "turn_limit" and out["primitive_calls"] == 3 and len(ex.calls) == 3
    r, ex = runner(tick_cap=25)
    out = r.run("for i in range(10):\n    hold(10)")
    assert out["status"] == "turn_limit" and len(ex.calls) == 3          # 30 ticks >= 25 after the 3rd call
    out = r.run("hold(10)")                                               # caps are per turn
    assert out["status"] == "ok"


def test_done_and_success_stop_program():
    r, ex = runner()
    out = r.run("hold(5)\ndone()\nhold(5)")
    assert out["status"] == "done" and ex.st.done_called and len(ex.calls) == 1 and r.done_called
    r, ex = runner(ex=FakeEx(stop_after_ticks=20))
    out = r.run("for i in range(10):\n    try:\n        hold(10)\n    except Exception:\n        pass\nprint('after')")
    assert out["status"] == "episode_stopped" and out["ok"] and len(ex.calls) == 2 and "after" not in out["stdout"]


def test_get_objects_only_with_obj():
    r, _ = runner()
    assert "NameError" in r.run("get_objects()")["error"]
    r, _ = runner(objects_fn=lambda: {"plate": {"position_m": [0.0, 0.1, 0.815]}})
    assert r.run("result = get_objects()['plate']['position_m']")["result_variable"] == [0.0, 0.1, 0.815]


def test_stdout_and_log_truncation():
    r, _ = runner()
    out = r.run("for i in range(5000):\n    print('line', i)")
    assert len(out["stdout"]) <= C.STDOUT_MAX + 100 and "truncated" in out["stdout"]
    out = r.run("for i in range(60):\n    hold(1)")
    assert len(out["primitive_log"]) == C.LOG_KEEP + 1 and "omitted" in out["primitive_log"][10]


# ── schemas / parsing / prompt ─────────────────────────────────────────────

def test_code_tool_schemas_strict():
    tools = C.code_tool_schemas()
    assert [t["name"] for t in tools] == ["RUN_CODE", "DONE"]
    for t in tools:
        p = t["parameters"]
        assert t["strict"] and p["additionalProperties"] is False and set(p["required"]) == set(p["properties"])
    assert "code" in tools[0]["parameters"]["required"]


def test_parse_code_call():
    base = {"reasoning": "r", "description": "d", "active_subgoal_id": "G1", "subgoal_status": "in_progress"}
    call, err = C.parse_code_call("RUN_CODE", json.dumps({**base, "code": "hold(5)"}))
    assert call and call.params == {"code": "hold(5)"} and err is None
    assert C.parse_code_call("RUN_CODE", json.dumps(base))[0] is None                       # missing code
    assert C.parse_code_call("RUN_CODE", json.dumps({**base, "code": "x", "extra": 1}))[0] is None
    assert C.parse_code_call("MOVE_EEF", json.dumps({**base, "params": {}}))[0] is None      # not a code-mode tool
    assert C.parse_code_call("RUN_CODE", '{"code": ')[0] is None
    call, _ = C.parse_code_call("DONE", json.dumps({**base, "params": {}}))
    assert call.name == "DONE"


def test_code_system_prompt():
    from imbench.vlm_eval.stage3.agent import system_prompt
    s = system_prompt(24, True, "code", obj=False)
    for frag in ("RUN_CODE", "move_eef(position_m", "at least 15 cm above the table surface", "z >= 0.95 m",
                 "EXTRINSIC X-Y-Z", "SIDE face", "No import", "persist", "PLAN block"):
        assert frag in s, frag
    assert "get_objects" not in s
    assert "get_objects()" in system_prompt(24, True, "code", obj=True)
    import re
    assert not re.search(r"\{[a-z_]+\}", s), re.findall(r"\{[a-z_]+\}", s)               # every slot filled
    assert system_prompt(24, True) == system_prompt(24, True, "primitives")                 # primitives unchanged


# ── full loop in sim (mock client) ──────────────────────────────────────────

def _run(calls, tmp_path, **kw):
    from imbench.vlm_eval.stage3.agent import EpisodeConfig, MockToolClient, run_episode
    cfg = EpisodeConfig(seed=0, plan_text=None, plan_meta={"plan_source": "none"}, record=False, mode="code", **kw)
    return run_episode(MockToolClient(calls), cfg, tmp_path, log=lambda *_: None)


def test_mock_reference_program_succeeds_in_sim(tmp_path):
    from imbench.vlm_eval.stage3.scripted_program import scripted_code_calls
    s = _run(scripted_code_calls(), tmp_path, obj=True)
    assert s["success"] and s["exit_reason"] == "success" and s["mode"] == "code" and s["steps"] == 3
    recs = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [r["executed"] for r in recs] == ["RUN_CODE"] * 3 and all(r["valid"] for r in recs)
    assert recs[2]["result"]["status"] == "episode_stopped"
    tools = json.loads((tmp_path / "tools.json").read_text())
    assert [t["name"] for t in tools] == ["RUN_CODE", "DONE"]
    p3 = json.loads((tmp_path / "step03_prompt.json").read_text())["content"]
    assert any("ran RUN_CODE program" in c.get("text", "") for c in p3)


def test_code_mode_failures_and_done(tmp_path):
    calls = [{"malformed": True}, {"name": "RUN_CODE", "args": {"code": "import os"}},
             {"name": "RUN_CODE", "args": {"code": "hold(10)\ndone()"}}]
    s = _run(calls, tmp_path)
    recs = [json.loads(l) for l in (tmp_path / "steps.jsonl").read_text().splitlines()]
    assert [r["executed"] for r in recs] == ["HOLD", "RUN_CODE", "RUN_CODE"]
    assert recs[1]["result"]["status"] == "rejected" and recs[2]["result"]["status"] == "done"
    assert s["exit_reason"] == "done" and not s["success"]
