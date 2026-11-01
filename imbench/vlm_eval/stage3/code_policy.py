"""Code-as-Policies mode for Stage 3: the VLM writes a Python program per turn instead of one primitive.

The program runs in a restricted namespace that exposes the same 8 primitives (as functions), robot
state read-outs, numpy and math. Primitive arguments are validated with the same pydantic validators as
the tool-calling mode, so the motor interface is identical; only the control flow (loops, conditionals,
arithmetic on read-outs, closed-loop corrections inside one turn) is new.

Guards (best effort, not a security boundary; the episode already runs in its own subprocess):
  - AST check: no import, no names/attributes starting with "_", no exec/eval/open/..., no file-I/O
    attributes (np.save, arr.tofile, ...), no bare `except:` / `except BaseException` (so the limits
    below cannot be swallowed), no global/nonlocal.
  - Restricted builtins.
  - Per-turn limits: control ticks, primitive calls, wall-clock time, code length, stdout length.
Variables and functions the program defines persist across turns of the same episode.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import math
import signal
import threading
import time
import traceback
from typing import Callable, Literal

import numpy as np
from pydantic import ValidationError

from imbench.vlm_eval.stage3 import primitives as P
from imbench.vlm_eval.stage3.tools import SUBGOAL_STATUS, VALIDATORS, _P, parse_call, tool_schemas

CODE_MAX_CHARS = 12_000
TURN_TICK_CAP = 3_000            # control ticks (30 s sim) per RUN_CODE turn
TURN_MAX_PRIMITIVES = 120
TURN_WALL_S = 600.0              # wall clock per turn (sim with 4 rendered cameras is ~10-30 ms/tick)
STDOUT_MAX = 4_000
LOG_KEEP = 40                    # primitive-call log entries returned to the model (first 10 + last 30)

BANNED_NAMES = {"exec", "eval", "compile", "open", "__import__", "globals", "locals", "vars", "getattr",
                "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview", "type",
                "object", "super", "classmethod", "staticmethod", "property", "dir", "id", "hash"}
BANNED_ATTRS = {"load", "loads", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "fromfile", "tofile",
                "genfromtxt", "fromregex", "memmap", "ctypeslib", "ctypes", "lib", "testing", "f2py", "distutils",
                "os", "sys", "DataSource", "dump", "dumps", "system", "popen", "frame", "f_globals", "gi_frame",
                "cr_frame", "tb_frame", "func_globals"}
SAFE_BUILTINS = {n: getattr(builtins, n) for n in (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset", "int",
    "isinstance", "len", "list", "map", "max", "min", "pow", "range", "repr", "reversed", "round", "set",
    "slice", "sorted", "str", "sum", "tuple", "zip", "Exception", "ValueError", "RuntimeError", "ZeroDivisionError",
    "IndexError", "KeyError", "TypeError", "ArithmeticError", "AssertionError")}


# ── limits that the program must not be able to catch ─────────────────────

class _Abort(BaseException):
    """Base for limits/stops. Derives from BaseException so `except Exception` in model code can't catch it."""


class EpisodeStopped(_Abort):
    pass


class TurnLimit(_Abort):
    pass


class DoneCalled(_Abort):
    pass


# ── static check ───────────────────────────────────────────────────────────

def check_code(code: str) -> str | None:
    """Return an error message if the program is not allowed, else None."""
    if not isinstance(code, str) or not code.strip():
        return "empty program"
    if len(code) > CODE_MAX_CHARS:
        return f"program too long ({len(code)} > {CODE_MAX_CHARS} chars)"
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as e:
        return f"SyntaxError: {e.msg} (line {e.lineno})"
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return f"line {line}: import is not allowed (np, math and the robot API are already available)"
        if isinstance(node, (ast.Global, ast.Nonlocal)):
            return f"line {line}: global/nonlocal is not allowed"
        if isinstance(node, ast.Name) and (node.id.startswith("_") or node.id in BANNED_NAMES):
            return f"line {line}: name {node.id!r} is not allowed"
        if isinstance(node, ast.Attribute) and (node.attr.startswith("_") or node.attr in BANNED_ATTRS):
            return f"line {line}: attribute {node.attr!r} is not allowed"
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name.startswith("_"):
            return f"line {line}: names starting with '_' are not allowed"
        if isinstance(node, (ast.AsyncFunctionDef, ast.Await, ast.AsyncFor, ast.AsyncWith, ast.ClassDef)):
            return f"line {line}: async code and class definitions are not allowed"
        if isinstance(node, ast.ExceptHandler):
            t = node.type
            names = [t] if t is not None and not isinstance(t, ast.Tuple) else (t.elts if t is not None else [])
            if t is None or any(isinstance(n, ast.Name) and n.id in ("BaseException",) for n in names):
                return f"line {line}: bare `except:` / `except BaseException` is not allowed; catch Exception"
        if isinstance(node, ast.arg) and node.arg.startswith("_"):
            return f"line {line}: names starting with '_' are not allowed"
        if isinstance(node, ast.keyword) and node.arg and node.arg.startswith("_"):
            return f"line {line}: names starting with '_' are not allowed"
    return None


# ── runner ─────────────────────────────────────────────────────────────────

def _jsonable(x):
    if isinstance(x, np.ndarray):
        return [_jsonable(v) for v in x.tolist()]
    if isinstance(x, (np.floating, float)):
        return round(float(x), 5)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    return x


class CodeRunner:
    """Executes model programs against a PrimitiveExecutor; namespace persists across turns."""

    def __init__(self, ex: P.PrimitiveExecutor, state_fn: Callable[[], dict],
                 objects_fn: Callable[[], dict] | None = None, tick_cap: int = TURN_TICK_CAP,
                 max_primitives: int = TURN_MAX_PRIMITIVES, wall_s: float = TURN_WALL_S):
        self.ex, self.state_fn, self.objects_fn = ex, state_fn, objects_fn
        self.tick_cap, self.max_primitives, self.wall_s = tick_cap, max_primitives, wall_s
        self.ns: dict = {}                             # ONE namespace for the whole episode, so functions
                                                       # defined in an earlier turn see later globals
        self.done_called = False
        self._turn = None
        self._out = io.StringIO()
        # builtins are bound once: a function keeps the builtins of the turn that defined it
        self._builtins = {**SAFE_BUILTINS, "print": self._print}

    def _print(self, *args, sep=" ", end="\n"):
        if self._out.tell() < STDOUT_MAX * 2:
            self._out.write(sep.join(str(a) for a in args) + end)

    # API exposed to the program ------------------------------------------------
    def _prim(self, name: str, params: dict) -> dict:
        t = self._turn
        if self.ex.stopped:
            raise EpisodeStopped("episode already ended (success, termination or tick cap); program stopped")
        if t["n_prims"] >= self.max_primitives:
            raise TurnLimit(f"more than {self.max_primitives} primitive calls in one turn")
        if self.ex.st.ticks - t["tick0"] >= self.tick_cap:
            raise TurnLimit(f"turn used {self.tick_cap} control steps (limit per turn)")
        try:
            p = VALIDATORS[name](**_jsonable(params)).model_dump()
        except (ValidationError, TypeError) as e:
            if isinstance(e, ValidationError):
                d = e.errors()[0]
                msg = f"{'.'.join(map(str, d['loc']))}: {d['msg']}"
            else:
                msg = str(e)
            raise ValueError(f"{name.lower()}(): invalid arguments: {msg}") from None
        res = self.ex.run(name, p)
        t["n_prims"] += 1
        t["log"].append({"i": t["n_prims"], "call": name.lower(), "args": p, "result": res,
                         "ticks_after": self.ex.st.ticks})
        if self.ex.st.success:
            raise EpisodeStopped("task success detected by the environment; program stopped")
        if self.ex.stopped:
            raise EpisodeStopped("episode ended (termination or tick cap); program stopped")
        return res

    def _api(self) -> dict:
        def move_eef(position_m, orientation_rpy_deg=None, max_steps=P.DEFAULT_MAX_STEPS):
            return self._prim("MOVE_EEF", {"position_m": position_m, "orientation_rpy_deg": orientation_rpy_deg,
                                           "max_steps": max_steps})

        def close_gripper():
            return self._prim("CLOSE_GRIPPER", {})

        def open_gripper():
            return self._prim("OPEN_GRIPPER", {})

        def lift(dz_m):
            return self._prim("LIFT", {"dz_m": dz_m})

        def descend(target_z_m):
            return self._prim("DESCEND", {"target_z_m": target_z_m})

        def atomic_action(action7, n_ticks):
            return self._prim("ATOMIC_ACTION", {"action7": action7, "n_ticks": n_ticks})

        def hold(n_ticks):
            return self._prim("HOLD", {"n_ticks": n_ticks})

        def done():
            self.done_called = True
            self.ex.done()
            raise DoneCalled("done() called")

        def get_state():
            return self.state_fn()

        def eef_pos():
            return np.array(self.ex.pos)

        def eef_rpy():
            return np.array(self.ex.rpy)

        def gripper_width():
            return float(self.ex.width)

        def force_world():
            return np.array(self.ex.force_world())

        def tactile():
            return float(self.ex.tactile_max())

        def elapsed_steps():
            return int(self.ex.st.ticks)

        api = {"move_eef": move_eef, "close_gripper": close_gripper, "open_gripper": open_gripper, "lift": lift,
               "descend": descend, "atomic_action": atomic_action, "hold": hold, "done": done,
               "get_state": get_state, "eef_pos": eef_pos, "eef_rpy": eef_rpy, "gripper_width": gripper_width,
               "force_world": force_world, "tactile": tactile, "elapsed_steps": elapsed_steps}
        if self.objects_fn is not None:
            api["get_objects"] = lambda: self.objects_fn()
        return api

    # execution ---------------------------------------------------------------
    def run(self, code: str) -> dict:
        t0_wall = time.time()
        self._turn = {"n_prims": 0, "tick0": self.ex.st.ticks, "log": []}
        err = check_code(code)
        if err:
            return self._result(False, f"REJECTED before running: {err}", "", None, "rejected", t0_wall)
        out = self._out = io.StringIO()
        ns = self.ns
        api = self._api()
        ns.pop("result", None)                         # `result` is per turn
        ns.update({"__builtins__": self._builtins, "np": np, "math": math, **api})
        api_names = set(api) | {"np", "math", "__builtins__"}      # re-injected every turn: can't be shadowed

        timer_set = False
        if threading.current_thread() is threading.main_thread():
            def _alarm(signum, frame):
                raise TurnLimit(f"turn exceeded {self.wall_s:.0f} s wall clock (infinite loop?)")
            old = signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, self.wall_s)
            timer_set = True
        status, error = "ok", None
        try:
            exec(compile(code, "<policy>", "exec"), ns)  # noqa: S102 (sandboxed by check_code + builtins)
        except DoneCalled:
            status = "done"
        except EpisodeStopped as e:
            status, error = "episode_stopped", str(e)
        except TurnLimit as e:
            status, error = "turn_limit", str(e)
        except Exception as e:                           # program error: report a short traceback
            tb = traceback.extract_tb(e.__traceback__)
            lines = [f"line {fr.lineno}: {fr.line}" for fr in tb if fr.filename == "<policy>"]
            status, error = "error", f"{type(e).__name__}: {e}" + (f" (at {lines[-1]})" if lines else "")
        finally:
            if timer_set:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)
        self.user_names = sorted(k for k in ns if k not in api_names and k != "result")
        rv = ns.get("result")
        return self._result(status in ("ok", "done", "episode_stopped"), error, out.getvalue(), rv, status, t0_wall)

    def _result(self, ok, error, stdout, rv, status, t0_wall) -> dict:
        t = self._turn
        log = t["log"]
        if len(log) > LOG_KEEP:
            log = log[:10] + [{"omitted": f"{len(log) - LOG_KEEP} calls"}] + log[-(LOG_KEEP - 10):]
        if len(stdout) > STDOUT_MAX:
            stdout = stdout[:STDOUT_MAX // 2] + "\n...[stdout truncated]...\n" + stdout[-STDOUT_MAX // 2:]
        res = {"status": status, "ok": ok, "error": error, "primitive_calls": t["n_prims"],
               "ticks": self.ex.st.ticks - t["tick0"], "stdout": stdout, "primitive_log": log,
               "wall_s": round(time.time() - t0_wall, 1)}
        if rv is not None:
            try:
                res["result_variable"] = json.loads(json.dumps(_jsonable(rv), default=repr))
            except Exception:
                res["result_variable"] = repr(rv)[:500]
        if getattr(self, "user_names", None):
            res["persistent_names"] = self.user_names[:60]
        return res


# ── tool schema + parsing ──────────────────────────────────────────────────

def code_tool_schemas() -> list[dict]:
    meta = {
        "reasoning": {"type": "string", "description": "what you see, what the last result tells you, and what "
                                                       "this program will do"},
        "description": {"type": "string", "description": "one line: what this program should achieve"},
        "active_subgoal_id": {"type": "string", "description": "plan sub-goal being worked on, or 'none'"},
        "subgoal_status": {"type": "string", "enum": SUBGOAL_STATUS},
    }
    run = {"type": "function", "name": "RUN_CODE", "strict": True,
           "description": "Run a Python program that controls the robot through the robot API (see the system "
                          "prompt). It runs to completion before you get the next observation.",
           "parameters": {"type": "object", "additionalProperties": False,
                          "required": [*meta, "code"],
                          "properties": {**meta, "code": {"type": "string", "description": "Python source"}}}}
    done = next(t for t in tool_schemas() if t["name"] == "DONE")
    return [run, done]


class CodeCall(_P):
    name: Literal["RUN_CODE"]
    reasoning: str
    description: str
    active_subgoal_id: str
    subgoal_status: Literal["in_progress", "achieved", "abandoned"]
    code: str

    @property
    def params(self) -> dict:
        return {"code": self.code}


def parse_code_call(name: str | None, arguments_json: str | None):
    """(call, None) or (None, error). RUN_CODE -> CodeCall; DONE -> tools.ToolCall."""
    if name == "DONE":
        return parse_call(name, arguments_json)
    if not name:
        return None, "no function call in response"
    if name != "RUN_CODE":
        return None, f"unknown tool {name!r}"
    try:
        args = json.loads(arguments_json or "")
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"malformed arguments JSON: {e}"
    if not isinstance(args, dict):
        return None, "arguments are not a JSON object"
    try:
        return CodeCall(name=name, **args), None
    except (ValidationError, TypeError) as e:
        detail = e.errors()[0] if isinstance(e, ValidationError) else {"loc": (), "msg": str(e)}
        return None, f"off-schema call: {'.'.join(map(str, detail['loc']))}: {detail['msg']}"
