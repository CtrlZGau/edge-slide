"""Strict JSON tool schemas for the 8 primitives + validation of returned calls.

Every call carries {reasoning, description, active_subgoal_id, subgoal_status, params}.
Fixed-length arrays (position_m: 3, orientation_rpy_deg: 3, action7: 7) are validated here
(not via minItems/maxItems in the schema); an off-schema call is executed as HOLD by the agent.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, confloat, conint, conlist

SUBGOAL_STATUS = ["in_progress", "achieved", "abandoned"]
NUM = {"type": "number"}


def _arr(desc: str) -> dict:
    return {"type": "array", "items": NUM, "description": desc}


PARAMS = {
    "MOVE_EEF": ({
        "position_m": _arr("target gripper position [x, y, z], metres, world frame"),
        "orientation_rpy_deg": {"type": ["array", "null"], "items": NUM,
                                "description": "target [roll, pitch, yaw] in degrees, extrinsic XYZ about world "
                                               "axes; null = keep the current orientation"},
        "max_steps": {"type": "integer", "description": "control-step budget (100 steps = 1 s), e.g. 300"},
    }, "Move the gripper in a straight line to a target pose (position + optional orientation)."),
    "CLOSE_GRIPPER": ({}, "Close the parallel jaws (holds the current pose while closing)."),
    "OPEN_GRIPPER": ({}, "Open the parallel jaws fully (holds the current pose while opening)."),
    "LIFT": ({"dz_m": {"type": "number", "description": "vertical displacement in metres (+ up)"}},
             "Move straight up (or down for negative dz_m), keeping x, y and orientation."),
    "DESCEND": ({"target_z_m": {"type": "number", "description": "absolute target gripper z in metres"}},
                "Move vertically to an absolute z, keeping x, y and orientation."),
    "ATOMIC_ACTION": ({
        "action7": _arr("raw normalised delta [dx, dy, dz, droll, dpitch, dyaw, grip], each in [-1, 1]"),
        "n_ticks": {"type": "integer", "description": "number of control steps to repeat it (<= 200)"},
    }, "Send one raw normalised 7-D delta action for n_ticks control steps (fine nudges only)."),
    "HOLD": ({"n_ticks": {"type": "integer", "description": "control steps to hold the current pose (<= 200)"}},
             "Hold the current pose and gripper state."),
    "DONE": ({}, "Declare the task complete. Ends the episode."),
}


def tool_schemas() -> list[dict]:
    tools = []
    for name, (params, desc) in PARAMS.items():
        tools.append({
            "type": "function", "name": name, "description": desc, "strict": True,
            "parameters": {
                "type": "object", "additionalProperties": False,
                "required": ["reasoning", "description", "active_subgoal_id", "subgoal_status", "params"],
                "properties": {
                    "reasoning": {"type": "string", "description": "why this primitive and these values, "
                                                                   "given the observation and the last results"},
                    "description": {"type": "string", "description": "one line: what this call should achieve"},
                    "active_subgoal_id": {"type": "string", "description": "plan sub-goal being worked on, "
                                                                           "or 'none'"},
                    "subgoal_status": {"type": "string", "enum": SUBGOAL_STATUS},
                    "params": {"type": "object", "additionalProperties": False,
                               "required": list(params), "properties": params},
                },
            },
        })
    return tools


# ── validation ─────────────────────────────────────────────────────────────

class _P(BaseModel):
    model_config = ConfigDict(extra="forbid")


Vec3 = conlist(float, min_length=3, max_length=3)
Vec7 = conlist(confloat(ge=-1.0, le=1.0), min_length=7, max_length=7)


class MoveEef(_P):
    position_m: Vec3
    orientation_rpy_deg: Vec3 | None
    max_steps: conint(ge=1, le=2000)


class NoParams(_P):
    pass


class Lift(_P):
    dz_m: float


class Descend(_P):
    target_z_m: float


class Atomic(_P):
    action7: Vec7
    n_ticks: conint(ge=1, le=200)


class Hold(_P):
    n_ticks: conint(ge=1, le=200)


VALIDATORS = {"MOVE_EEF": MoveEef, "CLOSE_GRIPPER": NoParams, "OPEN_GRIPPER": NoParams, "LIFT": Lift,
              "DESCEND": Descend, "ATOMIC_ACTION": Atomic, "HOLD": Hold, "DONE": NoParams}


class ToolCall(_P):
    name: str
    reasoning: str
    description: str
    active_subgoal_id: str
    subgoal_status: Literal["in_progress", "achieved", "abandoned"]
    params: dict


def parse_call(name: str | None, arguments_json: str | None) -> tuple[ToolCall | None, str | None]:
    """Return (call, None) if valid, else (None, error)."""
    if not name:
        return None, "no function call in response"
    if name not in VALIDATORS:
        return None, f"unknown tool {name!r}"
    try:
        args = json.loads(arguments_json or "")
    except (json.JSONDecodeError, TypeError) as e:
        return None, f"malformed arguments JSON: {e}"
    if not isinstance(args, dict):
        return None, "arguments are not a JSON object"
    try:
        call = ToolCall(name=name, **args)
        p = VALIDATORS[name](**call.params)
    except (ValidationError, TypeError) as e:
        detail = e.errors()[0] if isinstance(e, ValidationError) else {"loc": (), "msg": str(e)}
        return None, f"off-schema call: {'.'.join(map(str, detail['loc']))}: {detail['msg']}"
    call.params = p.model_dump()
    return call, None
