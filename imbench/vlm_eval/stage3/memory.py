"""Working memory (ReactMemory) and rolling frame history for the Stage-3 agent."""

from __future__ import annotations

import json
from collections import deque


CODE_SHOW = 3000          # chars of a past program shown in working memory
LOG_SHOW = 12             # primitive-log entries of a past program shown in working memory (the latest turn shows all)


def _brief_code_result(r: dict) -> dict:
    """Older turns: keep status/error/stdout and the tail of the primitive log."""
    r = dict(r)
    log = r.get("primitive_log") or []
    if len(log) > LOG_SHOW:
        r["primitive_log"] = [{"omitted": f"{len(log) - LOG_SHOW} earlier calls"}] + log[-LOG_SHOW:]
    return r


class ReactMemory:
    """Sliding window of the last H (abbreviated state, primitive + params, result, reward delta)."""

    def __init__(self, H: int = 12):
        self.H = H
        self.entries: deque[dict] = deque(maxlen=H)

    def add(self, step: int, state: dict, primitive: str, params: dict, result: dict, reward_delta: float,
            note: str = ""):
        self.entries.append({"step": step, "state_before": state, "primitive": primitive, "params": params,
                             "result": result, "reward_delta": round(float(reward_delta), 3), "note": note})

    def render(self) -> str:
        if not self.entries:
            return "WORKING MEMORY: (empty: this is the first step)"
        L = [f"WORKING MEMORY (last {len(self.entries)} of at most {self.H} steps, oldest first):"]
        for e in self.entries:
            s = e["state_before"]
            L.append(f"- step {e['step']}: state before: eef {s['eef_pos_m']} rpy {s['eef_rpy_deg']} "
                     f"gripper {s['gripper_opening_m']} m")
            if "code" in e["params"]:                     # Code-as-Policies turn
                code = e["params"]["code"]
                if len(code) > CODE_SHOW:
                    code = code[:CODE_SHOW] + "\n# ...[program truncated in memory]..."
                L.append(f"  ran RUN_CODE program:" + (f"  [{e['note']}]" if e["note"] else ""))
                L.append("    " + code.replace("\n", "\n    "))
                full = e is self.entries[-1]
                L.append(f"  result: {json.dumps(e['result'] if full else _brief_code_result(e['result']))}")
            else:
                L.append(f"  called {e['primitive']} {json.dumps(e['params'])}" + (f"  [{e['note']}]" if e["note"] else ""))
                L.append(f"  result: {json.dumps(e['result'])}")
            L.append(f"  reward change: {e['reward_delta']:+.3f}")
        return "\n".join(L)


class FrameHistory:
    """Rolling window of the last H observation batches (list of (camera, image path))."""

    def __init__(self, H: int = 12):
        self.H = H
        self.batches: deque[tuple[int, list[tuple[str, str]]]] = deque(maxlen=H)

    def add(self, step: int, frames: list[tuple[str, str]]):
        self.batches.append((step, frames))

    def __len__(self):
        return len(self.batches)
