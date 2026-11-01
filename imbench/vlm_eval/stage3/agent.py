"""Stage-3 ReAct agent: the VLM picks one primitive per step, guided by a PLAN block.

Per call the prompt is self-contained (slots): system = task spec + robot reference + plan
instructions; user = PLAN block, frame history (last H batches, oldest first), current observation
(text + up to 4 views), working memory (last H steps).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from imbench.vlm_eval.client import BudgetExceeded, image_part, text_part
from imbench.vlm_eval.common import load_prompt
from imbench.vlm_eval.stage3 import primitives as P
from imbench.vlm_eval.stage3.memory import FrameHistory, ReactMemory
from imbench.vlm_eval.stage3.tools import parse_call, tool_schemas

MODES = ("primitives", "code")

def _success_spec() -> tuple[float, float]:
    """(table_z, lift_height) from the task YAML; success = plate centre z >= table_z + lift_height."""
    from imbench.tasks.base_task import load_task_config
    scene = load_task_config("ib-p1-edge-slide-v1")["scene"]
    return float(scene["table"]["height"]), float(scene["lift_height"])


TABLE_Z, LIFT_HEIGHT = _success_spec()
TASK_TEXT = (f"Lift the plate so that it is at least {LIFT_HEIGHT * 100:.0f} cm above the table surface "
             f"(plate centre z >= {TABLE_Z + LIFT_HEIGHT:.2f} m) while you hold it.")
VIEWS = ("exo_right_cam", "exo_left_cam", "ego_cam", "gripper_cam")
IMG_W = 512
HIST_IMG_W = 256
FALLBACK_HOLD = 20                 # ticks of HOLD when a call fails or is invalid

# All values measured in recon / primitive report (see README "Stage 3").
ROBOT_REF = {
    "max_opening_m": 0.08,
    "plus_x": "away from the robot base toward the far side of the table (rightward in exo_right_cam, leftward "
              "in exo_left_cam)",
    "plus_y": "the robot's left, toward exo_left_cam (−y is toward exo_right_cam)",
    "table_z": "0.80 m",
    "safe_z": "0.95 m",
    "grasp_range": "0.81–0.86 m for the gripper point when touching objects on the table (with the gripper "
                   "pointing down, below ≈0.808 m the fingertips press into the table)",
    "tip_below_m": 0.008,
    "convention": "EXTRINSIC X-Y-Z about the fixed world axes (roll about world x, then pitch about world y, then "
                  "yaw about world z)",
    "down_rpy": "(180, 0, 0)",
    "jaw_axis_down": "y",
    "pos_speed_m": 0.003,
    "rot_speed_deg": 0.6,
    "t_10cm": 70,
    "t_90deg": 240,
    "pos_output_max": P.POS_OUTPUT_MAX,
    "rot_output_max_deg": round(float(np.degrees(P.ROT_OUTPUT_MAX)), 1),
}

PLAN_INSTRUCTIONS = {
    True: ("A PLAN block is given in each observation. Treat it as guidance: follow its sub-goals in order, keep "
           "track of which sub-goal is active, and deviate only with a reason stated in your reasoning. Report "
           "the active sub-goal id and its status on every call."),
    False: "No plan is given. Decide the steps yourself. Use active_subgoal_id \"none\".",
}


def system_prompt(K: int, has_plan: bool, mode: str = "primitives", obj: bool = False) -> str:
    if mode == "primitives":
        return load_prompt("stage3_system.md").format(max_steps=K, task=TASK_TEXT,
                                                      plan_instructions=PLAN_INSTRUCTIONS[has_plan], **ROBOT_REF)
    if mode != "code":
        raise ValueError(f"unknown mode {mode!r}")
    from imbench.vlm_eval.stage3 import code_policy as C
    objects_api = ("\nget_objects(): the same objects dict as in your observation (plate pose and extents, table "
                   "top and edges), re-measured at the moment you call it." if obj else "")
    return load_prompt("stage3_code_system.md").format(
        max_steps=K, task=TASK_TEXT, plan_instructions=PLAN_INSTRUCTIONS[has_plan], objects_api=objects_api,
        turn_ticks=C.TURN_TICK_CAP, turn_seconds=C.TURN_TICK_CAP / P.CONTROL_HZ, turn_prims=C.TURN_MAX_PRIMITIVES,
        **ROBOT_REF)


# ── observation ─────────────────────────────────────────────────────────────

def _r(v, n=3):
    return [round(float(x), n) for x in v]


def camera_calibration(task) -> dict:
    sim = task.env.sim
    out = {}
    for label, cfg in task.camera_cfgs.items():
        cid = sim.model.camera_name2id(cfg["robosuite_name"])
        R = np.array(sim.data.cam_xmat[cid]).reshape(3, 3)
        out[label] = {"position_m": _r(sim.data.cam_xpos[cid]), "view_direction": _r(-R[:, 2]),
                      "image_up_direction": _r(R[:, 1]), "fovy_deg": float(sim.model.cam_fovy[cid]),
                      "resolution": [cfg["width"], cfg["height"]]}
    return out


def object_info(task) -> dict:
    """--obj: per-object names, poses and extents from MuJoCo geometry (privileged)."""
    sim = task.env.sim
    out = {}
    bid = task._find_body_id(sim, "cube")
    gid = [i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == bid][0]
    out["plate"] = {"position_m": _r(sim.data.body_xpos[bid]),
                    "rpy_deg": _r(P.quat_to_rpy_deg(np.roll(sim.data.body_xquat[bid], -1)), 1),
                    "extents_m": _r(2 * sim.model.geom_size[gid])}
    tid = sim.model.geom_name2id("table_collision")
    tpos = sim.data.geom_xpos[tid]
    tsz = sim.model.geom_size[tid]
    out["table"] = {"top_surface_z_m": round(float(tpos[2] + tsz[2]), 3), "center_xy_m": _r(tpos[:2]),
                    "extents_m": _r(2 * tsz), "edges": {"x_min": round(float(tpos[0] - tsz[0]), 3),
                                                        "x_max": round(float(tpos[0] + tsz[0]), 3),
                                                        "y_min": round(float(tpos[1] - tsz[1]), 3),
                                                        "y_max": round(float(tpos[1] + tsz[1]), 3)}}
    return out


def state_summary(ex: P.PrimitiveExecutor) -> dict:
    o = ex.obs
    ft = np.asarray(o.get("force_torque", np.zeros(6)), float)
    return {"eef_pos_m": _r(ex.pos), "eef_quat_xyzw": _r(o["robot0_eef_quat"]), "eef_rpy_deg": _r(ex.rpy, 1),
            "gripper_opening_m": round(ex.width, 4),
            "gripper_openness_frac": round(ex.width / ROBOT_REF["max_opening_m"], 3),
            "gripper_command_register": "closed" if ex.reg.value > 0 else "open",
            "ft_sensor_frame": {"force_N": _r(ft[:3], 2), "torque_Nm": _r(ft[3:], 3)},
            "ft_world_force_N": _r(ex.force_world(), 2),
            "tactile_max": {"left": round(float(np.max(o.get("robot0_tactile_left", 0))), 3),
                            "right": round(float(np.max(o.get("robot0_tactile_right", 0))), 3)}}


def save_views(ex: P.PrimitiveExecutor, d: Path, tag: str) -> list[tuple[str, str]]:
    d.mkdir(parents=True, exist_ok=True)
    out = []
    for cam in VIEWS:
        key = f"rgb_{cam}"
        if key not in ex.obs:
            continue
        im = Image.fromarray(ex.obs[key])
        im = im.resize((IMG_W, round(im.height * IMG_W / im.width)), Image.LANCZOS)
        p = d / f"{tag}_{cam}.jpg"
        im.save(p, quality=88)
        out.append((cam, str(p)))
    return out


def _compact(key: str, v) -> str:
    """One line per field (nested dicts one level down) to keep the observation short."""
    if isinstance(v, dict):
        return f"{key}:\n" + "\n".join(f"  {k}: {json.dumps(x, separators=(', ', ': '))}" for k, x in v.items())
    return f"{key}: {json.dumps(v)}"


class RenderError(RuntimeError):
    pass


def _img_sig(path: str) -> tuple:
    a = np.asarray(Image.open(path).convert("L").resize((32, 24)), dtype=np.float32)
    return (round(float(a.mean()), 1), round(float(a.std()), 1))


def _hist_image(path: str) -> dict:
    """History frames: downscaled, low detail."""
    import base64
    import io
    im = Image.open(path)
    im = im.resize((HIST_IMG_W, round(im.height * HIST_IMG_W / im.width)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=80)
    return {"type": "input_image", "image_url": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(),
            "detail": "low"}


def build_user_content(step: int, K: int, plan_text: str | None, hist: FrameHistory, state: dict,
                       calib: dict, objects: dict | None, views: list[tuple[str, str]], mem: ReactMemory,
                       ticks: int, tail: str = "Choose the next primitive.") -> tuple[list[dict], list[dict]]:
    """Return (API content, JSON-safe record with image paths)."""
    parts, rec = [], []

    def add_text(t):
        parts.append(text_part(t))
        rec.append(text_part(t))

    add_text(f"PLAN\n{plan_text}" if plan_text else "PLAN\n(no plan provided)")
    if len(hist):
        add_text(f"FRAME HISTORY: the last {len(hist)} observation batches before now, oldest first "
                 f"(low resolution).")
        for s, frames in hist.batches:
            add_text(f"[history: start of step {s}]")
            for cam, p in frames:
                parts.append(_hist_image(p))
                rec.append({"type": "input_image", "file": p, "detail": "low", "label": f"step {s} {cam}"})
    obs_txt = {"step": f"{step} of {K}", "elapsed_control_steps": ticks, "elapsed_s": round(ticks / P.CONTROL_HZ, 2),
               "robot_state": state, "camera_calibration": calib}
    if objects is not None:
        obs_txt["objects"] = objects
    add_text("CURRENT OBSERVATION\n" + "\n".join(_compact(k, v) for k, v in obs_txt.items()))
    for cam, p in views:
        add_text(f"[current view: {cam}]")
        parts.append(image_part(p, detail="high"))
        rec.append({"type": "input_image", "file": p, "detail": "high", "label": f"current {cam}"})
    add_text(mem.render())
    add_text(tail)
    return parts, rec


# ── mock client (offline loop tests) ───────────────────────────────────────

class MockToolClient:
    """Replays a list of tool calls. Items: {"name", "args"} | {"malformed": True} | {"unknown": True} |
    {"timeout": True}. When exhausted, returns DONE."""

    def __init__(self, calls: list[dict], ledger=None):
        from imbench.vlm_eval.client import Ledger
        self.calls = list(calls)
        self.i = 0
        self.ledger = ledger or Ledger()
        self.effort = "mock"
        self.model = "mock"

    def call_tools(self, system, content, tools):
        self.ledger.check()
        item = self.calls[self.i] if self.i < len(self.calls) else {"name": "DONE", "args": {"params": {}}}
        self.i += 1
        if item.get("timeout"):
            raise RuntimeError("API call failed after 3 retries: mock timeout")
        if item.get("malformed"):
            name, args = "MOVE_EEF", '{"reasoning": "x", "params": {"position_m": [0.1, '
        elif item.get("unknown"):
            name, args = "TELEPORT", json.dumps({"reasoning": "x"})
        else:
            a = {"reasoning": "mock replay", "description": item["name"], "active_subgoal_id": "none",
                 "subgoal_status": "in_progress", **item["args"]}
            name, args = item["name"], json.dumps(a)
        self.ledger.add(0.0)
        return {"raw": {"mock": True, "name": name, "arguments": args}, "usage": {"input_tokens": 0,
                "output_tokens": 0, "mock": True}, "cost_usd": 0.0, "model_snapshot": "mock", "effort": "mock",
                "tool_name": name, "arguments_json": args, "reasoning_summary": []}


# ── episode ────────────────────────────────────────────────────────────────

@dataclass
class EpisodeConfig:
    seed: int
    plan_text: str | None
    plan_meta: dict
    obj: bool = False
    K: int = 24
    H: int = 12
    tick_cap: int = 10_000
    record: bool = True
    mode: str = "primitives"          # "primitives" (one tool call = one primitive) | "code" (Code-as-Policies)


def run_episode(client, cfg: EpisodeConfig, out_dir: Path, log=print) -> dict:
    import imbench as ib
    out_dir.mkdir(parents=True, exist_ok=True)
    task = ib.make("ib-p1-edge-slide-v1", obs_mode="combo", max_episode_steps=cfg.tick_cap)
    obs, _ = task.reset(seed=cfg.seed)
    ex = P.PrimitiveExecutor(task, obs, record_cam=("exo_right_cam", "exo_left_cam") if cfg.record else None,
                             tick_cap=cfg.tick_cap)
    system = system_prompt(cfg.K, cfg.plan_text is not None, cfg.mode, cfg.obj)
    code_mode = cfg.mode == "code"
    if code_mode:
        from imbench.vlm_eval.stage3 import code_policy as C
        tools = C.code_tool_schemas()
        parse = C.parse_code_call
        runner = C.CodeRunner(ex, lambda: state_summary(ex), (lambda: object_info(task)) if cfg.obj else None)
        tail = "Write the next program (RUN_CODE), or call DONE."
    else:
        tools, parse, runner, tail = tool_schemas(), parse_call, None, "Choose the next primitive."
    mem, hist = ReactMemory(cfg.H), FrameHistory(cfg.H)
    (out_dir / "system_prompt.md").write_text(system)
    (out_dir / "tools.json").write_text(json.dumps(tools, indent=1))
    steps_f = out_dir / "steps.jsonl"
    steps_f.write_text("")
    exit_reason, total_cost, t0 = "max_steps", 0.0, time.time()
    reward_prev = 0.0

    for step in range(1, cfg.K + 1):
        if ex.stopped:
            exit_reason = "success" if ex.st.success else ("terminated" if ex.st.terminated else "tick_cap")
            break
        state = state_summary(ex)
        views = save_views(ex, out_dir / "frames", f"step{step:02d}")
        if len(views) > 1 and len({_img_sig(p) for _, p in views}) == 1:
            # all camera views identical: broken offscreen render (seen when a 2nd env is built in one
            # process). Abort before spending an API call on garbage images.
            task.close()
            raise RenderError(f"step {step}: all {len(views)} camera views are identical (broken render)")
        objects = object_info(task) if cfg.obj else None
        parts, rec = build_user_content(step, cfg.K, cfg.plan_text, hist, state, camera_calibration(task),
                                        objects, views, mem, ex.st.ticks, tail)
        (out_dir / f"step{step:02d}_prompt.json").write_text(json.dumps({"system": "system_prompt.md",
                                                                          "content": rec}, indent=1))
        rec_step = {"step": step, "state_before": state}
        try:
            resp = client.call_tools(system, parts, tools)
        except BudgetExceeded as e:
            exit_reason = "budget"
            rec_step["error"] = str(e)
            _append(steps_f, rec_step)
            break
        except Exception as e:                       # retries exhausted / timeout
            if type(e).__name__ in ("BadRequestError", "AuthenticationError", "PermissionDeniedError",
                                    "NotFoundError", "QuotaExhausted"):
                task.close()                         # request/config error: abort, don't fake an episode
                raise
            resp = {"raw": {"error": str(e)}, "usage": {}, "cost_usd": 0.0, "tool_name": None,
                    "arguments_json": None, "reasoning_summary": [], "effort": getattr(client, "effort", None),
                    "model_snapshot": None, "call_error": str(e)}
        (out_dir / f"step{step:02d}_raw.json").write_text(json.dumps(resp["raw"], indent=1, default=str))
        total_cost += resp.get("cost_usd", 0.0)
        call, err = parse(resp.get("tool_name"), resp.get("arguments_json"))
        if resp.get("call_error"):
            err = f"API failure: {resp['call_error']}"
        if call is None:
            name, params, note = "HOLD", {"n_ticks": FALLBACK_HOLD}, f"FALLBACK HOLD ({err})"
        else:
            name, params, note = call.name, call.params, ""
        hist.add(step, views)
        if name == "RUN_CODE":
            result = runner.run(params["code"])
        else:
            result = ex.run(name, params)
        reward = ex.st.reward
        mem.add(step, state, name, params, result, reward - reward_prev, note)
        reward_prev = reward
        rec_step.update({
            "tool_name_returned": resp.get("tool_name"), "arguments_returned": resp.get("arguments_json"),
            "valid": call is not None, "error": err, "executed": name, "params": params,
            "reasoning": call.reasoning if call else None, "description": call.description if call else None,
            "active_subgoal_id": call.active_subgoal_id if call else None,
            "subgoal_status": call.subgoal_status if call else None,
            "reasoning_summary": resp.get("reasoning_summary"), "result": result,
            "state_after": state_summary(ex), "reward": reward, "success": ex.st.success,
            "ticks": ex.st.ticks, "usage": resp.get("usage"), "cost_usd": resp.get("cost_usd", 0.0),
            "effort": resp.get("effort"), "model_snapshot": resp.get("model_snapshot")})
        _append(steps_f, rec_step)
        brief = (f"[{result['status']}] {result['primitive_calls']} prims {(result.get('error') or '')[:70]}"
                 if name == "RUN_CODE" else json.dumps(params)[:110])
        log(f"  step {step:2d}: {name:13s} {brief}  -> "
            f"{'conv' if result.get('converged') else ''} reward={reward:.0f} ticks={ex.st.ticks} "
            f"${total_cost:.3f}{'  ' + note if note else ''}")
        if ex.st.success:
            exit_reason = "success"
            break
        if name == "DONE" or (name == "RUN_CODE" and result.get("status") == "done"):
            exit_reason = "done"
            break
        if ex.st.terminated:
            exit_reason = "terminated"
            break
        if ex.st.ticks >= cfg.tick_cap:
            exit_reason = "tick_cap"
            break

    frames = ex.st.frames
    summary = {"seed": cfg.seed, "success": bool(ex.st.success), "exit_reason": exit_reason,
               "steps": sum(1 for _ in open(steps_f)), "ticks": ex.st.ticks, "cost_usd": round(total_cost, 4),
               "plate_final_pos_m": _r(ex.obs["object_pos"]), "wall_s": round(time.time() - t0, 1),
               "obj": cfg.obj, "mode": cfg.mode, **cfg.plan_meta}
    task.close()
    if frames:
        import imageio
        imageio.mimsave(out_dir / f"episode_{'success' if summary['success'] else 'failure'}.mp4", frames, fps=25)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _append(path: Path, rec: dict):
    with path.open("a") as f:
        f.write(json.dumps(rec, default=str) + "\n")
