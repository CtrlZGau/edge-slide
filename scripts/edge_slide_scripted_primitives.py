"""Scripted edge-slide solver that uses ONLY the 8 Stage-3 primitives (feasibility check, no model).

Privileged state (plate pose and size from MuJoCo) is read only to choose primitive parameters.

    python scripts/edge_slide_scripted_primitives.py --seeds 0 1 2 3 4 5 6 7 8 9 --video-seeds 0

Strategy: close the jaws into a compact pusher, lower the fingertips beside the plate, MOVE_EEF
sideways along a line through the plate centre so the fingertips push its trailing side face toward
the nearer side edge (y = +-0.25) until the lip overhangs, lift
clear, move outside the edge, turn the gripper horizontal (pointing back at the table, jaws opening
vertically), open, slide in to straddle the lip, CLOSE_GRIPPER, LIFT.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from imbench.vlm_eval.stage3 import primitives as P  # noqa: E402

TASK_ID = "ib-p1-edge-slide-v1"
TABLE_Z = 0.80
TABLE_HALF = 0.25            # table edges at x, y = +-0.25 (full_size 0.5)
TIP_BELOW_SITE = 0.008       # fingertips extend ~8 mm past the eef site (measured: press-on-table stop at z=0.808)
SAFE_Z = 0.95

# tunables
OVERHANG = 0.035             # desired lip overhang beyond the edge (m), along the push direction
EDGE_MARGIN = 0.012          # fingertip clearance outside the table edge for the lower jaw
LIFT_DZ = 0.25
REORIENT_X = 0.05           # x where the side-grasp reorientation is reliably reachable (probe)


def plate_info(task):
    sim = task.env.sim
    bid = task._find_body_id(sim, "cube")
    gid = [i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == bid][0]
    hl, hw, hh = sim.model.geom_size[gid]
    pos = np.array(sim.data.body_xpos[bid])
    yaw = R.from_matrix(sim.data.body_xmat[bid].reshape(3, 3)).as_euler("xyz")[2]
    return pos, yaw, float(hl), float(hw), float(hh)


def corners(pos, yaw, hl, hw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[pos[0] + c * a - s * b, pos[1] + s * a + c * b] for a, b in
                     [(hl, hw), (hl, -hw), (-hl, hw), (-hl, -hw)]])


def extent_y(yaw, hl, hw):
    return abs(hl * np.sin(yaw)) + abs(hw * np.cos(yaw))


def overhang(task, side):
    pos, yaw, hl, hw, _ = plate_info(task)
    return side * pos[1] + extent_y(yaw, hl, hw) - TABLE_HALF


def lip_x(task, side, y_min):
    """Mean x of the plate area beyond |y| > y_min on the given side (grid sampling)."""
    pos, yaw, hl, hw, _ = plate_info(task)
    a, b = np.meshgrid(np.linspace(-hl, hl, 61), np.linspace(-hw, hw, 41))
    c, s = np.cos(yaw), np.sin(yaw)
    x = pos[0] + c * a - s * b
    y = pos[1] + s * a + c * b
    m = side * y > y_min
    return float(x[m].mean()) if m.any() else float(pos[0]), int(m.sum())


def side_grasp_rpy(side) -> np.ndarray:
    """Gripper horizontal, approach axis (local z) pointing back at the table (-side*y), jaws closing
    along world z (local y). Only this closing sign is reachable (probe): rpy = (side*90, 0, 0)."""
    z = np.array([0.0, -side, 0.0])
    y = np.array([0.0, 0.0, side])
    x = np.cross(y, z)
    return R.from_matrix(np.column_stack([x, y, z])).as_euler(P.RPY_CONVENTION, degrees=True)


def solve(seed: int, record: bool = False, log=print) -> dict:
    import imbench as ib
    task = ib.make(TASK_ID, obs_mode="combo", max_episode_steps=10_000)
    obs, _ = task.reset(seed=seed)
    ex = P.PrimitiveExecutor(task, obs, record_cam=("exo_right_cam", "exo_left_cam") if record else None)
    seq = []

    def call(name, **params):
        if ex.stopped:
            return None
        res = ex.run(name, params)
        seq.append({"primitive": name, "params": {k: (np.round(v, 4).tolist() if isinstance(v, np.ndarray) else v)
                                                  for k, v in params.items()}, "result": res})
        log(f"  {name:13s} {json.dumps(seq[-1]['params'])} -> "
            f"{ {k: res[k] for k in list(res)[:4]} }")
        return res

    pos, yaw, hl, hw, hh = plate_info(task)
    side = 1.0 if pos[1] >= 0 else -1.0
    down_rpy = ex.rpy.copy()
    log(f"seed {seed}: plate c=({pos[0]:.3f},{pos[1]:.3f}) yaw={np.degrees(yaw):.0f}deg size="
        f"{2 * hl * 100:.1f}x{2 * hw * 100:.1f}x{2 * hh * 100:.1f}cm -> push toward y={side * TABLE_HALF:+.2f}")

    # 1. slide the plate toward the edge by pushing its trailing SIDE face through the centre.
    #    (Top-face friction dragging cannot work here: MuJoCo uses max(geom friction), the table geom has
    #    mu=1.0 and the fingertip collision geoms mu=1.0, so pressing harder never beats table friction.)
    push_z = TABLE_Z + 0.005 + TIP_BELOW_SITE           # fingertips ~5 mm above the table surface
    for k in range(4):
        o = overhang(task, side)
        if o >= OVERHANG - 0.004:
            break
        pos, yaw, hl, hw, hh = plate_info(task)
        d = min(hl / max(abs(np.sin(yaw)), 1e-6), hw / max(abs(np.cos(yaw)), 1e-6))   # centre -> trailing face
        ey = extent_y(yaw, hl, hw)
        target_cy = side * (TABLE_HALF + OVERHANG - ey)
        start_y = pos[1] - side * (d + 0.025)
        end_y = target_cy - side * (d - 0.004)           # fingertip ends 4 mm "inside" the face: keeps pushing
        call("MOVE_EEF", position_m=[pos[0], start_y, SAFE_Z - 0.07], orientation_rpy_deg=down_rpy, max_steps=400)
        if k == 0:
            call("CLOSE_GRIPPER")
        call("DESCEND", target_z_m=push_z, max_steps=200)
        # conservative first push (the closed fingers extend ~1-2 cm ahead of the eef site), then
        # closed-loop 1 cm increments with the overhang re-measured between moves
        y_cmd = end_y - side * 0.03
        call("MOVE_EEF", position_m=[pos[0], y_cmd, push_z], orientation_rpy_deg=down_rpy, max_steps=600)
        for _ in range(12):
            o = overhang(task, side)
            if o >= OVERHANG - 0.003 or ex.stopped:
                break
            y_cmd = ex.pos[1] + side * float(np.clip(OVERHANG - o, 0.003, 0.01))
            call("MOVE_EEF", position_m=[ex.pos[0], y_cmd, push_z], orientation_rpy_deg=down_rpy, max_steps=200)
        call("LIFT", dz_m=0.08)
        log(f"    pass {k}: overhang {overhang(task, side) * 100:.1f} cm")
    o = overhang(task, side)

    # 2. side grasp of the exposed lip
    y_site = side * (TABLE_HALF + EDGE_MARGIN + TIP_BELOW_SITE)
    gx, n = lip_x(task, side, abs(y_site))
    mid_z = plate_info(task)[0][2]
    g_rpy = side_grasp_rpy(side)
    call("LIFT", dz_m=max(0.0, SAFE_Z - ex.pos[2]))
    # turn the gripper horizontal at a well-conditioned spot (x=REORIENT_X; near the base, e.g. x=-0.15,
    # the rotation stalls), then translate with the orientation held
    call("MOVE_EEF", position_m=[REORIENT_X, side * 0.40, SAFE_Z], orientation_rpy_deg=down_rpy, max_steps=400)
    call("OPEN_GRIPPER")
    call("MOVE_EEF", position_m=[REORIENT_X, side * 0.40, SAFE_Z], orientation_rpy_deg=g_rpy, max_steps=600)
    call("MOVE_EEF", position_m=[gx, side * 0.40, mid_z], orientation_rpy_deg=g_rpy, max_steps=500)
    call("MOVE_EEF", position_m=[gx, y_site, mid_z], orientation_rpy_deg=g_rpy, max_steps=400)
    rc = call("CLOSE_GRIPPER")
    call("LIFT", dz_m=LIFT_DZ, max_steps=500)
    call("HOLD", n_ticks=50)
    if ex.st.success:
        call("DONE")
    out = {"seed": seed, "success": ex.st.success, "overhang_cm": round(o * 100, 2), "lip_samples": n,
           "plate_z": round(float(ex.obs["object_pos"][2]), 3), "ticks": ex.st.ticks,
           "terminated": ex.st.terminated, "object_between": (rc or {}).get("object_between_fingers"),
           "sequence": seq}
    frames = ex.st.frames
    task.close()
    return out, frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--video-seeds", type=int, nargs="*", default=[0])
    ap.add_argument("--out", default="runs/stage3_scripted")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for s in a.seeds:
        r, frames = solve(s, record=s in a.video_seeds)
        print(f"==> seed {s}: success={r['success']} overhang={r['overhang_cm']}cm plate_z={r['plate_z']} "
              f"ticks={r['ticks']}", flush=True)
        results.append(r)
        if frames:
            import imageio
            imageio.mimsave(out / f"seed{s}_{'success' if r['success'] else 'failure'}.mp4", frames, fps=25)
    (out / "results.json").write_text(json.dumps(results, indent=1))
    sr = np.mean([r["success"] for r in results])
    print(f"\nSUCCESS {sum(r['success'] for r in results)}/{len(results)} = {sr * 100:.0f}%")


if __name__ == "__main__":
    main()
