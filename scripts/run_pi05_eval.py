"""Evaluate pi0.5 (via an openpi policy server) on an IMBench task with rollout videos.

Start the server first (in the openpi repo):
    uv run scripts/serve_policy.py --env LIBERO

Then:
    python scripts/run_pi05_eval.py --n-episodes 50
    python scripts/run_pi05_eval.py --policy random --n-episodes 5   # baseline / pipeline check

Each episode writes ``<out>/ep<seed>_<success|failure>.mp4``: a 2x2 grid of all
task cameras plus the two 224x224 images the model actually receives, with
step / action / score overlaid. ``results.json`` holds per-episode metrics.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import imageio
import numpy as np

import imbench as ib
from imbench.policy.base_policy import PolicyInterface

# LIBERO geometry (libero_tabletop_manipulation.py / mounted_panda.py): Panda base at
# x=-0.66 on the default mount (z=0.912), agentview camera at the pose below. We place
# a camera at the same pose *relative to the robot base* and express the state in
# LIBERO's frame, so image directions, state and OSC deltas line up with training.
LIBERO_BASE = np.array([-0.66, 0.0, 0.912])
LIBERO_AGENTVIEW_POS = np.array([0.6586, 0.0, 1.6103])
LIBERO_AGENTVIEW_QUAT = [0.6380, 0.3048, 0.3048, 0.6380]  # (w, x, y, z)
LIBERO_INIT_QPOS = np.array([0, -0.161037389, 0.0, -2.44459747, 0.0, 2.22675220, np.pi / 4])
# Built-in robosuite cameras we can repurpose; the first one the task doesn't use wins.
SPARE_CAMS = ["frontview", "agentview", "sideview", "birdview"]


def libero_cam_config(task_id: str) -> dict:
    """Camera override for a ``libero_cam`` on a robosuite camera the task leaves free.
    The pose is a placeholder until :func:`place_libero_cam` knows the base position."""
    from imbench.tasks.base_task import load_task_config
    used = {c.get("robosuite_name") for c in load_task_config(task_id).get("obs", {}).get("cameras", {}).values()
            if c.get("enabled", False)}
    free = next(c for c in SPARE_CAMS if c not in used)
    return {"enabled": True, "robosuite_name": free, "pos": LIBERO_AGENTVIEW_POS.tolist(),
            "quat": LIBERO_AGENTVIEW_QUAT, "fovy": 45, "width": 256, "height": 256,
            "depth": False, "save": False}


def place_libero_cam(env) -> np.ndarray:
    """Move ``libero_cam`` to LIBERO's base-relative pose; returns the state offset
    (LIBERO base - IMBench base) to add to ``robot0_eef_pos``. Applied on next reset."""
    sim = env._env.sim
    base = np.array(sim.data.body_xpos[sim.model.body_name2id("robot0_base")])
    offset = LIBERO_BASE - base
    env._custom_cameras["libero_cam"]["pos"] = (LIBERO_AGENTVIEW_POS - offset).tolist()
    return offset


class RandomPolicy(PolicyInterface):
    def act(self, obs):
        return np.random.uniform(-1, 1, size=7)


class NoopPolicy(PolicyInterface):
    def act(self, obs):
        return np.array([0, 0, 0, 0, 0, 0, -1.0])


def _label(img, text, org=(6, 18), scale=0.5):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def compose_frame(obs, policy, step, action, score):
    # libero_cam first, then the task's own cameras; 2x2 grid, black tile if fewer.
    cams = sorted((k[4:] for k in obs if k.startswith("rgb_")), key=lambda c: c != "libero_cam")[:4]
    tiles = []
    for cam in cams:
        img = obs[f"rgb_{cam}"]
        if img.shape[0] == img.shape[1]:  # square (libero_cam): letterbox instead of stretching
            sq = cv2.resize(img, (240, 240))
            img = np.zeros((240, 320, 3), np.uint8)
            img[:, 40:280] = sq
        else:
            img = cv2.resize(img, (320, 240))
        tiles.append(_label(img.copy(), cam))
    tiles += [np.zeros((240, 320, 3), np.uint8)] * (4 - len(tiles))
    grid = np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)

    side = np.zeros((480, 224, 3), np.uint8)
    main_in = getattr(policy, "last_main_img", None)
    wrist_in = getattr(policy, "last_wrist_img", None)
    if main_in is not None:
        side[:224] = _label(main_in.copy(), "model: image", scale=0.4)
    if wrist_in is not None:
        side[232:456] = _label(wrist_in.copy(), "model: wrist", scale=0.4)
    frame = np.concatenate([grid, side], 1)

    a = " ".join(f"{x:+.2f}" for x in action)
    _label(frame, f"step {step}  score {score:.2f}", org=(6, 470))
    _label(frame, f"a=[{a}]", org=(250, 470), scale=0.42)
    return frame


def plate_overhang(env) -> tuple[float, bool]:
    """Edge-slide only: how far (m) the plate's furthest corner sticks out past the
    table edge (negative = fully on the table), and whether it fell off."""
    sim = env.sim
    body = env._find_body_id(sim, "cube")
    geom = next(i for i in range(sim.model.ngeom) if sim.model.geom_bodyid[i] == body)
    hx, hy = sim.model.geom_size[geom][:2]
    pos = sim.data.body_xpos[body]
    rot = sim.data.body_xmat[body].reshape(3, 3)
    corners = pos[:2] + (np.array([[sx * hx, sy * hy, 0.0] for sx in (-1, 1) for sy in (-1, 1)]) @ rot.T)[:, :2]
    table_c = np.asarray(env._table_offset[:2], dtype=np.float64)
    table_half = np.asarray(env._table_full_size[:2], dtype=np.float64) / 2
    overhang = float(np.max(np.abs(corners - table_c) - table_half))
    fell = bool(pos[2] < env._table_height - 0.05)
    return overhang, fell


def undo_task_flip(obs):
    """Invert MirrorPickPlaceV1._flip_cameras (180-degree rotation of every frame but bird-eye)."""
    for k in obs:
        if (k.startswith("rgb_") or k.startswith("depth_")) and "bird" not in k:
            obs[k] = obs[k][::-1, ::-1]
    return obs


def _write_summary(out, args, prompt, episodes):
    summary = {
        "task": args.task, "policy": args.policy, "prompt": prompt, "args": vars(args),
        "n_episodes_done": len(episodes),
        "success_rate": float(np.mean([e["success"] for e in episodes])),
        "mean_length": float(np.mean([e["length"] for e in episodes])),
        "mean_min_eef_obj_dist": float(np.mean([e["min_eef_obj_dist"] for e in episodes])),
        "episodes": episodes,
    }
    if args.push_metric:
        summary["push_success_def"] = (f"final plate overhang >= {args.push_threshold} m past the "
                                       f"table edge and plate not fallen off")
        summary["mean_final_overhang"] = float(np.mean([e["final_overhang"] for e in episodes]))
        summary["fell_off_rate"] = float(np.mean([e["fell_off"] for e in episodes]))
    (out / "results.json").write_text(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="ib-p1-edge-slide-v1")
    p.add_argument("--policy", choices=["pi05", "random", "noop"], default="pi05")
    p.add_argument("--prompt", default=None, help="defaults to configs/tasks/vla_prompts.json, else the task DESCRIPTION")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=600, help="at --control-freq")
    p.add_argument("--control-freq", type=int, default=20, help="LIBERO trains at 20 Hz")
    p.add_argument("--main-cam", default="libero_cam")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--video-every", type=int, default=1, help="save video every N episodes (0 = never)")
    p.add_argument("--out", default=None)
    p.add_argument("--push-metric", action="store_true",
                   help="edge-slide: score the push sub-task (plate overhang) instead of the lift")
    p.add_argument("--push-threshold", type=float, default=0.03, help="overhang (m) counted as a graspable lip")
    p.add_argument("--state-dx", type=float, default=0.0, help="debug: extra x shift (m) added to the state offset")
    p.add_argument("--cam-dx", type=float, default=0.0, help="debug: extra x shift (m) of libero_cam")
    p.add_argument("--wrist-cam", default="gripper_cam")
    p.add_argument("--wrist-res", type=int, default=256,
                   help="render the wrist camera square at this size like LIBERO (0 = task's own resolution)")
    p.add_argument("--table-height", type=float, default=None,
                   help="override scene.table.height (LIBERO's table top is at 0.90, ~level with the robot base)")
    p.add_argument("--kp", type=float, default=None, help="override OSC stiffness (LIBERO used robosuite's default 150)")
    p.add_argument("--libero-init-pose", action="store_true",
                   help="start the Panda at LIBERO's initial joint configuration (process-local override)")
    p.add_argument("--undo-task-flip", action="store_true",
                   help="ib-m-mirror-pick-place: rotate the task's 180-degree-flipped camera frames back upright")
    args = p.parse_args()

    if args.libero_init_pose:
        # LIBERO (mounted_panda.py) starts the arm higher with a near-vertical gripper;
        # robosuite's default starts lower, tilted ~7 deg forward. Every LIBERO demo began
        # from this pose. Robots are rebuilt on each hard reset, so patch the model class.
        from robosuite.models.robots.manipulators.panda_robot import Panda
        Panda.init_qpos = property(lambda self: LIBERO_INIT_QPOS.copy())
    cam_overrides = {"libero_cam": libero_cam_config(args.task)}
    if args.wrist_res:
        # LIBERO rendered the (identical, fovy 75) wrist camera at 256x256; a 4:3 frame
        # padded to 224x224 shrinks the cube by ~25% and biases the final alignment.
        cam_overrides[args.wrist_cam] = {"width": args.wrist_res, "height": args.wrist_res}
    env = ib.make(
        args.task, obs_mode="combo", max_episode_steps=args.max_steps,
        config_overrides={"control_freq": args.control_freq, "obs": {"cameras": cam_overrides},
                          **({"controller": {"kp": args.kp}} if args.kp is not None else {}),
                          **({"scene": {"table": {"height": args.table_height}}} if args.table_height is not None else {})},
    )
    state_offset = place_libero_cam(env)
    state_offset[0] += args.state_dx
    env._custom_cameras["libero_cam"]["pos"][0] += args.cam_dx
    if args.prompt is None:
        # Prefer the curated VLA prompt; fall back to the task description.
        vla_prompts = json.loads((Path(__file__).parents[1] / "configs/tasks/vla_prompts.json").read_text())
        args.prompt = vla_prompts.get(args.task) or env.DESCRIPTION.split(":", 1)[-1]
        args.prompt = args.prompt.strip().rstrip(".").lower()  # LIBERO-style instruction
    prompt = args.prompt

    if args.policy == "pi05":
        from imbench.policy.pi05_policy import Pi05Policy
        policy = Pi05Policy(
            prompt=prompt, host=args.host, port=args.port, main_cam=args.main_cam, wrist_cam=args.wrist_cam,
            mirror=not args.no_mirror, replan_steps=args.replan_steps,
            state_pos_offset=tuple(state_offset),
        )
    else:
        policy = RandomPolicy() if args.policy == "random" else NoopPolicy()

    out = Path(args.out or f"outputs/pi05_eval/{args.task}/{args.policy}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"task={args.task} policy={args.policy} prompt={prompt!r} -> {out}")

    episodes = []
    for ep in range(args.n_episodes):
        seed = args.seed_offset + ep
        record = args.video_every > 0 and ep % args.video_every == 0
        obs, info = env.reset(seed=seed)
        if args.undo_task_flip:
            obs = undo_task_flip(obs)
        policy.reset()
        frames, max_score, t0 = [], 0.0, time.time()
        min_dist = float("inf")
        traj = {"action": [], "eef_pos": [], "eef_quat": [], "object_pos": []}
        if args.push_metric:
            init_overhang, _ = plate_overhang(env)
            max_overhang, fell = init_overhang, False
            traj["overhang"] = []
        done = False
        while not done:
            action = policy.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            if args.undo_task_flip:
                obs = undo_task_flip(obs)
            max_score = max(max_score, info["task_score"])
            if args.push_metric:
                overhang, fell = plate_overhang(env)
                if not fell:
                    max_overhang = max(max_overhang, overhang)
                traj["overhang"].append(overhang)
                # Report overhang in the video's score slot.
                info = {**info, "task_score": overhang}
            traj["action"].append(action)
            traj["eef_pos"].append(obs["robot0_eef_pos"].copy())
            traj["eef_quat"].append(obs["robot0_eef_quat"].copy())
            traj["object_pos"].append(obs.get("object_pos", np.full(3, np.nan)).copy())
            if "object_pos" in obs:
                min_dist = min(min_dist, float(np.linalg.norm(obs["robot0_eef_pos"] - obs["object_pos"])))
            if record:
                frames.append(compose_frame(obs, policy, info["step"], action, info["task_score"]))
            done = terminated or truncated
        success = info["task_score"] > 0.9
        ep_res = {
            "seed": seed, "success": bool(success), "final_score": float(info["task_score"]),
            "max_score": max_score, "min_eef_obj_dist": round(min_dist, 4), "length": info["step"], "terminated": bool(terminated),
            "object_pos_final": [float(x) for x in obs.get("object_pos", [])],
            "wall_time_s": round(time.time() - t0, 1),
        }
        if args.push_metric:
            final_overhang = float(traj["overhang"][-1])
            success = (not fell) and final_overhang >= args.push_threshold
            ep_res.update({
                "success": bool(success), "final_score": final_overhang,
                "initial_overhang": round(init_overhang, 4), "final_overhang": round(final_overhang, 4),
                "max_overhang": round(max_overhang, 4), "fell_off": fell,
                "plate_xy_displacement": round(float(np.linalg.norm(
                    traj["object_pos"][-1][:2] - traj["object_pos"][0][:2])), 4),
            })
        episodes.append(ep_res)
        _write_summary(out, args, prompt, episodes)  # keep results.json valid if the run is stopped early
        np.savez_compressed(out / f"ep{seed:03d}_traj.npz", **{k: np.asarray(v) for k, v in traj.items()})
        if record:
            name = out / f"ep{seed:03d}_{'success' if success else 'failure'}.mp4"
            imageio.mimwrite(name, frames, fps=args.control_freq, macro_block_size=1)
        print(f"[{ep + 1}/{args.n_episodes}] seed={seed} success={success} "
              f"len={info['step']} min_dist={min_dist:.3f} ({ep_res['wall_time_s']}s)", flush=True)

    summary = _write_summary(out, args, prompt, episodes)
    print(f"success_rate={summary['success_rate']:.2%} over {len(episodes)} episodes -> {out}/results.json")
    env.close()


if __name__ == "__main__":
    main()
