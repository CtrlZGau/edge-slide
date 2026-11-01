"""Evaluate GR00T N1.7 / MolmoAct2 (LIBERO checkpoints) exactly like ``run_pi05_eval.py``.

The env setup (``libero_cam``), push/overhang metric, video layout and
``results.json`` schema are IMPORTED from ``scripts/run_pi05_eval.py`` so the
protocol cannot drift from the pi0.5 runs in ``outputs/pi05_eval/main_v2_libero_cam``.
The episode loop below is a copy of pi0.5's, plus: resume from an existing
``results.json``, per-query inference latency, and the model/server recorded.

    python scripts/run_vla_eval.py --policy groot     --n-episodes 20 --out outputs/vla_eval/main_v2_libero_cam/groot_default_prompt
    python scripts/run_vla_eval.py --policy molmoact2 --n-episodes 20 --out outputs/vla_eval/main_v2_libero_cam/molmoact2_default_prompt

Per output dir (same as pi0.5): ``results.json``, ``ep<seed>_traj.npz``,
``ep<seed>_<success|failure>.mp4``; plus ``run_config.json`` (git SHA + task diff).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import imbench as ib  # noqa: E402
import run_pi05_eval as pi05  # noqa: E402  (plate_overhang, compose_frame, _write_summary)
from imbench.policy.vla_clients import LIBERO_CAM  # noqa: E402

DEFAULT_PORTS = {"groot": 5555, "molmoact2": 5556, "pi05": 8000}


def git_info() -> dict:
    def run(*cmd):
        try:
            return subprocess.run(cmd, cwd=REPO, capture_output=True, text=True, timeout=10).stdout
        except Exception:
            return ""
    return {
        "sha": run("git", "rev-parse", "HEAD").strip(),
        "dirty_files": run("git", "status", "--short").splitlines(),
        "task_diff": run("git", "diff", "--", "configs/tasks", "imbench/tasks"),
    }


def make_policy(args, prompt):
    kw = dict(host=args.host, port=args.port, main_cam=args.main_cam,
              mirror=not args.no_mirror, replan_steps=args.replan_steps)
    if args.policy == "groot":
        from imbench.policy.vla_clients import GrootLiberoPolicy
        return GrootLiberoPolicy(prompt, **kw)
    if args.policy == "molmoact2":
        from imbench.policy.vla_clients import MolmoAct2LiberoPolicy
        return MolmoAct2LiberoPolicy(prompt, **kw)
    if args.policy == "pi05":
        from imbench.policy.pi05_policy import Pi05Policy
        return Pi05Policy(prompt=prompt, **kw)
    return pi05.RandomPolicy() if args.policy == "random" else pi05.NoopPolicy()


def main():
    p = argparse.ArgumentParser()
    # ---- identical to run_pi05_eval.py ----
    p.add_argument("--task", default="ib-p1-edge-slide-v1")
    p.add_argument("--policy", choices=["groot", "molmoact2", "pi05", "random", "noop"], required=True)
    p.add_argument("--prompt", default=None, help="defaults to the task DESCRIPTION")
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--seed-offset", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=600, help="at --control-freq")
    p.add_argument("--control-freq", type=int, default=20, help="LIBERO trains at 20 Hz")
    p.add_argument("--main-cam", default="libero_cam")
    p.add_argument("--no-mirror", action="store_true")
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default=None, help="default: groot 5555, molmoact2 5556, pi05 8000")
    p.add_argument("--video-every", type=int, default=1, help="save video every N episodes (0 = never)")
    p.add_argument("--out", default=None)
    p.add_argument("--push-metric", action="store_true",
                   help="edge-slide: score the push sub-task (plate overhang) instead of the lift")
    p.add_argument("--push-threshold", type=float, default=0.03, help="overhang (m) counted as a graspable lip")
    # ---- additions ----
    p.add_argument("--no-resume", action="store_true", help="ignore an existing results.json in --out")
    p.add_argument("--wrist-native", type=int, default=0, metavar="PX",
                   help="render gripper_cam natively at PX x PX (LIBERO eye_in_hand is 256x256); "
                        "0 = task default 640x480, padded to square (original protocol)")
    args = p.parse_args()
    args.port = args.port or DEFAULT_PORTS.get(args.policy, 0)

    cameras = {"libero_cam": LIBERO_CAM}
    if args.wrist_native:
        cameras["gripper_cam"] = {"enabled": True, "robosuite_name": "robot0_eye_in_hand",
                                  "width": args.wrist_native, "height": args.wrist_native,
                                  "depth": False, "save": True}
    env = ib.make(
        args.task, obs_mode="combo", max_episode_steps=args.max_steps,
        config_overrides={"control_freq": args.control_freq, "obs": {"cameras": cameras}},
    )
    prompt = args.prompt or env.DESCRIPTION.split(":", 1)[-1].strip().lower()
    policy = make_policy(args, prompt)

    out = Path(args.out or REPO / "outputs" / "vla_eval" / args.task / f"{args.policy}_{time.strftime('%Y%m%d_%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_config.json").write_text(json.dumps(
        {"args": vars(args), "prompt": prompt, "libero_cam": LIBERO_CAM,
         "started": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_info()}, indent=2))

    episodes = []
    if not args.no_resume and (out / "results.json").exists():
        episodes = json.loads((out / "results.json").read_text()).get("episodes", [])
    done_seeds = {e["seed"] for e in episodes}
    print(f"task={args.task} policy={args.policy} prompt={prompt!r} -> {out}"
          f"{f'  (resuming: {len(done_seeds)} done)' if done_seeds else ''}", flush=True)

    for ep in range(args.n_episodes):
        seed = args.seed_offset + ep
        if seed in done_seeds:
            continue
        record = args.video_every > 0 and ep % args.video_every == 0
        obs, info = env.reset(seed=seed)
        try:
            policy.reset(episode_seed=seed)
        except TypeError:
            policy.reset()
        frames, max_score, t0 = [], 0.0, time.time()
        min_dist = float("inf")
        traj = {"action": [], "eef_pos": [], "object_pos": []}
        if args.push_metric:
            init_overhang, _ = pi05.plate_overhang(env)
            max_overhang, fell = init_overhang, False
            traj["overhang"] = []
        done = False
        while not done:
            action = policy.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            max_score = max(max_score, info["task_score"])
            if args.push_metric:
                overhang, fell = pi05.plate_overhang(env)
                if not fell:
                    max_overhang = max(max_overhang, overhang)
                traj["overhang"].append(overhang)
                info = {**info, "task_score": overhang}
            traj["action"].append(action)
            traj["eef_pos"].append(obs["robot0_eef_pos"].copy())
            traj["object_pos"].append(obs.get("object_pos", np.full(3, np.nan)).copy())
            if "object_pos" in obs:
                min_dist = min(min_dist, float(np.linalg.norm(obs["robot0_eef_pos"] - obs["object_pos"])))
            if record:
                frames.append(pi05.compose_frame(obs, policy, info["step"], action, info["task_score"]))
            done = terminated or truncated
        success = info["task_score"] > 0.9
        lat = getattr(policy, "infer_ms", None) or []
        ep_res = {
            "seed": seed, "success": bool(success), "final_score": float(info["task_score"]),
            "max_score": max_score, "min_eef_obj_dist": round(min_dist, 4), "length": info["step"],
            "terminated": bool(terminated),
            "object_pos_final": [float(x) for x in obs.get("object_pos", [])],
            "wall_time_s": round(time.time() - t0, 1),
            "infer_ms_mean": round(float(np.mean(lat)), 1) if lat else None,
            "n_queries": len(lat) if lat else None,
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
        episodes.sort(key=lambda e: e["seed"])
        pi05._write_summary(out, args, prompt, episodes)
        np.savez_compressed(out / f"ep{seed:03d}_traj.npz", **{k: np.asarray(v) for k, v in traj.items()})
        if record:
            name = out / f"ep{seed:03d}_{'success' if success else 'failure'}.mp4"
            imageio.mimwrite(name, frames, fps=args.control_freq, macro_block_size=1)
        print(f"[{ep + 1}/{args.n_episodes}] seed={seed} success={success} len={info['step']} "
              f"min_dist={min_dist:.3f} ({ep_res['wall_time_s']}s, infer {ep_res['infer_ms_mean'] or 0} ms)",
              flush=True)

    summary = pi05._write_summary(out, args, prompt, episodes)
    print(f"success_rate={summary['success_rate']:.2%} over {len(episodes)} episodes -> {out}/results.json")
    env.close()


if __name__ == "__main__":
    main()
