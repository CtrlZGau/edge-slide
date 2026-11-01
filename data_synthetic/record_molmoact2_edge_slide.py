"""
record_molmoact2_edge_slide.py
==============================
Record edge-slide demonstrations in the exact observation/action convention
that ``scripts/run_vla_eval.py --policy molmoact2`` feeds MolmoAct 2, for
LoRA fine-tuning (e.g. from ``allenai/MolmoAct2-LIBERO``).

Per step (at the eval control rate, default 20 Hz):

* ``image``        libero_cam render, mirrored + resize_with_pad to 256x256
* ``wrist_image``  gripper_cam render, mirrored + resize_with_pad to 256x256
* ``state``        8-D: eef pos in LIBERO world frame (IMBench pos + (-0.25, 0, 0)),
                   eef axis-angle, 2 gripper finger qpos
* ``action``       7-D IMBench cartesian delta in [-1, 1], gripper -1 open / +1 close
                   (exactly what env.step() received)

Image/state preprocessing is imported from ``imbench.policy.vla_clients`` and the
camera (``LIBERO_CAM``) from ``imbench.policy.vla_clients`` so training and evaluation can't drift.

Output (stage 1, raw; convert with ``package_molmoact2_lerobot.py``)::

    <out>/episode_000000/{data.npz, image.mp4, wrist_image.mp4, meta.json}
    <out>/progress.json   # next seed to try -> safe resume, no duplicate seeds

Videos are lossless (libx264rgb, crf 0) so stage 2 re-encodes only once.
Only successful episodes are kept. Seeds start at --seed-start (default 100000)
so the eval seeds 0..N-1 stay held out.

    python data_synthetic/record_molmoact2_edge_slide.py --n-episodes 150 \\
        --out demos_synthetic/molmoact2/raw_edge_slide_20hz --human-noise 1.0
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import imageio.v2 as imageio  # noqa: E402

import imbench as ib  # noqa: E402
from imbench.policy.vla_clients import LIBERO_CAM, quat2axisangle, resize_with_pad  # noqa: E402
from data_synthetic.single_arm_pipeline.edge_slide_teleop_policy import (  # noqa: E402
    EdgeSlideTeleopPolicy,
)

TASK = "ib-p1-edge-slide-v1"
STATE_POS_OFFSET = np.array([-0.25, 0.0, 0.0])   # = _LiberoChunkPolicy default
IMG = 256                                        # = MolmoAct2LiberoPolicy resize


def prep(img: np.ndarray) -> np.ndarray:
    """Same as _LiberoChunkPolicy._prep (mirror=True, resize=256)."""
    return resize_with_pad(np.ascontiguousarray(img[:, ::-1]), IMG, IMG)


def state8(obs) -> np.ndarray:
    """Same as _LiberoChunkPolicy._state."""
    return np.concatenate([
        obs["robot0_eef_pos"] + STATE_POS_OFFSET,
        quat2axisangle(obs["robot0_eef_quat"]),
        obs["robot0_gripper_qpos"],
    ]).astype(np.float32)


def write_lossless(path: Path, frames, fps: int):
    w = imageio.get_writer(str(path), fps=fps, codec="libx264rgb", pixelformat="rgb24",
                           macro_block_size=1, ffmpeg_log_level="error",
                           output_params=["-crf", "0", "-preset", "veryfast"])
    for f in frames:
        w.append_data(f)
    w.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-episodes", type=int, default=150, help="successful episodes wanted (total)")
    ap.add_argument("--out", default=str(REPO / "demos_synthetic/molmoact2/raw_edge_slide_20hz"))
    ap.add_argument("--control-freq", type=int, default=20, help="must match run_vla_eval --control-freq")
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--seed-start", type=int, default=100000)
    ap.add_argument("--human-noise", type=float, default=1.0)
    ap.add_argument("--prompt", default=None,
                    help="instruction stored with every frame (default: derived from the task "
                         "DESCRIPTION exactly like run_vla_eval.py's default prompt)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    env = ib.make(
        TASK, obs_mode="combo", max_episode_steps=args.max_steps,
        config_overrides={
            "control_freq": args.control_freq,
            "obs": {"cameras": {
                "libero_cam": LIBERO_CAM,
                # Not used by the model; skipping them only saves render time
                # (cameras don't affect physics).
                "ego_cam": {"enabled": False},
                "exo_right_cam": {"enabled": False},
                "exo_left_cam": {"enabled": False},
            }},
        },
    )
    prompt = args.prompt or env.DESCRIPTION.split(":", 1)[-1].strip().lower()
    policy = EdgeSlideTeleopPolicy(human_noise=args.human_noise, verbose=False,
                                   control_freq=args.control_freq)

    prog_path = out / "progress.json"
    prog = json.loads(prog_path.read_text()) if prog_path.exists() else {
        "next_seed": args.seed_start, "attempts": 0, "failures": []}
    have = sorted(p for p in out.glob("episode_*") if (p / "meta.json").exists())
    n_ok = len(have)
    print(f"[molmoact2-rec] prompt={prompt!r} control={args.control_freq} Hz  "
          f"have {n_ok} episodes, next seed {prog['next_seed']}  -> {out}", flush=True)

    budget = max(3 * args.n_episodes, args.n_episodes + 20)
    while n_ok < args.n_episodes and prog["attempts"] < budget:
        seed = prog["next_seed"]
        t0 = time.time()
        obs, info = env.reset(seed=seed)
        policy.reset()
        imgs, wrists, states, actions = [], [], [], []
        done = False
        while not done:
            imgs.append(prep(obs["rgb_libero_cam"]))
            wrists.append(prep(obs["rgb_gripper_cam"]))
            states.append(state8(obs))
            a = np.clip(policy.act(obs, env=env), -1.0, 1.0)
            actions.append(a.astype(np.float32))
            obs, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated
        ok = info.get("task_score", 0.0) >= 0.9
        prog["attempts"] += 1
        prog["next_seed"] = seed + 1
        if ok:
            ep_dir = out / f"episode_{n_ok:06d}"
            tmp = out / f".tmp_{n_ok:06d}"
            tmp.mkdir(exist_ok=True)
            T = len(actions)
            np.savez_compressed(
                tmp / "data.npz",
                state=np.stack(states), action=np.stack(actions),
                timestamp=(np.arange(T) / args.control_freq).astype(np.float32),
            )
            write_lossless(tmp / "image.mp4", imgs, args.control_freq)
            write_lossless(tmp / "wrist_image.mp4", wrists, args.control_freq)
            (tmp / "meta.json").write_text(json.dumps({
                "seed": seed, "length": T, "task": prompt, "success": True,
                "control_freq": args.control_freq, "human_noise": args.human_noise,
                "policy": "EdgeSlideTeleopPolicy", "recoveries": policy._recoveries,
                "plate_half_extents": [float(x) for x in policy._half],
                "style": {k: float(v) for k, v in policy._style.items()},
            }, indent=2))
            tmp.rename(ep_dir)          # atomic: an episode dir is always complete
            n_ok += 1
            status = "SUCCESS"
        else:
            prog["failures"].append(seed)
            status = f"FAILURE phase={policy.phase}"
        prog_path.write_text(json.dumps(prog, indent=2))
        print(f"  seed {seed}  steps={len(actions)}  {status}  "
              f"[{n_ok}/{args.n_episodes}]  {time.time() - t0:.1f}s", flush=True)

    env.close()
    print(f"[molmoact2-rec] done: {n_ok} episodes, {prog['attempts']} attempts, "
          f"{len(prog['failures'])} failures", flush=True)


if __name__ == "__main__":
    main()
