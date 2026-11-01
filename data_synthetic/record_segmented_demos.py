"""
record_segmented_demos.py
=========================
Record scripted teleop-style demos for the Diffusion Policy datasets, in the
same observation/action convention as ``record_molmoact2_edge_slide.py`` (and
``scripts/run_vla_eval.py``), plus per-frame **phase labels** so full
edge-slide episodes can be split into a push segment (A) and a pick segment (B).

    --policy edge_slide  ib-p1-edge-slide-v1, EdgeSlideTeleopPolicy(edge_mode=closest, ±Y)
    --policy cube_pick   ib-p1-cube-pick-v1,  CubePickTeleopPolicy

Per step (20 Hz by default):
  image / wrist_image  libero_cam / gripper_cam, mirrored + resize_with_pad 256 (= eval)
  state   (8,)         eef pos + (-0.25,0,0), eef axis-angle, finger qpos (= eval)
  action  (7,)         exactly what env.step() received
  phase   (int)        index into meta["phase_names"]
  object  (7,)         object pos + quat (xyzw), privileged, for analysis only
  overhang (float)     plate overhang past the nearest table edge (edge-slide only)

Output: <out>/episode_<seed>/{data.npz,image.mp4,wrist_image.mp4,meta.json}
(lossless video). Directory names carry the seed, so parallel workers with
disjoint seed ranges can share one --out. Resumable: seeds already attempted
are listed in <out>/attempts_<seed_start>.jsonl.

Kept episodes: task success AND zero policy recoveries (recoveries interleave
push and grasp phases, which would break the A/B split).

    .venv/bin/python data_synthetic/record_segmented_demos.py --policy edge_slide \\
        --seed-start 200000 --seed-end 200100 --out demos_synthetic/dp/raw_edge_slide
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import imbench as ib  # noqa: E402
from imbench.policy.vla_clients import LIBERO_CAM  # noqa: E402
from data_synthetic.record_molmoact2_edge_slide import prep, state8, write_lossless  # noqa: E402
from data_synthetic.single_arm_pipeline.edge_slide_teleop_policy import EdgeSlideTeleopPolicy  # noqa: E402
from data_synthetic.single_arm_pipeline.cube_pick_teleop_policy import CubePickTeleopPolicy  # noqa: E402
from run_pi05_eval import plate_overhang  # noqa: E402  (read-only import)

POLICIES = {
    "edge_slide": dict(
        task="ib-p1-edge-slide-v1",
        phases=["start", "approach_push", "descend_push", "push", "clear", "to_pregrasp",
                "descend_pregrasp", "approach_grasp", "close", "lift", "back_off",
                "return_to_push", "hold"],
    ),
    "cube_pick": dict(
        task="ib-p1-cube-pick-v1",
        phases=["start", "approach", "descend", "close", "lift", "back_off", "hold"],
    ),
}


def make_policy(name, args):
    kw = dict(human_noise=args.human_noise, verbose=False, control_freq=args.control_freq)
    if name == "edge_slide":
        return EdgeSlideTeleopPolicy(edge_mode=args.edge_mode, allowed_edges=tuple(args.allowed_edges), **kw)
    return CubePickTeleopPolicy(**kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", choices=sorted(POLICIES), required=True)
    ap.add_argument("--seed-start", type=int, required=True)
    ap.add_argument("--seed-end", type=int, required=True, help="exclusive")
    ap.add_argument("--out", required=True)
    ap.add_argument("--control-freq", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=1000)
    ap.add_argument("--human-noise", type=float, default=1.0)
    ap.add_argument("--edge-mode", default="closest")
    ap.add_argument("--allowed-edges", nargs="+", default=["+y", "-y"])
    args = ap.parse_args()

    spec = POLICIES[args.policy]
    phase_id = {p: i for i, p in enumerate(spec["phases"])}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / f"attempts_{args.seed_start}.jsonl"
    done_seeds = set()
    if log_path.exists():
        done_seeds = {json.loads(l)["seed"] for l in log_path.read_text().splitlines() if l.strip()}

    env = ib.make(
        spec["task"], obs_mode="combo", max_episode_steps=args.max_steps,
        config_overrides={
            "control_freq": args.control_freq,
            "obs": {"cameras": {
                "libero_cam": LIBERO_CAM,
                "ego_cam": {"enabled": False},
                "exo_right_cam": {"enabled": False},
                "exo_left_cam": {"enabled": False},
            }},
        },
    )
    policy = make_policy(args.policy, args)
    n_ok = n_try = 0
    for seed in range(args.seed_start, args.seed_end):
        if seed in done_seeds:
            continue
        t0 = time.time()
        obs, info = env.reset(seed=seed)
        policy.reset()
        imgs, wrists, states, actions, phases, objs, ovh = [], [], [], [], [], [], []
        done = False
        while not done:
            imgs.append(prep(obs["rgb_libero_cam"]))
            wrists.append(prep(obs["rgb_gripper_cam"]))
            states.append(state8(obs))
            objs.append(np.r_[obs["object_pos"], obs["object_quat"]].astype(np.float32))
            if args.policy == "edge_slide":
                ovh.append(plate_overhang(env)[0])
            a = np.clip(policy.act(obs, env=env), -1.0, 1.0)
            # phase that produced this action (act() may advance the phase internally)
            phases.append(phase_id[policy.phase])
            actions.append(a.astype(np.float32))
            obs, _, terminated, truncated, info = env.step(a)
            done = terminated or truncated
        success = info.get("task_score", 0.0) >= 0.9
        keep = success and policy._recoveries == 0
        rec = dict(seed=seed, success=bool(success), recoveries=int(policy._recoveries),
                   kept=bool(keep), length=len(actions), final_phase=policy.phase,
                   edge=getattr(policy, "edge", None) if args.policy == "edge_slide" else None,
                   secs=round(time.time() - t0, 1))
        if keep:
            ep = out / f"episode_{seed:07d}"
            tmp = out / f".tmp_{seed:07d}"
            tmp.mkdir(exist_ok=True)
            T = len(actions)
            np.savez_compressed(
                tmp / "data.npz", state=np.stack(states), action=np.stack(actions),
                phase=np.array(phases, np.int16), object=np.stack(objs),
                overhang=np.array(ovh if ovh else [np.nan] * T, np.float32),
                timestamp=(np.arange(T) / args.control_freq).astype(np.float32))
            write_lossless(tmp / "image.mp4", imgs, args.control_freq)
            write_lossless(tmp / "wrist_image.mp4", wrists, args.control_freq)
            (tmp / "meta.json").write_text(json.dumps({
                "seed": seed, "length": T, "task_id": spec["task"], "policy": type(policy).__name__,
                "success": True, "recoveries": 0, "control_freq": args.control_freq,
                "human_noise": args.human_noise, "edge": rec["edge"],
                "object_half_extents": [float(x) for x in policy._half],
                "style": {k: (np.asarray(v).tolist()) for k, v in policy._style.items()},
                "phase_names": spec["phases"],
            }, indent=2))
            tmp.rename(ep)
            n_ok += 1
        n_try += 1
        with open(log_path, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"  seed {seed} len={rec['length']} edge={rec['edge']} success={success} "
              f"rec={rec['recoveries']} kept={keep}  [{n_ok}/{n_try}] {rec['secs']}s", flush=True)
    env.close()
    print(f"[rec] done {args.policy} seeds {args.seed_start}-{args.seed_end}: kept {n_ok}/{n_try}", flush=True)


if __name__ == "__main__":
    main()
