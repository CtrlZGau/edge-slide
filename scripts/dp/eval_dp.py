"""
eval_dp.py
==========
Closed-loop evaluation of the multi-task Diffusion Policy served by
``dp_server.py`` (IMBench venv; one client per server). Observation convention =
``record_segmented_demos.py`` (libero_cam + gripper_cam, mirrored, 256; 8-D state; 20 Hz).

Modes (seeds are held out: training used 200000+ / 300000+):
  push   task A on ib-p1-edge-slide-v1. Success = the A->B hand-off condition is reached
         within --max-a steps: plate overhang >= 3 cm past a table edge (corner metric,
         run_pi05_eval.plate_overhang), not fallen, and the gripper has lifted clear
         (eef z >= table + 7 cm, the end-of-``clear`` pose where the training A segment ends).
  pick   task B standalone: the scripted EdgeSlideTeleopPolicy runs the push until its
         ``to_pregrasp`` phase (= exactly where training B segments start), then the DP
         takes over. Success = env task_score (plate >= 15 cm up with gripper contact).
  chain  DP A until hand-off, then DP B (server reset with the B task id). Success = lift.
  cube   task C on ib-p1-cube-pick-v1. Success = env task_score.

Writes <out>/episodes.jsonl (one line per seed, resumable) and <out>/videos/*.mp4.

    .venv/bin/python scripts/dp/eval_dp.py --mode chain --port 5600 --seeds 0 50 --out outputs/dp_eval/x/chain
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import imageio.v2 as imageio  # noqa: E402

import imbench as ib  # noqa: E402
from imbench.policy.vla_clients import LIBERO_CAM, ZmqMsgpackClient  # noqa: E402
from data_synthetic.record_molmoact2_edge_slide import prep, state8  # noqa: E402
from data_synthetic.single_arm_pipeline.edge_slide_teleop_policy import EdgeSlideTeleopPolicy  # noqa: E402
from run_pi05_eval import plate_overhang  # noqa: E402  (read-only import)

TASK_ID = {"push": 0, "pick": 1, "cube": 2}
TABLE_Z = 0.80
HANDOFF_OVERHANG = 0.03
HANDOFF_EEF_Z = TABLE_Z + 0.07


def make_env(task, max_steps):
    return ib.make(task, obs_mode="combo", max_episode_steps=max_steps, config_overrides={
        "control_freq": 20,
        "obs": {"cameras": {"libero_cam": LIBERO_CAM, "ego_cam": {"enabled": False},
                            "exo_right_cam": {"enabled": False}, "exo_left_cam": {"enabled": False}}}})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["push", "pick", "chain", "cube"], required=True)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--seeds", type=int, nargs=2, default=[0, 50], help="[start, end)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-a", type=int, default=400, help="DP steps allowed for the push (A)")
    ap.add_argument("--max-b", type=int, default=600, help="DP steps allowed for the pick (B)")
    ap.add_argument("--max-c", type=int, default=500, help="DP steps allowed for the cube pick (C)")
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()

    out = Path(args.out); (out / "videos").mkdir(parents=True, exist_ok=True)
    log = out / "episodes.jsonl"
    done = {json.loads(l)["seed"] for l in log.read_text().splitlines()} if log.exists() else set()
    task = "ib-p1-cube-pick-v1" if args.mode == "cube" else "ib-p1-edge-slide-v1"
    env = make_env(task, 2000)
    cli = ZmqMsgpackClient(args.host, args.port, timeout_ms=120_000)
    assert cli.ping(), f"no DP server on port {args.port}"
    scripted = EdgeSlideTeleopPolicy(human_noise=1.0, verbose=False, control_freq=20, edge_mode="closest")

    for seed in range(*args.seeds):
        if seed in done:
            continue
        t0 = time.time()
        obs, _ = env.reset(seed=seed)
        frames, infer_s = [], []
        rec = {"seed": seed, "mode": args.mode}

        def frame(o, label):
            if not args.no_video:
                f = np.concatenate([prep(o["rgb_libero_cam"]), prep(o["rgb_gripper_cam"])], 1)
                f[:14, :, :] = (f[:14] * 0.3).astype(np.uint8)  # dark strip; label is in the filename/json
                frames.append(f)

        def dp_steps(o, task_name, max_steps, stop):
            """Run the DP for up to max_steps; stop(o, info) -> True ends early. Returns (o, info, n, stopped)."""
            cli.call("reset", {"task_id": TASK_ID[task_name]})
            info = {"task_score": 0.0}
            for n in range(max_steps):
                frame(o, task_name)
                r = cli.call("act", {"image": prep(o["rgb_libero_cam"]), "wrist_image": prep(o["rgb_gripper_cam"]),
                                     "state": state8(o)})
                if r["new_chunk"]:
                    infer_s.append(r["secs"])
                a = np.clip(np.asarray(r["action"], np.float64), -1.0, 1.0)
                o, _, term, trunc, info = env.step(a)
                if stop(o, info) or term or trunc:
                    return o, info, n + 1, True
            return o, info, max_steps, False

        def handoff(o, info):
            ovh, fell = plate_overhang(env)
            return (not fell) and ovh >= HANDOFF_OVERHANG and o["robot0_eef_pos"][2] >= HANDOFF_EEF_Z

        def lifted(o, info):
            return info.get("task_score", 0.0) >= 0.9

        if args.mode in ("push", "chain"):
            obs, info, n_a, _ = dp_steps(obs, "push", args.max_a, handoff)
            ovh, fell = plate_overhang(env)
            a_ok = (not fell) and ovh >= HANDOFF_OVERHANG and obs["robot0_eef_pos"][2] >= HANDOFF_EEF_Z
            rec.update(a_success=bool(a_ok), a_steps=n_a, overhang_after_a=round(ovh, 4), fell_after_a=fell,
                       plate_y_after_a=round(float(obs["object_pos"][1]), 4))
            if args.mode == "push":
                rec["success"] = bool(a_ok)
            elif not a_ok:
                rec.update(success=False, b_steps=0)
            else:
                obs, info, n_b, _ = dp_steps(obs, "pick", args.max_b, lifted)
                rec.update(success=bool(info.get("task_score", 0) >= 0.9), b_steps=n_b)
        elif args.mode == "pick":
            scripted.reset()
            n_s = 0
            while True:
                a = np.clip(scripted.act(obs, env=env), -1, 1)
                if scripted.phase == "to_pregrasp":
                    break           # hand over before executing the first B action, as in the data split
                frame(obs, "scripted")
                obs, _, term, trunc, info = env.step(a)
                n_s += 1
                if term or trunc or n_s > 600:
                    break
            ovh, fell = plate_overhang(env)
            rec.update(scripted_steps=n_s, edge=scripted.edge, overhang_at_handoff=round(ovh, 4),
                       scripted_reached_handoff=scripted.phase == "to_pregrasp")
            if scripted.phase != "to_pregrasp":
                rec.update(success=False, note="scripted push failed; excluded from B success rate", excluded=True)
            else:
                obs, info, n_b, _ = dp_steps(obs, "pick", args.max_b, lifted)
                rec.update(success=bool(info.get("task_score", 0) >= 0.9), b_steps=n_b)
        else:
            obs, info, n_c, _ = dp_steps(obs, "cube", args.max_c, lifted)
            rec.update(success=bool(info.get("task_score", 0) >= 0.9), c_steps=n_c)

        rec["object_pos_final"] = [round(float(x), 4) for x in obs["object_pos"]]
        rec["infer_s_mean"] = round(float(np.mean(infer_s)), 3) if infer_s else None
        rec["wall_s"] = round(time.time() - t0, 1)
        if frames:
            tag = "ok" if rec["success"] else "FAIL"
            w = imageio.get_writer(str(out / "videos" / f"seed{seed:03d}_{tag}.mp4"), fps=20,
                                   macro_block_size=1, ffmpeg_log_level="error")
            for f in frames:
                w.append_data(f)
            w.close()
        with open(log, "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
    env.close()


if __name__ == "__main__":
    main()
