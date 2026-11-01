"""Observation bundle for the VLM eval: multi-view frames + non-privileged metadata.

build_bundle(seed, out_dir) resets ib-p1-edge-slide-v1 with ``seed``, holds the
arm still (zero delta, gripper open) and captures 3 temporally spaced frames
from each of the 4 configured cameras. No object poses are exposed: only
proprioception, gripper state and the wrist F/T reading.

    python -m imbench.vlm_eval.observation --seed 0 --out runs/bundle_check
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image, ImageDraw

TASK_ID = "ib-p1-edge-slide-v1"
TASK_DESCRIPTION = "Lift the plate above the table."

# label -> human-readable description of where the camera is
CAMERAS = {
    "exo_right_cam": "external camera, right side of the table, elevated, looking at the table centre",
    "exo_left_cam": "external camera, left side of the table, elevated, looking at the table centre",
    "ego_cam": "front camera on the far side of the table, facing back toward the robot, looking down at the table",
    "gripper_cam": "wrist camera on the gripper, looking down along the gripper's approach axis",
}
FRAME_STEPS = (50,)   # env steps at 100 Hz -> one frame at t = 0.5 s (3 spaced frames were identical: static scene)
IMG_WIDTH = 512
JPEG_QUALITY = 90
HOLD_ACTION = np.array([0, 0, 0, 0, 0, 0, -1.0])  # zero delta, gripper fully open
PANDA_MAX_APERTURE_M = 0.08  # 2 x 0.04 m finger travel (PandaGripper joint ranges)

CONTROLLER_INTERFACE = {
    "robot": "single Franka Emika Panda 7-DoF arm, fixed base",
    "end_effector": "parallel-jaw gripper (Panda hand), two fingers, max opening "
                    f"{PANDA_MAX_APERTURE_M:.2f} m; no suction",
    "controller": "OSC_POSE operational-space controller, delta commands in the robot base frame",
    "action_space": {
        "dim": 7,
        "layout": ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"],
        "range": [-1.0, 1.0],
        "scaling": "per step, +-1 maps to +-0.05 m translation and +-0.5 rad rotation targets",
        "gripper": "continuous: -1 fully open, +1 fully closed; intermediate values give partial openings",
    },
    "control_frequency_hz": 100,
    "observables": [
        "eef_pos (m, world frame)", "eef_quat (xyzw)", "gripper finger joint positions",
        "gripper openness (0 closed .. 1 fully open)", "wrist force/torque sensor (N, N*m)",
        "joint positions/velocities", "fingertip tactile arrays (8x8 per finger)",
        "RGB cameras: " + ", ".join(CAMERAS),
    ],
}


def _sensor_snapshot(obs: dict) -> dict:
    gq = np.asarray(obs["robot0_gripper_qpos"], dtype=float)
    ft = np.asarray(obs["force_torque"], dtype=float)
    r3 = lambda a: [round(float(x), 3) for x in a]
    return {
        "eef_pos_m": r3(obs["robot0_eef_pos"]),
        "eef_quat_xyzw": r3(obs["robot0_eef_quat"]),
        "gripper_finger_qpos_m": r3(gq),
        "gripper_opening_m": round(float(abs(gq[0]) + abs(gq[1])), 4),
        "gripper_openness_frac": round(float((abs(gq[0]) + abs(gq[1])) / PANDA_MAX_APERTURE_M), 3),
        "wrist_force_N": r3(ft[:3]),
        "wrist_torque_Nm": r3(ft[3:]),
        "tactile_max_right": round(float(np.max(obs["robot0_tactile_right"])), 4),
        "tactile_max_left": round(float(np.max(obs["robot0_tactile_left"])), 4),
    }


def _save_jpeg(img: np.ndarray, path: Path) -> None:
    im = Image.fromarray(img)
    h = round(im.height * IMG_WIDTH / im.width)
    im.resize((IMG_WIDTH, h), Image.LANCZOS).save(path, "JPEG", quality=JPEG_QUALITY)


def _contact_sheet(frames: list[dict], out_dir: Path, path: Path) -> None:
    cams = list(dict.fromkeys(f["camera"] for f in frames))
    idxs = sorted({f["frame_idx"] for f in frames})
    tiles = {(f["camera"], f["frame_idx"]): Image.open(out_dir / f["file"]) for f in frames}
    w, h = next(iter(tiles.values())).size
    pad = 24
    sheet = Image.new("RGB", (w * len(idxs), (h + pad) * len(cams)), "white")
    d = ImageDraw.Draw(sheet)
    for r, cam in enumerate(cams):
        for c, i in enumerate(idxs):
            y = r * (h + pad)
            sheet.paste(tiles[(cam, i)], (c * w, y + pad))
            d.text((c * w + 4, y + 6), f"{cam}  frame {i}", fill="black")
    sheet.save(path)


def build_bundle(seed: int, out_dir: str | Path) -> dict:
    """Reset with ``seed``, capture frames + metadata into ``out_dir``. Returns the bundle dict."""
    import imbench as ib

    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)

    task = ib.make(TASK_ID, obs_mode="combo")
    try:
        obs, _ = task.reset(seed=seed)
        frames, sensors = [], []
        step = 0
        for idx, target in enumerate(FRAME_STEPS):
            while step < target:
                obs, *_ = task.step(HOLD_ACTION)
                step += 1
            for cam in CAMERAS:
                fname = f"frames/{cam}_f{idx}.jpg"
                _save_jpeg(obs[f"rgb_{cam}"], out_dir / fname)
                frames.append({"camera": cam, "frame_idx": idx, "t_s": step / task.control_freq,
                               "file": fname})
            sensors.append({"frame_idx": idx, "t_s": step / task.control_freq, **_sensor_snapshot(obs)})
    finally:
        task.close() if hasattr(task, "close") else None

    bundle = {
        "task_id": TASK_ID,
        "seed": seed,
        "frames": frames,
        "metadata": {
            "task_description": TASK_DESCRIPTION,
            "controller_interface": CONTROLLER_INTERFACE,
            "cameras": CAMERAS,
            "sensor_snapshot": sensors,
            "capture": f"arm held still (zero delta, gripper open); frames at t = "
                       f"{[s['t_s'] for s in sensors]} s",
        },
    }
    (out_dir / "bundle.json").write_text(json.dumps(bundle, indent=2))
    _contact_sheet(frames, out_dir, out_dir / "contact_sheet.png")
    return bundle


def load_or_build(seed: int, out_dir: str | Path) -> dict:
    p = Path(out_dir) / "bundle.json"
    if p.exists():
        return json.loads(p.read_text())
    return build_bundle(seed, out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/bundle_check")
    a = ap.parse_args()
    b = build_bundle(a.seed, Path(a.out) / f"seed_{a.seed}")
    print(json.dumps(b["metadata"]["sensor_snapshot"][-1], indent=2))
    print("contact sheet:", Path(a.out) / f"seed_{a.seed}" / "contact_sheet.png")
