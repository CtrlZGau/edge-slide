"""
package_molmoact2_lerobot.py
============================
Stage 2: turn the raw episodes written by ``record_molmoact2_edge_slide.py``
into a LeRobot **v3.0** dataset that MolmoAct 2's trainer
(``launch_scripts/train_lerobot.py``) can read directly.

Run with the LeRobot venv (LeRobot 0.5.0 needs numpy 2 / torch, so it can't
live in the IMBench venv)::

    ~/venvs/lerobot/bin/python data_synthetic/package_molmoact2_lerobot.py \\
        --raw demos_synthetic/molmoact2/raw_edge_slide_20hz \\
        --root demos_synthetic/molmoact2/lerobot \\
        --repo-id imbench/edge_slide_teleop_20hz

The dataset is written to ``<root>/<repo-id>`` (the layout MolmoAct's loader
expects with ``LEROBOT_DATA_ROOT=<root>``). Feature keys match
``allenai/MolmoAct2-LIBERO-Dataset``:

    observation.images.image        video 256x256x3 (libero_cam, eval-preprocessed)
    observation.images.wrist_image  video 256x256x3 (gripper_cam, eval-preprocessed)
    observation.state               float32 (8,)
    action                          float32 (7,)

Nothing is uploaded anywhere (no push_to_hub).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FEATURES = {
    "observation.images.image": {
        "dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.wrist_image": {
        "dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {
        "dtype": "float32", "shape": (8,),
        "names": ["eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az",
                  "finger_left", "finger_right"]},
    "action": {
        "dtype": "float32", "shape": (7,),
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--root", required=True, help="LEROBOT_DATA_ROOT")
    ap.add_argument("--repo-id", default="imbench/edge_slide_teleop_20hz")
    ap.add_argument("--vcodec", default="h264")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    raw = Path(args.raw)
    eps = sorted(p for p in raw.glob("episode_*") if (p / "meta.json").exists())
    if not eps:
        raise SystemExit(f"no episodes in {raw}")
    metas = [json.loads((p / "meta.json").read_text()) for p in eps]
    fps = {m["control_freq"] for m in metas}
    if len(fps) != 1:
        raise SystemExit(f"mixed control rates in {raw}: {fps}")
    fps = fps.pop()

    out = Path(args.root) / args.repo_id
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)

    ds = LeRobotDataset.create(
        repo_id=args.repo_id, fps=fps, features=FEATURES, root=out,
        robot_type="panda", use_videos=True, vcodec=args.vcodec,
        image_writer_threads=8,
    )

    sidecar = []
    for i, (ep, meta) in enumerate(zip(eps, metas)):
        d = np.load(ep / "data.npz")
        img = iio.imread(ep / "image.mp4", plugin="pyav")
        wri = iio.imread(ep / "wrist_image.mp4", plugin="pyav")
        T = len(d["action"])
        assert len(img) == len(wri) == T == len(d["state"]), (ep, len(img), len(wri), T)
        for t in range(T):
            ds.add_frame({
                "observation.images.image": img[t],
                "observation.images.wrist_image": wri[t],
                "observation.state": d["state"][t].astype(np.float32),
                "action": d["action"][t].astype(np.float32),
                "task": meta["task"],
            })
        ds.save_episode()
        sidecar.append({"episode_index": i, "source": ep.name, **meta})
        print(f"  [{i + 1}/{len(eps)}] {ep.name}  seed={meta['seed']}  frames={T}", flush=True)

    ds.finalize()
    with open(out / "meta" / "imbench_episodes.jsonl", "w") as f:
        for r in sidecar:
            f.write(json.dumps(r) + "\n")
    print(f"done: {len(eps)} episodes, {sum(m['length'] for m in metas)} frames @ {fps} Hz -> {out}")


if __name__ == "__main__":
    main()
