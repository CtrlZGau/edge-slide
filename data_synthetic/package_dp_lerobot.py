"""
package_dp_lerobot.py
=====================
Package raw episodes from ``record_segmented_demos.py`` into ONE LeRobot v3.0
multi-task dataset for Diffusion Policy training (run in ~/venvs/lerobot):

  task 0  A  edge-slide push segment  "push the flat plate to the edge of the table"
  task 1  B  edge-slide pick segment  "pick up the flat plate from the edge of the table"
  task 2  C  cube pick                "pick up the cube from the table"

Each full edge-slide episode is split at the first ``to_pregrasp`` frame
(end of the push's ``clear`` hop): A = [0, split), B = [split, T). B therefore
starts exactly where A ends, so A->B can be chained at eval time.

Features (the image/state/action keys match the MolmoAct2/LIBERO convention, so
the dataset can also fine-tune a VLA; filter by ``task_index``):

  observation.images.image        video 256x256 (libero_cam, eval-preprocessed)
  observation.images.wrist_image  video 256x256 (gripper_cam, eval-preprocessed)
  observation.state               float32 (8,)
  observation.environment_state   float32 (3,)  one-hot task id (DP conditioning)
  action                          float32 (7,)

A sidecar ``meta/imbench_episodes.jsonl`` records the source episode, seed,
segment, edge, object size and split frame of every dataset episode.

    ~/venvs/lerobot/bin/python data_synthetic/package_dp_lerobot.py \\
        --raw-edge demos_synthetic/dp/raw_edge_slide --raw-cube demos_synthetic/dp/raw_cube \\
        --root demos_synthetic/dp/lerobot --repo-id imbench/dp_multitask_20hz
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import imageio.v3 as iio
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASKS = [
    ("push", "push the flat plate to the edge of the table"),
    ("pick", "pick up the flat plate from the edge of the table"),
    ("cube", "pick up the cube from the table"),
]

FEATURES = {
    "observation.images.image": {
        "dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.images.wrist_image": {
        "dtype": "video", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "observation.state": {
        "dtype": "float32", "shape": (8,),
        "names": ["eef_x", "eef_y", "eef_z", "eef_ax", "eef_ay", "eef_az",
                  "finger_left", "finger_right"]},
    "observation.environment_state": {
        "dtype": "float32", "shape": (3,), "names": ["task_push", "task_pick", "task_cube"]},
    "action": {
        "dtype": "float32", "shape": (7,),
        "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]},
}


def episodes(raw: Path, limit: int | None):
    eps = sorted(p for p in raw.glob("episode_*") if (p / "meta.json").exists())
    return eps[:limit] if limit else eps


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-edge", required=True)
    ap.add_argument("--raw-cube", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--repo-id", default="imbench/dp_multitask_20hz")
    ap.add_argument("--n-edge", type=int, default=None, help="use the first N edge episodes (by seed)")
    ap.add_argument("--n-cube", type=int, default=None)
    ap.add_argument("--vcodec", default="h264")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--shard", type=int, default=None,
                    help="write only episodes i::num_shards to <repo-id>_shard<i> (run shards in "
                         "parallel, then --merge)")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge", action="store_true",
                    help="aggregate <repo-id>_shard0..N-1 into <repo-id> (videos/data are copied, "
                         "not re-encoded) and concatenate their sidecars")
    args = ap.parse_args()
    if args.merge:
        return merge(args)
    if args.shard is not None:
        args.repo_id = f"{args.repo_id}_shard{args.shard}"

    out = Path(args.root) / args.repo_id
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    ds = LeRobotDataset.create(repo_id=args.repo_id, fps=20, features=FEATURES, root=out,
                               robot_type="panda", use_videos=True, vcodec=args.vcodec,
                               image_writer_threads=8)
    onehot = np.eye(len(TASKS), dtype=np.float32)
    sidecar, counts = [], {k: [0, 0] for k, _ in TASKS}

    def add(seg, src, meta, d, img, wri, lo, hi, extra):
        ti = [k for k, _ in TASKS].index(seg)
        for t in range(lo, hi):
            ds.add_frame({
                "observation.images.image": img[t],
                "observation.images.wrist_image": wri[t],
                "observation.state": d["state"][t].astype(np.float32),
                "observation.environment_state": onehot[ti],
                "action": d["action"][t].astype(np.float32),
                "task": TASKS[ti][1],
            })
        ds.save_episode()
        sidecar.append({"episode_index": len(sidecar), "segment": seg, "task": TASKS[ti][1],
                        "source": src.name, "seed": meta["seed"], "frames": [lo, hi],
                        "edge": meta.get("edge"), "object_half_extents": meta["object_half_extents"],
                        **extra})
        counts[seg][0] += 1
        counts[seg][1] += hi - lo

    def load(ep):
        meta = json.loads((ep / "meta.json").read_text())
        d = np.load(ep / "data.npz")
        img = iio.imread(ep / "image.mp4", plugin="pyav")
        wri = iio.imread(ep / "wrist_image.mp4", plugin="pyav")
        T = len(d["action"])
        assert meta["control_freq"] == 20 and len(img) == len(wri) == T == len(d["state"]), ep
        return meta, d, img, wri, T

    edge_eps = episodes(Path(args.raw_edge), args.n_edge)
    cube_eps = episodes(Path(args.raw_cube), args.n_cube)
    if args.shard is not None:
        edge_eps = edge_eps[args.shard::args.num_shards]
        cube_eps = cube_eps[args.shard::args.num_shards]
    for k, ep in enumerate(edge_eps):
        meta, d, img, wri, T = load(ep)
        names = meta["phase_names"]
        hits = np.flatnonzero(d["phase"] == names.index("to_pregrasp"))
        if len(hits) == 0:
            raise SystemExit(f"{ep}: no to_pregrasp phase")
        split = int(hits[0])
        ovh = float(d["overhang"][split])
        add("push", ep, meta, d, img, wri, 0, split, {"split": split, "overhang_at_split": ovh})
        add("pick", ep, meta, d, img, wri, split, T, {"split": split, "overhang_at_split": ovh})
        print(f"  edge [{k + 1}/{len(edge_eps)}] {ep.name} edge={meta['edge']} split={split}/{T} "
              f"overhang={ovh:.3f}", flush=True)
    for k, ep in enumerate(cube_eps):
        meta, d, img, wri, T = load(ep)
        add("cube", ep, meta, d, img, wri, 0, T, {})
        print(f"  cube [{k + 1}/{len(cube_eps)}] {ep.name} T={T}", flush=True)

    ds.finalize()
    with open(out / "meta" / "imbench_episodes.jsonl", "w") as f:
        for r in sidecar:
            f.write(json.dumps(r) + "\n")
    print("done:", {k: f"{v[0]} episodes / {v[1]} frames" for k, v in counts.items()}, "->", out, flush=True)


def merge(args):
    from lerobot.datasets.aggregate import aggregate_datasets

    root = Path(args.root)
    shard_ids = [f"{args.repo_id}_shard{i}" for i in range(args.num_shards)]
    out = root / args.repo_id
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"{out} exists (use --overwrite)")
        shutil.rmtree(out)
    aggregate_datasets(shard_ids, args.repo_id, roots=[root / r for r in shard_ids], aggr_root=out)
    rows = []
    for r in shard_ids:
        for line in (root / r / "meta" / "imbench_episodes.jsonl").read_text().splitlines():
            row = json.loads(line)
            row["shard"], row["shard_episode_index"] = r, row["episode_index"]
            row["episode_index"] = len(rows)
            rows.append(row)
    ds = LeRobotDataset(args.repo_id, root=out)
    assert ds.meta.total_episodes == len(rows), (ds.meta.total_episodes, len(rows))
    with open(out / "meta" / "imbench_episodes.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"merged {len(shard_ids)} shards: {ds.meta.total_episodes} episodes, "
          f"{ds.meta.total_frames} frames -> {out}", flush=True)


if __name__ == "__main__":
    main()
