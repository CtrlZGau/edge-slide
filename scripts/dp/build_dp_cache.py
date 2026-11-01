"""
build_dp_cache.py
=================
Decode a LeRobot v3.0 dataset (from ``package_dp_lerobot.py``) ONCE into a
compact training cache, because random-access video decoding through LeRobot's
pyav backend costs ~160 ms/sample here (torchcodec can't load: no system FFmpeg
shared libs), which would starve the GPU.

Cache layout (<out>/):
  meta.json                     fps, image size, keys, task names, episode table
  lowdim.npz                    state (N,8), action (N,7), env_state (N,3),
                                episode_index (N,), frame_index (N,), task_index (N,)
  <cam>.bin / <cam>_off.npy     JPEG (q=95) frames at --size x --size, concatenated;
                                offsets (N+1,) into the .bin, in global frame order

Frames are resized 256 -> --size with cv2.INTER_AREA. Every cached frame is
cross-checked (exact count, timestamp mapping) and a few are compared against
LeRobotDataset's own decode.

    ~/venvs/lerobot/bin/python scripts/dp/build_dp_cache.py \\
        --dataset demos_synthetic/dp/lerobot/imbench/dp_multitask_20hz --out demos_synthetic/dp/cache128
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

CAMS = ["observation.images.image", "observation.images.wrist_image"]


def decode_file(args):
    path, size, quality = args
    out = []
    with av.open(str(path)) as c:
        for fr in c.decode(video=0):
            img = fr.to_ndarray(format="rgb24")
            if img.shape[0] != size:
                img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, quality])
            assert ok
            out.append(buf.tobytes())
    return str(path), out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--quality", type=int, default=95)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = Path(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((root / "meta/info.json").read_text())
    fps = info["fps"]
    eps = pd.concat([pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet"))])
    eps = eps.sort_values("episode_index").reset_index(drop=True)
    data = pd.concat([pd.read_parquet(p) for p in sorted((root / "data").rglob("*.parquet"))])
    data = data.sort_values("index").reset_index(drop=True)
    N = len(data)
    assert N == info["total_frames"] and (data["index"].values == np.arange(N)).all()

    low = dict(
        state=np.stack(data["observation.state"].values).astype(np.float32),
        action=np.stack(data["action"].values).astype(np.float32),
        env_state=np.stack(data["observation.environment_state"].values).astype(np.float32),
        episode_index=data["episode_index"].values.astype(np.int32),
        frame_index=data["frame_index"].values.astype(np.int32),
        task_index=data["task_index"].values.astype(np.int16),
    )
    np.savez(out / "lowdim.npz", **low)

    vtmpl = info["video_path"]
    for cam in CAMS:
        files = {}
        for _, e in eps.iterrows():
            p = root / vtmpl.format(video_key=cam, chunk_index=int(e[f"videos/{cam}/chunk_index"]),
                                    file_index=int(e[f"videos/{cam}/file_index"]))
            files.setdefault(p, None)
        decoded = {}
        with ProcessPoolExecutor(args.workers) as ex:
            for path, frames in ex.map(decode_file, [(p, args.size, args.quality) for p in files]):
                decoded[path] = frames
                print(f"  {cam}: decoded {Path(path).name} ({len(frames)} frames)", flush=True)
        blobs = [None] * N
        for _, e in eps.iterrows():
            p = str(root / vtmpl.format(video_key=cam, chunk_index=int(e[f"videos/{cam}/chunk_index"]),
                                        file_index=int(e[f"videos/{cam}/file_index"])))
            f0 = int(round(float(e[f"videos/{cam}/from_timestamp"]) * fps))
            L = int(e["length"])
            g0 = int(e["dataset_from_index"])
            assert f0 + L <= len(decoded[p]), (cam, e["episode_index"], f0, L, len(decoded[p]))
            for k in range(L):
                blobs[g0 + k] = decoded[p][f0 + k]
        assert all(b is not None for b in blobs)
        off = np.zeros(N + 1, np.int64)
        off[1:] = np.cumsum([len(b) for b in blobs])
        with open(out / f"{cam}.bin", "wb") as f:
            for b in blobs:
                f.write(b)
        np.save(out / f"{cam}_off.npy", off)
        print(f"{cam}: {N} frames, {off[-1] / 1e9:.2f} GB", flush=True)
        del decoded, blobs

    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    task_names = [None] * len(tasks)
    for name, row in tasks.iterrows():
        task_names[int(row["task_index"])] = name
    meta = dict(
        source=str(root), fps=fps, size=args.size, jpeg_quality=args.quality, cams=CAMS, n_frames=N,
        task_names=task_names,
        episodes=[dict(episode_index=int(e["episode_index"]), start=int(e["dataset_from_index"]),
                       length=int(e["length"])) for _, e in eps.iterrows()],
    )
    (out / "meta.json").write_text(json.dumps(meta))

    # Spot-check against LeRobot's own decoder.
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(info.get("repo_id", "x"), root=root, video_backend="pyav")
    rng = np.random.default_rng(0)
    errs = []
    for i in rng.integers(0, N, 12):
        s = ds[int(i)]
        assert np.allclose(s["action"].numpy(), low["action"][i])
        for cam in CAMS:
            off = np.load(out / f"{cam}_off.npy")
            with open(out / f"{cam}.bin", "rb") as f:
                f.seek(off[i]); b = f.read(off[i + 1] - off[i])
            mine = cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1].astype(np.float32)
            ref = (s[cam].numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            ref = cv2.resize(ref, (args.size, args.size), interpolation=cv2.INTER_AREA).astype(np.float32)
            errs.append(np.abs(mine - ref).mean())
    print(f"spot-check vs LeRobotDataset decode: mean-abs-err max {max(errs):.2f} (0-255)", flush=True)
    assert max(errs) < 4.0, "cache frames don't match the dataset"


if __name__ == "__main__":
    main()
