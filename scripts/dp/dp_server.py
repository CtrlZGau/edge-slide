#!/usr/bin/env python3
"""Serve a multi-task Diffusion Policy checkpoint (``train_dp.py``) over ZMQ/msgpack.

Runs in ~/venvs/lerobot (LeRobot/torch), not the IMBench venv. One eval client per server.

Endpoints (same wire protocol as scripts/vla_eval/molmoact2_server.py):
  ping
  reset  {task_id}                        clear obs/action queues, set the task one-hot
  act    {image, wrist_image, state}      -> {action (7,), new_chunk: bool}
         image / wrist_image: 256x256x3 uint8, already mirrored + resize_with_pad
         (= record_molmoact2_edge_slide.prep); state: 8-D (= state8).

Images get the same preprocessing as training: INTER_AREA resize to the cache
size, then a JPEG (q=95) round trip, then the saved MEAN_STD stats.
Weights: EMA (default) or raw.

    ~/venvs/lerobot/bin/python scripts/dp/dp_server.py --ckpt outputs/dp/multitask_v1/checkpoints/step0200000 --port 5600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
import zmq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_dp import CAMS, Normalizer, make_config  # noqa: E402

from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--port", type=int, default=5600)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--weights", choices=["ema", "model"], default="ema")
    ap.add_argument("--inference-steps", type=int, default=None, help="override denoising steps (default: as trained, 100)")
    ap.add_argument("--scheduler", choices=["DDPM", "DDIM"], default="DDPM",
                    help="DDIM reuses the DDPM-trained eps-predictor with fewer steps (paper A.4: DDIM, 16 steps, real-world)")
    args = ap.parse_args()

    ck = Path(args.ckpt)
    targs = argparse.Namespace(**json.loads((ck / "train_args.json").read_text()))
    if args.inference_steps:
        targs.inference_steps = args.inference_steps
    targs.scheduler = args.scheduler
    cfgj = json.loads((ck / "config.json").read_text())
    size = int(cfgj["image_size"])
    cfg = make_config(targs, size)
    policy = DiffusionPolicy(cfg)
    state = torch.load(ck / "state.pt", map_location="cpu")
    policy.load_state_dict(state["ema" if args.weights == "ema" else "model"])
    policy = policy.cuda().eval()
    norm = Normalizer(json.loads((ck / "stats.json").read_text())).cuda()
    n_task = 3
    print(f"[dp-server] {ck} ({args.weights}, step {state['step']}), img {size}, "
          f"{cfg.noise_scheduler_type} steps {cfg.num_inference_steps}, port {args.port}", flush=True)
    cur = {"task": 0}

    def prep_img(im):
        im = np.asarray(im, np.uint8)
        if im.shape[0] != size:
            im = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", im[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
        im = cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]
        return torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1)[None]   # (1,3,H,W) uint8

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    while True:
        req = msgpack.unpackb(sock.recv(), object_hook=mnp.decode, raw=False)
        ep, data = req.get("endpoint"), req.get("data") or {}
        try:
            if ep == "ping":
                resp = {"status": "ok"}
            elif ep == "reset":
                cur["task"] = int(data.get("task_id", 0))
                policy.reset()
                resp = {"status": "ok"}
            elif ep == "act":
                t0 = time.time()
                onehot = torch.zeros(1, n_task); onehot[0, cur["task"]] = 1.0
                b = {
                    "observation.state": torch.as_tensor(np.asarray(data["state"], np.float32))[None],
                    "observation.environment_state": onehot,
                    CAMS[0]: prep_img(data["image"]),
                    CAMS[1]: prep_img(data["wrist_image"]),
                }
                b = norm({k: v.cuda() for k, v in b.items()})
                new_chunk = len(policy._queues["action"]) == 0
                with torch.no_grad():
                    a = policy.select_action(b)
                a = norm.unnorm_action(a)[0].float().cpu().numpy()
                resp = {"action": a, "new_chunk": new_chunk, "secs": time.time() - t0}
            else:
                resp = {"error": f"unknown endpoint {ep}"}
        except Exception as e:  # report, keep serving
            resp = {"error": repr(e)}
        sock.send(msgpack.packb(resp, default=mnp.encode))


if __name__ == "__main__":
    main()
