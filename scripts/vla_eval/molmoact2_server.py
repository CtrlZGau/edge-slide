#!/usr/bin/env python3
"""Serve allenai/MolmoAct2-LIBERO over ZMQ (GR00T PolicyServer wire protocol).

Runs in its own venv (torch + transformers + trust_remote_code), NOT the IMBench venv:

    python -m venv ~/venvs/molmoact2 && source ~/venvs/molmoact2/bin/activate
    pip install torch transformers pillow numpy huggingface_hub pyzmq msgpack msgpack-numpy
    python scripts/vla_eval/molmoact2_server.py --port 5556

Endpoints: ping, reset {seed}, get_action {images:[main, wrist], state(8,), task} -> {actions (H, 7)}.
Inference call copied from the model card (https://huggingface.co/allenai/MolmoAct2-LIBERO).
"""

from __future__ import annotations

import argparse
import time

import msgpack
import msgpack_numpy as mnp
import numpy as np
import torch
import zmq
from PIL import Image


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="allenai/MolmoAct2-LIBERO")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32",
                   help="model card default is float32; bfloat16 fits in <16 GB")
    p.add_argument("--num-steps", type=int, default=10, help="flow solver steps")
    p.add_argument("--norm-tag", default="libero",
                   help="key in the checkpoint's norm_stats.json (fine-tuned IMBench model: imbench_edge_slide)")
    p.add_argument("--no-cuda-graph", action="store_true")
    p.add_argument("--device", default="cuda", help="cuda | cpu (cpu: smoke tests only, forces fp32, no CUDA graph)")
    args = p.parse_args()
    if args.device == "cpu":
        args.dtype, args.no_cuda_graph = "float32", True

    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    # bfloat16: load weights in bf16 too (halves VRAM; needed on 24 GB cards if fp32 doesn't fit).
    load_dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, dtype=load_dtype,
    ).to(args.device).eval()
    print(f"loaded {args.model} as {load_dtype}; device {args.device}", flush=True)

    def predict(images, state, task):
        kw = dict(
            processor=processor,
            images=[Image.fromarray(np.asarray(im, dtype=np.uint8)) for im in images],
            task=task,
            state=np.asarray(state, dtype=np.float32),
            norm_tag=args.norm_tag,
            inference_action_mode="continuous",
            enable_depth_reasoning=False,
            num_steps=args.num_steps,
            normalize_language=True,
            enable_cuda_graph=not args.no_cuda_graph,
        )
        with torch.inference_mode():
            if args.dtype == "bfloat16":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model.predict_action(**kw)
            else:
                out = model.predict_action(**kw)
        a = out.actions
        a = a.detach().float().cpu().numpy() if torch.is_tensor(a) else np.asarray(a, dtype=np.float32)
        return a.reshape(-1, a.shape[-1])  # (H, 7)

    # Warm up (CUDA-graph capture makes the first calls slow).
    dummy = np.zeros((256, 256, 3), np.uint8)
    for _ in range(3):
        print("warm-up action chunk shape:", predict([dummy, dummy], np.zeros(8), "pick up the plate").shape)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{args.host}:{args.port}")
    print(f"MolmoAct2 server ({args.model}, norm_tag={args.norm_tag}) listening on {args.host}:{args.port}", flush=True)

    while True:
        req = msgpack.unpackb(sock.recv(), object_hook=mnp.decode, raw=False)
        ep, data = req.get("endpoint", "get_action"), req.get("data") or {}
        try:
            if ep == "ping":
                resp = {"status": "ok"}
            elif ep == "reset":
                if data.get("seed") is not None:
                    torch.manual_seed(int(data["seed"]))
                    np.random.seed(int(data["seed"]) % 2**32)
                resp = {"status": "ok"}
            elif ep == "get_action":
                t0 = time.perf_counter()
                actions = predict(data["images"], data["state"], data["task"])
                resp = {"actions": actions.astype(np.float32),
                        "infer_ms": (time.perf_counter() - t0) * 1e3}
            else:
                resp = {"error": f"unknown endpoint {ep!r}"}
        except Exception as e:  # keep serving; client raises on "error"
            resp = {"error": f"{type(e).__name__}: {e}"}
        sock.send(msgpack.packb(resp, default=mnp.encode))


if __name__ == "__main__":
    main()
