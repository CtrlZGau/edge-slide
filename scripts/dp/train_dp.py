"""
train_dp.py
===========
Multi-task Diffusion Policy (CNN U-Net, Chi et al. 2023) on the IMBench
edge-push / edge-pick / cube-pick cache built by ``build_dp_cache.py``.

The network is LeRobot 0.5's ``DiffusionPolicy`` (a port of the official
implementation). This script supplies what ``lerobot-train`` lacks here:
  * EMA weights (official DP: EMAModel power 0.75, max 0.9999), used for eval;
  * a JPEG cache loader (video decode via pyav is ~160 ms/sample on this box);
  * task conditioning: the 3-way one-hot task id is fed as
    ``observation.environment_state``, which DP concatenates into the global
    FiLM conditioning of every U-Net block.

Normalisation follows LeRobot's DP defaults (state/action MIN_MAX to [-1,1],
images per-channel MEAN_STD); stats are computed from the training set and saved
with every checkpoint.

    ~/venvs/lerobot/bin/python scripts/dp/train_dp.py --cache demos_synthetic/dp/cache128 \\
        --out outputs/dp/multitask_v1 --steps 200000 --wandb
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy

CAMS = ["observation.images.image", "observation.images.wrist_image"]


# ── data ────────────────────────────────────────────────────────────────────
class CacheDataset(Dataset):
    """Samples (obs window of n_obs, action window of horizon) like LeRobot's
    delta_timestamps: obs at t-(n_obs-1)..t, actions at t-(n_obs-1)..t-(n_obs-1)+horizon-1,
    clamped to the episode with an ``action_is_pad`` mask. The last
    ``drop_n_last`` frames of each episode are not used as sample anchors
    (as in the original implementation / LeRobot's drop_n_last_frames)."""

    def __init__(self, cache: Path, n_obs: int, horizon: int, drop_n_last: int, episodes=None):
        self.cache = Path(cache)
        self.meta = json.loads((self.cache / "meta.json").read_text())
        low = np.load(self.cache / "lowdim.npz")
        self.state, self.action, self.env = low["state"], low["action"], low["env_state"]
        self.offs = {c: np.load(self.cache / f"{c}_off.npy") for c in CAMS}
        self._bins = None
        self.n_obs, self.horizon = n_obs, horizon
        eps = self.meta["episodes"]
        if episodes is not None:
            keep = set(episodes)
            eps = [e for e in eps if e["episode_index"] in keep]
        anchors, starts, ends = [], [], []
        for e in eps:
            s, L = e["start"], e["length"]
            n = max(1, L - drop_n_last)
            anchors.append(np.arange(s, s + n))
            starts.append(np.full(n, s)); ends.append(np.full(n, s + L - 1))
        self.anchor = np.concatenate(anchors)
        self.ep_start = np.concatenate(starts)
        self.ep_end = np.concatenate(ends)

    def __len__(self):
        return len(self.anchor)

    def _img(self, cam, i):
        if self._bins is None:   # open lazily in each worker
            self._bins = {c: np.memmap(self.cache / f"{c}.bin", np.uint8, "r") for c in CAMS}
        o = self.offs[cam]
        buf = np.asarray(self._bins[cam][o[i]:o[i + 1]])
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)[:, :, ::-1]

    def __getitem__(self, k):
        t, s, e = self.anchor[k], self.ep_start[k], self.ep_end[k]
        obs_idx = np.clip(np.arange(t - self.n_obs + 1, t + 1), s, e)
        a_raw = np.arange(t - self.n_obs + 1, t - self.n_obs + 1 + self.horizon)
        a_idx = np.clip(a_raw, s, e)
        out = {
            "observation.state": torch.from_numpy(self.state[obs_idx]),
            "observation.environment_state": torch.from_numpy(self.env[obs_idx]),
            "action": torch.from_numpy(self.action[a_idx]),
            "action_is_pad": torch.from_numpy((a_raw < s) | (a_raw > e)),
        }
        for c in CAMS:
            imgs = np.stack([self._img(c, i) for i in obs_idx])           # (n_obs,H,W,3) uint8
            out[c] = torch.from_numpy(np.ascontiguousarray(imgs)).permute(0, 3, 1, 2)  # uint8 (n_obs,3,H,W)
        return out


def compute_stats(ds: CacheDataset, n_img=4000, seed=0):
    st = {
        "observation.state": {"min": ds.state.min(0), "max": ds.state.max(0)},
        "action": {"min": ds.action.min(0), "max": ds.action.max(0)},
    }
    rng = np.random.default_rng(seed)
    idx = rng.choice(ds.meta["n_frames"], size=min(n_img, ds.meta["n_frames"]), replace=False)
    for c in CAMS:
        px = np.stack([ds._img(c, int(i)) for i in idx]).reshape(-1, 3).astype(np.float64) / 255.0
        st[c] = {"mean": px.mean(0), "std": px.std(0)}
    return {k: {kk: np.asarray(vv, np.float32).tolist() for kk, vv in v.items()} for k, v in st.items()}


class Normalizer(torch.nn.Module):
    """LeRobot DP default modes: VISUAL MEAN_STD, STATE/ACTION MIN_MAX -> [-1,1]; env one-hot as is."""

    def __init__(self, stats):
        super().__init__()
        for k, v in stats.items():
            for kk, vv in v.items():
                self.register_buffer(f"{k.replace('.', '_')}__{kk}", torch.tensor(vv, dtype=torch.float32))

    def _g(self, k, kk):
        return getattr(self, f"{k.replace('.', '_')}__{kk}")

    def _mm(self, k, x):
        lo, hi = self._g(k, "min"), self._g(k, "max")
        return 2 * (x - lo) / (hi - lo).clamp_min(1e-8) - 1

    def unnorm_action(self, a):
        lo, hi = self._g("action", "min"), self._g("action", "max")
        return (a + 1) / 2 * (hi - lo).clamp_min(1e-8) + lo

    def forward(self, b):
        b = dict(b)
        b["observation.state"] = self._mm("observation.state", b["observation.state"].float())
        if "action" in b:
            b["action"] = self._mm("action", b["action"].float())
        for c in CAMS:
            x = b[c].float() / 255.0
            m = self._g(c, "mean").view(3, 1, 1); s = self._g(c, "std").view(3, 1, 1)
            b[c] = (x - m) / s
        return b


# ── model ───────────────────────────────────────────────────────────────────
def make_config(a, img_size):
    feats_in = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(8,)),
        "observation.environment_state": PolicyFeature(type=FeatureType.ENV, shape=(3,)),
    }
    for c in CAMS:
        feats_in[c] = PolicyFeature(type=FeatureType.VISUAL, shape=(3, img_size, img_size))
    crop = int(round(img_size * a.crop_ratio))
    return DiffusionConfig(
        input_features=feats_in,
        output_features={"action": PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        n_obs_steps=a.n_obs, horizon=a.horizon, n_action_steps=a.n_action,
        drop_n_last_frames=a.horizon - a.n_action - a.n_obs + 1,
        vision_backbone="resnet18", pretrained_backbone_weights=None, use_group_norm=True,
        spatial_softmax_num_keypoints=32, use_separate_rgb_encoder_per_camera=True,
        crop_shape=(crop, crop), crop_is_random=True,
        down_dims=(512, 1024, 2048), kernel_size=5, n_groups=8, diffusion_step_embed_dim=128,
        use_film_scale_modulation=True,
        noise_scheduler_type=getattr(a, "scheduler", "DDPM"), num_train_timesteps=100, beta_schedule="squaredcos_cap_v2",
        beta_start=1e-4, beta_end=0.02, prediction_type="epsilon", clip_sample=True,
        clip_sample_range=1.0, num_inference_steps=a.inference_steps,
        optimizer_lr=a.lr, optimizer_betas=(0.95, 0.999), optimizer_eps=1e-8,
        optimizer_weight_decay=1e-6, scheduler_name="cosine", scheduler_warmup_steps=a.warmup,
        device="cuda",
    )


class EMA:
    """Official DP EMAModel: decay = 1 - (1 + step/inv_gamma)^-power, clamped to [0, max]."""

    def __init__(self, model, power=0.75, inv_gamma=1.0, max_value=0.9999):
        self.m = copy.deepcopy(model).eval().requires_grad_(False)
        self.power, self.inv_gamma, self.max_value, self.step = power, inv_gamma, max_value, 0

    @torch.no_grad()
    def update(self, model):
        self.step += 1
        d = 1 - (1 + self.step / self.inv_gamma) ** -self.power
        d = min(max(d, 0.0), self.max_value)
        for pe, p in zip(self.m.parameters(), model.parameters()):
            pe.mul_(d).add_(p.detach(), alpha=1 - d)
        for be, b in zip(self.m.buffers(), model.buffers()):
            be.copy_(b)


def cosine_with_warmup(opt, warmup, total):
    def f(s):
        if s < warmup:
            return s / max(1, warmup)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, (s - warmup) / max(1, total - warmup))))
    return torch.optim.lr_scheduler.LambdaLR(opt, f)


def save_ckpt(path: Path, policy, ema, opt, sched, step, stats, cfg_dict, args):
    path.mkdir(parents=True, exist_ok=True)
    torch.save({"model": policy.state_dict(), "ema": ema.m.state_dict(), "ema_step": ema.step,
                "opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, path / "state.pt")
    (path / "stats.json").write_text(json.dumps(stats))
    (path / "config.json").write_text(json.dumps(cfg_dict, indent=2))
    (path / "train_args.json").write_text(json.dumps(vars(args), indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--n-obs", type=int, default=2)
    ap.add_argument("--horizon", type=int, default=16)
    ap.add_argument("--n-action", type=int, default=8)
    ap.add_argument("--crop-ratio", type=float, default=0.9)
    ap.add_argument("--inference-steps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--save-every", type=int, default=20_000)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (off by default: fp32 like the paper)")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--wandb-project", default="imbench-dp")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    ds = CacheDataset(Path(args.cache), args.n_obs, args.horizon,
                      drop_n_last=args.horizon - args.n_action - args.n_obs + 1)
    stats_path = out / "stats.json"
    stats = json.loads(stats_path.read_text()) if stats_path.exists() else compute_stats(ds)
    stats_path.write_text(json.dumps(stats))
    cfg = make_config(args, ds.meta["size"])
    policy = DiffusionPolicy(cfg).cuda().train()
    norm = Normalizer(stats).cuda()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"[dp] {len(ds)} samples from {len(ds.meta['episodes'])} episodes; "
          f"params {n_params / 1e6:.1f} M; steps {args.steps} batch {args.batch} "
          f"(~{args.steps * args.batch / len(ds):.1f} epochs)", flush=True)

    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, betas=(0.95, 0.999), eps=1e-8,
                            weight_decay=1e-6)
    sched = cosine_with_warmup(opt, args.warmup, args.steps)
    ema = EMA(policy)
    step = 0
    last = out / "checkpoints" / "last"
    if args.resume and (last / "state.pt").exists():
        ck = torch.load(last / "state.pt", map_location="cpu")   # CPU: don't keep a 4 GB copy on the GPU
        policy.load_state_dict(ck["model"]); ema.m.load_state_dict(ck["ema"]); ema.step = ck["ema_step"]
        opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); step = ck["step"]
        del ck
        torch.cuda.empty_cache()
        print(f"[dp] resumed at step {step}", flush=True)

    cfg_dict = {k: (v if isinstance(v, (int, float, str, bool, list, tuple, type(None))) else str(v))
                for k, v in vars(cfg).items() if k not in ("input_features", "output_features")}
    cfg_dict["image_size"] = ds.meta["size"]
    cfg_dict["cams"] = CAMS
    cfg_dict["task_names"] = ds.meta["task_names"]

    run = None
    if args.wandb:
        import wandb
        run = wandb.init(entity="gauthamgb85-bits-pilani", project=args.wandb_project,
                         name=out.name, config={**vars(args), "n_params": n_params,
                                                "n_samples": len(ds)},
                         resume="allow", id=(out / "wandb_id").read_text().strip()
                         if (out / "wandb_id").exists() else None)
        (out / "wandb_id").write_text(run.id)

    g = torch.Generator(); g.manual_seed(args.seed + step)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=args.workers,
                    pin_memory=True, drop_last=True, persistent_workers=True, prefetch_factor=4,
                    generator=g)
    t0, tl, losses = time.time(), time.time(), []
    log = open(out / "train_log.jsonl", "a")
    while step < args.steps:
        for batch in dl:
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            batch = norm(batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=args.amp):
                loss, _ = policy.forward(batch)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(policy.parameters(), 10.0)
            opt.step(); sched.step(); ema.update(policy)
            step += 1
            losses.append(loss.item())
            if step % args.log_every == 0:
                dt = time.time() - tl; tl = time.time()
                rec = {"step": step, "loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0],
                       "grad_norm": float(gn), "s_per_step": dt / args.log_every,
                       "ema_decay": 1 - (1 + ema.step) ** -0.75,
                       "eta_h": (args.steps - step) * dt / args.log_every / 3600}
                losses = []
                log.write(json.dumps(rec) + "\n"); log.flush()
                if step % (args.log_every * 10) == 0:
                    print(f"[dp] step {step} loss {rec['loss']:.4f} lr {rec['lr']:.2e} "
                          f"{rec['s_per_step']:.3f}s/step eta {rec['eta_h']:.1f}h", flush=True)
                if run:
                    run.log(rec, step=step)
            if step % args.save_every == 0 or step == args.steps:
                save_ckpt(out / "checkpoints" / f"step{step:07d}", policy, ema, opt, sched, step,
                          stats, cfg_dict, args)
                save_ckpt(last, policy, ema, opt, sched, step, stats, cfg_dict, args)
                print(f"[dp] saved step {step}", flush=True)
            if step >= args.steps:
                break
    print(f"[dp] done {step} steps in {(time.time() - t0) / 3600:.2f} h", flush=True)
    if run:
        run.finish()


if __name__ == "__main__":
    main()
