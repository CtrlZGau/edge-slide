# IMBench edge-slide: environment, training, evaluation and results

This is a self-contained repo for task **T01 edge-slide** (`ib-p1-edge-slide-v1`) from IMBench.

**The task:** a flat plate (3 cm thick, 8–15 cm wide) lies on a table. It is too wide for the Panda's 8 cm gripper, so the robot has to slide it until it overhangs a table edge, pinch the exposed lip from the side, and lift it.

| Part | What | Result |
|---|---|---|
| 1 | Zero-shot VLAs: π0.5, GR00T N1.7, MolmoAct2-LIBERO | 0/20, 0/20, 1/20 (default prompt); 0 on every other setting |
| 2 | MolmoAct2 LoRA fine-tune (150 synthetic demos) | 6/10 pick-up, 5/10 push |
| 3 | GPT-6 Astra: Stage 1/2 (constraints, plan) and Stage 3 (primitives / Code-as-Policies) | Stage 1/2: all required rubric items covered; Stage 3: 1/6 intended-strategy successes (code mode with object poses) |
| 4 | Multi-task Diffusion Policy (push / pick / cube), step 80k, seeds 0–19 | push→pick chained 16/20, pick 20/20, push 17/20, cube 12/20 |

## Checkpoints

Both repos are **private** Hugging Face repos under `Ctrlgb`, so log in first with `hf auth login`.

| Model | Link | Download |
|---|---|---|
| Diffusion Policy, multi-task, step 80k (EMA weights, 1.1 GB): the Part 4 results | [Ctrlgb/dp-imbench-edge-slide-multitask-step80k](https://huggingface.co/Ctrlgb/dp-imbench-edge-slide-multitask-step80k) | `hf download Ctrlgb/dp-imbench-edge-slide-multitask-step80k --local-dir outputs/dp/multitask_v1/checkpoints/step0080000` |
| MolmoAct2 LoRA fine-tune, step 3000: the Part 2 results | [Ctrlgb/molmoact2-imbench-edge-slide-step3000](https://huggingface.co/Ctrlgb/molmoact2-imbench-edge-slide-step3000) | `hf download Ctrlgb/molmoact2-imbench-edge-slide-step3000 --exclude "training_state/*" --local-dir checkpoints/molmoact2_step3000` |

The datasets and the other DP checkpoints are not in git. `ARTIFACTS.md` says where they are and how to rebuild them.

---

## Layout

```
imbench/                 simulator package (trimmed from IMBench)
  registry.py            ib-p1-edge-slide-v1, ib-p1-cube-pick-v1, ib-m-mirror-pick-place-v1 (VLA sanity check)
  tasks/, envs/, utils/  task base class, robosuite envs, tactile sensors
  policy/                VLA clients (GR00T / MolmoAct2 over ZMQ, π0.5 over openpi websocket)
  vlm_eval/              GPT-6 Astra evaluation (own README.md and TRANSFER.md)
configs/tasks/           task YAMLs
data_synthetic/          scripted demonstrators, recorders, LeRobot v3.0 packagers
scripts/
  run_vla_eval.py        VLA eval client (all three VLAs)
  run_pi05_eval.py       π0.5 sanity-check runs + shared helpers (plate_overhang, video frames)
  edge_slide_scripted_primitives.py   scripted solver built from the 8 Stage-3 primitives
  vla_eval/              MolmoAct2 policy server, suite runner, GPU-machine setup/run
  dp/                    DP cache builder, trainer, policy server, eval client, eval launchers
training/molmoact2/      MolmoAct2 fine-tune: setup/train/convert scripts, upstream patch, configs, train log, W&B files
tests/                   117 tests (env, Astra stages, primitives, Stage-3 agent, code sandbox)
outputs/                 VLA / MolmoAct2 / DP results (every episode: results.json, trajectories, videos)
runs/                    GPT-6 Astra results
demos_synthetic/         dataset metadata (the data itself: ARTIFACTS.md)
manifests/               sha256 of the large artifacts
```

---

## Setup

Three Python environments are needed, because their numpy/torch versions conflict. They talk over ZMQ/msgpack, with one eval client per policy server.

```bash
# 1. Simulation venv: env, data recording, eval clients, GPT-6 Astra eval, tests
uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python -r requirements-sim.txt
uv pip install --python .venv/bin/python -e . --no-deps
export MUJOCO_GL=egl                      # headless rendering (Ubuntu: apt install libegl1 libgl1 libglvnd0)

# 2. LeRobot venv: dataset packaging, DP training and DP policy server
uv venv -p 3.12 ~/venvs/lerobot
uv pip install --python ~/venvs/lerobot/bin/python torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python ~/venvs/lerobot/bin/python -r requirements-lerobot.txt

# 3. VLA servers (GR00T N1.7 via Isaac-GR00T, MolmoAct2 venv): one script, run on a ≥24 GB GPU
HF_TOKEN=<token with access to nvidia/Cosmos-Reason2-2B> bash scripts/vla_eval/vm_setup.sh
```

To check the install:

```bash
.venv/bin/python -m pytest -o addopts="" tests/          # 117 passed, ~3.5 min
```

Scripts default to `.venv/bin/python` and `~/venvs/lerobot/bin/python`. Override them with `SIM_PY=…` and `LEROBOT_PY=…`.

---

## Reproduce

All commands run from the repo root. Every eval resumes where it left off (finished episodes are skipped), and writes `results.json`, per-episode trajectories and videos to the path shown.

**Eval protocol, shared by parts 1, 2 and 4:**
- 20 Hz control; 600 steps (1200 for "60 s").
- Seeds `0..N-1`, held out from all training data.
- `libero_cam` + wrist camera, mirrored, 256 px; 8-D state; 7-D delta actions.

### 1. Zero-shot VLAs: `outputs/vla_eval/main_v2_libero_cam/`, `outputs/pi05_eval/main_v2_libero_cam/`

`run_main_v2.sh` runs the full suite: 20 default-prompt + 20 strategy-prompt + 6 push-only + 5 push-only-60 s episodes.

```bash
# π0.5: start the openpi LIBERO server (github.com/Physical-Intelligence/openpi), then run the suite
(cd ~/openpi && uv run scripts/serve_policy.py --env LIBERO) &              # port 8000
OUT=outputs/pi05_eval/main_v2_libero_cam bash scripts/vla_eval/run_main_v2.sh pi05

# GR00T N1.7 and MolmoAct2: starts each server in turn, runs the suite, stops it (24 GB GPU -> bf16 MolmoAct2)
HF_TOKEN=<token> MOLMO_DTYPE=bfloat16 bash scripts/vla_eval/vm_run_all.sh

# GR00T rerun on a fresh machine (outputs/vla_eval/main_v2_libero_cam_rerun_20261101/)
HF_TOKEN=<token> MODELS=groot OUT=outputs/vla_eval/main_v2_libero_cam_rerun_20261101 bash scripts/vla_eval/vm_run_all.sh
```

**Sanity checks on a cube pick-up** (they confirm the observation/action pipeline):

```bash
# π0.5 on ib-m-mirror-pick-place-v1 -> 3/10   (outputs/pi05_eval/ib-m-mirror-pick-place-v1/pi05_upright)
.venv/bin/python scripts/run_pi05_eval.py --task ib-m-mirror-pick-place-v1 --undo-task-flip --wrist-res 0 \
    --prompt "pick up the cube and lift it" --max-steps 400 --n-episodes 10 \
    --out outputs/pi05_eval/ib-m-mirror-pick-place-v1/pi05_upright
#   same without --undo-task-flip (camera images left rotated 180°) -> 0/10   (.../pi05_task_flipped)

# GR00T N1.7 on ib-p1-cube-pick-v1, GR00T server on :5555 -> 0/10 padded wrist cam, 0/10 native 256 px
.venv/bin/python scripts/run_vla_eval.py --policy groot --task ib-p1-cube-pick-v1 --n-episodes 10 \
    --out outputs/vla_eval/sanity/groot_cube_pick_padded10
.venv/bin/python scripts/run_vla_eval.py --policy groot --task ib-p1-cube-pick-v1 --n-episodes 10 --wrist-native 256 \
    --out outputs/vla_eval/sanity/groot_cube_pick_wrist256
```

### 2. MolmoAct2 LoRA fine-tune: `outputs/molmoact2_ft/eval_results/results/step3000/`

```bash
# data: 150 successful scripted demos (seeds 100000+), recorded and packaged as LeRobot v3.0
bash demos_synthetic/molmoact2/run_pipeline.sh

# training (1x A100 80 GB, ~8 h). WS = work dir; set WANDB_ENTITY to log to W&B
WS=/workspace bash training/molmoact2/imbench_ft/setup.sh      # clones MolmoAct2 @6070080 + applies our patch
WS=/workspace bash training/molmoact2/imbench_ft/train.sh
WS=/workspace bash training/molmoact2/imbench_ft/convert.sh 3000   # -> $WS/eval/hf/step3000

# or skip training and use the released model (see Checkpoints):
#   hf download Ctrlgb/molmoact2-imbench-edge-slide-step3000 --exclude "training_state/*" --local-dir <dir>

# eval: 10 episodes per setting -> 6/10 pick, 3/10 strategy, 5/10 push, 5/10 push 60 s
~/venvs/molmoact2/bin/python scripts/vla_eval/molmoact2_server.py --model <dir> --norm-tag imbench_edge_slide --dtype bfloat16 &
OUT=outputs/molmoact2_ft/eval_results/results/step3000 PREFIX=molmoact2ft N=10 PUSH_N=10 PUSH60_N=10 \
    bash scripts/vla_eval/run_main_v2.sh molmoact2
```

### 3. GPT-6 Astra: `runs/gpt-6-astra/`

The key comes from `OPENAI_API_KEY`, or from `~/openai_astra_key.txt` (`chmod 600`). Each Stage-3 episode costs about $2–3. Full protocol: `imbench/vlm_eval/README.md`.

```bash
# Stage 1 + 2 (effort max), seed 0
.venv/bin/python -m imbench.vlm_eval.run_edge_slide --seeds 0 --samples 1 --budget-usd 10

# Stage 3, 8 primitives (effort low), plan from seed 0's Stage 2; add --obj for object poses   (P1–P6)
.venv/bin/python -m imbench.vlm_eval.run_stage3 --seeds 0 1 2 --plan-source stage2 --stage2-seed 0 --budget-usd 15 [--obj]

# Stage 3, Code-as-Policies; C1 = with --obj, seed 0 -> success
.venv/bin/python -m imbench.vlm_eval.run_stage3 --mode code --seeds 0 --plan-source stage2 --stage2-seed 0 --obj --budget-usd 8

# checks that need no API key
.venv/bin/python -m imbench.vlm_eval.stage3.primitive_report --seeds 0 1 2                     # 138/138 primitive checks
.venv/bin/python scripts/edge_slide_scripted_primitives.py --seeds 0 1 2 3 4 5 6 7 8 9        # scripted solver 10/10
```

### 4. Multi-task Diffusion Policy: `outputs/dp/multitask_v1/`, `outputs/dp_eval/`

```bash
# D1. record demos (sim venv). Edge-slide: 12 workers x 100 seeds; cube: 8 workers x 140 seeds
for i in $(seq 0 11); do s=$((200000 + 100*i)); .venv/bin/python data_synthetic/record_segmented_demos.py --policy edge_slide \
    --seed-start $s --seed-end $((s+100)) --out demos_synthetic/dp/raw_edge_slide & done; wait
for i in $(seq 0 7); do s=$((300000 + 140*i)); .venv/bin/python data_synthetic/record_segmented_demos.py --policy cube_pick \
    --seed-start $s --seed-end $((s+140)) --out demos_synthetic/dp/raw_cube & done; wait

# D2. package into one LeRobot v3.0 dataset: push / pick / cube segments, 3453 episodes (20 shards, then merge)
pack() { ~/venvs/lerobot/bin/python data_synthetic/package_dp_lerobot.py --raw-edge demos_synthetic/dp/raw_edge_slide \
           --raw-cube demos_synthetic/dp/raw_cube --root demos_synthetic/dp/lerobot --num-shards 20 "$@"; }
for i in $(seq 0 19); do pack --shard $i & done; wait; pack --merge

# D3. decode once into a 128 px JPEG training cache
~/venvs/lerobot/bin/python scripts/dp/build_dp_cache.py \
    --dataset demos_synthetic/dp/lerobot/imbench/dp_multitask_20hz --out demos_synthetic/dp/cache128

# D4. train (12 GB GPU; 200k-step schedule; the reported checkpoint is step0080000; --resume continues from checkpoints/last)
~/venvs/lerobot/bin/python scripts/dp/train_dp.py --cache demos_synthetic/dp/cache128 --out outputs/dp/multitask_v1 --steps 200000

#     or skip D1-D4 and use the released checkpoint (see Checkpoints):
#     hf download Ctrlgb/dp-imbench-edge-slide-multitask-step80k --local-dir outputs/dp/multitask_v1/checkpoints/step0080000

# D5. final eval, step 80k, DDPM-100, seeds 0-19 -> outputs/dp_eval/v1_step80k_ddpm100_s20/FINAL_SUMMARY.txt
bash scripts/dp/eval_s20.sh outputs/dp/multitask_v1/checkpoints/step0080000 outputs/dp_eval/v1_step80k_ddpm100_s20

#     quick check of any checkpoint (DDIM-16, seeds 0-9, cube + chain), as in outputs/dp_eval/v1_step*k_ddim16
bash scripts/dp/eval_ckpt.sh outputs/dp/multitask_v1/checkpoints/step0080000 outputs/dp_eval/v1_step80k_ddim16 5601 0 10 ddim:16 cube chain
```

The DP hyperparameters are listed in `outputs/dp/multitask_v1/checkpoints/step0080000/train_args.json` and `config.json`.

---

## Provenance

The code comes from IMBench `main` @ `fb807d5`, plus the working-tree edits these experiments ran against. What changed relative to IMBench:
- **`imbench/` is trimmed to the tasks above.** The other tasks, the suction gripper and edge-slide v2 were removed; none of them is used here.
- **Scripts are repo-relative** instead of hard-coding `/home/gb/IMBench` or `/workspace`:
  - the MolmoAct2 training scripts take `WS`
  - `vm_setup.sh` installs from `requirements-sim.txt`
  - `run_main_v2.sh` gained the `N`, `PREFIX`, `PORT` and `EXTRA` overrides used for the fine-tuned eval
- **Removed one-off helpers** (no effect on any result):
  - laptop↔VM sync scripts
  - the mock server
  - W&B mirroring
  - checkpoint pruning
  - training-time auto-eval loops
  - pod orchestration scripts
- **Removed superseded or partial results:**
  - older π0.5 protocol runs
  - an aborted DP eval
  - a partial DP eval that ran out of GPU memory
  - one-episode "extended prompt" side runs
- Paths inside `outputs/**/run_config.json` are the original machines' paths, kept as a record.

## Secrets

No credentials are in this repo; it was scanned for HF, AWS, W&B, OpenAI and SSH keys. The only matches were OpenAI `encrypted_content` reasoning blobs in `runs/**/*_raw.json`, which are opaque model output.

`training/molmoact2/wandb/` records the run's W&B entity and the account email.
