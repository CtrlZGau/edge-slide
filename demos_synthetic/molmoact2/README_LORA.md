# MolmoAct 2 LoRA data: edge-slide (IMBench)

Not in git (see ARTIFACTS.md); rebuild with `run_pipeline.sh`.

## What's here

| Path | Contents |
|---|---|
| `lerobot/imbench/edge_slide_teleop_20hz/` | **LeRobot v3.0 dataset for training.** 150 successful episodes at 20 fps |
| `raw_edge_slide_20hz/` | Stage-1 raw recording (lossless mp4 + npz + per-episode meta). The dataset can be rebuilt from it |
| `run_pipeline.sh`, `pipeline.log` | The command that produced it |

Producer scripts: `data_synthetic/record_molmoact2_edge_slide.py` (stage 1, IMBench venv) and
`data_synthetic/package_molmoact2_lerobot.py` (stage 2, `~/venvs/lerobot` = LeRobot 0.5.0).

## Conventions (identical to `scripts/run_vla_eval.py --policy molmoact2`)

| Key | Shape | Meaning |
|---|---|---|
| `observation.images.image` | 256×256×3 video | `libero_cam` (LIBERO agentview pose, +0.25 m in x), mirrored horizontally, `resize_with_pad` |
| `observation.images.wrist_image` | 256×256×3 video | `gripper_cam` 640×480, mirrored, `resize_with_pad` (letterboxed) |
| `observation.state` | 8 | eef pos + (-0.25, 0, 0) (LIBERO world frame), eef axis-angle, 2 finger qpos |
| `action` | 7 | IMBench cartesian delta at 20 Hz in [-1, 1]; gripper -1 open / +1 close |
| `task` | str | `"pick up the flat plate from the table"` (= run_vla_eval's default prompt from the task DESCRIPTION) |

- Control rate is 20 Hz, the same as the eval (`--control-freq 20`). A +1 action moves the hand about 14 mm and rotates it about 7.4° per step.
- Demonstrator: `EdgeSlideTeleopPolicy` (one push to a size-aware overhang, then a side grasp and lift), with human-like noise 1.0 recorded in the actions. Only successes are kept.
- **Seeds 100000+** are used, so the eval seeds `0..N-1` are held out. Per-episode seed, plate size and style are in `meta/imbench_episodes.jsonl`.
- The 100 Hz datasets in `demos_synthetic/{ib-p1-edge-slide-v1,teleop_hn1}/` use seeds `0, 7, 14, ...`, which **overlap eval seeds**. Don't mix them in without changing the eval seeds.

## Fine-tuning (MolmoAct 2 repo, `experiments/`)

1. Register the mixture in `launch_scripts/data_mixtures.py`:

```python
def build_molmoact2_imbench_edge_slide():
    return build_single_lerobot_mixture(
        name="imbench_edge_slide",
        tag="imbench_edge_slide",
        repo_ids=["imbench/edge_slide_teleop_20hz"],
        action_key="action",
        state_keys=["observation.state"],
        camera_keys=[
            "observation.images.image",
            "observation.images.wrist_image",
        ],
        normalize_gripper=False,
        setup_type="single franka robotic arm on a table in imbench",
        control_mode="delta end-effector pose",
        action_horizon=10,
        n_action_steps=10,
    )


MOLMOACT2_LEROBOT_MIXTURES["imbench_edge_slide"] = build_molmoact2_imbench_edge_slide
```

2. Point the loader at the local data: `export LEROBOT_DATA_ROOT=<path>/demos_synthetic/molmoact2/lerobot`
   (it resolves `${LEROBOT_DATA_ROOT}/imbench/edge_slide_teleop_20hz`).

3. LoRA from the LIBERO checkpoint (closest embodiment and conventions). This is the README recipe with batch settings for 1 GPU.
   Adjust `--nproc-per-node`, the batch sizes and `--max_duration` to the GPU you rent:

```bash
export EXP_NAME="molmoact2-imbench-edge-slide-lora"
torchrun --standalone --nproc-per-node=1 \
  launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-LIBERO \
  imbench_edge_slide \
  --max_duration=10000 \
  --device_batch_size=2 --global_batch_size=32 \
  --num_workers=4 --pin_memory=true --data.timeout=900 \
  --save_interval=2000 --save_folder="checkpoints/finetune/${EXP_NAME}" \
  --packing=false --dynamic_seq_len=true \
  --ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head \
  --lora_enable=true --lora_rank=64 \
  --llm_learning_rate=5e-5 --vit_learning_rate=5e-5 \
  --connector_learning_rate=5e-5 --action_expert_learning_rate=5e-5
```

   Run a smoke test first (`--max_duration=20 --device_batch_size=1 --global_batch_size=1 --num_workers=0`).
   Use `stepX-merged` as the inference checkpoint.

4. Convert the merged checkpoint (`training/molmoact2/imbench_ft/convert.sh`) and evaluate it with
   `scripts/vla_eval/molmoact2_server.py --model <hf dir> --norm-tag imbench_edge_slide --dtype bfloat16` plus
   `scripts/vla_eval/run_main_v2.sh` (see the top-level README).
