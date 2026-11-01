#!/usr/bin/env bash
# LoRA fine-tune of MolmoAct2-LIBERO on IMBench edge-slide (default prompt only, 150 demos @ 20 Hz).
# Exactly the run reported (1x A100 80 GB, ~8 h):   bash training/molmoact2/imbench_ft/train.sh
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
source "$MOLMO_VENV/bin/activate"
cd "$MOLMO_SRC/experiments"
export PYTHONPATH="$PWD:$PWD/lerobot/src:${PYTHONPATH:-}"
WANDB_ARGS=()
[ "$WANDB_MODE" = online ] && WANDB_ARGS=(--wandb.name=molmoact2-imbench-edge-slide-lora-r64 --wandb.entity="$WANDB_ENTITY" --wandb.project="$WANDB_PROJECT")
torchrun --standalone --nproc-per-node=1 launch_scripts/train_lerobot.py \
  allenai/MolmoAct2-LIBERO imbench_edge_slide "${WANDB_ARGS[@]}" \
  --max_duration=3000 --device_batch_size=8 --global_batch_size=32 --log_interval=20 \
  --num_workers=8 --pin_memory=true --data.timeout=900 \
  --save_interval=1000 --save_num_checkpoints_to_keep=1 \
  --save_folder="$CKPT_DIR" \
  --packing=false --dynamic_seq_len=true \
  --ft_vlm=true --ft_action_expert=true --ft_embedding=lm_head \
  --lora_enable=true --lora_rank=64 \
  --llm_learning_rate=5e-5 --vit_learning_rate=5e-5 \
  --connector_learning_rate=5e-5 --action_expert_learning_rate=5e-5 2>&1 | tee "$WS/logs/train.log"
echo "TRAIN_EXIT=${PIPESTATUS[0]}"
