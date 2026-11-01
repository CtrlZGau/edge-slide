# Sourced by setup.sh / train.sh / convert.sh. Paths for the MolmoAct2 LoRA fine-tune on IMBench edge-slide.
# WS is the work directory (a 200 GB network volume on the pod used here). Override any variable before sourcing.
export WS=${WS:-/workspace}
export REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}   # this repository
export HF_HOME=${HF_HOME:-$WS/hf}
export HF_HUB_ENABLE_HF_TRANSFER=0
export LEROBOT_DATA_ROOT=$WS/data/lerobot          # contains imbench/edge_slide_teleop_20hz
export LEROBOT_VIDEO_BACKEND=pyav
export MOLMO_DATA_DIR=$WS/molmo_data
export MOLMO_SRC=$WS/molmoact2                      # allenai/molmoact2 clone
export MOLMO_VENV=${MOLMO_VENV:-$WS/venvs/molmoact2}
export CKPT_DIR=$WS/checkpoints/finetune/molmoact2-imbench-edge-slide-lora-r64
# W&B is optional: set WANDB_ENTITY (and WANDB_PROJECT) to log, otherwise logging is disabled.
export WANDB_PROJECT=${WANDB_PROJECT:-imbench-molmoact2}
[ -n "${WANDB_ENTITY:-}" ] && export WANDB_MODE=online || export WANDB_MODE=disabled
