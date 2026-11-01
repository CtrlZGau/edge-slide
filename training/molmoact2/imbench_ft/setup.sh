#!/usr/bin/env bash
# One-time setup on the training GPU machine: MolmoAct2 at the pinned commit + our data-mixture patch,
# its venv, the base checkpoint, and the LeRobot dataset in place.
#   WS=/workspace bash training/molmoact2/imbench_ft/setup.sh
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$WS/logs" "$HF_HOME" "$LEROBOT_DATA_ROOT/imbench" "$MOLMO_DATA_DIR"
if [ ! -d "$MOLMO_SRC" ]; then
  git clone https://github.com/allenai/molmoact2 "$MOLMO_SRC"
  git -C "$MOLMO_SRC" checkout 6070080a20321b4f498ab30f28e1d09ac465edb7
  git -C "$MOLMO_SRC" apply "$REPO/training/molmoact2/molmoact2_data_mixtures.patch"   # registers imbench_edge_slide
fi
# dataset: built by demos_synthetic/molmoact2/run_pipeline.sh (see ARTIFACTS.md)
ln -sfn "$REPO/demos_synthetic/molmoact2/lerobot/imbench/edge_slide_teleop_20hz" "$LEROBOT_DATA_ROOT/imbench/edge_slide_teleop_20hz"
# base checkpoint download in parallel with the install
( uvx --from "huggingface_hub[cli]" hf download allenai/MolmoAct2-LIBERO > "$WS/logs/download.log" 2>&1 && echo DOWNLOAD_DONE >> "$WS/logs/download.log" ) &
[ -d "$MOLMO_VENV" ] || uv venv --python 3.12 "$MOLMO_VENV"
source "$MOLMO_VENV/bin/activate"
cd "$MOLMO_SRC/experiments"
uv pip install -e ".[all]"
uv pip install -e "./lerobot[async]"
python -c "import torch, transformers, peft, lerobot; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('transformers', transformers.__version__, 'peft', peft.__version__, 'lerobot', lerobot.__version__)"
wait
echo SETUP_DONE
