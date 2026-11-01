#!/usr/bin/env bash
# One-time VM setup for the GR00T N1.7 + MolmoAct2 IMBench eval. Idempotent; safe to re-run.
# Run on the GPU machine from the repo root:   HF_TOKEN=<token> bash scripts/vla_eval/vm_setup.sh
# Installs: system libs, uv, Simulation venv, Isaac-GR00T (+ LIBERO ckpt), MolmoAct2 venv (+ ckpt).
# Nothing here uploads data anywhere; only downloads code/checkpoints.
set -euo pipefail

: "${HF_TOKEN:?export HF_TOKEN (needs access to gated nvidia/Cosmos-Reason2-2B)}"
IMB=${IMB:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GR00T=${GR00T:-$HOME/Isaac-GR00T}
MOLMO_VENV=${MOLMO_VENV:-$HOME/venvs/molmoact2}
log() { echo -e "\n\033[1;34m### $*\033[0m"; }

log "GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

log "System packages"
SUDO=$([ "$(id -u)" = 0 ] && echo "" || echo sudo)
$SUDO apt-get update -qq
$SUDO env DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
  git git-lfs tmux rsync ffmpeg libegl1 libegl1-mesa-dev libgl1 libglu1-mesa libosmesa6 python3-venv build-essential >/dev/null
ffmpeg -version | head -1

log "uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

log "IMBench venv"
cd "$IMB"
[ -d .venv ] || uv venv -p 3.12 .venv
uv pip install --python .venv/bin/python -q -r requirements-sim.txt
uv pip install --python .venv/bin/python -q -e . --no-deps
MUJOCO_GL=egl .venv/bin/python - <<'EOF'
import os, numpy as np, imbench as ib
e = ib.make("ib-p1-edge-slide-v1", obs_mode="combo")
o, _ = e.reset(seed=0)
assert o["rgb_ego_cam"].mean() > 5, "EGL render is black"
print("IMBench EGL render OK", o["rgb_ego_cam"].shape)
EOF

log "Isaac-GR00T"
[ -d "$GR00T" ] || git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T "$GR00T"
cd "$GR00T"
git lfs install --local && git lfs pull   # wheels in scripts/deployment are LFS objects
git log -1 --format="GR00T commit %h %cd"
uv sync --python 3.12 && uv cache clean   # cache is ~16 GB; pod disks are small
export HF_TOKEN
uv run python -c "import gr00t; print('gr00t import OK')"
uv run hf download nvidia/GR00T-N1.7-LIBERO --include "libero_10/*" --exclude "libero_10/global_step*" --local-dir checkpoints/GR00T-N1.7-LIBERO
uv run hf download nvidia/Cosmos-Reason2-2B >/dev/null && echo "Cosmos-Reason2-2B access OK"
ls checkpoints/GR00T-N1.7-LIBERO/libero_10

if [ "${SKIP_MOLMO:-0}" = 1 ]; then log "Skipping MolmoAct2 (SKIP_MOLMO=1)"; log "Setup complete"; exit 0; fi
log "MolmoAct2 venv"
[ -d "$MOLMO_VENV" ] || uv venv -p 3.12 "$MOLMO_VENV"
uv pip install --python "$MOLMO_VENV/bin/python" -q torch transformers accelerate pillow numpy huggingface_hub pyzmq msgpack msgpack-numpy
"$MOLMO_VENV/bin/python" -c "from huggingface_hub import snapshot_download as d; print(d('allenai/MolmoAct2-LIBERO'))"

log "Setup complete"
