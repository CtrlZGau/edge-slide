#!/usr/bin/env bash
# Record 150 MolmoAct2-format edge-slide episodes (20 Hz, eval-identical obs), then package as LeRobot v3.0.
#   SIM_PY     : python of the simulation venv (default .venv/bin/python)
#   LEROBOT_PY : python of the LeRobot 0.5.0 venv (default ~/venvs/lerobot/bin/python)
#   The original run prepended PYTHONPATH with an overlay that only contained pyarrow 17.0.0;
#   installing pyarrow==17.0.0 in the sim venv is equivalent.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SIM_PY=${SIM_PY:-.venv/bin/python}
LEROBOT_PY=${LEROBOT_PY:-$HOME/venvs/lerobot/bin/python}
"$SIM_PY" -u data_synthetic/record_molmoact2_edge_slide.py \
    --n-episodes 150 --human-noise 1.0 --out demos_synthetic/molmoact2/raw_edge_slide_20hz
"$LEROBOT_PY" -u data_synthetic/package_molmoact2_lerobot.py \
    --raw demos_synthetic/molmoact2/raw_edge_slide_20hz \
    --root demos_synthetic/molmoact2/lerobot --repo-id imbench/edge_slide_teleop_20hz --overwrite \
    2> demos_synthetic/molmoact2/package_ffmpeg.log
echo PIPELINE_DONE
