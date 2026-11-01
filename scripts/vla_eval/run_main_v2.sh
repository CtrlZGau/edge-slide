#!/usr/bin/env bash
# The edge-slide VLA suite: default prompt, strategy prompt, push-only (30 s), push-only (60 s).
# Needs the model's policy server running (see README). Resumable: finished episodes are skipped.
#
#   bash scripts/vla_eval/run_main_v2.sh pi05|groot|molmoact2
#   env overrides:  OUT (results dir)  N (episodes per prompt, 20)  PUSH_N (6)  PUSH60_N (5)
#                   PREFIX (folder prefix, default = model)  PORT  EXTRA (extra run_vla_eval.py args)
set -euo pipefail

MODEL=${1:?usage: run_main_v2.sh pi05|groot|molmoact2}
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=${OUT:-$ROOT/outputs/vla_eval/main_v2_libero_cam}
PY=${PY:-$ROOT/.venv/bin/python}
N=${N:-20}
PUSH_N=${PUSH_N:-6}
PUSH60_N=${PUSH60_N:-5}
PREFIX=${PREFIX:-$MODEL}
EXTRA=${EXTRA:-}
[ -n "${PORT:-}" ] && EXTRA="$EXTRA --port $PORT"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

STRATEGY="slide the flat plate to the edge of the table, then grasp it and lift it up"
PUSH="push the flat plate to the edge of the table with the gripper"

run() {  # name, extra args...
  local name=$1; shift
  echo "=== $PREFIX / $name  $(date '+%F %T') ==="
  # shellcheck disable=SC2086
  "$PY" "$ROOT/scripts/run_vla_eval.py" --policy "$MODEL" --out "$OUT/${PREFIX}_${name}" "$@" $EXTRA \
    2>&1 | grep --line-buffered -vE "robosuite (WARNING|INFO)" | tee -a "$OUT/${PREFIX}_${name}.log"
}

mkdir -p "$OUT"
#    name             pi0.5 episodes   args
run default_prompt    --n-episodes "$N"                                 # 'pick up the flat plate from the table'
run strategy_prompt   --n-episodes "$N" --prompt "$STRATEGY"
run push_only         --n-episodes "$PUSH_N"   --prompt "$PUSH" --push-metric
run push_only_60s     --n-episodes "$PUSH60_N" --prompt "$PUSH" --push-metric --max-steps 1200
echo "=== $PREFIX suite done $(date '+%F %T') ==="
