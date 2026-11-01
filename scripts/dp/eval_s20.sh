#!/usr/bin/env bash
# 20-seed (0-19) DDPM-100 eval of one checkpoint: 6 server/client pairs in parallel (7 runs out of memory on a 12 GB GPU).
#   bash scripts/dp/eval_s20.sh outputs/dp/multitask_v1/checkpoints/step0080000 outputs/dp_eval/v1_step80k_ddpm100_s20
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SIM_PY=${SIM_PY:-.venv/bin/python}
CKPT=$1; OUT=$2; STEPS=${3:-100}
mkdir -p "$OUT"
run() { scripts/dp/eval_ckpt.sh "$CKPT" "$OUT/$1" "$2" "$3" "$4" "$STEPS" "$5" > "$OUT/$1.summary" 2>&1; }
run chain_a 5621 0 7 chain &
run chain_b 5622 7 14 chain &
run chain_c 5623 14 20 chain &
run cube 5624 0 20 cube &
run pick_a 5625 0 10 pick &
run pick_b 5626 10 20 pick &
wait
"$SIM_PY" scripts/dp/summarize_eval.py "$OUT" | tee "$OUT/FINAL_SUMMARY.txt"
