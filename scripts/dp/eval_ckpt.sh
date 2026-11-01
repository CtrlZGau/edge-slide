#!/usr/bin/env bash
# STEPS may be "100" (DDPM, as trained) or "ddim:16" (DDIM, 16 steps).
# Evaluate one DP checkpoint on all modes: start one dp_server (LeRobot venv),
# run eval_dp.py clients sequentially against it (one client per server), then stop
# the server by its PID.
#
#   scripts/dp/eval_ckpt.sh <ckpt_dir> <out_dir> <port> <seed_start> <seed_end> <inference_steps> [modes...]
#   e.g. scripts/dp/eval_ckpt.sh outputs/dp/multitask_v1/checkpoints/step0040000 outputs/dp_eval/v1_step40k 5601 0 20 100 cube chain pick push
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LEROBOT_PY=${LEROBOT_PY:-$HOME/venvs/lerobot/bin/python}
SIM_PY=${SIM_PY:-.venv/bin/python}
CKPT=$1; OUT=$2; PORT=$3; S0=$4; S1=$5; STEPS=$6; shift 6
MODES=${*:-cube chain pick push}
mkdir -p "$OUT"
cd "$REPO"
SCHED=DDPM; case "$STEPS" in ddim:*) SCHED=DDIM; STEPS=${STEPS#ddim:};; esac
setsid "$LEROBOT_PY" scripts/dp/dp_server.py --ckpt "$CKPT" --port "$PORT" \
    --inference-steps "$STEPS" --scheduler "$SCHED" > "$OUT/server.log" 2>&1 &
SPID=$!
trap 'kill $SPID 2>/dev/null' EXIT
for _ in $(seq 120); do grep -q "\[dp-server\]" "$OUT/server.log" && break; sleep 2; done
grep -q "\[dp-server\]" "$OUT/server.log" || { echo "server failed to start"; tail -5 "$OUT/server.log"; exit 1; }
for m in $MODES; do
    "$SIM_PY" scripts/dp/eval_dp.py --mode "$m" --port "$PORT" --seeds "$S0" "$S1" \
        --out "$OUT/$m" > "$OUT/$m.log" 2>&1
    python3 - "$OUT/$m/episodes.jsonl" "$m" <<'EOF'
import json, sys
r = [json.loads(l) for l in open(sys.argv[1])] if __import__("os").path.exists(sys.argv[1]) else []
r = [x for x in r if not x.get("excluded")]
ok = sum(x["success"] for x in r)
extra = ""
if sys.argv[2] == "chain" and r:
    extra = f"  (push stage {sum(x['a_success'] for x in r)}/{len(r)})"
print(f"{sys.argv[2]:6s} {ok}/{len(r)} = {100 * ok / max(1, len(r)):.0f}%{extra}", flush=True)
EOF
done
