#!/usr/bin/env bash
# Run the full main_v2_libero_cam suite for GR00T then MolmoAct2 on the VM, one model on the GPU at a time.
# Launch inside tmux so it survives SSH disconnects:
#   tmux new -d -s eval 'HF_TOKEN=<token> bash scripts/vla_eval/vm_run_all.sh 2>&1 | tee eval_all.log'
# Re-running resumes (finished episodes are skipped). MODELS="molmoact2" to run just one.
# Results: outputs/vla_eval/main_v2_libero_cam/  (local disk only; nothing is uploaded)
set -uo pipefail
# GR00T loads the gated nvidia/Cosmos-Reason2-2B processor at server start -> needs the token.
: "${HF_TOKEN:?export HF_TOKEN (GR00T server needs gated nvidia/Cosmos-Reason2-2B)}"
export HF_TOKEN HF_HUB_ENABLE_HF_TRANSFER=0

IMB=${IMB:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
GR00T=${GR00T:-$HOME/Isaac-GR00T}
MOLMO_VENV=${MOLMO_VENV:-$HOME/venvs/molmoact2}
MODELS=${MODELS:-"groot molmoact2"}
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH" MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
mkdir -p "$IMB/outputs/vla_eval/main_v2_libero_cam"

wait_ping() {  # port, pid, timeout_s
  local port=$1 pid=$2 t=${3:-1800}
  for ((i = 0; i < t; i += 10)); do
    kill -0 "$pid" 2>/dev/null || { echo "server process exited"; return 1; }
    "$IMB/.venv/bin/python" - "$port" <<'EOF' && return 0
import sys
from imbench.policy.vla_clients import ZmqMsgpackClient
c = ZmqMsgpackClient("localhost", int(sys.argv[1]), timeout_ms=5000)
sys.exit(0 if c.ping() else 1)
EOF
    sleep 10
  done
  echo "server on :$port did not come up in ${t}s"; return 1
}

start_server() {
  case $1 in
    groot)
      (cd "$GR00T" && exec uv run python gr00t/eval/run_gr00t_server.py \
        --model-path checkpoints/GR00T-N1.7-LIBERO/libero_10 --embodiment-tag LIBERO_PANDA \
        --use-sim-policy-wrapper --port 5555) > "$HOME/server_groot.log" 2>&1 &
      SERVER_PID=$!; wait_ping 5555 "$SERVER_PID" ;;
    molmoact2)
      "$MOLMO_VENV/bin/python" "$IMB/scripts/vla_eval/molmoact2_server.py" --port 5556 --dtype "${MOLMO_DTYPE:-float32}" \
        > "$HOME/server_molmoact2.log" 2>&1 &
      SERVER_PID=$!; wait_ping 5556 "$SERVER_PID" ;;
  esac
}

for m in $MODELS; do
  echo "##### $m: starting server $(date '+%F %T')"
  if start_server "$m"; then
    bash "$IMB/scripts/vla_eval/run_main_v2.sh" "$m"
  else
    echo "##### $m: server failed — see ~/server_$m.log"; tail -30 "$HOME/server_$m.log"
  fi
  kill "$SERVER_PID" 2>/dev/null; sleep 5; pkill -P "$SERVER_PID" 2>/dev/null || true
  nvidia-smi --query-gpu=memory.used --format=csv
done
echo "##### ALL DONE $(date '+%F %T')"
