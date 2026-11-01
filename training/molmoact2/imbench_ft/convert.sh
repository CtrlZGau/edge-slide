#!/usr/bin/env bash
# Convert a merged training checkpoint into a Hugging Face model dir that molmoact2_server.py can load (CPU only).
#   bash training/molmoact2/imbench_ft/convert.sh 3000            -> $WS/eval/hf/step3000
# (The released step-3000 model on HF is already converted; see ARTIFACTS.md.)
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
S=${1:?usage: convert.sh STEP}
SRC=$CKPT_DIR/step${S}-merged
DST=${2:-$WS/eval/hf/step${S}}
SNAP=$(ls -d "$HF_HOME"/hub/models--allenai--MolmoAct2-LIBERO/snapshots/*/ | head -1)
[ -f "$DST/DONE" ] && { echo "step$S already converted"; exit 0; }
rm -rf "$DST.tmp"
source "$MOLMO_VENV/bin/activate"
cd "$MOLMO_SRC/experiments"
export PYTHONPATH="$PWD:$PWD/lerobot/src:${PYTHONPATH:-}" CUDA_VISIBLE_DEVICES="" WANDB_MODE=disabled
python -m olmo.hf_model.convert_molmoact2_to_hf "$SRC" "$DST.tmp"
# processor / tokenizer / remote code from the base HF repo (only files the converter did not write)
for f in "$SNAP"*; do b=$(basename "$f"); case $b in *.safetensors|model.safetensors.index.json|config.json|norm_stats.json|README.md|.gitattributes) continue;; esac; [ -e "$DST.tmp/$b" ] || cp -rL "$f" "$DST.tmp/"; done
cp "$SRC/config.yaml" "$DST.tmp/olmo_config.yaml"
mv "$DST.tmp" "$DST" && touch "$DST/DONE" && echo "converted step$S -> $DST"
