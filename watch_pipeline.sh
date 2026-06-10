#!/bin/bash
# Drives the full Modal pipeline (prepare -> pretrain -> SFT) and logs only
# milestones with timestamps. When everything finishes it pulls the SFT
# weights + tokenizer back to ./modal_out so they can be tested locally.
#
# Usage: ./watch_pipeline.sh [extra modal-run args...]
set -uo pipefail

DIR_NAME="${DIR_NAME:-chat90m}"
SFT_DIR_NAME="${SFT_DIR_NAME:-chat90m_sft}"
LOG=watch_pipeline.log
: > "$LOG"

stamp() { while IFS= read -r line; do echo "$(date +%H:%M:%S) $line"; done; }

echo "$(date +%H:%M:%S) launching pipeline (dir=$DIR_NAME, sft=$SFT_DIR_NAME)" | tee -a "$LOG"

modal run modal_app.py \
    --force-prepare --max-train-docs 600000 \
    --batchsize 64 --grad-accum 2 \
    --save-every 1000 --val-every 1000 --warmup-steps 300 \
    --dir-name "$DIR_NAME" \
    --run-sft --sft-epochs 1 --sft-dir-name "$SFT_DIR_NAME" \
    "$@" 2>&1 |
  grep --line-buffered -E \
    'tokenized [0-9]+0000 docs|wrote .* tokens|Reusing|Saved tokenizer|exhausted|prepared|total params|step [0-9]+ \| Loss|Validation Loss|Saved checkpoint|finished|Test Loss|SFT val|saved .*sft|conversations|committed|Artifacts|Error|error|Traceback|preemption|interrupted' |
  stamp | tee -a "$LOG"

status=$?
echo "$(date +%H:%M:%S) modal run exited with status $status" | tee -a "$LOG"

if [ $status -eq 0 ]; then
  echo "$(date +%H:%M:%S) pulling artifacts to ./modal_out ..." | tee -a "$LOG"
  modal volume get --force myowntransformer-data "/saved/$SFT_DIR_NAME" "./modal_out/$SFT_DIR_NAME" 2>&1 | tail -1 | stamp | tee -a "$LOG"
  modal volume get --force myowntransformer-data "/saved/$DIR_NAME/ckpt_final.pt" "./modal_out/$DIR_NAME/ckpt_final.pt" 2>&1 | tail -1 | stamp | tee -a "$LOG"
  modal volume get --force myowntransformer-data /data_cache/cosmopedia/tokenizer.json ./modal_out/tokenizer.json 2>&1 | tail -1 | stamp | tee -a "$LOG"
  echo "$(date +%H:%M:%S) DONE — weights in ./modal_out/$SFT_DIR_NAME" | tee -a "$LOG"
else
  echo "$(date +%H:%M:%S) PIPELINE FAILED — check modal app logs" | tee -a "$LOG"
fi
