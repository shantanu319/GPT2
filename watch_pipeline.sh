#!/bin/bash
# Launches the server-side pipeline (prepare -> pretrain -> sft) DETACHED on
# Modal, then polls the volume for the final SFT checkpoint. Survives laptop
# sleep/disconnects: the run lives entirely on Modal; this script only watches.
#
# Usage: ./watch_pipeline.sh            # launch + watch
#        SKIP_LAUNCH=1 ./watch_pipeline.sh   # just watch an existing run
set -uo pipefail

DIR_NAME="${DIR_NAME:-chat90m}"
SFT_DIR_NAME="${SFT_DIR_NAME:-chat90m_sft}"
VOLUME=myowntransformer-data
LOG=watch_pipeline.log
: > "$LOG"

note() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

if [ -z "${SKIP_LAUNCH:-}" ]; then
  note "launching detached pipeline (dir=$DIR_NAME, sft=$SFT_DIR_NAME)"
  modal run --detach modal_app.py::pipeline \
      --force-prepare \
      --dir-name "$DIR_NAME" --sft-dir-name "$SFT_DIR_NAME" \
      > /dev/null 2>&1 &
  sleep 60
fi

note "watching volume for /saved/$SFT_DIR_NAME/sft_final.pt (poll: 5 min)"
while true; do
  if modal volume ls "$VOLUME" "saved/$SFT_DIR_NAME" 2>/dev/null | grep -q sft_final.pt; then
    note "sft_final.pt found — pulling artifacts"
    break
  fi
  ckpts=$(modal volume ls "$VOLUME" "saved/$DIR_NAME" 2>/dev/null | grep -c 'ckpt_' || true)
  tasks=$(modal app list --json 2>/dev/null | python3 -c \
    "import json,sys; print(sum(a.get('Tasks',0) for a in json.load(sys.stdin) if a.get('State','').startswith('ephemeral')))" \
    2>/dev/null || echo "?")
  note "poll: pretrain_ckpts=$ckpts running_tasks=$tasks"
  if [ "$tasks" = "0" ]; then
    note "WARNING: no running tasks — pipeline likely dead; relaunch with ./watch_pipeline.sh"
  fi
  sleep 300
done

modal volume get --force "$VOLUME" "/saved/$SFT_DIR_NAME" "./modal_out/$SFT_DIR_NAME" 2>&1 | tail -1 | tee -a "$LOG"
modal volume get --force "$VOLUME" "/saved/$DIR_NAME/ckpt_final.pt" "./modal_out/$DIR_NAME/ckpt_final.pt" 2>&1 | tail -1 | tee -a "$LOG"
modal volume get --force "$VOLUME" /data_cache/cosmopedia/tokenizer.json ./modal_out/tokenizer.json 2>&1 | tail -1 | tee -a "$LOG"
note "DONE — weights in ./modal_out/$SFT_DIR_NAME"
