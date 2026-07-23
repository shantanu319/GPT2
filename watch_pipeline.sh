#!/bin/bash
# Launches the server-side pipeline (prepare -> pretrain -> sft) DETACHED on the
# current vast.ai instance, then polls for the final SFT checkpoint and pulls
# artifacts into ./vast_out. Survives laptop sleep/disconnects: the run lives
# entirely on the instance (nohup); this script only watches.
#
# Usage: ./watch_pipeline.sh                  # launch + watch
#        SKIP_LAUNCH=1 ./watch_pipeline.sh    # just watch an existing run
set -uo pipefail

DIR_NAME="${DIR_NAME:-vast_run}"
SFT_DIR_NAME="${SFT_DIR_NAME:-vast_run_sft}"
REMOTE=/root/myowntransformer
LOG=watch_pipeline.log
: > "$LOG"

note() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

if [ -z "${SKIP_LAUNCH:-}" ]; then
  note "launching detached pipeline (dir=$DIR_NAME, sft=$SFT_DIR_NAME)"
  python3 vast_train.py pipeline --dir-name "$DIR_NAME" --sft-dir-name "$SFT_DIR_NAME" \
      >>"$LOG" 2>&1 || { note "launch failed — is the instance up? (vast_train.py create && push)"; exit 1; }
fi

note "watching for saved/$SFT_DIR_NAME/sft_final.pt (poll: 5 min)"
while true; do
  if python3 vast_train.py ssh "test -f $REMOTE/saved/$SFT_DIR_NAME/sft_final.pt" >/dev/null 2>&1; then
    note "sft_final.pt found — pulling artifacts"
    break
  fi
  tail_line=$(python3 vast_train.py ssh "tail -1 $REMOTE/pipeline.log 2>/dev/null" 2>/dev/null | tail -1)
  note "poll: ${tail_line:-no pipeline.log yet}"
  if ! python3 vast_train.py ssh "pgrep -f remote_pipeline.sh >/dev/null" >/dev/null 2>&1; then
    note "WARNING: pipeline process not running — relaunch with ./watch_pipeline.sh"
  fi
  sleep 300
done

python3 vast_train.py pull 2>&1 | tail -3 | tee -a "$LOG"
note "DONE — weights in ./vast_out/saved/$SFT_DIR_NAME"
