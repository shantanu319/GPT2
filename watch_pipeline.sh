#!/bin/bash
# Launches the server-side pipeline (prepare -> pretrain -> sft -> dpo) DETACHED
# on the current vast.ai instance, then polls for the final DPO checkpoint and
# pulls artifacts into ./vast_out. Survives laptop sleep/disconnects: the run
# lives entirely on the instance (nohup); this script only watches.
#
# Usage: ./watch_pipeline.sh                  # launch + watch
#        SKIP_LAUNCH=1 ./watch_pipeline.sh    # just watch an existing run
#
# Time-boxing: if the pipeline hasn't finished within MAX_HOURS (default 7),
# the watcher stops waiting, pulls whatever checkpoints exist (periodic
# ckpt_step*/ckpt_best/sft/dpo finals — pull grabs all of saved/), and exits.
# Set DESTROY_ON_TIMEOUT=1 to also stop the meter automatically.
set -uo pipefail

DIR_NAME="${DIR_NAME:-vast_run}"
SFT_DIR_NAME="${SFT_DIR_NAME:-vast_run_sft}"
DPO_DIR_NAME="${DPO_DIR_NAME:-vast_run_dpo}"
MAX_HOURS="${MAX_HOURS:-7}"
DESTROY_ON_TIMEOUT="${DESTROY_ON_TIMEOUT:-0}"
REMOTE=/root/myowntransformer
LOG=watch_pipeline.log
: > "$LOG"

note() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

if [ -z "${SKIP_LAUNCH:-}" ]; then
  note "launching detached pipeline (dir=$DIR_NAME, sft=$SFT_DIR_NAME, dpo=$DPO_DIR_NAME)"
  python3 vast_train.py pipeline --dir-name "$DIR_NAME" --sft-dir-name "$SFT_DIR_NAME" \
      --dpo-dir-name "$DPO_DIR_NAME" >>"$LOG" 2>&1 || { note "launch failed — is the instance up? (vast_train.py create && push)"; exit 1; }
fi

note "watching for saved/$DPO_DIR_NAME/dpo_final.pt (poll: 5 min, deadline: ${MAX_HOURS}h)"
SECONDS=0
while true; do
  if python3 vast_train.py ssh "test -f $REMOTE/saved/$DPO_DIR_NAME/dpo_final.pt" >/dev/null 2>&1; then
    note "dpo_final.pt found — pulling artifacts"
    break
  fi
  if (( SECONDS >= MAX_HOURS * 3600 )); then
    note "TIMEOUT: ${MAX_HOURS}h reached without dpo_final.pt — pulling latest checkpoints instead"
    python3 vast_train.py ssh "ls -t $REMOTE/saved/*/*.pt 2>/dev/null | head -5; tail -3 $REMOTE/pipeline.log 2>/dev/null" \
      2>/dev/null | tee -a "$LOG"
    python3 vast_train.py pull 2>&1 | tail -3 | tee -a "$LOG"
    if [ "$DESTROY_ON_TIMEOUT" = "1" ]; then
      note "destroying instance (DESTROY_ON_TIMEOUT=1)"
      python3 vast_train.py destroy 2>&1 | tail -2 | tee -a "$LOG"
    else
      note "instance still running (meter on) — stop it with: python vast_train.py destroy"
    fi
    note "DONE (timeout) — latest weights in ./vast_out/saved/"
    exit 0
  fi
  tail_line=$(python3 vast_train.py ssh "tail -1 $REMOTE/pipeline.log 2>/dev/null" 2>/dev/null | tail -1)
  note "poll: ${tail_line:-no pipeline.log yet}"
  if ! python3 vast_train.py ssh "pgrep -f remote_pipeline.sh >/dev/null" >/dev/null 2>&1; then
    note "WARNING: pipeline process not running — relaunch with ./watch_pipeline.sh"
  fi
  sleep 300
done

python3 vast_train.py pull 2>&1 | tail -3 | tee -a "$LOG"
note "DONE — weights in ./vast_out/saved/$DPO_DIR_NAME"
