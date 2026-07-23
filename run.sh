#!/usr/bin/env bash
# One-shot pipeline: prepare data (if needed) -> train -> plot.
#
# All knobs are env-var overridable. Defaults target a ~13M-param
# cosmopedia run that should finish overnight on an M3 Air CPU.
#
#   ./run.sh                             # full pipeline with defaults
#   EPOCHS=3 D_MODEL=128 ./run.sh        # override a few knobs
#   FORCE_PREPARE=1 ./run.sh             # re-run prepare even if .bin exists

set -euo pipefail

# --- Data prep knobs ---
DATA_DIR="${DATA_DIR:-data_cache/cosmopedia}"
VOCAB_SIZE="${VOCAB_SIZE:-32000}"
BPE_TRAIN_DOCS="${BPE_TRAIN_DOCS:-10000}"
MAX_TRAIN_DOCS="${MAX_TRAIN_DOCS:-}"     # empty = full train stream

# --- Model knobs ---
D_MODEL="${D_MODEL:-256}"
N_LAYERS="${N_LAYERS:-6}"
HEADS="${HEADS:-4}"

# --- Training knobs ---
EPOCHS="${EPOCHS:-1}"
BATCHSIZE="${BATCHSIZE:-32}"
SEQLEN="${SEQLEN:-256}"
LR="${LR:-3e-4}"
MUON_LR="${MUON_LR:-0.02}"
WARMUP_STEPS="${WARMUP_STEPS:-200}"
SAVE_EVERY="${SAVE_EVERY:-500}"
PRINTEVERY="${PRINTEVERY:-50}"

# --- Misc ---
FORCE_PREPARE="${FORCE_PREPARE:-0}"
OPEN_PLOT="${OPEN_PLOT:-1}"

# --- Step 1: prepare data ---
if [[ "${FORCE_PREPARE}" == "1" ]] || [[ ! -f "${DATA_DIR}/train.bin" ]]; then
    echo "=== Preparing data (BPE + .bin shards) -> ${DATA_DIR}"
    prepare_args=(
        --output-dir "${DATA_DIR}"
        --vocab-size "${VOCAB_SIZE}"
        --bpe-train-docs "${BPE_TRAIN_DOCS}"
    )
    [[ -n "${MAX_TRAIN_DOCS}" ]] && prepare_args+=(--max-train-docs "${MAX_TRAIN_DOCS}")
    python3 prepare.py "${prepare_args[@]}"
else
    echo "=== Data already prepared in ${DATA_DIR} (set FORCE_PREPARE=1 to redo)"
fi

# --- Step 2: train + validate + test (train.py does all three) ---
echo "=== Training"
MPLBACKEND=Agg python3 -u train.py \
    -data_dir "${DATA_DIR}" \
    -d_model "${D_MODEL}" \
    -n_layers "${N_LAYERS}" \
    -heads "${HEADS}" \
    -batchsize "${BATCHSIZE}" \
    -seqlen "${SEQLEN}" \
    -epochs "${EPOCHS}" \
    -lr "${LR}" \
    -muon_lr "${MUON_LR}" \
    -warmup_steps "${WARMUP_STEPS}" \
    -save_every "${SAVE_EVERY}" \
    -printevery "${PRINTEVERY}"

# --- Step 3: surface the plot ---
if [[ -f learning_curves.png ]]; then
    echo "=== Loss curves: learning_curves.png"
    if [[ "${OPEN_PLOT}" == "1" ]] && command -v open >/dev/null 2>&1; then
        open learning_curves.png
    fi
else
    echo "warning: learning_curves.png was not produced"
fi
