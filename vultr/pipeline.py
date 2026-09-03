import os
import shlex

from vultr.remote import REMOTE_ROOT


PYTHON = "/opt/myowntransformer/bin/python"
DATA_DIR = "data_cache/cosmopedia"
S3_VARS = ("VULTR_S3_HOSTNAME", "VULTR_S3_ACCESS_KEY", "VULTR_S3_SECRET_KEY")

# Every pretrain.train knob the pipeline forwards, as (flag, type, default).
# Defaults track pretrain/config.py except where a cluster run differs: the
# stack is dense (the KDA hybrid measured ~half the throughput of dense on an
# A100 and has never been trained at scale here), -muon_per_head follows Kimi
# K3, the schedule is sized for one pass over a large corpus with SmolLM2's
# 20% decay, and checkpoints land often enough that a preemption is cheap.
TRAIN_FLAGS = [
    ("d_model", int, 512),
    ("n_layers", int, 30),
    ("heads", int, 8),
    ("kv_heads", int, 2),
    ("loops", int, 1),
    ("dropout", float, 0.0),
    ("tied", int, 1),
    ("value_residual", int, 1),
    ("unet_skips", int, 1),
    ("attn_res", int, 0),
    ("kda", int, 0),
    ("swa", int, 0),
    ("batchsize", int, 128),
    ("seqlen", int, 1024),
    ("grad_accum", int, 1),
    ("grad_ckpt", int, 0),
    ("lr", float, 3e-4),
    ("muon_lr", float, 0.03),
    ("embed_lr", float, 3e-3),
    ("scalar_lr", float, 0.01),
    ("muon_impl", str, "local"),
    ("muon_per_head", int, 1),
    ("schedule", str, "wsd"),
    ("decay_frac", float, 0.2),
    ("momentum_warmup", int, 300),
    ("norm", float, 2.0),
    ("shuffle", int, 1),
    ("doc_mask", int, 1),
    ("ce_chunk", int, 16384),
    ("epochs", int, 1),
    ("warmup_steps", int, 1000),
    ("save_every", int, 2000),
    ("val_every", int, 2000),
    ("val_batches", int, 50),
    ("early_stop", int, 0),
    ("early_stop_delta", float, 0.005),
    ("early_stop_cooldown", int, 300),
]

PER_RANK = {"batchsize"}


def add_train_args(parser):
    """Expose every TRAIN_FLAGS entry as --flag-name on the pipeline command."""
    for name, kind, default in TRAIN_FLAGS:
        note = " (per GPU)" if name in PER_RANK else ""
        parser.add_argument(f"--{name.replace('_', '-')}", dest=name, type=kind,
                            default=default, help=f"pretrain.train -{name}{note}")


def train_flag_string(args):
    quote = shlex.quote
    parts = []
    for name, kind, default in TRAIN_FLAGS:
        value = kind(getattr(args, name, default))
        parts.append(f"-{name} {quote(str(value))}")
    return " ".join(parts)


def build_pipeline(args):
    quote = shlex.quote
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_export = f"export HF_TOKEN={quote(hf_token)}" if hf_token else ""
    # Pulling a prepared corpus beats rebuilding it: prep is download-bound,
    # and this box bills by the hour. Falls through to prepare if nothing lands.
    prefix = getattr(args, "corpus_prefix", "") or ""
    anneal = getattr(args, "anneal_prefix", "") or ""
    exports = "\n".join(f"export {name}={quote(os.environ.get(name, ''))}"
                         for name in S3_VARS) if prefix or anneal else ""
    fetch = ""
    if prefix:
        cap = getattr(args, "corpus_shards", 0)
        fetch = f'''if ! prepared "$DATA"; then
  step "fetch corpus from object storage"
  "$PY" -m vultr.storage down --from-env --prefix {quote(prefix)} \\
    --max-shards {cap} \\
    --data-dir "$DATA" || step "no corpus in storage; will tokenize instead"
fi'''
    anneal_fetch = anneal_flag = ""
    if anneal:
        # The decay-phase corpus is only ever fetched: tokenizing it here would
        # bill the GPU box for a CPU job, so a missing one stops the run.
        anneal_fetch = f'''ANNEAL=data_cache/{quote(anneal)}
if ! prepared "$ANNEAL"; then
  step "fetch anneal corpus from object storage"
  "$PY" -m vultr.storage down --from-env --prefix {quote(anneal)} --data-dir "$ANNEAL"
fi'''
        anneal_flag = '-anneal_dir "$ANNEAL"'
    prepare_args = f"--max-train-docs {args.max_train_docs}" if args.max_train_docs else ""
    train_args = train_flag_string(args)
    gpus = str(getattr(args, "gpus", "auto"))
    return f"""#!/bin/bash
set -euo pipefail
cd {quote(REMOTE_ROOT)}
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
{hf_export}
PY={quote(PYTHON)}
DATA={quote(DATA_DIR)}
DIR={quote(args.dir_name)}
SFT={quote(args.sft_dir_name)}
DPO={quote(args.dpo_dir_name)}
step() {{ echo "[pipeline $(date +%H:%M:%S)] $*"; }}

GPUS={quote(gpus)}
if [[ "$GPUS" == "auto" ]]; then
  GPUS=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
fi
[[ "$GPUS" -ge 1 ]] 2>/dev/null || GPUS=1
if [[ "$GPUS" -gt 1 ]]; then
  LAUNCH=("$PY" -m torch.distributed.run --standalone --nproc-per-node="$GPUS")
else
  LAUNCH=("$PY")
fi
step "launching on $GPUS GPU(s)"

prepared() {{ "$PY" -c "import json,sys;sys.exit(0 if json.load(open('$1/train_manifest.json')).get('complete') else 1)" 2>/dev/null; }}
{exports}
{fetch}
if ! prepared "$DATA"; then
  step prepare
  "$PY" -m pretrain.prepare --output-dir "$DATA" {prepare_args}
fi
{anneal_fetch}
if [[ ! -f "saved/$DIR/ckpt_final.pt" ]]; then
  latest=$(ls "saved/$DIR"/ckpt_step*.pt 2>/dev/null | sort -V | tail -1 || true)
  resume=()
  [[ -n "$latest" ]] && resume=(-resume "$latest")
  step pretrain
  "${{LAUNCH[@]}}" -m pretrain.train -data_dir "$DATA" -dir_name "$DIR" \
    {train_args} {anneal_flag} "${{resume[@]}}"
fi

if [[ ! -f "$DATA/sft_train.bin" ]]; then
  step sft_prepare
  "$PY" -m sft.sft_prepare --output-dir "$DATA"
fi

if [[ ! -f "saved/$SFT/sft_final.pt" ]]; then
  step sft
  "${{LAUNCH[@]}}" -m sft.finetune --checkpoint "saved/$DIR/ckpt_final.pt" \
    --data-dir "$DATA" --dir-name "$SFT" --epochs {args.sft_epochs}
fi

if [[ ! -f "$DATA/dpo_train.bin" ]]; then
  step dpo_prepare
  "$PY" -m dpo.dpo_prepare --output-dir "$DATA"
fi

if [[ ! -f "saved/$DPO/dpo_final.pt" ]]; then
  step dpo
  "${{LAUNCH[@]}}" -m dpo.dpo --checkpoint "saved/$SFT/sft_final.pt" \
    --data-dir "$DATA" --dir-name "$DPO" --epochs {args.dpo_epochs}
fi

echo "PIPELINE COMPLETE"
"""
