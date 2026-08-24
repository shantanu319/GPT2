import os
import shlex

from vultr.remote import REMOTE_ROOT


PYTHON = "/opt/myowntransformer/bin/python"
DATA_DIR = "data_cache/cosmopedia"


def build_pipeline(args):
    quote = shlex.quote
    hf_token = os.environ.get("HF_TOKEN", "")
    hf_export = f"export HF_TOKEN={quote(hf_token)}" if hf_token else ""
    prepare_args = f"--max-train-docs {args.max_train_docs}" if args.max_train_docs else ""
    train_args = (
        f"-d_model {args.d_model} -n_layers {args.n_layers} -heads {args.heads} "
        f"-kv_heads {args.kv_heads} -batchsize {args.batchsize} -seqlen {args.seqlen} "
        f"-grad_accum {args.grad_accum} -grad_ckpt {int(args.grad_ckpt)} "
        f"-epochs {args.epochs} -warmup_steps {args.warmup_steps} "
        f"-save_every {args.save_every} -val_every {args.val_every}"
    )
    return f"""#!/bin/bash
set -euo pipefail
cd {quote(REMOTE_ROOT)}
export MPLBACKEND=Agg
{hf_export}
PY={quote(PYTHON)}
DATA={quote(DATA_DIR)}
DIR={quote(args.dir_name)}
SFT={quote(args.sft_dir_name)}
DPO={quote(args.dpo_dir_name)}
step() {{ echo "[pipeline $(date +%H:%M:%S)] $*"; }}

if [[ ! -f "$DATA/train.bin" ]]; then
  step prepare
  "$PY" -u -m pretrain.prepare --output-dir "$DATA" {prepare_args}
fi

if [[ ! -f "saved/$DIR/ckpt_final.pt" ]]; then
  latest=$(ls "saved/$DIR"/ckpt_step*.pt 2>/dev/null | sort -V | tail -1 || true)
  resume=()
  [[ -n "$latest" ]] && resume=(-loadname "$latest")
  step pretrain
  "$PY" -u -m pretrain.train -data_dir "$DATA" -dir_name "$DIR" \
    {train_args} "${{resume[@]}}"
fi

if [[ ! -f "$DATA/sft_train.bin" ]]; then
  step sft_prepare
  "$PY" -u -m sft.sft_prepare --output-dir "$DATA"
fi

if [[ ! -f "saved/$SFT/sft_final.pt" ]]; then
  step sft
  "$PY" -u -m sft.finetune --checkpoint "saved/$DIR/ckpt_final.pt" \
    --data-dir "$DATA" --dir-name "$SFT" --epochs {args.sft_epochs}
fi

if [[ ! -f "$DATA/dpo_train.bin" ]]; then
  step dpo_prepare
  "$PY" -u -m dpo.dpo_prepare --output-dir "$DATA"
fi

if [[ ! -f "saved/$DPO/dpo_final.pt" ]]; then
  step dpo
  "$PY" -u -m dpo.dpo --checkpoint "saved/$SFT/sft_final.pt" \
    --data-dir "$DATA" --dir-name "$DPO" --epochs {args.dpo_epochs}
fi

echo "PIPELINE COMPLETE"
"""
