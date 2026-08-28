"""Tokenize the corpus on a cheap CPU box and park it in object storage.

Prep is download-bound, not compute-bound, so running it on the GPU box means
paying $11.92/hr to wait on HuggingFace — and paying again after a preemption.
This provisions a small instance in the bucket's own region, runs prepare,
uploads the shards, and destroys the box in a finally.
"""
import os
import time

from vultr.api import client_from_env
from vultr.lifecycle import destroy_state, provision
from vultr.pipeline import PYTHON, S3_VARS
from vultr.remote import REMOTE_ROOT, rsync, run_remote
from vultr.storage import ensure_subscription

# Measured on this corpus: ~3.3 characters per token, and bins are uint16.
BYTES_PER_TOKEN_CACHED = 3.3
BYTES_PER_TOKEN_BIN = 2
TOKENS_PER_DOC = 1000


def required_disk_gb(max_train_docs, headroom=1.4):
    """The fetch cache holds raw text and the bins hold uint16, and a capped
    run keeps both. Headroom covers the OS, the repo, and the estimate itself.
    """
    tokens = max_train_docs * TOKENS_PER_DOC
    total = tokens * (BYTES_PER_TOKEN_CACHED + BYTES_PER_TOKEN_BIN)
    return max(30, int(total / 1e9 * headroom))


def _bootstrap(state):
    run_remote(
        state,
        "cloud-init status --wait && export DEBIAN_FRONTEND=noninteractive && "
        "apt-get update -qq && apt-get install -y -qq python3-venv rsync && "
        "python3 -m venv /opt/myowntransformer && "
        f"{PYTHON} -m pip install -q --upgrade pip && "
        f"{PYTHON} -m pip install -q --index-url https://download.pytorch.org/whl/cpu "
        "torch==2.11.0 && "
        f"{PYTHON} -m pip install -q datasets boto3",
    )


def build_script(args, subscription):
    import shlex
    quote = shlex.quote
    data = f"data_cache/{args.prefix}"
    exports = "\n".join(
        f"export {name}={quote(value)}" for name, value in (
            ("HF_TOKEN", os.environ.get("HF_TOKEN", "")),
            ("VULTR_S3_HOSTNAME", subscription["s3_hostname"]),
            ("VULTR_S3_ACCESS_KEY", subscription["s3_access_key"]),
            ("VULTR_S3_SECRET_KEY", subscription["s3_secret_key"]),
        ) if value)
    return f"""#!/bin/bash
set -euo pipefail
cd {quote(REMOTE_ROOT)}
export PYTHONUNBUFFERED=1
{exports}
step() {{ echo "[prep $(date +%H:%M:%S)] $*"; }}
DATA={quote(data)}

step "tokenizing up to {args.max_train_docs} docs"
{PYTHON} -m pretrain.prepare --output-dir "$DATA" \
  --max-train-docs {args.max_train_docs} --shard-tokens {args.shard_tokens}

step "shards on disk"
ls -la "$DATA"/*.bin "$DATA"/*.json | tail -20
du -sh "$DATA"

step "uploading to object storage"
{PYTHON} -m vultr.storage up --from-env --data-dir "$DATA" \
  --prefix {quote(args.prefix)} --workers {args.workers}

step "removing the fetch cache so it is not re-uploaded"
rm -rf "$DATA/fetch_cache"
echo "PREP COMPLETE"
"""


def prep(args):
    api = client_from_env()
    subscription = ensure_subscription(api, args.label_storage, args.region)
    region = subscription.get("region") or args.region
    disk = args.disk or required_disk_gb(args.max_train_docs)
    print(f"corpus needs ~{disk} GB; placing prep in {region} next to the bucket")

    # provision does the selecting; it just needs the floors this job requires.
    args.region, args.compute, args.metal = region, True, False
    args.min_ram, args.min_disk, args.label = 2048, disk, "mot-prep"
    started = time.time()
    state = None
    try:
        api, state = provision(args, bootstrap_instance=False)
        print(f"[prep] {state['plan']} up at {state['ssh_host']}; installing runtime...")
        _bootstrap(state)
        run_remote(state, f"mkdir -p {REMOTE_ROOT}")
        rsync(state, "./", f"root@{state['ssh_host']}:{REMOTE_ROOT}/")
        remote_data = f"{REMOTE_ROOT}/data_cache/{args.prefix}"
        run_remote(state, f"mkdir -p {remote_data}")
        if args.tokenizer:
            # prepare reuses an existing tokenizer.json and only trains one when
            # absent. Training is slow in pure Python, and a fresh vocab would
            # make every existing checkpoint incompatible with the new corpus.
            print(f"[prep] reusing {args.tokenizer}")
            rsync(state, args.tokenizer, f"root@{state['ssh_host']}:{remote_data}/")
        script = build_script(args, subscription)
        run_remote(state, f"cat > {REMOTE_ROOT}/prep.sh && chmod 700 {REMOTE_ROOT}/prep.sh",
                   input=script.encode())
        run_remote(state, f"cd {REMOTE_ROOT} && ./prep.sh")
        print(f"[prep] done in {(time.time() - started) / 60:.1f} min; "
              f"corpus is at prefix {args.prefix}")
    finally:
        if state and not args.keep:
            print("[prep] destroying the instance to stop billing...")
            destroy_state(api, state)
        elif state:
            print(f"[prep] --keep left {state['id']} running")
