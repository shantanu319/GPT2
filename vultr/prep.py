"""Tokenize the corpus on a cheap CPU box and park it in object storage.

Prep is download-bound, not compute-bound, so running it on the GPU box means
paying $11.92/hr to wait on HuggingFace — and paying again after a preemption.
This provisions a small instance in the bucket's own region, runs prepare,
uploads the shards, and destroys the box in a finally.
"""
import math
import os
import time

from vultr.api import client_from_env
from vultr.lifecycle import destroy_state, provision
from vultr.pipeline import PYTHON, S3_VARS
from vultr.remote import REMOTE_ROOT, rsync, run_remote
from vultr.storage import ensure_subscription

# Measured over all five sources at 32.6M docs: 5.4 KB of cached text per doc,
# i.e. 4.1 bytes per token. Bins are uint16.
BYTES_PER_TOKEN_CACHED = 4.1
BYTES_PER_TOKEN_BIN = 2
TOKENS_PER_DOC = 1313   # measured: 3.0B tokens from 2.28M docs
# 500M-token shards sealed every 24.5s on 32 cores. The 20k-doc sample said
# 128k, but that run was short enough to be dominated by HuggingFace metadata
# and cold parquet opens rather than by tokenizing.
TOKENS_PER_CORE_SEC = 638_000


def required_disk_gb(max_train_docs, headroom=1.4):
    """The fetch cache holds raw text and the bins hold uint16, and a capped
    run keeps both. Headroom covers the OS, the repo, and the estimate itself.
    """
    tokens = max_train_docs * TOKENS_PER_DOC
    total = tokens * (BYTES_PER_TOKEN_CACHED + BYTES_PER_TOKEN_BIN)
    return max(30, int(total / 1e9 * headroom))


def required_vcpu(max_train_docs, target_hours=1):
    """Cores needed to tokenize in target_hours.

    The selector minimises $/hr with floors only on RAM and disk, which alone
    picks the slowest box that fits. Vultr prices these flat per core up to 32,
    so a shorter wall-clock is free and the target should be aggressive.

    This sizes tokenizing only. Fetch is capped at one worker per source, so
    it does not shrink with cores and sets the floor on total runtime.
    """
    core_seconds = max_train_docs * TOKENS_PER_DOC / TOKENS_PER_CORE_SEC
    return max(1, math.ceil(core_seconds / (target_hours * 3600)))


def _bootstrap(state):
    run_remote(
        state,
        "cloud-init status --wait && export DEBIAN_FRONTEND=noninteractive && "
        "apt-get update -qq && apt-get install -y -qq python3-venv rsync && "
        "python3 -m venv /opt/myowntransformer && "
        f"{PYTHON} -m pip install -q --upgrade pip && "
        f"{PYTHON} -m pip install -q --index-url https://download.pytorch.org/whl/cpu "
        "torch==2.11.0 && "
        f"{PYTHON} -m pip install -q datasets zstandard boto3",
    )


# sft_prepare and dpo_prepare are single-threaded and independent of each
# other, so two cores run both at once; their outputs are ~2 GB. Both import
# torch for ~300 MB of RSS each before streaming, which a 2 GB box does not
# comfortably hold alongside the datasets buffers.
POST_VCPU, POST_DISK_GB, POST_RAM_MB = 2, 40, 4096


def _pretrain_body(args, quote):
    return f"""step "uploading sealed shards as they land"
{PYTHON} -m vultr.storage up --from-env --follow --data-dir "$DATA" \
  --prefix {quote(args.prefix)} --workers {args.workers} &
UPLOADER=$!

step "tokenizing up to {args.max_train_docs} docs"
{PYTHON} -m pretrain.prepare --output-dir "$DATA" \
  --max-train-docs {args.max_train_docs} --shard-tokens {args.shard_tokens}

step "shards on disk"
ls -la "$DATA"/*.bin "$DATA"/*.json | tail -20
du -sh "$DATA"

step "waiting for the upload to drain"
wait $UPLOADER

step "dropping the fetch cache"
rm -rf "$DATA/fetch_cache"
"""


def _post_body(args, quote):
    """SFT and DPO only need tokenizer.json, which prep already rsyncs up, so
    both stream straight from HuggingFace and neither waits on the other."""
    return f"""step "sft_prepare and dpo_prepare (independent, run together)"
{PYTHON} -m sft.sft_prepare --output-dir "$DATA" &
SFT=$!
{PYTHON} -m dpo.dpo_prepare --output-dir "$DATA" &
DPO=$!
wait $SFT
wait $DPO

step "artifacts on disk"
ls -la "$DATA"/sft_*.bin "$DATA"/dpo_*.bin
du -sh "$DATA"

step "uploading to object storage"
{PYTHON} -m vultr.storage up --from-env --data-dir "$DATA" \
  --prefix {quote(args.prefix)} --workers {args.workers}"""


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
    body = (_post_body if getattr(args, "stage", "pretrain") == "post"
            else _pretrain_body)(args, quote)
    return f"""#!/bin/bash
set -euo pipefail
cd {quote(REMOTE_ROOT)}
export PYTHONUNBUFFERED=1
{exports}
step() {{ echo "[prep $(date +%H:%M:%S)] $*"; }}
DATA={quote(data)}

{body}
echo "PREP COMPLETE"
"""


def _run_detached(state, poll=60):
    """Run prep.sh under nohup and poll its log.

    A three-hour foreground ssh loses the run and, via the finally, the box
    and its work to one dropped connection. Detached, a drop costs a retry.
    """
    log = f"{REMOTE_ROOT}/prep.log"
    run_remote(state, f"cd {REMOTE_ROOT} && rm -f prep.log && "
                      "setsid nohup ./prep.sh > prep.log 2>&1 < /dev/null & sleep 1")
    last = None
    while True:
        time.sleep(poll)
        result = run_remote(
            state,
            f"tail -n 1 {log}; pgrep -f '[p]rep[.]sh' > /dev/null && echo __RUNNING__",
            check=False, capture_output=True)
        if result.returncode != 0:
            print("[prep] ssh unreachable; retrying")
            continue
        output = result.stdout.decode(errors="replace")
        line = output.replace("__RUNNING__", "").strip()
        if line and line != last:
            print(f"[prep] {line}")
            last = line
        if "__RUNNING__" not in output:
            done = run_remote(state, f"grep -q '^PREP COMPLETE' {log} && echo YES || echo NO",
                              capture_output=True).stdout
            return b"YES" in done


def prep(args):
    api = client_from_env()
    subscription = ensure_subscription(api, args.label_storage, args.region)
    region = subscription.get("region") or args.region
    post = getattr(args, "stage", "pretrain") == "post"
    disk = args.disk or (POST_DISK_GB if post else required_disk_gb(args.max_train_docs))
    print(f"{'sft/dpo' if post else 'corpus'} needs ~{disk} GB; "
          f"placing prep in {region} next to the bucket")

    # provision does the selecting; it just needs the floors this job requires.
    args.region, args.compute, args.metal = region, True, False
    cores = args.vcpu or (POST_VCPU if post else required_vcpu(args.max_train_docs))
    args.min_ram = POST_RAM_MB if post else 2048
    args.min_disk, args.label = disk, "mot-prep"
    args.min_vcpu = cores
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
        if not _run_detached(state):
            run_remote(state, f"tail -n 40 {REMOTE_ROOT}/prep.log", check=False)
            raise RuntimeError("prep did not reach PREP COMPLETE; see log above")
        print(f"[prep] done in {(time.time() - started) / 60:.1f} min; "
              f"{'sft/dpo' if post else 'corpus'} is at prefix {args.prefix}")
    finally:
        if state and not args.keep:
            print("[prep] destroying the instance to stop billing...")
            destroy_state(api, state)
        elif state:
            print(f"[prep] --keep left {state['id']} running")
