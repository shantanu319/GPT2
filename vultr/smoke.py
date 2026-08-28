import os
import time

from vultr.lifecycle import destroy_state, provision
from vultr.remote import REMOTE_ROOT, rsync, run_remote


def _bootstrap(state, compute=False):
    swap = ("(swapon --show --noheadings | grep -q . || (fallocate -l 2G /swapfile && "
            "chmod 600 /swapfile && mkswap /swapfile >/dev/null && swapon /swapfile)) && "
            if compute else "")
    runtime = ("/opt/myowntransformer/bin/pip install -q --index-url "
               "https://download.pytorch.org/whl/cpu torch==2.11.0 && "
               "/opt/myowntransformer/bin/pip install -q matplotlib" if compute else
               "/opt/myowntransformer/bin/pip install -q torch==2.11.0 matplotlib")
    run_remote(
        state,
        "cloud-init status --wait && export DEBIAN_FRONTEND=noninteractive && "
        "apt-get update -qq && apt-get install -y -qq python3-venv rsync && "
        f"{swap}"
        "python3 -m venv /opt/myowntransformer && "
        "/opt/myowntransformer/bin/pip install -q --upgrade pip && " + runtime,
    )


def _make_data(state):
    """Train lands as two numbered shards so the smoke also covers the sharded
    read path the big run depends on."""
    script = r"""from pathlib import Path
import numpy as np
from core.tokenizer import BPETokenizer

root = Path("data_cache/smoke")
root.mkdir(parents=True, exist_ok=True)
specials = {"<|im_start|>": 256, "<|im_end|>": 257, "<|endoftext|>": 258}
BPETokenizer(special_tokens=specials).save(root / "tokenizer.json")
rng = np.random.default_rng(1337)
for index in range(2):
    rng.integers(0, 259, 2048, dtype=np.uint16).tofile(root / f"train_{index:05d}.bin")
for split in ("val", "test"):
    rng.integers(0, 259, 1024, dtype=np.uint16).tofile(root / f"{split}.bin")
print("synthetic smoke shards ready")
"""
    run_remote(
        state,
        f"cd {REMOTE_ROOT} && /opt/myowntransformer/bin/python - <<'PY'\n{script}PY",
    )


def _gpu_count(state):
    result = run_remote(state, "nvidia-smi -L 2>/dev/null | wc -l",
                        check=False, capture_output=True)
    try:
        return int((result.stdout or b"0").decode().strip())
    except ValueError:
        return 0


def smoke(args):
    args.label = "mot-vultr-smoke"
    started = time.time()
    api = state = None
    try:
        try:
            api, state = provision(args, bootstrap_instance=False)
        except RuntimeError as error:
            if "support request for access to this product" not in str(error):
                raise
            print("[smoke] GPU access is not enabled; falling back to shared CPU compute")
            args.compute = True
            args.plan = None
            # Also drop --metal: re-selecting with it set would pick the
            # cheapest bare-metal plan that has a live region, which is the
            # 8x A100 box at $11.92/hr.
            args.metal = False
            api, state = provision(args, bootstrap_instance=False)
        print("[smoke] installing the minimal runtime...")
        _bootstrap(state, compute=args.compute)
        device_check = ("import torch; print('torch', torch.__version__, 'CPU ready')" if args.compute else
                        "import torch; assert torch.cuda.is_available(); "
                        "print('torch', torch.__version__, 'CUDA ready')")
        print(f"[smoke] validating {'CPU' if args.compute else 'GPU'} compute...")
        gpu_check = "" if args.compute else "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && "
        run_remote(state, f"{gpu_check}/opt/myowntransformer/bin/python -c \"{device_check}\"")
        print("[smoke] pushing the repository and generating synthetic data...")
        run_remote(state, f"mkdir -p {REMOTE_ROOT}")
        rsync(state, "./", f"root@{state['ssh_host']}:{REMOTE_ROOT}/")
        _make_data(state)
        ranks = max(1, args.ranks)
        gpus = 0 if args.compute else _gpu_count(state)
        # NCCL needs a GPU per rank and resolve_device hands rank N cuda:N, so
        # a box with fewer GPUs than ranks runs the check on CPU over gloo.
        on_gpu = gpus >= ranks
        launcher = ("-m torch.distributed.run --standalone "
                    f"--nproc-per-node={ranks}" if ranks > 1 else "")
        backend = "nccl" if on_gpu and ranks > 1 else ("gloo" if ranks > 1 else "single")
        print(f"[smoke] running tiny eager training on {ranks} rank(s) "
              f"({backend}, {'GPU' if on_gpu else 'CPU'})...")
        run_remote(
            state,
            f"cd {REMOTE_ROOT} && MPLBACKEND=Agg PYTHONUNBUFFERED=1 "
            f"/opt/myowntransformer/bin/python {launcher} "
            "-m pretrain.train -data_dir data_cache/smoke -d_model 32 -n_layers 1 "
            "-heads 1 -kv_heads 1 -batchsize 2 -seqlen 32 -epochs 1 -warmup_steps 2 "
            "-save_every 0 -val_every 0 -printevery 8 -no_compile -dir_name smoke"
            + ("" if on_gpu else " -no_cuda"),
        )
        destination = os.path.join(args.out, "smoke")
        os.makedirs(destination, exist_ok=True)
        checkpoint = os.path.join(destination, "ckpt_final.pt")
        if os.path.exists(checkpoint):
            os.remove(checkpoint)
        rsync(state, f"root@{state['ssh_host']}:{REMOTE_ROOT}/saved/smoke/", f"{destination}/")
        if not os.path.isfile(checkpoint) or not os.path.getsize(checkpoint):
            raise RuntimeError("smoke checkpoint is missing or empty")
        elapsed = (time.time() - started) / 60
        print(f"[smoke] PASS in {elapsed:.1f} min: {checkpoint}")
        print(f"[smoke] Vultr minimum one-hour charge: ${state['hourly_cost']:.3f}")
    finally:
        if state and not args.keep:
            print("[smoke] destroying the instance to stop billing...")
            destroy_state(api, state)
        elif state:
            print(f"[smoke] --keep left instance {state['id']} running")
