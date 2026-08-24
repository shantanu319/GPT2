import os
import time

from vultr.lifecycle import destroy_state, provision
from vultr.remote import REMOTE_ROOT, rsync, run_remote


def _bootstrap(state):
    run_remote(
        state,
        "cloud-init status --wait && export DEBIAN_FRONTEND=noninteractive && "
        "apt-get update -qq && apt-get install -y -qq python3-venv rsync && "
        "python3 -m venv /opt/myowntransformer && "
        "/opt/myowntransformer/bin/pip install -q --upgrade pip && "
        "/opt/myowntransformer/bin/pip install -q torch==2.11.0 matplotlib",
    )


def _make_data(state):
    script = r"""from pathlib import Path
import numpy as np
from core.tokenizer import BPETokenizer

root = Path("data_cache/smoke")
root.mkdir(parents=True, exist_ok=True)
specials = {"<|im_start|>": 256, "<|im_end|>": 257, "<|endoftext|>": 258}
BPETokenizer(special_tokens=specials).save(root / "tokenizer.json")
rng = np.random.default_rng(1337)
for split in ("train", "val", "test"):
    rng.integers(0, 259, 1024, dtype=np.uint16).tofile(root / f"{split}.bin")
print("synthetic smoke shards ready")
"""
    run_remote(
        state,
        f"cd {REMOTE_ROOT} && /opt/myowntransformer/bin/python - <<'PY'\n{script}PY",
    )


def smoke(args):
    args.min_vram = 2
    args.label = "mot-vultr-smoke"
    started = time.time()
    api = state = None
    try:
        api, state = provision(args, bootstrap_instance=False)
        print("[smoke] installing the minimal runtime...")
        _bootstrap(state)
        print("[smoke] validating the GPU...")
        run_remote(
            state,
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && "
            "/opt/myowntransformer/bin/python -c \"import torch; "
            "assert torch.cuda.is_available(); print('torch', torch.__version__, 'CUDA ready')\"",
        )
        print("[smoke] pushing the repository and generating synthetic data...")
        run_remote(state, f"mkdir -p {REMOTE_ROOT}")
        rsync(state, "./", f"root@{state['ssh_host']}:{REMOTE_ROOT}/")
        _make_data(state)
        print("[smoke] running tiny eager CUDA training...")
        run_remote(
            state,
            f"cd {REMOTE_ROOT} && MPLBACKEND=Agg /opt/myowntransformer/bin/python -u "
            "-m pretrain.train -data_dir data_cache/smoke -d_model 32 -n_layers 1 "
            "-heads 1 -kv_heads 1 -batchsize 2 -seqlen 32 -epochs 1 -warmup_steps 2 "
            "-save_every 0 -val_every 0 -printevery 8 -no_compile -dir_name smoke",
        )
        destination = os.path.join(args.out, "smoke")
        os.makedirs(destination, exist_ok=True)
        rsync(state, f"root@{state['ssh_host']}:{REMOTE_ROOT}/saved/smoke/", f"{destination}/")
        checkpoint = os.path.join(destination, "ckpt_final.pt")
        if not os.path.getsize(checkpoint):
            raise RuntimeError("smoke checkpoint is empty")
        elapsed = (time.time() - started) / 60
        print(f"[smoke] PASS in {elapsed:.1f} min: {checkpoint}")
        print(f"[smoke] Vultr minimum one-hour charge: ${state['hourly_cost']:.3f}")
    finally:
        if state and not args.keep:
            print("[smoke] destroying the instance to stop billing...")
            destroy_state(api, state)
        elif state:
            print(f"[smoke] --keep left instance {state['id']} running")
