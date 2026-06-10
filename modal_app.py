"""Modal entrypoint: prepare a corpus + train MyOwnTransformer on a GPU.

One-time setup:
    pip install modal
    modal setup         # opens a browser to link your account

Usage:
    # One-shot: prepare (if needed) -> train with ~90M defaults on H100.
    modal run modal_app.py

    # Override knobs:
    modal run modal_app.py --d-model 640 --n-layers 14 --heads 10 \
        --seqlen 1024 --batchsize 128 --epochs 1 --warmup-steps 1000

    # Rebuild the tokenizer + .bin shards (otherwise we reuse the volume copy)
    modal run modal_app.py --force-prepare

    # Just prep, no training:
    modal run modal_app.py::prepare

    # Just train (assumes data already on the volume):
    modal run modal_app.py::train --dir-name my_run

    # Detach (don't block the shell; stream logs with `modal app logs`):
    modal run --detached modal_app.py

    # Pull checkpoints + learning_curves back to ./modal_out/
    modal volume get myowntransformer-data /saved ./modal_out
"""
import modal


APP_NAME = "myowntransformer"
VOLUME_NAME = "myowntransformer-data"
VOL_MOUNT = "/vol"
SAVE_ROOT = f"{VOL_MOUNT}/saved"


DATA_DIR = f"{VOL_MOUNT}/data_cache/cosmopedia"

# torch 2.11 matches what the user runs locally; torch.optim.Muon needs >= 2.9.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.11.0",
        "numpy",
        "datasets",
        "matplotlib",
    )
    .add_local_dir(
        ".",
        remote_path="/root/src",
        ignore=[
            "data_cache",
            "saved",
            "chat/target",
            "**/__pycache__",
            ".git",
            ".pytest_cache",
            "*.bin",
            "learning_curves.png",
            ".claude",
            "modal_out",
        ],
    )
)

app = modal.App(APP_NAME, image=image)
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# HF_TOKEN raises HuggingFace streaming rate limits ~40x for anonymous reads.
# Create once with: modal secret create huggingface HF_TOKEN=hf_...
hf_secret = modal.Secret.from_name("huggingface")


@app.function(
    volumes={VOL_MOUNT: vol},
    cpu=4.0,
    timeout=60 * 60 * 6,
    secrets=[hf_secret],
    retries=modal.Retries(max_retries=5, initial_delay=10.0),
)
def prepare(
    force: bool = False,
    vocab_size: int = 32000,
    bpe_train_docs: int = 10000,
    max_train_docs: int = 0,
    holdout_period: int = 500,
):
    """Stream the mixed corpus (see prepare.SOURCES), train BPE, emit train/val/test.bin into the volume.

    max_train_docs=0 means no cap (full stream)."""
    import os
    import subprocess

    if not force and os.path.exists(f"{DATA_DIR}/train.bin"):
        print(f"{DATA_DIR}/train.bin already exists — skipping (pass --force-prepare to rebuild)")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    os.chdir("/root/src")
    cmd = [
        "python", "-u", "prepare.py",
        "--output-dir", DATA_DIR,
        "--vocab-size", str(vocab_size),
        "--bpe-train-docs", str(bpe_train_docs),
        "--holdout-period", str(holdout_period),
    ]
    if max_train_docs > 0:
        cmd += ["--max-train-docs", str(max_train_docs)]
    subprocess.run(cmd, check=True)
    vol.commit()
    print(f"Data prepared and committed to volume `{VOLUME_NAME}:{DATA_DIR}`")


@app.function(
    volumes={VOL_MOUNT: vol},
    gpu="H100",
    timeout=60 * 60 * 24,
)
def train(
    d_model: int = 640,
    n_layers: int = 14,
    heads: int = 10,
    kv_heads: int = 5,
    loops: int = 1,
    batchsize: int = 128,
    grad_accum: int = 1,
    seqlen: int = 1024,
    epochs: int = 1,
    lr: float = 3e-4,
    muon_lr: float = 0.03,
    warmup_steps: int = 1000,
    save_every: int = 2000,
    val_every: int = 2000,
    printevery: int = 50,
    dir_name: str = "modal_run",
):
    """Run train.py on a GPU, writing checkpoints + plot into the volume."""
    import os
    import shutil
    import subprocess

    if not os.path.exists(f"{DATA_DIR}/train.bin"):
        raise FileNotFoundError(
            f"no {DATA_DIR}/train.bin on the volume — "
            f"run `modal run modal_app.py::prepare` first"
        )

    os.chdir("/root/src")

    # train.py hard-codes `saved/<dir_name>/` relative to CWD. Symlink that into
    # the volume so periodic checkpoints land on persistent storage directly.
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if not os.path.lexists("/root/src/saved"):
        os.symlink(SAVE_ROOT, "/root/src/saved")

    env = {**os.environ, "MPLBACKEND": "Agg"}
    subprocess.run(
        [
            "python", "-u", "train.py",
            "-data_dir", DATA_DIR,
            "-dir_name", dir_name,
            "-d_model", str(d_model),
            "-n_layers", str(n_layers),
            "-heads", str(heads),
            "-kv_heads", str(kv_heads),
            "-loops", str(loops),
            "-batchsize", str(batchsize),
            "-grad_accum", str(grad_accum),
            "-seqlen", str(seqlen),
            "-epochs", str(epochs),
            "-lr", str(lr),
            "-muon_lr", str(muon_lr),
            "-warmup_steps", str(warmup_steps),
            "-save_every", str(save_every),
            "-val_every", str(val_every),
            "-printevery", str(printevery),
        ],
        check=True, env=env,
    )

    plot_src = "/root/src/learning_curves.png"
    if os.path.exists(plot_src):
        shutil.copy(plot_src, f"{SAVE_ROOT}/{dir_name}_learning_curves.png")
    vol.commit()
    print(f"Artifacts saved to volume `{VOLUME_NAME}:/saved/{dir_name}/`")
    print(f"Download with: modal volume get {VOLUME_NAME} /saved ./modal_out")


@app.function(
    volumes={VOL_MOUNT: vol},
    cpu=4.0,
    timeout=60 * 60 * 4,
    secrets=[hf_secret],
    retries=modal.Retries(max_retries=5, initial_delay=10.0),
)
def sft_prepare(
    force: bool = False,
    max_conversations: int = 0,
    holdout_period: int = 200,
):
    """Tokenize smol-smoltalk into sft_*.bin shards on the volume."""
    import os
    import subprocess

    if not force and os.path.exists(f"{DATA_DIR}/sft_train.bin"):
        print(f"{DATA_DIR}/sft_train.bin already exists — skipping (--force-sft-prepare to rebuild)")
        return

    os.chdir("/root/src")
    cmd = [
        "python", "-u", "sft_prepare.py",
        "--output-dir", DATA_DIR,
        "--holdout-period", str(holdout_period),
    ]
    if max_conversations > 0:
        cmd += ["--max-conversations", str(max_conversations)]
    subprocess.run(cmd, check=True)
    vol.commit()
    print(f"SFT data committed to `{VOLUME_NAME}:{DATA_DIR}`")


@app.function(
    volumes={VOL_MOUNT: vol},
    gpu="H100",
    timeout=60 * 60 * 12,
)
def sft(
    checkpoint: str = "modal_run/ckpt_final.pt",
    epochs: int = 2,
    batchsize: int = 64,
    seqlen: int = 512,
    lr: float = 3e-5,
    muon_lr: float = 0.003,
    warmup_steps: int = 100,
    save_every: int = 1000,
    val_every: int = 500,
    dir_name: str = "sft_run",
):
    """Fine-tune a pretrained checkpoint on chat data. `checkpoint` is relative
    to the volume's /saved root."""
    import os
    import subprocess

    ckpt_path = f"{SAVE_ROOT}/{checkpoint}"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"no checkpoint at {ckpt_path}")
    if not os.path.exists(f"{DATA_DIR}/sft_train.bin"):
        raise FileNotFoundError(
            f"no {DATA_DIR}/sft_train.bin — run `modal run modal_app.py::sft_prepare` first")

    os.chdir("/root/src")
    os.makedirs(SAVE_ROOT, exist_ok=True)
    if not os.path.lexists("/root/src/saved"):
        os.symlink(SAVE_ROOT, "/root/src/saved")

    subprocess.run(
        [
            "python", "-u", "finetune.py",
            "--checkpoint", ckpt_path,
            "--data-dir", DATA_DIR,
            "--epochs", str(epochs),
            "--batchsize", str(batchsize),
            "--seqlen", str(seqlen),
            "--lr", str(lr),
            "--muon-lr", str(muon_lr),
            "--warmup-steps", str(warmup_steps),
            "--save-every", str(save_every),
            "--val-every", str(val_every),
            "--dir-name", dir_name,
        ],
        check=True,
    )
    vol.commit()
    print(f"SFT checkpoints saved to `{VOLUME_NAME}:/saved/{dir_name}/`")
    print(f"Download with: modal volume get {VOLUME_NAME} /saved/{dir_name} ./modal_out/{dir_name}")


@app.function(
    volumes={VOL_MOUNT: vol},
    gpu="H100",
    timeout=60 * 60 * 12,
    secrets=[hf_secret],
    retries=modal.Retries(max_retries=5, initial_delay=10.0),
)
def pipeline(
    force_prepare: bool = False,
    max_train_docs: int = 200000,
    batchsize: int = 64,
    grad_accum: int = 2,
    save_every: int = 1000,
    val_every: int = 1000,
    warmup_steps: int = 300,
    dir_name: str = "chat90m",
    sft_epochs: int = 1,
    sft_dir_name: str = "chat90m_sft",
):
    """Server-side chain: prepare -> train -> sft_prepare -> sft.

    Runs entirely on one detached worker so a local disconnect can't kill it.
    Every stage skips itself if its artifact already exists, so retries resume
    from the last finished stage."""
    import os

    marker = f"{DATA_DIR}/prepare_done.marker"
    if not os.path.exists(marker):
        prepare.local(force=force_prepare, max_train_docs=max_train_docs)
        open(marker, "w").write("ok")
        vol.commit()
    if not os.path.exists(f"{SAVE_ROOT}/{dir_name}/ckpt_final.pt"):
        train.local(
            batchsize=batchsize, grad_accum=grad_accum, save_every=save_every,
            val_every=val_every, warmup_steps=warmup_steps, dir_name=dir_name,
        )
    sft_prepare.local()
    if not os.path.exists(f"{SAVE_ROOT}/{sft_dir_name}/sft_final.pt"):
        sft.local(checkpoint=f"{dir_name}/ckpt_final.pt",
                  epochs=sft_epochs, dir_name=sft_dir_name)
    print("PIPELINE COMPLETE")


@app.local_entrypoint()
def main(
    force_prepare: bool = False,
    vocab_size: int = 32000,
    bpe_train_docs: int = 10000,
    max_train_docs: int = 0,
    holdout_period: int = 500,
    d_model: int = 640,
    n_layers: int = 14,
    heads: int = 10,
    kv_heads: int = 5,
    loops: int = 1,
    batchsize: int = 128,
    grad_accum: int = 1,
    seqlen: int = 1024,
    epochs: int = 1,
    lr: float = 3e-4,
    muon_lr: float = 0.03,
    warmup_steps: int = 1000,
    save_every: int = 2000,
    val_every: int = 2000,
    printevery: int = 50,
    dir_name: str = "modal_run",
    run_sft: bool = False,
    sft_epochs: int = 2,
    sft_dir_name: str = "sft_run",
):
    prepare.remote(
        force=force_prepare,
        vocab_size=vocab_size,
        bpe_train_docs=bpe_train_docs,
        max_train_docs=max_train_docs,
        holdout_period=holdout_period,
    )
    train.remote(
        d_model=d_model,
        n_layers=n_layers,
        heads=heads,
        kv_heads=kv_heads,
        loops=loops,
        batchsize=batchsize,
        grad_accum=grad_accum,
        seqlen=seqlen,
        epochs=epochs,
        lr=lr,
        muon_lr=muon_lr,
        warmup_steps=warmup_steps,
        save_every=save_every,
        val_every=val_every,
        printevery=printevery,
        dir_name=dir_name,
    )
    if run_sft:
        sft_prepare.remote()
        sft.remote(
            checkpoint=f"{dir_name}/ckpt_final.pt",
            epochs=sft_epochs,
            dir_name=sft_dir_name,
        )
