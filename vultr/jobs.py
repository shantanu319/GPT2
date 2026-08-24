import os
import shlex
import subprocess

from vultr.pipeline import DATA_DIR, build_pipeline
from vultr.remote import REMOTE_ROOT, load_state, rsync, run_remote, ssh_prefix


PULL_DIR = "vultr_out"


def push(args):
    state = load_state()
    run_remote(state, f"mkdir -p {REMOTE_ROOT}")
    rsync(state, "./", f"root@{state['ssh_host']}:{REMOTE_ROOT}/")
    print(f"pushed repository to {REMOTE_ROOT}")


def ssh(args):
    state = load_state()
    if not args.command:
        print(shlex.join(ssh_prefix(state)))
        return
    result = run_remote(state, shlex.join(args.command), check=False)
    raise SystemExit(result.returncode)


def pipeline(args):
    state = load_state()
    script = build_pipeline(args)
    run_remote(
        state, f"cat > {REMOTE_ROOT}/remote_pipeline.sh && chmod 700 {REMOTE_ROOT}/remote_pipeline.sh",
        input=script.encode(),
    )
    run_remote(
        state,
        f"cd {REMOTE_ROOT} && nohup bash remote_pipeline.sh > pipeline.log 2>&1 & echo $!",
    )
    print("pipeline detached; use `python3 vultr/vultr_train.py status` to monitor it")


def pull(args):
    state = load_state()
    os.makedirs(args.out, exist_ok=True)
    host = f"root@{state['ssh_host']}:{REMOTE_ROOT}"
    rsync(state, f"{host}/saved/", f"{args.out}/saved/")
    print(f"pulled saved/ to {args.out}/saved/")
    for remote_path in [f"{DATA_DIR}/tokenizer.json", "pipeline.log", "learning_curves.png"]:
        try:
            rsync(state, f"{host}/{remote_path}", f"{args.out}/")
        except subprocess.CalledProcessError:
            pass
    print(f"artifacts are in {args.out}/")
