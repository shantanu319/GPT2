import json
import os
import shlex
import subprocess
import time


REMOTE_ROOT = "/root/myowntransformer"
STATE_FILE = ".vultr_instance.json"
SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "ConnectTimeout=15",
    "-o", "LogLevel=ERROR",
]
RSYNC_EXCLUDES = [
    ".git", ".claude", "__pycache__", "**/__pycache__", ".pytest_cache",
    "inference/chat/target", "saved", "modal_out", "vast_out", "vultr_out",
    "data_cache", ".env.local", ".vast_instance.json", STATE_FILE,
    "watch_pipeline.log", "watch_vultr_pipeline.log", ".DS_Store",
]


def load_state(required=True):
    if not os.path.exists(STATE_FILE):
        if required:
            raise RuntimeError(f"no {STATE_FILE}; run create first")
        return None
    with open(STATE_FILE) as handle:
        return json.load(handle)


def save_state(state):
    with open(STATE_FILE, "w") as handle:
        json.dump(state, handle, indent=2)


def clear_state():
    if os.path.exists(STATE_FILE):
        os.remove(STATE_FILE)


def ssh_prefix(state):
    identity = os.path.expanduser(state["ssh_private_key"])
    return ["ssh", "-i", identity, *SSH_OPTS, f"root@{state['ssh_host']}"]


def run_remote(state, command, check=True, **kwargs):
    return subprocess.run(ssh_prefix(state) + [command], check=check, **kwargs)


def rsync(state, source, destination, excludes=()):
    ssh = shlex.join(ssh_prefix(state)[:-1])
    command = ["rsync", "-az", "-e", ssh]
    for pattern in [*RSYNC_EXCLUDES, *excludes]:
        command.extend(["--exclude", pattern])
    subprocess.run([*command, source, destination], check=True)


def ensure_ssh_key(api, public_key_path):
    path = os.path.expanduser(public_key_path)
    with open(path) as handle:
        public_key = handle.read().strip()
    keys = api.request("GET", "/ssh-keys?per_page=500").get("ssh_keys", [])
    existing = next((key for key in keys if key.get("ssh_key", "").strip() == public_key), None)
    if existing:
        return existing["id"], False
    result = api.request("POST", "/ssh-keys", {"name": "myowntransformer", "ssh_key": public_key})
    return result["ssh_key"]["id"], True


def wait_ready(api, instance_id, ssh_private_key, timeout=20 * 60):
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        info = api.request("GET", f"/instances/{instance_id}")["instance"]
        status = (info.get("status"), info.get("power_status"), info.get("server_status"))
        if status != last:
            print(f"  instance {instance_id}: {' / '.join(str(item) for item in status)}", flush=True)
            last = status
        ip = info.get("main_ip")
        if status == ("active", "running", "ok") and ip and ip != "0.0.0.0":
            state = {"id": instance_id, "ssh_host": ip, "ssh_private_key": ssh_private_key}
            probe = subprocess.run(
                ssh_prefix(state) + ["true"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            if probe.returncode == 0:
                return state
        time.sleep(10)
    raise RuntimeError(f"instance {instance_id} not SSH-ready after {timeout // 60} minutes")
