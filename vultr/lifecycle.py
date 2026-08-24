import os
import sys
import time

from vultr.api import (
    client_from_env, list_gpu_plans, list_plans, select_compute_plan, select_plan,
)
from vultr.remote import clear_state, ensure_ssh_key, load_state, run_remote, save_state, wait_ready


GPU_OS_ID = 2284
DEFAULT_PUBLIC_KEY = "~/.ssh/id_ed25519.pub"
DEFAULT_PRIVATE_KEY = "~/.ssh/id_ed25519"


def print_plans(args):
    plans = sorted(list_gpu_plans(), key=lambda plan: (plan["hourly_cost"], plan["id"]))
    print(f"{'plan':<34} {'gpu':<14} {'VRAM':>6} {'$/hr':>7}  regions")
    for plan in plans:
        if plan.get("gpu_vram_gb", 0) < args.min_vram:
            continue
        regions = ",".join(plan.get("locations", [])) or "unavailable"
        print(f"{plan['id']:<34} {plan['gpu_type']:<14} "
              f"{plan['gpu_vram_gb']:>4}GB {plan['hourly_cost']:>7.3f}  {regions}")


def bootstrap(state):
    print("bootstrapping Python environment...")
    run_remote(
        state,
        "cloud-init status --wait && export DEBIAN_FRONTEND=noninteractive && "
        "apt-get update -qq && apt-get install -y -qq python3-venv rsync && "
        "python3 -m venv /opt/myowntransformer && "
        "/opt/myowntransformer/bin/pip install -q --upgrade pip && "
        "/opt/myowntransformer/bin/pip install -q torch==2.11.0 datasets matplotlib && "
        "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader && "
        "/opt/myowntransformer/bin/python -c \"import torch; assert torch.cuda.is_available(); "
        "print('torch', torch.__version__, 'CUDA ready')\"",
    )


def _delete_resource(api, path, retries=3):
    for attempt in range(retries):
        try:
            api.request("DELETE", path)
            return
        except RuntimeError as error:
            if "HTTP 404" in str(error):
                return
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)


def _confirm_instance_deleted(api, instance_id, attempts=12):
    for _ in range(attempts):
        try:
            api.request("GET", f"/instances/{instance_id}")
        except RuntimeError as error:
            if "HTTP 404" in str(error):
                return
            raise
        time.sleep(2)
    raise RuntimeError(f"instance {instance_id} still exists after delete")


def provision(args, bootstrap_instance=True):
    api = client_from_env()
    if getattr(args, "compute", False):
        plan, region = select_compute_plan(list_plans("vc2", api), region=args.region)
    else:
        plan, region = select_plan(
            list_gpu_plans(api), min_vram=args.min_vram, plan_id=args.plan, region=args.region
        )
    public_key = os.path.expanduser(args.ssh_public_key)
    key_id, key_created = ensure_ssh_key(api, public_key)
    state = None
    try:
        print(f"creating {plan['id']} in {region} (${plan['hourly_cost']:.3f}/hr)...")
        result = api.request("POST", "/instances", {
            "region": region,
            "plan": plan["id"],
            "os_id": args.os_id,
            "label": args.label,
            "hostname": args.label,
            "sshkey_id": [key_id],
            "activation_email": False,
        })
        instance_id = result["instance"]["id"]
        state = {
            "id": instance_id, "ssh_host": "", "ssh_private_key": args.ssh_private_key,
            "plan": plan["id"], "region": region,
            "hourly_cost": plan["hourly_cost"], "gpu": plan.get("gpu_type", "CPU"),
            "ssh_key_id": key_id, "ssh_key_created": key_created,
        }
        save_state(state)
        state.update(wait_ready(api, instance_id, args.ssh_private_key))
        save_state(state)
        if bootstrap_instance:
            bootstrap(state)
        print(f"instance ready: {state['ssh_host']} ({state['gpu']})")
        return api, state
    except BaseException:
        try:
            if state:
                destroy_state(api, state)
            elif key_created:
                _delete_resource(api, f"/ssh-keys/{key_id}")
        except Exception as cleanup_error:
            resource = state["id"] if state else key_id
            print(f"WARNING: cleanup incomplete for {resource}: {cleanup_error}", file=sys.stderr)
        raise


def destroy_state(api, state):
    errors = []
    try:
        _delete_resource(api, f"/instances/{state['id']}")
        _confirm_instance_deleted(api, state["id"])
    except Exception as error:
        errors.append(error)
    try:
        if state.get("ssh_key_created"):
            _delete_resource(api, f"/ssh-keys/{state['ssh_key_id']}")
    except Exception as error:
        errors.append(error)
    if errors:
        raise RuntimeError(f"cleanup incomplete for instance {state['id']}: {errors}")
    clear_state()
    print(f"destroyed instance {state['id']}; billing stopped")


def destroy(args):
    api = client_from_env()
    state = load_state()
    destroy_state(api, state)


def status(args):
    api = client_from_env()
    state = load_state()
    info = api.request("GET", f"/instances/{state['id']}")["instance"]
    print(f"instance {state['id']}: {info['status']} / {info['power_status']} / "
          f"{info['server_status']} | {state['plan']} @ ${state['hourly_cost']:.3f}/hr")
    if info["status"] == "active":
        run_remote(state, "cd /root/myowntransformer && tail -n 20 pipeline.log 2>/dev/null || true",
                   check=False)
