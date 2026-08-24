import os

from vultr.api import client_from_env, list_gpu_plans, select_plan
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
        "/opt/myowntransformer/bin/pip install -q torch==2.11.0 datasets matplotlib",
    )


def provision(args, bootstrap_instance=True):
    api = client_from_env()
    plan, region = select_plan(
        list_gpu_plans(api), min_vram=args.min_vram, plan_id=args.plan, region=args.region
    )
    public_key = os.path.expanduser(args.ssh_public_key)
    key_id, key_created = ensure_ssh_key(api, public_key)
    instance_id = None
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
        state = wait_ready(api, instance_id, args.ssh_private_key)
        state.update({
            "plan": plan["id"], "region": region,
            "hourly_cost": plan["hourly_cost"], "gpu": plan["gpu_type"],
            "ssh_key_id": key_id, "ssh_key_created": key_created,
        })
        save_state(state)
        if bootstrap_instance:
            bootstrap(state)
        print(f"instance ready: {state['ssh_host']} ({state['gpu']})")
        return api, state
    except Exception:
        if instance_id:
            api.request("DELETE", f"/instances/{instance_id}")
        if key_created:
            api.request("DELETE", f"/ssh-keys/{key_id}")
        clear_state()
        raise


def destroy_state(api, state):
    api.request("DELETE", f"/instances/{state['id']}")
    if state.get("ssh_key_created"):
        api.request("DELETE", f"/ssh-keys/{state['ssh_key_id']}")
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
