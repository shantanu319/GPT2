import os
import sys
import time
import uuid

from vultr.api import (
    client_from_env, list_gpu_plans, list_plans, select_compute_plan, select_live_plan,
    select_plan,
)
from vultr.remote import (
    claim_state, clear_state, ensure_ssh_key, load_state, run_remote, save_state, wait_ready,
)


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


def _find_instance_by_label(api, label, attempts=3):
    for attempt in range(attempts):
        instances = api.request("GET", "/instances?per_page=500").get("instances", [])
        match = next((item for item in instances if item.get("label") == label), None)
        if match:
            return match
        if attempt < attempts - 1:
            time.sleep(2 ** attempt)
    return None


def _resolved_state(provisional, instance):
    state = {
        **provisional, "id": instance["id"], "ssh_host": instance.get("main_ip", ""),
    }
    state.pop("status", None)
    return state


def _definitive_rejection(error):
    return any(f"HTTP {code}" in str(error) for code in (400, 401, 403, 404, 405, 409, 422))


def _resolve_tracked(api, tracked):
    if not tracked or "id" in tracked or not tracked.get("label"):
        return tracked
    recovered = _find_instance_by_label(api, tracked["label"])
    if not recovered:
        return tracked
    state = _resolved_state(tracked, recovered)
    save_state(state)
    return state


def provision(args, bootstrap_instance=True):
    api = client_from_env()
    if getattr(args, "compute", False):
        plan, region = select_live_plan(
            api, list_plans("vc2", api), "vc2",
            lambda plans: select_compute_plan(plans, region=args.region),
        )
    else:
        plan, region = select_live_plan(
            api, list_gpu_plans(api), "vcg",
            lambda plans: select_plan(
                plans, min_vram=args.min_vram, plan_id=args.plan, region=args.region
            ),
        )
    key_id = None
    key_created = False
    claimed = False
    provisional = None
    state = None
    try:
        public_key = os.path.expanduser(args.ssh_public_key)
        key_id, key_created = ensure_ssh_key(api, public_key)
        claim_state()
        claimed = True
        request_label = f"{args.label}-{uuid.uuid4().hex[:8]}"
        provisional = {
            "status": "provisioning", "label": request_label,
            "ssh_private_key": args.ssh_private_key, "plan": plan["id"], "region": region,
            "hourly_cost": plan["hourly_cost"], "gpu": plan.get("gpu_type", "CPU"),
            "ssh_key_id": key_id, "ssh_key_created": key_created,
        }
        save_state(provisional)
        print(f"creating {plan['id']} in {region} (${plan['hourly_cost']:.3f}/hr)...")
        result = api.request("POST", "/instances", {
            "region": region,
            "plan": plan["id"],
            "os_id": args.os_id,
            "label": request_label,
            "hostname": args.label,
            "sshkey_id": [key_id],
            "activation_email": False,
        })
        instance_id = result["instance"]["id"]
        state = _resolved_state(provisional, {"id": instance_id})
        save_state(state)
        state.update(wait_ready(api, instance_id, args.ssh_private_key))
        save_state(state)
        if bootstrap_instance:
            bootstrap(state)
        print(f"instance ready: {state['ssh_host']} ({state['gpu']})")
        return api, state
    except BaseException as provision_error:
        try:
            if state:
                destroy_state(api, state)
            elif claimed and _definitive_rejection(provision_error):
                try:
                    if key_created:
                        _delete_resource(api, f"/ssh-keys/{key_id}")
                finally:
                    clear_state()
            elif claimed:
                recovered = _find_instance_by_label(api, provisional["label"])
                if recovered:
                    state = _resolved_state(provisional, recovered)
                    save_state(state)
                    destroy_state(api, state)
                else:
                    print(f"WARNING: create outcome unresolved; kept {provisional['label']} in "
                          ".vultr_instance.json", file=sys.stderr)
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
    tracked = load_state(required=False)
    if tracked and tracked.get("id") == state["id"]:
        clear_state()
    print(f"destroyed instance {state['id']}; billing stopped")


def destroy(args):
    api = client_from_env()
    tracked = load_state(required=False)
    instance_id = getattr(args, "id", None)
    if not instance_id:
        tracked = _resolve_tracked(api, tracked)
    if instance_id and (not tracked or tracked.get("id") != instance_id):
        state = {"id": instance_id}
    else:
        state = tracked
    if not state or "id" not in state:
        raise RuntimeError("no tracked instance; pass --id for recovery")
    destroy_state(api, state)


def status(args):
    api = client_from_env()
    tracked = load_state(required=False)
    requested_id = getattr(args, "id", None)
    if not requested_id:
        tracked = _resolve_tracked(api, tracked)
    instance_id = requested_id or (tracked or {}).get("id")
    if not instance_id:
        raise RuntimeError("no tracked instance; pass --id for recovery")
    state = tracked if tracked and tracked.get("id") == instance_id else None
    info = api.request("GET", f"/instances/{instance_id}")["instance"]
    plan = (state or {}).get("plan", info.get("plan", "unknown plan"))
    rate = f" @ ${state['hourly_cost']:.3f}/hr" if state else ""
    print(f"instance {instance_id}: {info['status']} / {info['power_status']} / "
          f"{info['server_status']} | {plan}{rate}")
    if info["status"] == "active" and state and state.get("ssh_host"):
        run_remote(state, "cd /root/myowntransformer && tail -n 20 pipeline.log 2>/dev/null || true",
                   check=False)
