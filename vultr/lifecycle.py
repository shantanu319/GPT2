import os
import sys
import time
import uuid

from vultr.api import (
    INSTANCE, KINDS, METAL, client_from_env, hourly_cost, is_preemptible_only,
    list_gpu_plans, list_kind_plans, list_plans, per_device_vram, select_compute_plan,
    select_live_plan, select_plan,
)
from vultr.remote import (
    claim_state, clear_state, ensure_ssh_key, load_state, run_remote, save_state, wait_ready,
)


GPU_OS_ID = 2284
# Bare metal racks and boots real hardware, so it takes far longer than a VM.
METAL_READY_TIMEOUT = 45 * 60
DEFAULT_PUBLIC_KEY = "~/.ssh/id_ed25519.pub"
DEFAULT_PRIVATE_KEY = "~/.ssh/id_ed25519"


def print_plans(args):
    """Cloud GPU and bare metal in one table. The multi-GPU A100/H100/B200
    boxes are bare metal only, which is why /plans?type=vcg never shows them."""
    plans = [(INSTANCE, plan) for plan in list_gpu_plans()]
    plans += [(METAL, plan) for plan in list_kind_plans(METAL)
              if plan.get("gpu_type")]
    plans.sort(key=lambda pair: (hourly_cost(pair[1]), pair[1]["id"]))
    print(f"{'plan':<34} {'gpu':<16} {'count':>5} {'VRAM/GPU':>9} {'$/hr':>7} "
          f"{'kind':<9} regions")
    for kind, plan in plans:
        vram = per_device_vram(plan)
        if vram < args.min_vram:
            continue
        regions = ",".join(plan.get("locations", [])) or "unavailable"
        label = kind.name + ("*" if is_preemptible_only(plan) else "")
        print(f"{plan['id']:<34} {plan['gpu_type']:<16} "
              f"{str(plan.get('gpu_count', 1)):>5} {vram:>7g}GB "
              f"{hourly_cost(plan):>7.3f} {label:<9} {regions}")
    print("* preemptible only — Vultr can reclaim the box; the rate shown is "
          "the preemptible one")


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


def _confirm_instance_deleted(api, instance_id, kind=INSTANCE, attempts=12):
    for _ in range(attempts):
        try:
            api.request("GET", f"{kind.path}/{instance_id}")
        except RuntimeError as error:
            if "HTTP 404" in str(error):
                return
            raise
        time.sleep(2)
    raise RuntimeError(f"instance {instance_id} still exists after delete")


def _find_instance_by_label(api, label, kind=INSTANCE, attempts=3):
    for attempt in range(attempts):
        instances = api.request(
            "GET", f"{kind.path}?per_page=500").get(kind.items, [])
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


def state_kind(state):
    return KINDS.get((state or {}).get("kind", INSTANCE.name), INSTANCE)


def _resolve_tracked(api, tracked):
    if not tracked or "id" in tracked or not tracked.get("label"):
        return tracked
    recovered = _find_instance_by_label(api, tracked["label"], state_kind(tracked))
    if not recovered:
        return tracked
    state = _resolved_state(tracked, recovered)
    save_state(state)
    return state


def _select(args):
    """Pick (kind, plan, region). Bare metal has no per-region availability
    endpoint, so its plan locations are taken at face value."""
    api = client_from_env()
    if getattr(args, "metal", False) or str(getattr(args, "plan", "") or "").startswith("vbm-"):
        plan, region = select_plan(
            list_kind_plans(METAL, api), min_vram=args.min_vram,
            plan_id=args.plan, region=args.region,
        )
        return api, METAL, plan, region
    if getattr(args, "compute", False):
        plan, region = select_live_plan(
            api, list_plans("vc2", api), "vc2",
            lambda plans: select_compute_plan(plans, region=args.region),
        )
        return api, INSTANCE, plan, region
    plan, region = select_live_plan(
        api, list_gpu_plans(api), "vcg",
        lambda plans: select_plan(
            plans, min_vram=args.min_vram, plan_id=args.plan, region=args.region
        ),
    )
    return api, INSTANCE, plan, region


def provision(args, bootstrap_instance=True):
    api, kind, plan, region = _select(args)
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
            "hourly_cost": hourly_cost(plan), "gpu": plan.get("gpu_type", "CPU"),
            "ssh_key_id": key_id, "ssh_key_created": key_created, "kind": kind.name,
        }
        save_state(provisional)
        note = " PREEMPTIBLE — Vultr can reclaim it" if is_preemptible_only(plan) else ""
        print(f"creating {plan['id']} in {region} "
              f"(${hourly_cost(plan):.3f}/hr){note}...")
        result = api.request("POST", kind.path, {
            "region": region,
            "plan": plan["id"],
            "os_id": args.os_id,
            "label": request_label,
            "hostname": args.label,
            "sshkey_id": [key_id],
            "activation_email": False,
        })
        instance_id = result[kind.item]["id"]
        state = _resolved_state(provisional, {"id": instance_id})
        save_state(state)
        timeout = METAL_READY_TIMEOUT if kind is METAL else 20 * 60
        state.update(wait_ready(api, instance_id, args.ssh_private_key,
                                timeout=timeout, kind=kind))
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
                recovered = _find_instance_by_label(api, provisional["label"], kind)
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
    kind = state_kind(state)
    errors = []
    try:
        _delete_resource(api, f"{kind.path}/{state['id']}")
        _confirm_instance_deleted(api, state["id"], kind)
    except Exception as error:
        errors.append(error)
    try:
        if state.get("ssh_key_created"):
            _delete_resource(api, f"/ssh-keys/{state['ssh_key_id']}")
    except Exception as error:
        errors.append(error)
    if errors:
        raise RuntimeError(f"cleanup incomplete for {kind.name} {state['id']}: {errors}")
    tracked = load_state(required=False)
    same_id = tracked and tracked.get("id") == state["id"]
    same_label = tracked and state.get("label") and tracked.get("label") == state["label"]
    if same_id or same_label:
        clear_state()
    print(f"destroyed {kind.name} {state['id']}; billing stopped")


def destroy(args):
    api = client_from_env()
    tracked = load_state(required=False)
    instance_id = getattr(args, "id", None)
    if not instance_id:
        tracked = _resolve_tracked(api, tracked)
    if instance_id and tracked and "id" not in tracked:
        kind = state_kind(tracked)
        info = api.request("GET", f"{kind.path}/{instance_id}")[kind.item]
        if info.get("label") != tracked.get("label"):
            raise RuntimeError(f"instance {instance_id} does not match tracked label")
        state = _resolved_state(tracked, info)
    elif instance_id and (not tracked or tracked.get("id") != instance_id):
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
    kind = state_kind(state or tracked)
    info = api.request("GET", f"{kind.path}/{instance_id}")[kind.item]
    plan = (state or {}).get("plan", info.get("plan", "unknown plan"))
    rate = f" @ ${state['hourly_cost']:.3f}/hr" if state else ""
    print(f"{kind.name} {instance_id}: {info['status']} / {info['power_status']} / "
          f"{info['server_status']} | {plan}{rate}")
    if info["status"] == "active" and state and state.get("ssh_host"):
        run_remote(state, "cd /root/myowntransformer && tail -n 20 pipeline.log 2>/dev/null || true",
                   check=False)
