import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from fractions import Fraction


BASE_URL = "https://api.vultr.com/v2"


class VultrAPI:
    def __init__(self, token=None):
        self.token = token or ""

    def request(self, method, path, payload=None, auth=True):
        headers = {"Accept": "application/json"}
        if auth:
            if not self.token:
                raise RuntimeError("VULTR_API_KEY is not set")
            headers["Authorization"] = f"Bearer {self.token}"
        data = None
        if payload is not None:
            data = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{BASE_URL}{path}", data=data, headers=headers, method=method
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            raise RuntimeError(f"Vultr API {method} {path}: HTTP {error.code}: {detail}") from error
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as error:
            raise RuntimeError(f"Vultr API {method} {path}: transport error: {error}") from error
        return json.loads(body) if body else {}


def client_from_env(required=True):
    token = os.environ.get("VULTR_API_KEY", "")
    if required and not token:
        raise RuntimeError(
            "VULTR_API_KEY is not set; create one at "
            "https://my.vultr.com/settings/#settingsapi"
        )
    return VultrAPI(token)


@dataclass(frozen=True)
class Kind:
    """Cloud GPU and bare metal differ only in endpoint and JSON key names."""
    name: str
    path: str
    item: str
    items: str
    plans_path: str
    plans_key: str
    # The status fields this kind reports, and the values that mean ready.
    # Bare metal returns status alone -- no power_status or server_status.
    ready: tuple


INSTANCE = Kind("instance", "/instances", "instance", "instances",
                "/plans?type=vcg&per_page=500", "plans",
                (("status", "active"), ("power_status", "running"),
                 ("server_status", "ok")))
METAL = Kind("metal", "/bare-metals", "bare_metal", "bare_metals",
             "/plans-metal?per_page=500", "plans_metal",
             (("status", "active"),))
KINDS = {kind.name: kind for kind in (INSTANCE, METAL)}


def list_kind_plans(kind, client=None):
    """The plan catalog for a resource kind. The A100 and H100 boxes live in
    /plans-metal, which /plans?type=vcg does not return."""
    client = client or client_from_env(required=False)
    return client.request("GET", kind.plans_path, auth=False)[kind.plans_key]


def deployable(plan):
    return bool(plan.get("deploy_ondemand") or plan.get("deploy_preemptible"))


def hourly_cost(plan):
    """What the plan actually bills. The 8x A100 box is preemptible-only, so
    its on-demand hourly_cost is a rate you cannot deploy at."""
    if plan.get("deploy_ondemand"):
        return plan["hourly_cost"]
    return plan.get("hourly_cost_preemptible", plan["hourly_cost"])


def is_preemptible_only(plan):
    return not plan.get("deploy_ondemand") and bool(plan.get("deploy_preemptible"))


def list_plans(plan_type, client=None):
    client = client or client_from_env(required=False)
    result = client.request("GET", f"/plans?type={plan_type}&per_page=500", auth=False)
    return result["plans"]


def list_gpu_plans(client=None):
    return list_plans("vcg", client)


def plan_type_of(plan_id):
    """vc2-6c-16gb -> vc2. The availability endpoint is queried per type, so a
    mixed candidate list has to ask about each plan under its own type."""
    return plan_id.split("-")[0]


def select_live_plan(client, plans, plan_type, selector):
    remaining = [{**plan, "locations": list(plan.get("locations", []))} for plan in plans]
    while True:
        plan, region = selector(remaining)
        kind = plan_type or plan_type_of(plan["id"])
        path = f"/regions/{urllib.parse.quote(region)}/availability?type={kind}"
        available = client.request("GET", path, auth=False).get("available_plans", [])
        if plan["id"] in available:
            return plan, region
        for candidate in remaining:
            if candidate["id"] == plan["id"]:
                candidate["locations"].remove(region)
                break


def per_device_vram(plan):
    count = Fraction(str(plan.get("gpu_count") or 1))
    total = plan.get("gpu_vram_gb", 0)
    return total / float(count) if count > 1 else total


def select_plan(plans, min_vram=20, plan_id=None, region=None):
    candidates = [plan for plan in plans if deployable(plan)]
    if plan_id:
        candidates = [plan for plan in candidates if plan["id"] == plan_id]
    else:
        candidates = [plan for plan in candidates if per_device_vram(plan) >= min_vram]
    if region:
        candidates = [plan for plan in candidates if region in plan.get("locations", [])]
    else:
        candidates = [plan for plan in candidates if plan.get("locations")]
    if not candidates:
        target = plan_id or f">={min_vram} GB VRAM"
        where = f" in {region}" if region else ""
        raise RuntimeError(f"no on-demand Vultr GPU plan for {target}{where}")
    plan = min(candidates, key=lambda item: (hourly_cost(item), item["id"]))
    return plan, region or plan["locations"][0]


def select_compute_plan(plans, min_ram=1024, region=None, min_disk=0, plan_id=None):
    candidates = [
        plan for plan in plans
        if plan.get("deploy_ondemand")
        and plan.get("locations") and (not region or region in plan["locations"])
        and (plan["id"] == plan_id if plan_id else
             plan.get("ram", 0) >= min_ram and plan.get("disk", 0) >= min_disk)
    ]
    if not candidates:
        where = f" in {region}" if region else ""
        disk = f" and >={min_disk} GB disk" if min_disk else ""
        target = plan_id or f">={min_ram} MB RAM{disk}"
        raise RuntimeError(f"no on-demand Vultr compute plan for {target}{where}")
    plan = min(candidates, key=lambda item: (item["hourly_cost"], item["id"]))
    return plan, region or plan["locations"][0]
