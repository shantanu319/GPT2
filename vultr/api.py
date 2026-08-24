import json
import os
import urllib.error
import urllib.parse
import urllib.request


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


def list_plans(plan_type, client=None):
    client = client or client_from_env(required=False)
    result = client.request("GET", f"/plans?type={plan_type}&per_page=500", auth=False)
    return result["plans"]


def list_gpu_plans(client=None):
    return list_plans("vcg", client)


def select_plan(plans, min_vram=20, plan_id=None, region=None):
    candidates = [plan for plan in plans if plan.get("deploy_ondemand")]
    if plan_id:
        candidates = [plan for plan in candidates if plan["id"] == plan_id]
    else:
        candidates = [plan for plan in candidates if plan.get("gpu_vram_gb", 0) >= min_vram]
    if region:
        candidates = [plan for plan in candidates if region in plan.get("locations", [])]
    else:
        candidates = [plan for plan in candidates if plan.get("locations")]
    if not candidates:
        target = plan_id or f">={min_vram} GB VRAM"
        where = f" in {region}" if region else ""
        raise RuntimeError(f"no on-demand Vultr GPU plan for {target}{where}")
    plan = min(candidates, key=lambda item: (item["hourly_cost"], item["id"]))
    return plan, region or plan["locations"][0]


def select_compute_plan(plans, min_ram=1024, region=None):
    candidates = [
        plan for plan in plans
        if plan.get("deploy_ondemand") and plan.get("ram", 0) >= min_ram
        and plan.get("locations") and (not region or region in plan["locations"])
    ]
    if not candidates:
        where = f" in {region}" if region else ""
        raise RuntimeError(f"no on-demand Vultr compute plan with >={min_ram} MB RAM{where}")
    plan = min(candidates, key=lambda item: (item["hourly_cost"], item["id"]))
    return plan, region or plan["locations"][0]
