import io
import json
import urllib.error

from types import SimpleNamespace

import pytest

from vultr.api import VultrAPI, per_device_vram, select_compute_plan, select_live_plan, select_plan


class Response:
    def __init__(self, body=b"{}"):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def read(self):
        return self.body


def plans():
    return [
        {"id": "cheap", "hourly_cost": 0.06, "gpu_vram_gb": 2,
         "deploy_ondemand": True, "locations": ["ewr", "ord"]},
        {"id": "roomy", "hourly_cost": 0.90, "gpu_vram_gb": 32,
         "deploy_ondemand": True, "locations": ["sgp"]},
        {"id": "sold-out", "hourly_cost": 0.50, "gpu_vram_gb": 24,
         "deploy_ondemand": True, "locations": []},
    ]


def test_select_plan_chooses_cheapest_available_match():
    plan, region = select_plan(plans(), min_vram=20)
    assert (plan["id"], region) == ("roomy", "sgp")


def test_select_plan_applies_vram_floor_per_device():
    multi_gpu = {
        "id": "two-small", "hourly_cost": 0.50, "gpu_vram_gb": 32, "gpu_count": 2,
        "deploy_ondemand": True, "locations": ["ewr"],
    }
    plan, _ = select_plan([multi_gpu, *plans()], min_vram=20)
    assert plan["id"] == "roomy"
    assert per_device_vram(multi_gpu) == 16


def test_select_plan_honors_exact_plan_and_region():
    plan, region = select_plan(plans(), plan_id="cheap", region="ord")
    assert (plan["id"], region) == ("cheap", "ord")


def test_select_plan_rejects_unavailable_capacity():
    with pytest.raises(RuntimeError, match="no on-demand"):
        select_plan(plans(), plan_id="sold-out")


def test_select_compute_plan_skips_too_small_instances():
    compute = [
        {"id": "tiny", "hourly_cost": 0.005, "ram": 512,
         "deploy_ondemand": True, "locations": ["ewr"]},
        {"id": "viable", "hourly_cost": 0.007, "ram": 1024,
         "deploy_ondemand": True, "locations": ["ewr"]},
    ]
    plan, region = select_compute_plan(compute)
    assert (plan["id"], region) == ("viable", "ewr")


def test_select_live_plan_skips_regions_without_capacity():
    class Client:
        def request(self, method, path, auth=True):
            available = ["cheap"] if "/ord/" in path else []
            return {"available_plans": available}

    plan, region = select_live_plan(
        Client(), plans(), "vcg", lambda items: select_plan(items, plan_id="cheap")
    )
    assert (plan["id"], region) == ("cheap", "ord")


def test_request_sends_bearer_token_and_json(monkeypatch):
    captured = {}

    def open_request(request, timeout):
        captured.update(request=request, timeout=timeout)
        return Response(json.dumps({"ok": True}).encode())

    monkeypatch.setattr("urllib.request.urlopen", open_request)
    result = VultrAPI("secret").request("POST", "/instances", {"plan": "cheap"})
    request = captured["request"]
    assert result == {"ok": True}
    assert request.get_header("Authorization") == "Bearer secret"
    assert json.loads(request.data) == {"plan": "cheap"}
    assert captured["timeout"] == 30


def test_request_surfaces_api_error_without_token(monkeypatch):
    error = urllib.error.HTTPError(
        "url", 403, "Forbidden", {}, io.BytesIO(b'{"error":"denied"}')
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(RuntimeError, match='HTTP 403.*denied'):
        VultrAPI("secret").request("GET", "/instances")


def test_request_normalizes_transport_errors_for_cleanup_retries(monkeypatch):
    error = urllib.error.URLError("temporary failure")
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(error))
    with pytest.raises(RuntimeError, match="transport error.*temporary failure"):
        VultrAPI("secret").request("DELETE", "/instances/test")


def test_authenticated_request_requires_key():
    with pytest.raises(RuntimeError, match="VULTR_API_KEY"):
        VultrAPI().request("GET", "/instances")


def test_plan_selection_accepts_a_preemptible_only_plan():
    """The 8x A100 box has deploy_ondemand false; filtering on that alone
    would hide the only multi-GPU plan Vultr actually rents."""
    from vultr.api import hourly_cost, is_preemptible_only, select_plan
    plan = {"id": "vbm-112c-2048gb-8-a100-gpu", "deploy_ondemand": False,
            "deploy_preemptible": True, "hourly_cost": 22.4,
            "hourly_cost_preemptible": 11.92, "gpu_count": 8,
            "gpu_vram_gb": 640, "locations": ["fra"]}
    chosen, region = select_plan([plan], min_vram=40)
    assert chosen["id"] == plan["id"] and region == "fra"
    assert hourly_cost(plan) == 11.92
    assert is_preemptible_only(plan)


def test_hourly_cost_prefers_the_on_demand_rate_when_deployable():
    from vultr.api import hourly_cost
    plan = {"deploy_ondemand": True, "hourly_cost": 1.671,
            "hourly_cost_preemptible": 0.8}
    assert hourly_cost(plan) == 1.671


def test_cheapest_plan_is_ranked_by_the_rate_actually_billed():
    from vultr.api import select_plan
    cheap_preemptible = {"id": "metal", "deploy_ondemand": False,
                         "deploy_preemptible": True, "hourly_cost": 22.4,
                         "hourly_cost_preemptible": 11.92, "gpu_count": 8,
                         "gpu_vram_gb": 640, "locations": ["fra"]}
    dearer_ondemand = {"id": "cloud", "deploy_ondemand": True,
                       "hourly_cost": 13.368, "gpu_count": 8,
                       "gpu_vram_gb": 384, "locations": ["fra"]}
    chosen, _ = select_plan([cheap_preemptible, dearer_ondemand], min_vram=40)
    assert chosen["id"] == "metal"


def test_metal_and_instance_kinds_use_distinct_endpoints():
    from vultr.api import INSTANCE, KINDS, METAL
    assert METAL.path == "/bare-metals" and METAL.item == "bare_metal"
    assert INSTANCE.path == "/instances" and INSTANCE.item == "instance"
    assert set(KINDS) == {"instance", "metal"}


def test_state_kind_defaults_to_instance_for_pre_metal_state_files():
    from vultr.api import INSTANCE, METAL
    from vultr.lifecycle import state_kind
    assert state_kind({"id": "x"}) is INSTANCE
    assert state_kind({"id": "x", "kind": "metal"}) is METAL
    assert state_kind(None) is INSTANCE


def test_bare_metal_readiness_uses_only_the_field_it_reports():
    """Bare metal has no power_status/server_status. Matching the instance
    tuple against it would never succeed and wait_ready would hang."""
    from vultr.api import INSTANCE, METAL
    assert METAL.ready == (("status", "active"),)
    assert dict(INSTANCE.ready)["power_status"] == "running"


def test_wait_ready_accepts_a_bare_metal_that_reports_status_alone(monkeypatch):
    from vultr import remote
    from vultr.api import METAL
    info = {"bare_metal": {"status": "active", "main_ip": "10.0.0.1"}}
    monkeypatch.setattr(remote.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0))
    api = SimpleNamespace(request=lambda method, path: info)
    state = remote.wait_ready(api, "metal-1", "key", timeout=5, kind=METAL)
    assert state["ssh_host"] == "10.0.0.1"


def test_wait_ready_still_requires_all_three_fields_for_an_instance(monkeypatch):
    from vultr import remote
    from vultr.api import INSTANCE
    half_up = {"instance": {"status": "active", "power_status": "stopped",
                            "server_status": "ok", "main_ip": "10.0.0.1"}}
    monkeypatch.setattr(remote.subprocess, "run",
                        lambda *a, **k: SimpleNamespace(returncode=0))
    monkeypatch.setattr(remote.time, "sleep", lambda _: None)
    api = SimpleNamespace(request=lambda method, path: half_up)
    with pytest.raises(RuntimeError, match="not SSH-ready"):
        remote.wait_ready(api, "i-1", "key", timeout=1, kind=INSTANCE)
