import io
import json
import urllib.error

import pytest

from vultr.api import VultrAPI, select_plan


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


def test_select_plan_honors_exact_plan_and_region():
    plan, region = select_plan(plans(), plan_id="cheap", region="ord")
    assert (plan["id"], region) == ("cheap", "ord")


def test_select_plan_rejects_unavailable_capacity():
    with pytest.raises(RuntimeError, match="no on-demand"):
        select_plan(plans(), plan_id="sold-out")


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


def test_authenticated_request_requires_key():
    with pytest.raises(RuntimeError, match="VULTR_API_KEY"):
        VultrAPI().request("GET", "/instances")
