from types import SimpleNamespace

import pytest

from vultr import lifecycle


PLAN = {
    "id": "vcg-test", "hourly_cost": 0.059, "gpu_vram_gb": 2,
    "gpu_type": "NVIDIA_A16", "deploy_ondemand": True, "locations": ["ewr"],
}


class API:
    def __init__(self):
        self.calls = []

    def request(self, method, path, payload=None, auth=True):
        self.calls.append((method, path, payload))
        if (method, path) == ("POST", "/instances"):
            return {"instance": {"id": "instance-1"}}
        if method == "GET" and path.startswith("/instances/"):
            raise RuntimeError("HTTP 404: gone")
        return {}


def args():
    return SimpleNamespace(
        min_vram=2, plan=None, region=None, os_id=2284, label="smoke",
        ssh_public_key="~/.ssh/id_ed25519.pub", ssh_private_key="~/.ssh/id_ed25519",
    )


def arrange(monkeypatch, api, wait_result=None):
    monkeypatch.setattr(lifecycle, "client_from_env", lambda: api)
    monkeypatch.setattr(lifecycle, "list_gpu_plans", lambda client: [PLAN])
    monkeypatch.setattr(lifecycle, "ensure_ssh_key", lambda *unused: ("key-1", True))
    monkeypatch.setattr(lifecycle, "wait_ready", lambda *unused: wait_result or {
        "id": "instance-1", "ssh_host": "192.0.2.1", "ssh_private_key": "key-path",
    })
    monkeypatch.setattr(lifecycle, "save_state", lambda state: None)
    monkeypatch.setattr(lifecycle, "clear_state", lambda: None)


def test_provision_passes_ephemeral_key_and_records_cost(monkeypatch):
    api = API()
    arrange(monkeypatch, api)
    monkeypatch.setattr(lifecycle, "bootstrap", lambda state: None)

    _, state = lifecycle.provision(args())

    create = next(call for call in api.calls if call[:2] == ("POST", "/instances"))
    assert create[2]["sshkey_id"] == ["key-1"]
    assert create[2]["activation_email"] is False
    assert state["hourly_cost"] == 0.059
    assert state["ssh_key_created"] is True


def test_provision_cleans_up_instance_and_key_when_readiness_fails(monkeypatch):
    api = API()
    arrange(monkeypatch, api)
    monkeypatch.setattr(
        lifecycle, "wait_ready", lambda *unused: (_ for _ in ()).throw(RuntimeError("timeout"))
    )

    with pytest.raises(RuntimeError, match="timeout"):
        lifecycle.provision(args())

    assert ("DELETE", "/instances/instance-1", None) in api.calls
    assert ("DELETE", "/ssh-keys/key-1", None) in api.calls


def test_destroy_removes_only_pipeline_owned_key(monkeypatch):
    api = API()
    monkeypatch.setattr(lifecycle, "clear_state", lambda: None)
    state = {"id": "instance-1", "ssh_key_id": "key-1", "ssh_key_created": True}
    lifecycle.destroy_state(api, state)
    assert api.calls == [
        ("DELETE", "/instances/instance-1", None),
        ("GET", "/instances/instance-1", None),
        ("DELETE", "/ssh-keys/key-1", None),
    ]

    api.calls.clear()
    lifecycle.destroy_state(api, {**state, "ssh_key_created": False})
    assert api.calls == [
        ("DELETE", "/instances/instance-1", None),
        ("GET", "/instances/instance-1", None),
    ]
