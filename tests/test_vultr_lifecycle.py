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
        if method == "GET" and path.startswith("/regions/"):
            return {"available_plans": ["vcg-test"]}
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
    monkeypatch.setattr(lifecycle, "claim_state", lambda: None)
    monkeypatch.setattr(lifecycle, "wait_ready", lambda *unused, **kw: wait_result or {
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


def test_gpu_bootstrap_fails_closed_without_cuda(monkeypatch):
    commands = []
    monkeypatch.setattr(lifecycle, "run_remote", lambda state, command: commands.append(command))
    lifecycle.bootstrap({})
    assert "nvidia-smi" in commands[0]
    assert "assert torch.cuda.is_available()" in commands[0]


def test_provision_cleans_up_instance_and_key_when_readiness_fails(monkeypatch):
    api = API()
    arrange(monkeypatch, api)
    monkeypatch.setattr(
        lifecycle, "wait_ready", lambda *unused, **kw: (_ for _ in ()).throw(RuntimeError("timeout"))
    )

    with pytest.raises(RuntimeError, match="timeout"):
        lifecycle.provision(args())

    assert ("DELETE", "/instances/instance-1", None) in api.calls
    assert ("DELETE", "/ssh-keys/key-1", None) in api.calls


def test_refused_second_provision_preserves_tracked_state(monkeypatch):
    api = API()
    arrange(monkeypatch, api)
    cleared = []
    monkeypatch.setattr(
        lifecycle, "claim_state", lambda: (_ for _ in ()).throw(RuntimeError("already tracks live"))
    )
    monkeypatch.setattr(lifecycle, "clear_state", lambda: cleared.append(True))

    with pytest.raises(RuntimeError, match="already tracks live"):
        lifecycle.provision(args())

    assert not cleared
    assert ("DELETE", "/ssh-keys/key-1", None) in api.calls


def test_ambiguous_create_response_recovers_instance_by_unique_label(monkeypatch):
    class AmbiguousAPI(API):
        label = None

        def request(self, method, path, payload=None, auth=True):
            if (method, path) == ("POST", "/instances"):
                self.calls.append((method, path, payload))
                self.label = payload["label"]
                raise RuntimeError("transport error: response lost")
            if (method, path) == ("GET", "/instances?per_page=500"):
                self.calls.append((method, path, payload))
                return {"instances": [{"id": "recovered", "label": self.label}]}
            return super().request(method, path, payload, auth)

    api = AmbiguousAPI()
    arrange(monkeypatch, api)
    with pytest.raises(RuntimeError, match="response lost"):
        lifecycle.provision(args())
    assert ("DELETE", "/instances/recovered", None) in api.calls
    assert ("DELETE", "/ssh-keys/key-1", None) in api.calls


def test_unresolved_create_keeps_provisional_state_and_key(monkeypatch):
    class LostResponseAPI(API):
        def request(self, method, path, payload=None, auth=True):
            if (method, path) == ("POST", "/instances"):
                raise RuntimeError("transport error: response lost")
            if (method, path) == ("GET", "/instances?per_page=500"):
                return {"instances": []}
            return super().request(method, path, payload, auth)

    api = LostResponseAPI()
    arrange(monkeypatch, api)
    saved = []
    cleared = []
    monkeypatch.setattr(lifecycle, "save_state", lambda state: saved.append(state.copy()))
    monkeypatch.setattr(lifecycle, "clear_state", lambda: cleared.append(True))
    monkeypatch.setattr(lifecycle.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="response lost"):
        lifecycle.provision(args())
    assert saved[-1]["status"] == "provisioning"
    assert not cleared
    assert not any(call[:2] == ("DELETE", "/ssh-keys/key-1") for call in api.calls)


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


def test_destroy_by_id_preserves_a_different_tracked_instance(monkeypatch):
    api = API()
    cleared = []
    monkeypatch.setattr(lifecycle, "client_from_env", lambda: api)
    monkeypatch.setattr(lifecycle, "load_state", lambda required=False: {"id": "tracked"})
    monkeypatch.setattr(lifecycle, "clear_state", lambda: cleared.append(True))
    lifecycle.destroy(SimpleNamespace(id="recovery"))
    assert ("DELETE", "/instances/recovery", None) in api.calls
    assert not cleared


def test_destroy_by_id_cleans_up_provisional_metadata(monkeypatch):
    class RecoveryAPI(API):
        deleted = False

        def request(self, method, path, payload=None, auth=True):
            if (method, path) == ("GET", "/instances/recovery") and not self.deleted:
                self.calls.append((method, path, payload))
                return {"instance": {"id": "recovery", "label": "unique-label"}}
            if (method, path) == ("DELETE", "/instances/recovery"):
                self.deleted = True
            return super().request(method, path, payload, auth)

    api = RecoveryAPI()
    tracked = {
        "status": "provisioning", "label": "unique-label",
        "ssh_key_id": "key-1", "ssh_key_created": True,
    }
    cleared = []
    monkeypatch.setattr(lifecycle, "client_from_env", lambda: api)
    monkeypatch.setattr(lifecycle, "load_state", lambda required=False: tracked)
    monkeypatch.setattr(lifecycle, "clear_state", lambda: cleared.append(True))
    lifecycle.destroy(SimpleNamespace(id="recovery"))
    assert ("DELETE", "/instances/recovery", None) in api.calls
    assert ("DELETE", "/ssh-keys/key-1", None) in api.calls
    assert cleared


@pytest.mark.parametrize("instance_id", ["missing", "wrong-label"])
def test_destroy_by_id_rejects_unverified_provisional_match(monkeypatch, instance_id):
    class UnverifiedAPI(API):
        def request(self, method, path, payload=None, auth=True):
            if method == "GET" and path == "/instances/wrong-label":
                return {"instance": {"id": "wrong-label", "label": "somebody-else"}}
            return super().request(method, path, payload, auth)

    api = UnverifiedAPI()
    tracked = {"status": "provisioning", "label": "unique-label", "ssh_key_created": True}
    cleared = []
    monkeypatch.setattr(lifecycle, "client_from_env", lambda: api)
    monkeypatch.setattr(lifecycle, "load_state", lambda required=False: tracked)
    monkeypatch.setattr(lifecycle, "clear_state", lambda: cleared.append(True))
    with pytest.raises(RuntimeError):
        lifecycle.destroy(SimpleNamespace(id=instance_id))
    assert not any(call[0] == "DELETE" for call in api.calls)
    assert not cleared


def test_bootstrap_installs_drivers_only_when_nvidia_smi_is_missing(monkeypatch):
    """Bare metal deploys plain Ubuntu 24.04 — there is no GPU-enabled image
    in /os, and no CUDA marketplace app — so drivers may not be present."""
    captured = []
    monkeypatch.setattr(lifecycle, "run_remote",
                        lambda state, command: captured.append(command))
    lifecycle.bootstrap({"ssh_host": "h", "ssh_private_key": "k"})
    command = captured[0]
    assert "nvidia-smi >/dev/null 2>&1 ||" in command
    assert "ubuntu-drivers install --gpgpu" in command
    # the driver step must precede the CUDA assertion it exists to satisfy
    assert command.index("ubuntu-drivers") < command.index("torch.cuda.is_available")


def test_provision_explains_how_to_unlock_gpu_access(monkeypatch):
    """Every GPU product is gated behind a Vultr ticket — vcg- Cloud GPU and
    vbm-*-gpu bare metal alike — so the raw 400 needs a next step attached."""
    class Gated(API):
        def request(self, method, path, payload=None, auth=True):
            if method == "POST" and path.endswith("s"):
                raise RuntimeError(
                    f"Vultr API POST {path}: HTTP 400: " '{"error":"Server add '
                    'failed: Please open a support request for access to this '
                    'product."}')
            return super().request(method, path, payload, auth)

    arrange(monkeypatch, Gated())
    monkeypatch.setattr(lifecycle, "_delete_resource", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="my.vultr.com/support"):
        lifecycle.provision(args())


COMPUTE_CATALOG = {
    "vc2": [{"id": "vc2-1c-1gb", "deploy_ondemand": True, "ram": 1024, "disk": 25,
             "locations": ["ams"], "hourly_cost": 0.007}],
    "vhf": [{"id": "vhf-3c-8gb", "deploy_ondemand": True, "ram": 8192, "disk": 256,
             "locations": ["ams"], "hourly_cost": 0.066}],
    "vhp": [],
}


def _compute_args(**overrides):
    values = {"compute": True, "metal": False, "plan": None, "region": "ams",
              "min_vram": 0}
    values.update(overrides)
    return SimpleNamespace(**values)


def _compute_select(monkeypatch, args):
    monkeypatch.setattr(lifecycle, "client_from_env", lambda: None)
    monkeypatch.setattr(lifecycle, "list_plans",
                        lambda family, client=None: COMPUTE_CATALOG[family])
    monkeypatch.setattr(lifecycle, "select_live_plan",
                        lambda api, plans, kind, selector: selector(plans))
    return lifecycle._select(args)[2]


def test_compute_selection_respects_a_disk_floor(monkeypatch):
    """prep asked for 30 GB and got a 25 GB box, because the compute branch
    re-selected the cheapest plan and ignored the floors it was given."""
    plan = _compute_select(monkeypatch, _compute_args(min_ram=2048, min_disk=30))
    assert plan["id"] == "vhf-3c-8gb"


def test_compute_selection_still_defaults_to_the_cheapest(monkeypatch):
    assert _compute_select(monkeypatch, _compute_args())["id"] == "vc2-1c-1gb"


def test_compute_selection_honours_an_explicit_plan(monkeypatch):
    plan = _compute_select(monkeypatch, _compute_args(plan="vhf-3c-8gb"))
    assert plan["id"] == "vhf-3c-8gb"


def test_compute_selection_searches_beyond_vc2(monkeypatch):
    """vc2 alone offers no plan with real disk."""
    with pytest.raises(RuntimeError, match="no on-demand"):
        _compute_select(monkeypatch, _compute_args(min_disk=9999))
