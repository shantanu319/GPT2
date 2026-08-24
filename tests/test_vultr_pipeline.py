from types import SimpleNamespace

import pytest

from vultr.pipeline import build_pipeline
from vultr import jobs, smoke as smoke_module
from vultr import remote
from vultr.remote import ssh_prefix


def pipeline_args(**overrides):
    values = {
        "max_train_docs": 10, "d_model": 64, "n_layers": 2, "heads": 2,
        "kv_heads": 1, "batchsize": 4, "seqlen": 64, "epochs": 1,
        "warmup_steps": 2, "save_every": 0, "val_every": 0,
        "dir_name": "base", "sft_dir_name": "chat", "dpo_dir_name": "preference",
        "sft_epochs": 1, "dpo_epochs": 1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def smoke_args(tmp_path, keep=False):
    return SimpleNamespace(
        keep=keep, out=str(tmp_path), min_vram=99, label="old", compute=False,
        plan=None, region=None, os_id=2284,
        ssh_public_key="public", ssh_private_key="private",
    )


def test_pipeline_is_resumable_across_all_training_stages():
    script = build_pipeline(pipeline_args())
    assert '[[ ! -f "$DATA/train.bin" ]]' in script
    assert '[[ ! -f "saved/$DIR/ckpt_final.pt" ]]' in script
    assert '[[ ! -f "$DATA/sft_train.bin" ]]' in script
    assert '[[ ! -f "saved/$SFT/sft_final.pt" ]]' in script
    assert '[[ ! -f "$DATA/dpo_train.bin" ]]' in script
    assert '[[ ! -f "saved/$DPO/dpo_final.pt" ]]' in script
    assert "PIPELINE COMPLETE" in script


def test_pipeline_quotes_run_names():
    script = build_pipeline(pipeline_args(dir_name="run; touch /tmp/bad"))
    assert "DIR='run; touch /tmp/bad'" in script


def test_ssh_is_noninteractive_and_preserves_watcher_command(monkeypatch):
    state = {"ssh_private_key": "key", "ssh_host": "host"}
    captured = []
    monkeypatch.setattr(jobs, "load_state", lambda: state)
    monkeypatch.setattr(
        jobs, "run_remote",
        lambda current, command, check: captured.append(command) or SimpleNamespace(returncode=0),
    )
    with pytest.raises(SystemExit) as exit_info:
        jobs.ssh(SimpleNamespace(command=["test -f /root/checkpoint"]))
    assert exit_info.value.code == 0
    assert captured == ["test -f /root/checkpoint"]
    assert "BatchMode=yes" in ssh_prefix(state)


def test_state_claim_refuses_to_overwrite_an_active_instance(monkeypatch, tmp_path):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(remote, "STATE_FILE", str(state_file))
    remote.claim_state()
    assert remote.load_state() == {"status": "provisioning"}
    with pytest.raises(RuntimeError, match="already tracks provisioning"):
        remote.claim_state()


def test_interrupted_state_write_preserves_previous_record(monkeypatch, tmp_path):
    state_file = tmp_path / "state.json"
    monkeypatch.setattr(remote, "STATE_FILE", str(state_file))
    remote.save_state({"status": "provisioning", "label": "recover-me"})
    monkeypatch.setattr(
        remote.json, "dump", lambda *unused, **kwargs: (_ for _ in ()).throw(OSError("disk full"))
    )
    with pytest.raises(OSError, match="disk full"):
        remote.save_state({"id": "would-be-lost"})
    assert remote.load_state() == {"status": "provisioning", "label": "recover-me"}
    assert list(tmp_path.iterdir()) == [state_file]


def test_smoke_destroys_instance_when_bootstrap_fails(monkeypatch, tmp_path):
    state = {"id": "instance-1", "hourly_cost": 0.059}
    destroyed = []
    monkeypatch.setattr(smoke_module, "provision", lambda *args, **kwargs: ("api", state))
    monkeypatch.setattr(
        smoke_module, "_bootstrap", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(smoke_module, "destroy_state", lambda api, current: destroyed.append(current))

    with pytest.raises(RuntimeError, match="boom"):
        smoke_module.smoke(smoke_args(tmp_path))

    assert destroyed == [state]


def test_smoke_keep_flag_skips_cleanup(monkeypatch, tmp_path):
    state = {"id": "instance-1", "hourly_cost": 0.059}
    monkeypatch.setattr(smoke_module, "provision", lambda *args, **kwargs: ("api", state))
    monkeypatch.setattr(
        smoke_module, "_bootstrap", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(
        smoke_module, "destroy_state",
        lambda *unused: pytest.fail("--keep must not destroy the instance"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        smoke_module.smoke(smoke_args(tmp_path, keep=True))


def test_smoke_honors_an_explicit_vram_floor(monkeypatch, tmp_path):
    captured = []
    state = {"id": "instance-1", "hourly_cost": 0.059}
    monkeypatch.setattr(
        smoke_module, "provision",
        lambda args, **kwargs: captured.append(args.min_vram) or ("api", state),
    )
    monkeypatch.setattr(
        smoke_module, "_bootstrap",
        lambda *unused, **kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    monkeypatch.setattr(smoke_module, "destroy_state", lambda *unused: None)
    with pytest.raises(RuntimeError, match="stop"):
        smoke_module.smoke(smoke_args(tmp_path))
    assert captured == [99]


def test_smoke_does_not_accept_a_stale_local_checkpoint(monkeypatch, tmp_path):
    checkpoint = tmp_path / "smoke" / "ckpt_final.pt"
    checkpoint.parent.mkdir()
    checkpoint.write_bytes(b"old checkpoint")
    state = {"id": "instance-1", "hourly_cost": 0.059, "ssh_host": "host"}
    monkeypatch.setattr(smoke_module, "provision", lambda *args, **kwargs: ("api", state))
    monkeypatch.setattr(smoke_module, "_bootstrap", lambda *args, **kwargs: None)
    monkeypatch.setattr(smoke_module, "run_remote", lambda *args, **kwargs: None)
    monkeypatch.setattr(smoke_module, "rsync", lambda *args, **kwargs: None)
    monkeypatch.setattr(smoke_module, "_make_data", lambda *args: None)
    monkeypatch.setattr(smoke_module, "destroy_state", lambda *args: None)
    with pytest.raises(RuntimeError, match="missing or empty"):
        smoke_module.smoke(smoke_args(tmp_path))
    assert not checkpoint.exists()
