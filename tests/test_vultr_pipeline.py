from types import SimpleNamespace

import pytest

from vultr.pipeline import build_pipeline
from vultr import smoke as smoke_module


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
        keep=keep, out=str(tmp_path), min_vram=99, label="old",
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


def test_smoke_destroys_instance_when_bootstrap_fails(monkeypatch, tmp_path):
    state = {"id": "instance-1", "hourly_cost": 0.059}
    destroyed = []
    monkeypatch.setattr(smoke_module, "provision", lambda *args, **kwargs: ("api", state))
    monkeypatch.setattr(
        smoke_module, "_bootstrap", lambda unused: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(smoke_module, "destroy_state", lambda api, current: destroyed.append(current))

    with pytest.raises(RuntimeError, match="boom"):
        smoke_module.smoke(smoke_args(tmp_path))

    assert destroyed == [state]


def test_smoke_keep_flag_skips_cleanup(monkeypatch, tmp_path):
    state = {"id": "instance-1", "hourly_cost": 0.059}
    monkeypatch.setattr(smoke_module, "provision", lambda *args, **kwargs: ("api", state))
    monkeypatch.setattr(
        smoke_module, "_bootstrap", lambda unused: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    monkeypatch.setattr(
        smoke_module, "destroy_state",
        lambda *unused: pytest.fail("--keep must not destroy the instance"),
    )
    with pytest.raises(RuntimeError, match="boom"):
        smoke_module.smoke(smoke_args(tmp_path, keep=True))
