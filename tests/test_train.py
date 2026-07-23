import pytest
import torch

from train import EarlyStopper, build_vocab_indices, lr_factor, resolve_device


def test_resolve_device_prefers_cpu_when_disabled(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device(no_cuda=True).type == "cpu"


def test_resolve_device_falls_back_to_cpu_when_unavailable(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device(no_cuda=False).type == "cpu"


def test_build_vocab_indices_uses_target_device():
    indices = build_vocab_indices(6, torch.device("cpu"))
    assert indices.device.type == "cpu"
    assert torch.equal(indices, torch.arange(6))


def test_lr_factor_warmup_grows_linearly():
    assert lr_factor(0, 1000, warmup_steps=100) == pytest.approx(0.01)
    assert lr_factor(49, 1000, warmup_steps=100) == pytest.approx(0.5)
    assert lr_factor(99, 1000, warmup_steps=100) == pytest.approx(1.0)


def test_lr_factor_post_warmup_starts_at_peak():
    # First post-warmup step should be at peak (cosine(0) = 1)
    assert lr_factor(100, 1000, warmup_steps=100) == pytest.approx(1.0)


def test_lr_factor_decays_toward_min_ratio():
    # Final step should be near min_lr_ratio
    assert lr_factor(999, 1000, warmup_steps=100, min_lr_ratio=0.1) == pytest.approx(0.1, abs=0.01)


def test_lr_factor_never_below_min_ratio():
    # Clamped past total_steps
    assert lr_factor(5000, 1000, warmup_steps=100, min_lr_ratio=0.1) == pytest.approx(0.1)


def test_early_stopper_first_eval_is_improvement():
    s = EarlyStopper(patience=2)
    assert s.check(3.0) is True
    assert s.best == 3.0 and not s.triggered


def test_early_stopper_improvement_resets_patience():
    s = EarlyStopper(patience=2, min_delta=0.01)
    s.check(3.0)
    assert s.check(3.05) is False      # worse: bad eval 1
    assert s.bad_evals == 1
    assert s.check(2.9) is True        # >1% below best: resets
    assert s.bad_evals == 0 and not s.triggered


def test_early_stopper_triggers_after_patience():
    s = EarlyStopper(patience=3, min_delta=0.005)
    s.check(3.0)
    assert not s.check(3.01)
    assert not s.check(3.02)
    assert not s.check(3.0)            # third stagnant eval -> trigger
    assert s.triggered


def test_early_stopper_delta_is_relative():
    s = EarlyStopper(patience=1, min_delta=0.01)
    s.check(4.0)
    # 3.99 is only 0.25% below best — under the 1% bar, patience 1 -> trigger
    assert s.check(3.99) is False
    assert s.triggered


def test_run_lr_cooldown_anneals_to_zero():
    import argparse
    import numpy as np
    from model import Transformer
    from train import make_optimizers, run_lr_cooldown
    torch.manual_seed(0)
    V = 32
    opt = argparse.Namespace(
        train=np.random.randint(0, V, 4096, dtype=np.uint16),
        batchsize=4, seqlen=16, device=torch.device('cpu'), vocab_size=V,
        norm=2.0, printevery=1)
    model = Transformer(V, 32, 2, 4, 0.0)
    opt.optimizers = make_optimizers(model, muon_lr=0.01, adamw_lr=1e-3)
    before = [p.clone() for p in model.parameters()]
    run_lr_cooldown(model, opt, grad_accum=1, cooldown_steps=3)
    assert all(g['lr'] == 0.0 for o in opt.optimizers for g in o.param_groups)
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))
