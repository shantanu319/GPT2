import pytest
import torch

from pretrain.train import EarlyStopper, build_vocab_indices, lr_factor, resolve_device


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


def test_lr_factor_cosine_warmup_grows_linearly():
    assert lr_factor(0, 1000, warmup_steps=100, schedule='cosine') == pytest.approx(0.01)
    assert lr_factor(49, 1000, warmup_steps=100, schedule='cosine') == pytest.approx(0.5)
    assert lr_factor(99, 1000, warmup_steps=100, schedule='cosine') == pytest.approx(1.0)


def test_lr_factor_cosine_post_warmup_starts_at_peak():
    # First post-warmup step should be at peak (cosine(0) = 1)
    assert lr_factor(100, 1000, warmup_steps=100, schedule='cosine') == pytest.approx(1.0)


def test_lr_factor_cosine_decays_toward_min_ratio():
    # Final step should be near min_lr_ratio
    assert lr_factor(999, 1000, warmup_steps=100, schedule='cosine', min_lr_ratio=0.1) == pytest.approx(0.1, abs=0.01)


def test_lr_factor_cosine_never_below_min_ratio():
    # Clamped past total_steps
    assert lr_factor(5000, 1000, warmup_steps=100, schedule='cosine', min_lr_ratio=0.1) == pytest.approx(0.1)


def test_lr_factor_wsd_warmup_grows_linearly():
    assert lr_factor(0, 1000, warmup_steps=100, schedule='wsd') == pytest.approx(0.01)
    assert lr_factor(49, 1000, warmup_steps=100, schedule='wsd') == pytest.approx(0.5)
    assert lr_factor(99, 1000, warmup_steps=100, schedule='wsd') == pytest.approx(1.0)


def test_lr_factor_wsd_stable_phase_is_peak():
    # decay_frac 0.25 -> decay starts at step 750; peak holds until then
    for s in (100, 400, 749):
        assert lr_factor(s, 1000, warmup_steps=100, schedule='wsd', decay_frac=0.25) == 1.0


def test_lr_factor_wsd_decay_endpoints():
    assert lr_factor(750, 1000, warmup_steps=100, schedule='wsd', decay_frac=0.25) == pytest.approx(1.0)
    assert lr_factor(1000, 1000, warmup_steps=100, schedule='wsd', decay_frac=0.25) == pytest.approx(0.0)
    assert lr_factor(5000, 1000, warmup_steps=100, schedule='wsd', decay_frac=0.25) == pytest.approx(0.0)


def test_lr_factor_wsd_decay_is_monotonic():
    factors = [lr_factor(s, 1000, warmup_steps=100, schedule='wsd', decay_frac=0.25)
               for s in range(750, 1001)]
    assert all(f1 >= f2 for f1, f2 in zip(factors, factors[1:]))
    assert factors[-1] < factors[0]


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
    from core.model import Transformer
    from pretrain.train import make_optimizers, run_lr_cooldown
    torch.manual_seed(0)
    V = 32
    opt = argparse.Namespace(
        train=np.random.randint(0, V, 4096, dtype=np.uint16),
        batchsize=4, seqlen=16, device=torch.device('cpu'), vocab_size=V,
        norm=2.0, printevery=1)
    model = Transformer(V, 32, 2, 4, 0.0)
    opt.optimizers = make_optimizers(model, muon_lr=0.01, embed_lr=1e-3, scalar_lr=1e-3)
    before = [p.clone() for p in model.parameters()]
    run_lr_cooldown(model, opt, grad_accum=1, cooldown_steps=3)
    assert all(g['lr'] == 0.0 for o in opt.optimizers for g in o.param_groups)
    assert any(not torch.equal(b, p) for b, p in zip(before, model.parameters()))


def _softcapped_ce_reference(hidden, weight, targets, softcap):
    import torch.nn.functional as F
    lin = torch.nn.Linear(hidden.size(1), weight.size(0), bias=False)
    with torch.no_grad():
        lin.weight.copy_(weight)
    z = softcap * torch.tanh(lin(hidden) / softcap)
    loss = F.cross_entropy(z, targets)
    loss.backward()
    return loss.detach(), lin.weight.grad


def test_chunked_cross_entropy_matches_reference():
    from pretrain.fused_ce import _chunked_cross_entropy, chunked_cross_entropy
    torch.manual_seed(0)
    N, d, V, softcap = 130, 32, 257, 15.0
    weight = torch.randn(V, d)
    targets = torch.randint(0, V, (N,))

    ref_h = torch.randn(N, d, requires_grad=True)
    ref_loss, ref_dw = _softcapped_ce_reference(ref_h, weight, targets, softcap)

    # Chunked autograd path (ragged last chunk: 130 = 2*64 + 2)
    h = ref_h.detach().clone().requires_grad_(True)
    loss = _chunked_cross_entropy(h, weight, targets, softcap, 64)
    loss.backward()
    assert loss.item() == pytest.approx(ref_loss.item(), abs=1e-4)
    assert torch.allclose(h.grad, ref_h.grad, atol=1e-4)
    # weight grad: recompute against autograd through the same reference
    w = weight.clone().requires_grad_(True)
    loss_w = _chunked_cross_entropy(ref_h.detach(), w, targets, softcap, 64)
    loss_w.backward()
    assert torch.allclose(w.grad, ref_dw, atol=1e-4)

    # Public API falls back to the plain path on CPU but must agree on value
    h2 = ref_h.detach().clone().requires_grad_(True)
    loss2 = chunked_cross_entropy(h2, weight, targets, softcap, 64)
    loss2.backward()
    assert loss2.item() == pytest.approx(ref_loss.item(), abs=1e-4)
    assert torch.allclose(h2.grad, ref_h.grad, atol=1e-4)
