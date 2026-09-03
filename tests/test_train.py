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
    loss = _chunked_cross_entropy(h, weight, None, targets, softcap, 64)
    loss.backward()
    assert loss.item() == pytest.approx(ref_loss.item(), abs=1e-4)
    assert torch.allclose(h.grad, ref_h.grad, atol=1e-4)
    # weight grad: recompute against autograd through the same reference
    w = weight.clone().requires_grad_(True)
    loss_w = _chunked_cross_entropy(ref_h.detach(), w, None, targets, softcap, 64)
    loss_w.backward()
    assert torch.allclose(w.grad, ref_dw, atol=1e-4)

    # Public API falls back to the plain path on CPU but must agree on value
    h2 = ref_h.detach().clone().requires_grad_(True)
    loss2 = chunked_cross_entropy(h2, weight, None, targets, softcap, 64)
    loss2.backward()
    assert loss2.item() == pytest.approx(ref_loss.item(), abs=1e-4)
    assert torch.allclose(h2.grad, ref_h.grad, atol=1e-4)


def test_chunked_cross_entropy_backward_runs_in_the_autocast_dtype():
    """Backward runs outside the caller's autocast region, so it has to be told
    which dtype the forward's logits matmul used. Recomputing them in fp32
    instead differentiates at a different operating point from the loss, and
    pays fp32 matmul rates for three (N, V) x (V, d) products."""
    from torch.utils._python_dispatch import TorchDispatchMode

    from pretrain.fused_ce import _chunked_cross_entropy, _reference_cross_entropy

    class _MatmulDtypes(TorchDispatchMode):
        def __init__(self):
            self.dtypes = set()

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func.__name__.startswith('mm'):
                self.dtypes.add(args[0].dtype)
            return func(*args, **(kwargs or {}))

    torch.manual_seed(0)
    N, d, V, softcap = 512, 64, 2000, 15.0
    h0, w0 = torch.randn(N, d) * 0.5, torch.randn(V, d) * 0.02
    targets = torch.randint(0, V, (N,))

    h = h0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        loss = _chunked_cross_entropy(h, w, None, targets, softcap, 256)
    seen = _MatmulDtypes()
    with seen:
        loss.backward()
    assert seen.dtypes == {torch.bfloat16}, seen.dtypes

    # ...and it still lands on the unfused fallback it stands in for.
    ref_h = h0.clone().requires_grad_(True)
    ref_w = w0.clone().requires_grad_(True)
    with torch.autocast(device_type='cpu', dtype=torch.bfloat16):
        _reference_cross_entropy(ref_h, ref_w, None, targets, softcap).backward()
    assert torch.allclose(h.grad, ref_h.grad, atol=1e-5)
    assert torch.allclose(w.grad, ref_w.grad, atol=1e-4)


def test_chunked_cross_entropy_applies_the_lm_head_bias():
    """model.out is an nn.Linear with a bias, so the chunked path has to be
    handed it. Without it the fused training loss drifts from the logits the
    model actually produces at inference, and the bias never sees a gradient."""
    import torch.nn.functional as F

    from core.model import LOGIT_SOFTCAP, Transformer, nopeak_mask
    from pretrain.fused_ce import _chunked_cross_entropy

    torch.manual_seed(0)
    V, d = 64, 16
    model = Transformer(vocab=V, d_model=d, N=1, heads=2, dropout=0.0, kv_heads=1).eval()
    torch.nn.init.normal_(model.out.bias, std=0.5)
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    mask = nopeak_mask(8, torch.device('cpu'))

    ref = F.cross_entropy(model(x, mask).view(-1, V), y.reshape(-1))
    hidden = model.decoder(x, mask).reshape(-1, d)
    fused = _chunked_cross_entropy(hidden, model.out.weight, model.out.bias,
                                   y.reshape(-1), LOGIT_SOFTCAP, 4)
    assert torch.allclose(ref, fused, atol=1e-5), (ref.item(), fused.item())

    fused.backward()
    assert model.out.bias.grad is not None
    assert model.out.bias.grad.abs().sum() > 0


def _tiny_run(tmp_path, save_every=0, resume=None, restore=True, anneal=False):
    """train_model on a fixed 8-step corpus, set up the way main() does it.
    Returns (model, train_curve)."""
    import numpy as np
    from core.model import get_model
    from pretrain.config import parse_args
    from pretrain.train import make_optimizers, restore_optimizers, train_model
    torch.manual_seed(0)
    opt = parse_args([])
    opt.device = torch.device('cpu')
    opt.d_model, opt.n_layers, opt.heads, opt.kv_heads = 32, 2, 4, 2
    opt.batchsize, opt.seqlen, opt.epochs = 2, 16, 2
    opt.warmup_steps, opt.momentum_warmup, opt.printevery = 2, 3, 1
    opt.save_every, opt.val_every = save_every, 0
    opt.vocab_size, opt.eos_id, opt.model_config = 64, None, None
    opt.dir_name, opt.savename = str(tmp_path), 'ckpt'
    opt.loadname = resume
    rng = np.random.default_rng(0)
    opt.train = rng.integers(0, 64, 2 * 16 * 4, dtype=np.uint16)   # 4 batches/epoch
    opt.valid = rng.integers(0, 64, 2 * 16 * 2, dtype=np.uint16)
    opt.anneal = None
    if anneal:   # decay, and so the anneal corpus, from step 2 of 8
        opt.decay_frac = 0.75
        opt.anneal = rng.integers(0, 64, 2 * 16 * 3, dtype=np.uint16)
    opt.batches_per_epoch = len(opt.train) // (opt.batchsize * opt.seqlen)
    opt.total_steps = opt.epochs * opt.batches_per_epoch
    model = get_model(opt, opt.vocab_size)
    opt.optimizers = make_optimizers(model, muon_lr=0.01, embed_lr=1e-3, scalar_lr=1e-3)
    opt.start_step = restore_optimizers(opt.optimizers, resume) if (resume and restore) else 0
    train_curve, _ = train_model(model, opt)
    return model, train_curve


def test_resume_reproduces_the_uninterrupted_run(tmp_path):
    """Step 3 of 8 is mid-epoch: the resumed run must land on the same
    weights as the run that never stopped, which needs the optimizer state,
    the step, and the data position all restored."""
    import os
    straight, _ = _tiny_run(tmp_path / 'a', save_every=3)
    ckpt = os.path.join(str(tmp_path / 'a'), 'ckpt_step3.pt')
    resumed, curve = _tiny_run(tmp_path / 'b', resume=ckpt)
    assert curve[0][0] == 4, "the first logged step continues the count"
    assert all(torch.equal(a, b) for a, b in
               zip(straight.state_dict().values(), resumed.state_dict().values()))
    weights_only, _ = _tiny_run(tmp_path / 'c', resume=ckpt, restore=False)
    assert not all(torch.equal(a, b) for a, b in
                   zip(straight.state_dict().values(), weights_only.state_dict().values()))


def test_resume_reproduces_the_uninterrupted_run_with_an_anneal_corpus(tmp_path):
    """Resumed at step 3 with the decay phase already one batch into the
    anneal corpus, so the anneal feeder has to rejoin its own order too."""
    import os
    straight, _ = _tiny_run(tmp_path / 'a', save_every=3, anneal=True)
    ckpt = os.path.join(str(tmp_path / 'a'), 'ckpt_step3.pt')
    resumed, _ = _tiny_run(tmp_path / 'b', resume=ckpt, anneal=True)
    assert all(torch.equal(a, b) for a, b in
               zip(straight.state_dict().values(), resumed.state_dict().values()))


def test_decay_start_is_shared_by_the_schedule():
    from pretrain.train import decay_start
    start = decay_start(1000, 0.25)
    assert start == 750
    assert lr_factor(749, 1000, warmup_steps=1, decay_frac=0.25) == 1.0
    assert lr_factor(751, 1000, warmup_steps=1, decay_frac=0.25) < 1.0


def _batch_opt(main_token, anneal_token, decay_frac=0.5, grad_accum=1):
    import numpy as np
    from types import SimpleNamespace
    opt = SimpleNamespace(batchsize=2, seqlen=8, device=torch.device('cpu'), eos_id=None,
                          shuffle=1, grad_accum=grad_accum, decay_frac=decay_frac)
    opt.train = np.full(2 * 8 * 8, main_token, dtype=np.uint16)       # 8 batches
    opt.anneal = np.full(2 * 8 * 3, anneal_token, dtype=np.uint16)    # 3 per pass
    opt.batches_per_epoch = 8
    opt.total_steps = 8 // grad_accum
    return opt


def _firsts(batches):
    return [x[0, 0].item() for x, _ in batches]


def test_epoch_batches_switch_to_the_anneal_corpus_at_decay_start():
    from pretrain.train import cycle_feeder, epoch_batches
    opt = _batch_opt(1, 9)
    assert _firsts(epoch_batches(opt, 0, 0, cycle_feeder(opt, opt.anneal))) == [1] * 4 + [9] * 4
    assert _firsts(epoch_batches(opt, 0, 0)) == [1] * 8


def test_anneal_switch_counts_optimizer_steps_not_micro_batches():
    from pretrain.train import cycle_feeder, epoch_batches
    opt = _batch_opt(1, 9, grad_accum=2)   # 4 steps; decay from step 2 = micro 4
    assert _firsts(epoch_batches(opt, 0, 0, cycle_feeder(opt, opt.anneal))) == [1] * 4 + [9] * 4


def test_cycle_feeder_skip_continues_across_passes():
    import itertools
    import numpy as np
    from pretrain.train import cycle_feeder
    opt = _batch_opt(1, 9)
    opt.anneal = np.arange(2 * 8 * 3, dtype=np.uint16)   # distinct windows, 3 per pass
    full = _firsts(itertools.islice(cycle_feeder(opt, opt.anneal), 7))
    assert len(set(full[:3])) == 3 and sorted(full[:3]) == sorted(full[3:6])
    assert _firsts(itertools.islice(cycle_feeder(opt, opt.anneal, skip=4), 3)) == full[4:]
