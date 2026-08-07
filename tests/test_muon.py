import pytest
import torch

from pretrain.muon import Muon, _polar_express


def test_muon_rejects_1d_params():
    p = torch.nn.Parameter(torch.randn(10))
    opt = Muon([p])
    p.grad = torch.randn(10)
    with pytest.raises(RuntimeError):
        opt.step()


def test_muon_weight_decay_shrinks_params():
    """With zero grads the orthogonal update vanishes, leaving exactly p*(1 - lr*wd)."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8))
    p0 = p.detach().clone()
    opt = Muon([p], lr=0.1, weight_decay=0.5)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.allclose(p.detach(), p0 * (1 - 0.1 * 0.5), atol=1e-7)


def test_muon_weight_decay_off_by_default():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8))
    p0 = p.detach().clone()
    opt = Muon([p], lr=0.1)
    p.grad = torch.zeros_like(p)
    opt.step()
    assert torch.equal(p.detach(), p0)


def test_muon_bf16_input_smoke():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(16, 8, dtype=torch.bfloat16))
    opt = Muon([p], lr=0.05)
    before = p.detach().clone()
    p.grad = torch.randn(16, 8, dtype=torch.bfloat16)
    opt.step()
    assert p.dtype == torch.bfloat16
    assert torch.isfinite(p.detach()).all()
    assert not torch.equal(p.detach(), before)


def test_muon_reduces_loss_on_tall_matrix():
    """Muon should fit a linear regression Y = XW^T when W is (out, in) with out > in."""
    torch.manual_seed(0)
    d_in, d_out = 4, 16
    W_true = torch.randn(d_out, d_in)
    X = torch.randn(64, d_in)
    Y = X @ W_true.T

    model = torch.nn.Linear(d_in, d_out, bias=False)
    opt = Muon([model.weight], lr=0.1)

    initial_loss = ((model(X) - Y) ** 2).mean().item()
    for _ in range(200):
        loss = ((model(X) - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = ((model(X) - Y) ** 2).mean().item()

    assert final_loss < initial_loss * 0.01, f"{initial_loss:.4f} -> {final_loss:.4f}"


def test_muon_reduces_loss_on_wide_matrix():
    """Exercises the Newton-Schulz transpose branch (in > out)."""
    torch.manual_seed(0)
    d_in, d_out = 16, 4
    W_true = torch.randn(d_out, d_in)
    X = torch.randn(64, d_in)
    Y = X @ W_true.T

    model = torch.nn.Linear(d_in, d_out, bias=False)
    opt = Muon([model.weight], lr=0.1)

    initial_loss = ((model(X) - Y) ** 2).mean().item()
    for _ in range(200):
        loss = ((model(X) - Y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    final_loss = ((model(X) - Y) ** 2).mean().item()

    assert final_loss < initial_loss * 0.01, f"{initial_loss:.4f} -> {final_loss:.4f}"


def _applied_update(p0, p1, lr, scale):
    return (p0 - p1) / (lr * scale)


def test_muon_per_head_axis0_matches_per_slice_ns():
    """axis=0 (q/k/v projections): each head slice of the applied update must
    equal a standalone 2D Newton-Schulz on that slice (i.e. truly per-head),
    and be approximately orthogonal up to the fixed-step NS error."""
    torch.manual_seed(0)
    heads, d_k, fan_in, lr = 4, 8, 32, 0.1
    p = torch.nn.Parameter(torch.randn(heads * d_k, fan_in))
    p.muon_head_split = (heads, 0)
    p0 = p.detach().clone()
    g = torch.randn_like(p)
    opt = Muon([p], lr=lr, momentum=0.0)  # momentum 0 => update == grad
    p.grad = g.clone()
    opt.step()

    scale = max(1.0, d_k / fan_in) ** 0.5
    U = _applied_update(p0, p.detach(), lr, scale).view(heads, d_k, fan_in)
    for h in range(heads):
        reference = _polar_express(g.view(heads, d_k, fan_in)[h])
        # bf16 compute takes different rounding paths for 2D vs batched matmul
        assert torch.allclose(U[h], reference, atol=0.02, rtol=0.02), \
            f"head {h} not independently orthogonalized"
        sv = torch.linalg.svdvals(U[h])
        assert sv.min() > 0.8 and sv.max() < 1.25, f"head {h} not near-orthogonal: {sv}"


def test_muon_per_head_axis1_matches_per_slice_ns():
    """axis=1 (attn out projection): head slices split along input columns;
    each (fan_out, d_k) slice must match a standalone 2D Newton-Schulz."""
    torch.manual_seed(0)
    heads, d_k, fan_out, lr = 4, 8, 32, 0.1
    p = torch.nn.Parameter(torch.randn(fan_out, heads * d_k))
    p.muon_head_split = (heads, 1)
    p0 = p.detach().clone()
    g = torch.randn_like(p)
    opt = Muon([p], lr=lr, momentum=0.0)
    p.grad = g.clone()
    opt.step()

    scale = max(1.0, fan_out / d_k) ** 0.5
    U = _applied_update(p0, p.detach(), lr, scale).view(fan_out, heads, d_k)
    for h in range(heads):
        reference = _polar_express(g.view(fan_out, heads, d_k)[:, h, :])
        # bf16 compute takes different rounding paths for 2D vs batched matmul
        assert torch.allclose(U[:, h, :], reference, atol=0.02, rtol=0.02), \
            f"head {h} not independently orthogonalized"
        sv = torch.linalg.svdvals(U[:, h, :])
        assert sv.min() > 0.8 and sv.max() < 1.25, f"head {h} not near-orthogonal: {sv}"


def test_muon_per_head_matches_batched_polar_express():
    """The tagged update must equal the batched 3D _polar_express applied to the
    reshaped gradient (momentum 0, no weight decay)."""
    torch.manual_seed(0)
    heads, lr = 2, 0.05
    p = torch.nn.Parameter(torch.randn(16, 8))
    p.muon_head_split = (heads, 0)
    p0 = p.detach().clone()
    g = torch.randn_like(p)
    opt = Muon([p], lr=lr, momentum=0.0)
    p.grad = g.clone()
    opt.step()

    expected = p0 - lr * _polar_express(g.view(heads, 8, 8)).view(16, 8)
    assert torch.allclose(p.detach(), expected)


def test_muon_per_head_differs_from_fused():
    """Sanity that the tag changes the update (fused vs per-head NS differ)."""
    torch.manual_seed(0)
    g = torch.randn(16, 8)
    results = []
    for split in (None, (2, 0)):
        p = torch.nn.Parameter(torch.randn(16, 8))
        if split:
            p.muon_head_split = split
        opt = Muon([p], lr=0.1, momentum=0.0)
        p.grad = g.clone()
        opt.step()
        results.append(p.detach().clone())
    assert not torch.allclose(results[0], results[1])


def test_muon_per_head_rejects_indivisible_split():
    p = torch.nn.Parameter(torch.randn(8, 16))
    p.muon_head_split = (3, 0)  # 8 % 3 != 0
    opt = Muon([p], lr=0.1)
    p.grad = torch.randn_like(p)
    with pytest.raises(RuntimeError):
        opt.step()
