import pytest
import torch

from muon import Muon


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
