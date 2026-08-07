import torch
import torch.nn.functional as F

from core.kda import KimiDeltaAttention, kda_chunk, kda_recurrence


def _inputs(B=2, T=128, H=3, K=16, V=16, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K, generator=gen), dim=-1)
    v = torch.randn(B, T, H, V, generator=gen)
    g = F.logsigmoid(torch.randn(B, T, H, K, generator=gen))  # negative log-decay
    beta = torch.rand(B, T, H, generator=gen)
    return q, k, v, g, beta


def test_recurrence_matches_manual_delta_rule():
    """The einsum form must equal the explicit Diag/decay matrix form."""
    B, T, H, K, V = 1, 8, 1, 4, 4
    q, k, v, g, beta = _inputs(B=B, T=T, H=H, K=K, V=V)
    o, S = kda_recurrence(q, k, v, g, beta)

    S_man = torch.zeros(K, V)
    outs = []
    for t in range(T):
        Dm = torch.diag(g[0, t, 0].exp())
        Sd = Dm @ S_man
        err = v[0, t, 0] - Sd.T @ k[0, t, 0]
        S_man = Sd + beta[0, t, 0] * torch.outer(k[0, t, 0], err)
        outs.append(S_man.T @ (q[0, t, 0] * K ** -0.5))
    o_man = torch.stack(outs)

    assert torch.allclose(o[0, :, 0], o_man, atol=1e-5)
    assert torch.allclose(S[0, 0], S_man, atol=1e-5)


def test_chunk_matches_recurrence():
    q, k, v, g, beta = _inputs()  # T=128, two chunks of 64
    o_ref, S_ref = kda_recurrence(q, k, v, g, beta)
    o_chk, S_chk = kda_chunk(q, k, v, g, beta, chunk_size=64)
    assert torch.allclose(o_chk, o_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S_chk, S_ref, atol=1e-4, rtol=1e-4)


def test_chunk_matches_recurrence_single_chunk_and_initial_state():
    q, k, v, g, beta = _inputs(T=64, seed=3)
    S0 = torch.randn(2, 3, 16, 16, generator=torch.Generator().manual_seed(4)) * 0.1
    o_ref, S_ref = kda_recurrence(q, k, v, g, beta, initial_state=S0)
    o_chk, S_chk = kda_chunk(q, k, v, g, beta, initial_state=S0, chunk_size=64)
    assert torch.allclose(o_chk, o_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S_chk, S_ref, atol=1e-4, rtol=1e-4)


def test_chunk_rejects_ragged_T():
    q, k, v, g, beta = _inputs(T=96)
    try:
        kda_chunk(q, k, v, g, beta, chunk_size=64)
        assert False, "expected an assertion on ragged T"
    except AssertionError:
        pass


def test_state_carry_split_matches_full():
    """Two sequential segments with a hand-off state must equal one full scan."""
    q, k, v, g, beta = _inputs(T=96)
    o_full, S_full = kda_recurrence(q, k, v, g, beta)
    o1, S1 = kda_recurrence(q[:, :48], k[:, :48], v[:, :48], g[:, :48], beta[:, :48])
    o2, S2 = kda_recurrence(q[:, 48:], k[:, 48:], v[:, 48:], g[:, 48:], beta[:, 48:],
                            initial_state=S1)
    assert torch.allclose(torch.cat([o1, o2], dim=1), o_full, atol=1e-5)
    assert torch.allclose(S2, S_full, atol=1e-5)


def test_chunk_state_carry_split():
    q, k, v, g, beta = _inputs()  # T=128
    o_full, S_full = kda_chunk(q, k, v, g, beta, chunk_size=64)
    o1, S1 = kda_chunk(q[:, :64], k[:, :64], v[:, :64], g[:, :64], beta[:, :64],
                       chunk_size=64)
    o2, S2 = kda_chunk(q[:, 64:], k[:, 64:], v[:, 64:], g[:, 64:], beta[:, 64:],
                       initial_state=S1, chunk_size=64)
    assert torch.allclose(torch.cat([o1, o2], dim=1), o_full, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S2, S_full, atol=1e-4, rtol=1e-4)


def test_causality():
    """Outputs at positions < t must not depend on tokens at positions >= t."""
    q, k, v, g, beta = _inputs(T=64)
    t_half = 32
    gen = torch.Generator().manual_seed(99)
    q2, k2, v2, g2, beta2 = (x.clone() for x in (q, k, v, g, beta))
    for x2 in (q2, k2, v2, g2):
        x2[:, t_half:] = torch.randn(x2[:, t_half:].shape, generator=gen)
    beta2[:, t_half:] = torch.rand(beta2[:, t_half:].shape, generator=gen)
    o1, _ = kda_chunk(q, k, v, g, beta, chunk_size=64)
    o2, _ = kda_chunk(q2, k2, v2, g2, beta2, chunk_size=64)
    assert torch.allclose(o1[:, :t_half], o2[:, :t_half], atol=1e-5)


def test_module_forward_backward():
    torch.manual_seed(0)
    layer = KimiDeltaAttention(d_model=32, heads=4)
    x = torch.randn(2, 64, 32, requires_grad=True)  # T=64 -> chunk path
    o = layer(x)
    assert o.shape == (2, 64, 32)
    o.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_module_recurrence_path_backward():
    torch.manual_seed(0)
    layer = KimiDeltaAttention(d_model=32, heads=4)
    x = torch.randn(2, 20, 32, requires_grad=True)  # 20 % 64 != 0 -> recurrence path
    o = layer(x)
    o.sum().backward()
    assert torch.isfinite(x.grad).all()


def test_module_long_sequence_stays_finite():
    torch.manual_seed(0)
    layer = KimiDeltaAttention(d_model=32, heads=4).eval()
    with torch.no_grad():
        o = layer(torch.randn(1, 256, 32))
    assert torch.isfinite(o).all()


def test_module_cache_decode_matches_full_forward():
    """Prefill + single-token decode with the state cache must equal a single
    full forward (the property the KV cache has for full attention)."""
    torch.manual_seed(0)
    layer = KimiDeltaAttention(d_model=32, heads=4).eval()
    x = torch.randn(1, 20, 32)  # 20 % 64 != 0 -> recurrence in both paths
    with torch.no_grad():
        o_full = layer(x)
        layer.reset_cache()
        outs = [layer(x[:, :7], start_pos=0)]
        for t in range(7, 20):
            outs.append(layer(x[:, t:t + 1], start_pos=t))
        o_cached = torch.cat(outs, dim=1)
    assert torch.allclose(o_cached, o_full, atol=1e-4)
