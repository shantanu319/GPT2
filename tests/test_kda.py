from types import SimpleNamespace

import torch
import torch.nn.functional as F

from core.kda import KimiDeltaAttention, kda_chunk, kda_recurrence, kda_scan
from core.model import Transformer, get_model, nopeak_mask


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


def _per_segment_reference(q, k, v, g, beta, seg_ids):
    """Ground truth for segment resets: run the recurrence independently on
    each segment of each row, stitch outputs, keep each row's final state."""
    B, T = seg_ids.shape
    o_rows, S_rows = [], []
    for b in range(B):
        cuts = (seg_ids[b, 1:] != seg_ids[b, :-1]).nonzero().flatten() + 1
        bounds = [0] + cuts.tolist() + [T]
        outs = []
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            ob, Sb = kda_recurrence(q[b:b + 1, lo:hi], k[b:b + 1, lo:hi],
                                    v[b:b + 1, lo:hi], g[b:b + 1, lo:hi],
                                    beta[b:b + 1, lo:hi])
            outs.append(ob)
        o_rows.append(torch.cat(outs, dim=1))
        S_rows.append(Sb)
    return torch.cat(o_rows, dim=0), torch.cat(S_rows, dim=0)


def _seg_ids(*specs, T=128):
    """Build per-row segment ids; each spec is a list of (start, end, id)."""
    seg = torch.zeros(len(specs), T, dtype=torch.long)
    for b, runs in enumerate(specs):
        for lo, hi, sid in runs:
            seg[b, lo:hi] = sid
    return seg


def test_chunk_with_segments_matches_per_segment_recurrence():
    """Boundaries off chunk alignment and multiple segments per chunk."""
    q, k, v, g, beta = _inputs()  # B=2, T=128
    seg = _seg_ids([(40, 96, 1), (96, 128, 2)],
                   [(10, 20, 3), (20, 70, 1), (70, 128, 2)], T=128)
    o_ref, S_ref = _per_segment_reference(q, k, v, g, beta, seg)
    o_chk, S_chk = kda_chunk(q, k, v, g, beta, chunk_size=64, seg_ids=seg)
    assert torch.allclose(o_chk, o_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S_chk, S_ref, atol=1e-4, rtol=1e-4)


def test_chunk_with_chunk_aligned_boundary():
    q, k, v, g, beta = _inputs()
    seg = _seg_ids([(64, 128, 1)], [(64, 128, 1)], T=128)
    o_ref, S_ref = _per_segment_reference(q, k, v, g, beta, seg)
    o_chk, S_chk = kda_chunk(q, k, v, g, beta, chunk_size=64, seg_ids=seg)
    assert torch.allclose(o_chk, o_ref, atol=1e-4, rtol=1e-4)
    assert torch.allclose(S_chk, S_ref, atol=1e-4, rtol=1e-4)


def test_recurrence_with_segments_matches_per_segment():
    q, k, v, g, beta = _inputs(T=96)
    seg = _seg_ids([(30, 60, 1), (60, 96, 2)], [(48, 96, 1)], T=96)
    o_ref, S_ref = _per_segment_reference(q, k, v, g, beta, seg)
    o_rec, S_rec = kda_recurrence(q, k, v, g, beta, seg_ids=seg)
    assert torch.allclose(o_rec, o_ref, atol=1e-5)
    assert torch.allclose(S_rec, S_ref, atol=1e-5)


def test_single_segment_matches_no_segments():
    q, k, v, g, beta = _inputs()
    seg = torch.zeros(2, 128, dtype=torch.long)
    o0, S0 = kda_chunk(q, k, v, g, beta, chunk_size=64)
    o1, S1 = kda_chunk(q, k, v, g, beta, chunk_size=64, seg_ids=seg)
    assert torch.equal(o1, o0) and torch.equal(S1, S0)
    o2, S2 = kda_recurrence(q, k, v, g, beta)
    o3, S3 = kda_recurrence(q, k, v, g, beta, seg_ids=seg)
    assert torch.equal(o3, o2) and torch.equal(S3, S2)


def test_module_forward_backward_with_segments():
    torch.manual_seed(0)
    layer = KimiDeltaAttention(d_model=32, heads=4)
    x = torch.randn(2, 64, 32, requires_grad=True)
    seg = _seg_ids([(17, 64, 1)], [(32, 64, 1)], T=64)
    o = layer(x, seg_ids=seg)
    assert o.shape == (2, 64, 32)
    o.sum().backward()
    assert torch.isfinite(x.grad).all()
    for name, p in layer.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def _hybrid(N=4, kda=2, **kw):
    torch.manual_seed(0)
    return Transformer(vocab=48, d_model=32, N=N, heads=4, dropout=0.0,
                       kv_heads=2, value_residual=True, unet_skips=True,
                       kda=kda, **kw)


def test_hybrid_layer_pattern():
    m = _hybrid(N=8, kda=4)
    kinds = [layer.is_kda for layer in m.decoder.layers]
    assert kinds == [True, True, True, False, True, True, True, False]
    assert all(not layer.is_kda for layer in _hybrid(kda=0).decoder.layers)
    assert all(layer.is_kda for layer in _hybrid(kda=1).decoder.layers)


def test_value_residual_only_on_full_attention_layers():
    m = _hybrid(N=4, kda=2)  # KDA, MHA, KDA, MHA
    assert not hasattr(m.decoder.layers[0].attn_1, 'vres')
    assert not hasattr(m.decoder.layers[1].attn_1, 'vres')  # first MHA defines v1
    assert not hasattr(m.decoder.layers[2].attn_1, 'vres')
    assert hasattr(m.decoder.layers[3].attn_1, 'vres')


def test_hybrid_forward_backward():
    m = _hybrid()
    x = torch.randint(0, 48, (2, 20))
    logits = m(x, nopeak_mask(20, torch.device('cpu')))
    assert logits.shape == (2, 20, 48)
    logits.sum().backward()
    kda_params = [p for layer in m.decoder.layers if layer.is_kda
                  for p in layer.attn_1.parameters()]
    assert kda_params  # every KDA parameter must be in the gradient path
    for p in kda_params:
        assert p.grad is not None and torch.isfinite(p.grad).all()
    for name, p in m.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


def test_hybrid_get_model_zero_init():
    opt = SimpleNamespace(d_model=32, heads=4, n_layers=4, dropout=0.0,
                          kv_heads=2, loops=1, grad_ckpt=0, value_residual=1,
                          unet_skips=1, attn_res=0, kda=2, device='cpu',
                          loadname=None)
    m = get_model(opt, 48)
    for name, p in m.named_parameters():
        if name.endswith(('attn_1.out.weight', 'attn_1.o_proj.weight',
                          'ff.w_down.weight')):
            assert torch.all(p == 0), name


def test_hybrid_checkpoint_config_roundtrip():
    """The cfg dict embedded in checkpoints must rebuild the exact hybrid."""
    m = _hybrid()
    cfg = dict(vocab_size=48, d_model=32, n_layers=4, heads=4, dropout=0.0,
               kv_heads=2, loops=1, value_residual=True, unet_skips=True,
               attn_res=0, kda=2)
    m2 = Transformer(vocab=cfg['vocab_size'], d_model=cfg['d_model'],
                     N=cfg['n_layers'], heads=cfg['heads'], dropout=cfg['dropout'],
                     kv_heads=cfg.get('kv_heads'), loops=cfg.get('loops', 1),
                     value_residual=cfg.get('value_residual', False),
                     unet_skips=cfg.get('unet_skips', False),
                     attn_res=cfg.get('attn_res', 0), kda=cfg.get('kda', 0))
    m2.load_state_dict(m.state_dict())
    x = torch.randint(0, 48, (1, 12))
    with torch.no_grad():
        assert torch.equal(m(x, nopeak_mask(12, torch.device('cpu'))),
                           m2(x, nopeak_mask(12, torch.device('cpu'))))


def test_model_cache_decode_matches_full_forward():
    """Whole-model prefill + decode must equal one full forward — exercises the
    MHA KV cache and the KDA state cache side by side."""
    m = _hybrid().eval()
    x = torch.randint(0, 48, (1, 20))  # 20 % 64 != 0: KDA takes the recurrence path
    with torch.no_grad():
        o_full = m(x, nopeak_mask(20, torch.device('cpu')))
        m.reset_cache()
        outs = [m(x[:, :7], nopeak_mask(7, torch.device('cpu')), start_pos=0)]
        for t in range(7, 20):
            outs.append(m(x[:, t:t + 1], None, start_pos=t))
        o_cached = torch.cat(outs, dim=1)
    assert torch.allclose(o_cached, o_full, atol=1e-4)


def test_chunk_grads_finite_under_steep_decay():
    """A chunk whose cumulative log-decay far exceeds exp's fp32 range must
    still give finite grads: the intermediates that would overflow are all
    masked out, so they are clamped rather than allowed to reach inf."""
    q, k, v, g, beta = _inputs(T=64, seed=7)
    g = g * 10.0  # cumulative decay ~ -600 over the chunk
    q, k, v, g, beta = (t.detach().requires_grad_(True) for t in (q, k, v, g, beta))
    o, S = kda_chunk(q, k, v, g, beta, chunk_size=64)
    (o.sum() + S.sum()).backward()
    for name, t in zip(('q', 'k', 'v', 'g', 'beta'), (q, k, v, g, beta)):
        assert torch.isfinite(t.grad).all(), f"non-finite grad for {name}"


def test_scan_matches_recurrence_including_a_ragged_tail():
    """The cached path chunks the leading whole chunks and walks only the
    ragged tail, so it has to equal the plain recurrence for any T -- carried
    state included."""
    S0 = torch.randn(2, 3, 16, 16, generator=torch.Generator().manual_seed(5)) * 0.1
    for T in (64, 100, 192):
        q, k, v, g, beta = _inputs(T=T, seed=T)
        o_ref, S_ref = kda_recurrence(q, k, v, g, beta, initial_state=S0)
        o, S = kda_scan(q, k, v, g, beta, initial_state=S0, chunk_size=64)
        assert torch.allclose(o, o_ref, atol=1e-4, rtol=1e-4), T
        assert torch.allclose(S, S_ref, atol=1e-4, rtol=1e-4), T


def test_kda_cached_prefill_matches_token_by_token_decode():
    """Prefilling a span in one call and feeding it a token at a time have to
    leave the layer in the same place and emit the same outputs."""
    torch.manual_seed(0)
    layer = KimiDeltaAttention(d_model=32, heads=4).eval()
    x = torch.randn(1, 100, 32)
    with torch.no_grad():
        layer.reset_cache()
        whole = layer(x, start_pos=0)
        layer.reset_cache()
        stepped = torch.cat([layer(x[:, i:i + 1], start_pos=i) for i in range(x.size(1))],
                            dim=1)
    assert torch.allclose(whole, stepped, atol=1e-4)
