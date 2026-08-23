import torch
import torch.nn.functional as F

from core.model import (Transformer, nopeak_mask, window_attention,
                        window_block_mask, window_band_mask)


def _qkv(B=2, H=4, Hkv=2, T=64, D=8, seed=0):
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(B, H, T, D, generator=gen)
    k = torch.randn(B, Hkv, T, D, generator=gen)
    v = torch.randn(B, Hkv, T, D, generator=gen)
    return q, k, v


def _dense_window(q, k, v, window, seg_ids=None):
    """Reference: full T x T attention with a causal band mask."""
    T = q.size(2)
    groups = q.size(1) // k.size(1)
    k, v = (x.repeat_interleave(groups, dim=1) for x in (k, v))
    m = window_band_mask(T, T, 0, window, q.device)          # (1, T, T)
    if seg_ids is not None:
        m = m & (seg_ids[:, :, None] == seg_ids[:, None, :])
    return F.scaled_dot_product_attention(q, k, v, attn_mask=m.unsqueeze(1))


def test_window_attention_matches_dense_band():
    q, k, v = _qkv()
    W = 16
    masks = window_block_mask(W, q.device)
    out = window_attention(q, k, v, W, masks, enable_gqa=True)
    assert torch.allclose(out, _dense_window(q, k, v, W), atol=1e-5)


def test_window_attention_matches_dense_band_with_segments():
    q, k, v = _qkv(seed=1)
    B, T, W = q.size(0), q.size(2), 16
    seg = (torch.arange(T) // 22).expand(B, T).contiguous()  # boundaries mid-block
    masks = window_block_mask(W, q.device, seg)
    out = window_attention(q, k, v, W, masks, enable_gqa=True)
    assert torch.allclose(out, _dense_window(q, k, v, W, seg), atol=1e-5)


def test_window_two_blocks_matches_dense_band():
    """The smallest blocked case: one head block plus one tail block."""
    q, k, v = _qkv(T=32, seed=2)
    W = 16
    out = window_attention(q, k, v, W, window_block_mask(W, q.device), enable_gqa=True)
    assert torch.allclose(out, _dense_window(q, k, v, W), atol=1e-5)


def test_window_attention_handles_a_ragged_last_block():
    """T need not be a multiple of W -- the feeders serve seqlen - 1 tokens."""
    q, k, v = _qkv(T=55, seed=11)
    W = 16
    out = window_attention(q, k, v, W, window_block_mask(W, q.device), enable_gqa=True)
    assert torch.allclose(out, _dense_window(q, k, v, W), atol=1e-5)


def test_window_attention_ragged_with_segments():
    q, k, v = _qkv(T=55, seed=12)
    B, W = q.size(0), 16
    seg = (torch.arange(55) // 21).expand(B, 55).contiguous()
    out = window_attention(q, k, v, W, window_block_mask(W, q.device, seg),
                           enable_gqa=True)
    assert torch.allclose(out, _dense_window(q, k, v, W, seg), atol=1e-5)


def test_model_swa_ragged_seqlen_backward():
    B, T, W, V = 2, 55, 16, 48
    m = _model(W, kda=2, seed=13).train()
    seg = (torch.arange(T) // 21).expand(B, T).contiguous()
    m(torch.randint(0, V, (B, T)), None, seg_ids=seg).float().mean().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def _model(swa, vocab=48, d_model=32, N=4, heads=4, kv_heads=2, kda=0, seed=0):
    torch.manual_seed(seed)
    return Transformer(vocab, d_model, N, heads, 0.0, kv_heads=kv_heads,
                       value_residual=True, unet_skips=True, kda=kda, swa=swa).eval()


def test_model_swa_matches_explicit_band_mask():
    """A -swa W model must equal the same weights run with a dense band mask."""
    B, T, W, V = 2, 64, 16, 48
    win = _model(W)
    glob = _model(0)
    glob.load_state_dict(win.state_dict())
    x = torch.randint(0, V, (B, T))
    with torch.no_grad():
        a = win(x, None)
        b = glob(x, window_band_mask(T, T, 0, W, x.device))
    assert torch.allclose(a, b, atol=1e-5)


def test_model_swa_single_segment_matches_no_segments():
    """One document filling the window: doc masking must be a no-op."""
    B, T, W, V = 2, 64, 16, 48
    m = _model(W, kda=2, seed=3)
    x = torch.randint(0, V, (B, T))
    with torch.no_grad():
        a = m(x, None, seg_ids=torch.zeros(B, T, dtype=torch.long))
        b = m(x, None)
    assert torch.allclose(a, b, atol=1e-6)


def test_model_swa_segments_block_cross_document_attention():
    B, T, W, V = 2, 64, 16, 48
    m = _model(W, seed=9)
    x = torch.randint(0, V, (B, T))
    seg = (torch.arange(T) // 25).expand(B, T).contiguous()
    with torch.no_grad():
        seg_out = m(x, None, seg_ids=seg)
        plain = m(x, None)
    # The first document is identical either way; later ones must differ.
    assert torch.allclose(seg_out[:, :25], plain[:, :25], atol=1e-5)
    assert not torch.allclose(seg_out[:, 25:], plain[:, 25:], atol=1e-3)


def test_model_swa_backward():
    B, T, W, V = 2, 32, 8, 48
    m = _model(W, kda=2, seed=4).train()
    x = torch.randint(0, V, (B, T))
    seg = (torch.arange(T) // 13).expand(B, T).contiguous()
    loss = m(x, None, seg_ids=seg).float().mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)


def test_swa_cache_decode_matches_full_forward():
    """Token-by-token decode through the KV cache must equal one batched pass."""
    B, T, W, V = 1, 48, 16, 48
    m = _model(W, kda=2, seed=5)
    x = torch.randint(0, V, (B, T))
    with torch.no_grad():
        full = m(x, None)
        m.reset_cache()
        step = [m(x[:, i:i + 1], None, start_pos=i) for i in range(T)]
    assert torch.allclose(torch.cat(step, dim=1), full, atol=2e-4)


def test_swa_chunked_prefill_matches_token_by_token():
    B, W, V = 1, 16, 48
    m = _model(W, seed=6)
    ctx = torch.randint(0, V, (B, 32))
    new = torch.randint(0, V, (B, 20))
    with torch.no_grad():
        m.reset_cache()
        m(ctx, nopeak_mask(32, ctx.device))
        chunk = m(new, nopeak_mask(20, new.device, start_pos=32), start_pos=32)
        m.reset_cache()
        m(ctx, nopeak_mask(32, ctx.device))
        step = [m(new[:, i:i + 1], None, start_pos=32 + i) for i in range(20)]
    assert torch.allclose(torch.cat(step, dim=1), chunk, atol=2e-4)


def test_swa_shorter_than_window_is_global():
    """A sequence that fits inside one window must match plain causal attention."""
    B, T, W, V = 2, 8, 16, 48
    win = _model(W, seed=7)
    glob = _model(0, seed=7)
    glob.load_state_dict(win.state_dict())
    x = torch.randint(0, V, (B, T))
    with torch.no_grad():
        a = win(x, nopeak_mask(T, x.device))
        b = glob(x, nopeak_mask(T, x.device))
    assert torch.allclose(a, b, atol=1e-6)


def test_swa_kv_cache_stays_bounded():
    """A windowed layer's cache must not grow with the number of decode steps."""
    B, W, V = 1, 8, 48
    m = _model(W, seed=10)
    with torch.no_grad():
        m.reset_cache()
        for i in range(40):
            m(torch.randint(0, V, (B, 1)), None, start_pos=i)
    for layer in m.decoder.layers:
        assert layer.attn_1.k_cache[0].size(2) == W - 1


def test_swa_survives_checkpoint_config():
    from core.model import model_from_checkpoint
    m = _model(16, kda=2, seed=8)
    ckpt = {'model': m.state_dict(),
            'config': dict(vocab_size=48, d_model=32, n_layers=4, heads=4,
                           kv_heads=2, dropout=0.0, kda=2, swa=16,
                           value_residual=True, unet_skips=True)}
    rebuilt = model_from_checkpoint(ckpt, torch.device('cpu'), dtype=torch.float32)
    assert [l.attn_1.window for l in rebuilt.decoder.layers if not l.is_kda] == [16, 16]
