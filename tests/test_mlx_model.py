"""The MLX backend must produce the same logits as the torch model it mirrors,
for every architecture variant and on both the uncached and cached paths."""
import numpy as np
import pytest
import torch

import core.model as tmod

mx = pytest.importorskip("mlx.core", reason="mlx is only available on Apple silicon")
mmod = pytest.importorskip("core.mlx_model")

VARIANTS = {
    "plain": {},
    "value_residual": dict(value_residual=True),
    "unet_skips": dict(unet_skips=True),
    "attn_res": dict(attn_res=2),
    "loops": dict(loops=2),
    "kda_hybrid": dict(kda=2, value_residual=True),
    "kda_only": dict(kda=1),
    "everything": dict(kda=2, value_residual=True, unet_skips=True, attn_res=2, loops=2),
}
IDS = [3, 17, 42, 8, 59, 12, 7, 33, 21, 5, 44, 61]
CPU = torch.device("cpu")
TOL = 1e-4  # fp32 reassociation between the two runtimes


def _pair(**kw):
    torch.manual_seed(0)
    t = tmod.Transformer(64, 32, 4, 4, 0.0, kv_heads=2, **kw).eval()
    m = mmod.Transformer(64, 32, 4, 4, kv_heads=2, **kw)
    m.load_weights([(k, mx.array(v.detach().numpy())) for k, v in t.state_dict().items()])
    m.out.weight = m.decoder.embed.embed.weight
    return t, m, torch.tensor([IDS]), mx.array(np.array([IDS]))


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_uncached_logits_match_torch(variant):
    t, m, x, X = _pair(**VARIANTS[variant])
    with torch.no_grad():
        ref = t(x, tmod.nopeak_mask(x.size(1), CPU)).numpy()
    assert np.abs(ref - np.array(m(X))).max() < TOL


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_cached_decode_matches_torch(variant):
    t, m, x, X = _pair(**VARIANTS[variant])
    n = x.size(1)
    with torch.no_grad():
        t.reset_cache()
        ref = [t(x[:, :5], tmod.nopeak_mask(5, CPU), start_pos=0)[:, -1]]
        for i in range(5, n):
            ref.append(t(x[:, i:i + 1], None, start_pos=i)[:, -1])
    m.reset_cache()
    out = [m(X[:, :5], start_pos=0)[:, -1]]
    for i in range(5, n):
        out.append(m(X[:, i:i + 1], start_pos=i)[:, -1])
    err = np.abs(torch.cat(ref).numpy() - np.array(mx.concatenate(out))).max()
    assert err < TOL


def test_multi_token_prefill_at_offset_matches_torch():
    # Multi-turn ingest: a whole prompt appended to a warm cache in one forward.
    t, m, x, X = _pair(kda=2, value_residual=True, unet_skips=True)
    with torch.no_grad():
        t.reset_cache()
        t(x[:, :6], tmod.nopeak_mask(6, CPU), start_pos=0)
        ref = t(x[:, 6:], tmod.nopeak_mask(x.size(1) - 6, CPU, start_pos=6), start_pos=6)
    m.reset_cache()
    m(X[:, :6], start_pos=0)
    assert np.abs(ref.numpy() - np.array(m(X[:, 6:], start_pos=6))).max() < TOL


def test_reset_cache_restarts_generation():
    t, m, x, X = _pair()
    m.reset_cache()
    first = np.array(m(X[:, :5], start_pos=0))
    m.reset_cache()
    assert np.array_equal(first, np.array(m(X[:, :5], start_pos=0)))
