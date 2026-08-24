"""The MLX KDA ops must agree with the torch reference they were ported from."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

import core.kda as tkda

mx = pytest.importorskip("mlx.core", reason="mlx is only available on Apple silicon")
mkda = pytest.importorskip("core.mlx_kda")


def _inputs(T, B=1, H=4, K=16, seed=0):
    torch.manual_seed(seed)
    q = F.normalize(torch.randn(B, T, H, K), dim=-1)
    k = F.normalize(torch.randn(B, T, H, K), dim=-1)
    return q, k, torch.randn(B, T, H, K), -torch.rand(B, T, H, K) * 0.3, torch.rand(B, T, H)


def _mlx(*ts):
    return [mx.array(t.numpy()) for t in ts]


@pytest.mark.parametrize("T", [1, 2, 7, 64])
def test_recurrence_matches_torch(T):
    args = _inputs(T)
    o, S = tkda.kda_recurrence(*args)
    mo, mS = mkda.kda_recurrence(*_mlx(*args))
    assert np.abs(o.numpy() - np.array(mo)).max() < 1e-5
    assert np.abs(S.numpy() - np.array(mS)).max() < 1e-5


def test_chunk_matches_torch():
    args = _inputs(128)
    o, S = tkda.kda_chunk(*args)
    mo, mS = mkda.kda_chunk(*_mlx(*args))
    assert np.abs(o.numpy() - np.array(mo)).max() < 1e-5
    assert np.abs(S.numpy() - np.array(mS)).max() < 1e-5


def test_scan_ragged_tail_matches_torch():
    # 70 tokens = one whole chunk through kda_chunk plus a 6-token recurrence tail.
    args = _inputs(70)
    o, S = tkda.kda_scan(*args)
    mo, mS = mkda.kda_scan(*_mlx(*args))
    assert np.abs(o.numpy() - np.array(mo)).max() < 1e-5
    assert np.abs(S.numpy() - np.array(mS)).max() < 1e-5


def test_layer_cached_prefill_matches_torch():
    torch.manual_seed(0)
    t = tkda.KimiDeltaAttention(64, 4).eval()
    m = mkda.KimiDeltaAttention(64, 4)
    m.load_weights([(k, mx.array(v.detach().numpy())) for k, v in t.state_dict().items()])
    x = torch.randn(1, 70, 64)
    X = mx.array(x.numpy())
    with torch.no_grad():
        t.reset_cache()
        ref = torch.cat([t(x[:, :64], start_pos=0), t(x[:, 64:], start_pos=64)], dim=1)
    m.reset_cache()
    out = mx.concatenate([m(X[:, :64], start_pos=0), m(X[:, 64:], start_pos=64)], axis=1)
    assert np.abs(ref.numpy() - np.array(out)).max() < 1e-5
