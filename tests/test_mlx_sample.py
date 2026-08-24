"""The MLX sampling primitives must behave like the torch ones they stand in
for, and drive the shared decode loop to the same text."""
import numpy as np
import pytest
import torch

import core.model as tmod
from inference.sample import TorchBackend, generate
from inference.sample import top_p_filter as torch_top_p

mx = pytest.importorskip("mlx.core", reason="mlx is only available on Apple silicon")
mmod = pytest.importorskip("core.mlx_model")
msample = pytest.importorskip("inference.mlx_sample")


class _IdentityTokenizer:
    """Tokens are their own ids, so generate()'s output is the id sequence."""

    def encode(self, text):
        return [int(t) for t in text.split()]

    def decode(self, ids):
        return " ".join(str(i) for i in ids)


@pytest.mark.parametrize("top_p", [0.01, 0.5, 0.7, 0.9, 1.0])
def test_top_p_filter_matches_torch(top_p):
    probs = torch.tensor([0.4, 0.25, 0.2, 0.1, 0.04, 0.01])
    ref = torch_top_p(probs, top_p).numpy()
    out = np.array(msample.top_p_filter(mx.array(probs.numpy()), top_p))
    assert np.allclose(ref, out)


def test_sample_stays_in_the_nucleus():
    mx.random.seed(0)
    backend = msample.MLXBackend()
    logits = mx.array([[10.0, 9.0, -20.0, -20.0, -20.0]])
    for _ in range(50):
        tok = backend.sample(logits, temperature=1.0, top_p=0.9)
        assert tok.shape == (1, 1)
        assert tok.item() in (0, 1)


@pytest.mark.parametrize("kw", [{}, dict(kda=2, value_residual=True, unet_skips=True)])
def test_greedy_generation_matches_torch(kw):
    torch.manual_seed(0)
    t = tmod.Transformer(64, 32, 4, 4, 0.0, kv_heads=2, **kw).eval()
    m = mmod.Transformer(64, 32, 4, 4, kv_heads=2, **kw)
    m.load_weights([(k, mx.array(v.detach().numpy())) for k, v in t.state_dict().items()])
    m.out.weight = m.decoder.embed.embed.weight

    tok, prompt = _IdentityTokenizer(), "3 17 42 8"
    args = dict(max_tokens=24, temperature=0.01, top_p=1.0, max_context=32,
                stop_at_eos=False)
    ref = generate(t, tok, prompt, backend=TorchBackend(torch.device("cpu")), **args)
    out = generate(m, tok, prompt, backend=msample.MLXBackend(), **args)
    assert ref == out


def test_generation_survives_a_context_overflow():
    # max_context below max_tokens forces the KV window to be rebuilt mid-run.
    torch.manual_seed(0)
    t = tmod.Transformer(64, 32, 2, 2, 0.0).eval()
    m = mmod.Transformer(64, 32, 2, 2)
    m.load_weights([(k, mx.array(v.detach().numpy())) for k, v in t.state_dict().items()])
    m.out.weight = m.decoder.embed.embed.weight

    tok = _IdentityTokenizer()
    args = dict(max_tokens=40, temperature=0.01, top_p=1.0, max_context=16,
                stop_at_eos=False)
    ref = generate(t, tok, "3 17 42 8", backend=TorchBackend(torch.device("cpu")), **args)
    out = generate(m, tok, "3 17 42 8", backend=msample.MLXBackend(), **args)
    assert len(out.split()) == 44
    assert ref == out
