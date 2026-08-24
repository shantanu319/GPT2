import torch

from core.model import nopeak_mask


def test_nopeak_mask_shape():
    mask = nopeak_mask(5, torch.device("cpu"))
    assert mask.shape == (1, 5, 5)


def test_nopeak_mask_is_causal():
    """Position i can attend to positions j <= i only."""
    N = 5
    mask = nopeak_mask(N, torch.device("cpu"))[0]  # (N, N)
    for i in range(N):
        for j in range(N):
            expected = j <= i
            assert bool(mask[i, j].item()) is expected, (
                f"position {i} attending to {j}: expected {expected}, got {bool(mask[i, j].item())}"
            )


def test_nopeak_mask_with_start_pos_is_rectangular():
    """A chunk fed at start_pos sees the whole cache plus its own past."""
    start, size = 4, 3
    mask = nopeak_mask(size, torch.device("cpu"), start_pos=start)
    assert mask.shape == (1, size, start + size)
    for i in range(size):
        for j in range(start + size):
            assert bool(mask[0, i, j].item()) is (j <= start + i)


def test_chunked_prefill_matches_token_by_token():
    """Feeding a chunk into an existing cache must equal feeding it one token
    at a time -- the batched multi-turn prefill path in chat_server."""
    from core.model import Transformer
    from inference.sample import TorchBackend

    backend = TorchBackend(torch.device("cpu"))
    torch.manual_seed(0)
    model = Transformer(vocab=64, d_model=32, N=2, heads=2, dropout=0.0, kv_heads=1).eval()
    ctx, new = [3, 9, 14, 2, 7], [11, 5, 30, 1]

    with torch.no_grad():
        model.reset_cache()
        backend.prefill(model, ctx)
        for i, tok in enumerate(new):
            one = model(torch.tensor([[tok]]), None, start_pos=len(ctx) + i)[:, -1, :]

        model.reset_cache()
        backend.prefill(model, ctx)
        chunk = backend.prefill(model, new, start_pos=len(ctx))

    assert torch.allclose(one, chunk, atol=1e-5)
