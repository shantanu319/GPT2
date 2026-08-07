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
