import torch

from inference.sample import top_p_filter


def test_top_p_filter_zeros_out_tail():
    probs = torch.tensor([0.5, 0.2, 0.2, 0.05, 0.05])
    out = top_p_filter(probs, top_p=0.7)
    # Nucleus is {0.5, 0.2} (cumsum = 0.7), tail {0.2, 0.05, 0.05} is dropped
    assert out[0] > 0
    assert out[1] > 0
    assert out[2] == 0
    assert out[3] == 0
    assert out[4] == 0


def test_top_p_filter_always_keeps_top_token():
    # Even a very small top-p should keep at least the highest-prob token.
    probs = torch.tensor([1.0, 0.0, 0.0])
    out = top_p_filter(probs, top_p=0.01)
    assert out[0] > 0


def test_top_p_filter_at_1_is_passthrough():
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    out = top_p_filter(probs, top_p=1.0)
    assert torch.allclose(out, probs)


def test_top_p_filter_preserves_original_position():
    # The non-zero outputs should sit at the original indices of the sorted top-k.
    probs = torch.tensor([0.1, 0.6, 0.3])  # sorted order: idx 1 (0.6), idx 2 (0.3), idx 0 (0.1)
    out = top_p_filter(probs, top_p=0.8)
    # Nucleus = {0.6, 0.3} at original indices 1 and 2; idx 0 dropped.
    assert out[1] > 0
    assert out[2] > 0
    assert out[0] == 0
