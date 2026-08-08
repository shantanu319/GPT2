import torch

from core.model import Transformer, nopeak_mask, segment_mask, segment_pos_ids


def test_segment_mask_semantics():
    seg = torch.tensor([[0, 0, 1, 1, 2]])
    m = segment_mask(seg)[0]
    expected = torch.tensor([
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1],
    ], dtype=torch.bool)
    assert torch.equal(m, expected)


def test_segment_pos_ids_restart_at_boundaries():
    seg = torch.tensor([[0, 0, 1, 1, 2], [3, 3, 3, 4, 4]])
    pos = segment_pos_ids(seg)
    assert pos.tolist() == [[0, 1, 0, 1, 0], [0, 1, 2, 0, 1]]


def _hybrid():
    torch.manual_seed(0)
    # layer 0 KDA, layer 1 full attention (GQA + partial RoPE)
    return Transformer(vocab=48, d_model=32, N=2, heads=4, dropout=0.0,
                       kv_heads=2, value_residual=True, unet_skips=True, kda=2)


def test_document_isolation():
    """Outputs after a boundary must not depend on tokens before it — for both
    the SDPA layer (mask + RoPE reset) and the KDA layer (state reset)."""
    model = _hybrid().eval()
    x1 = torch.tensor([[1, 2, 3, 10, 11, 12, 13, 14]])
    x2 = torch.tensor([[4, 5, 6, 10, 11, 12, 13, 14]])  # same second document
    seg = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1]])
    with torch.no_grad():
        o1 = model(x1, None, seg_ids=seg)
        o2 = model(x2, None, seg_ids=seg)
        o1_plain = model(x1, nopeak_mask(8, torch.device('cpu')))
        o2_plain = model(x2, nopeak_mask(8, torch.device('cpu')))
    assert torch.allclose(o1[:, 3:], o2[:, 3:], atol=1e-6)
    assert not torch.allclose(o1_plain[:, 3:], o2_plain[:, 3:], atol=1e-6)


def test_no_boundary_window_matches_plain_causal():
    """A single-segment window with seg_ids must equal the old packed-causal
    path exactly (mask, RoPE positions and KDA state all coincide)."""
    model = _hybrid().eval()
    x = torch.randint(0, 48, (2, 12))
    seg = torch.zeros(2, 12, dtype=torch.long)
    with torch.no_grad():
        o_seg = model(x, None, seg_ids=seg)
        o_plain = model(x, nopeak_mask(12, torch.device('cpu')))
    assert torch.allclose(o_seg, o_plain, atol=1e-6)


def test_forward_backward_with_segments():
    model = _hybrid()
    x = torch.randint(0, 48, (2, 12))
    y = torch.randint(0, 48, (2, 12))
    seg = torch.tensor([[0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2]] * 2)
    logits = model(x, None, seg_ids=seg)
    assert logits.shape == (2, 12, 48)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, 48), y.view(-1))
    loss.backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name


def test_forward_backward_with_segments_grad_ckpt():
    torch.manual_seed(0)
    model = Transformer(vocab=48, d_model=32, N=2, heads=4, dropout=0.0,
                        kv_heads=2, kda=2, grad_ckpt=True)
    model.train()
    x = torch.randint(0, 48, (2, 12))
    seg = torch.tensor([[0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]] * 2)
    logits = model(x, None, seg_ids=seg)
    logits.sum().backward()
    for name, p in model.named_parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all(), name
