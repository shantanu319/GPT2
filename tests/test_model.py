import torch
import torch.nn.functional as F

from core.model import Transformer, nopeak_mask


def _tiny_transformer(vocab=32, d_model=32, n_layers=2, heads=2, dropout=0.0):
    return Transformer(vocab, d_model, n_layers, heads, dropout)


def test_forward_output_shape():
    torch.manual_seed(0)
    B, T, V = 2, 8, 32
    model = _tiny_transformer(vocab=V)
    x = torch.randint(0, V, (B, T))
    mask = nopeak_mask(T, torch.device("cpu"))
    logits = model(x, mask)
    assert logits.shape == (B, T, V)


def test_forward_no_nan():
    torch.manual_seed(0)
    B, T, V = 2, 8, 32
    model = _tiny_transformer(vocab=V)
    x = torch.randint(0, V, (B, T))
    mask = nopeak_mask(T, torch.device("cpu"))
    logits = model(x, mask)
    assert torch.isfinite(logits).all()


def test_backward_produces_gradients():
    torch.manual_seed(0)
    B, T, V = 2, 8, 32
    model = _tiny_transformer(vocab=V)
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    mask = nopeak_mask(T, torch.device("cpu"))
    logits = model(x, mask)
    loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
    loss.backward()

    trainable_params = {name: p for name, p in model.named_parameters() if p.requires_grad}
    missing_grad = [name for name, p in trainable_params.items() if p.grad is None]
    non_finite_grad = [name for name, p in trainable_params.items() if p.grad is not None and not torch.isfinite(p.grad).all()]

    assert not missing_grad, f"parameters missing gradients: {missing_grad}"
    assert not non_finite_grad, f"parameters with non-finite gradients: {non_finite_grad}"


def test_can_overfit_single_batch():
    """Loss should drop substantially when overfitting one tiny batch."""
    torch.manual_seed(0)
    B, T, V = 2, 8, 32
    model = _tiny_transformer(vocab=V, d_model=64, n_layers=2, heads=2)
    x = torch.randint(0, V, (B, T))
    y = torch.randint(0, V, (B, T))
    mask = nopeak_mask(T, torch.device("cpu"))
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    losses = []
    for _ in range(100):
        logits = model(x, mask)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    # Loss should meaningfully decrease from initial value
    assert losses[-1] < losses[0] * 0.5, f"loss did not drop: {losses[0]:.3f} -> {losses[-1]:.3f}"
