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


def test_model_from_checkpoint_keeps_one_dtype_and_a_tied_head(tmp_path):
    """assign=True replaces the parameter objects, which unties the head from
    the embedding; and the rope tables are non-persistent buffers, so nothing
    in the state dict converts them. Both have to be handled after the load."""
    from core.model import load_checkpoint, model_from_checkpoint, nopeak_mask

    torch.manual_seed(0)
    cfg = {'vocab_size': 32, 'd_model': 16, 'n_layers': 2, 'heads': 2,
           'dropout': 0.0, 'kv_heads': 1}
    model = Transformer(vocab=cfg['vocab_size'], d_model=cfg['d_model'],
                        N=cfg['n_layers'], heads=cfg['heads'], dropout=0.0,
                        kv_heads=cfg['kv_heads']).eval()
    path = tmp_path / 'ckpt.pt'
    torch.save({'model': model.state_dict(), 'config': cfg}, path)

    loaded = model_from_checkpoint(load_checkpoint(str(path)), torch.device('cpu'),
                                   dtype=torch.bfloat16)
    assert loaded.out.weight is loaded.decoder.embed.embed.weight
    dtypes = {p.dtype for p in loaded.parameters()} | {b.dtype for b in loaded.buffers()}
    assert dtypes == {torch.bfloat16}, dtypes

    x = torch.randint(0, cfg['vocab_size'], (1, 4))
    mask = nopeak_mask(4, torch.device('cpu'))
    with torch.no_grad():
        assert torch.equal(loaded(x, mask), model.to(torch.bfloat16)(x, mask))
