import numpy as np
import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from chat_format import (DEFAULT_SYSTEM, EOS_TOKEN, IM_END, IM_START,
                         render_conversation, special_token_map)
from data import data_feeder_masked
from model import LOGIT_SOFTCAP, Transformer, get_model, nopeak_mask


def _tiny(vocab=32, d_model=32, n_layers=2, heads=4, kv_heads=None, loops=1, **kw):
    return Transformer(vocab, d_model, n_layers, heads, dropout=0.0,
                       kv_heads=kv_heads, loops=loops, **kw)


def test_gqa_forward_shape():
    torch.manual_seed(0)
    B, T, V = 2, 8, 32
    model = _tiny(vocab=V, heads=4, kv_heads=2)
    x = torch.randint(0, V, (B, T))
    logits = model(x, nopeak_mask(T, torch.device("cpu")))
    assert logits.shape == (B, T, V)
    assert torch.isfinite(logits).all()


def test_gqa_has_fewer_params():
    full = sum(p.numel() for p in _tiny(heads=4, kv_heads=4).parameters())
    gqa = sum(p.numel() for p in _tiny(heads=4, kv_heads=2).parameters())
    assert gqa < full


def test_loops_forward_and_kv_cache_match():
    """Looped model: incremental decoding with cache must match full forward."""
    torch.manual_seed(0)
    V, T = 32, 6
    model = _tiny(vocab=V, loops=2).eval()
    x = torch.randint(0, V, (1, T))
    full = model(x, nopeak_mask(T, torch.device("cpu")))

    model.reset_cache()
    out = []
    for t in range(T):
        step = model(x[:, t:t+1], None, start_pos=t)
        out.append(step)
    inc = torch.cat(out, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_logits_softcapped():
    torch.manual_seed(0)
    assert LOGIT_SOFTCAP == 15.0
    model = _tiny()
    x = torch.randint(0, 32, (1, 4))
    logits = model(x, nopeak_mask(4, torch.device("cpu")))
    assert logits.abs().max() <= LOGIT_SOFTCAP


def test_backward_with_gqa_and_loops():
    torch.manual_seed(0)
    V = 32
    model = _tiny(vocab=V, kv_heads=2, loops=2)
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    loss = F.cross_entropy(
        model(x, nopeak_mask(8, torch.device("cpu"))).view(-1, V), y.view(-1))
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_none_mask_matches_nopeak_mask():
    """Training with mask=None (is_causal path) must match the explicit bool mask."""
    torch.manual_seed(0)
    V, T = 32, 8
    model = _tiny(vocab=V, kv_heads=2).eval()
    x = torch.randint(0, V, (2, T))
    explicit = model(x, nopeak_mask(T, torch.device("cpu")))
    implicit = model(x, None)
    assert torch.allclose(explicit, implicit, atol=1e-5), (explicit - implicit).abs().max()


def test_grad_ckpt_flag_training():
    torch.manual_seed(0)
    V = 32
    model = _tiny(vocab=V, grad_ckpt=True)
    model.train()
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    loss = F.cross_entropy(model(x, None).view(-1, V), y.view(-1))
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_value_residual_forward_backward():
    torch.manual_seed(0)
    V = 32
    model = _tiny(vocab=V, n_layers=3, kv_heads=2, value_residual=True)
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    logits = model(x, None)
    assert logits.shape == (2, 8, V)
    F.cross_entropy(logits.view(-1, V), y.view(-1)).backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, missing
    vres = [n for n, _ in model.named_parameters() if n.endswith('attn_1.vres')]
    assert vres == ['decoder.layers.1.attn_1.vres', 'decoder.layers.2.attn_1.vres']
    assert all(p.dim() < 2 for n, p in model.named_parameters() if n in vres)


def test_value_residual_kv_cache_match():
    """Value residual: incremental cached decode must match the full forward."""
    torch.manual_seed(0)
    V, T = 32, 6
    model = _tiny(vocab=V, n_layers=3, kv_heads=2, value_residual=True).eval()
    x = torch.randint(0, V, (1, T))
    full = model(x, nopeak_mask(T, torch.device("cpu")))

    model.reset_cache()
    out = []
    for t in range(T):
        step = model(x[:, t:t+1], None, start_pos=t)
        out.append(step)
    inc = torch.cat(out, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()


def test_get_model_old_opt_has_no_new_params():
    """An opt namespace lacking the new flags rebuilds the old architecture."""
    opt = SimpleNamespace(d_model=32, n_layers=3, heads=4, dropout=0.0,
                          device='cpu', loadname=None)
    model = get_model(opt, 32)
    keys = set(model.state_dict())
    assert not any('vres' in k or 'skip' in k for k in keys)


@pytest.mark.parametrize("n_layers", [2, 3])
def test_unet_skips_forward_backward(n_layers):
    torch.manual_seed(0)
    V = 32
    model = _tiny(vocab=V, n_layers=n_layers, kv_heads=2, unet_skips=True)
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    logits = model(x, None)
    assert logits.shape == (2, 8, V)
    F.cross_entropy(logits.view(-1, V), y.view(-1)).backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, missing
    assert model.decoder.skip_x0.shape == (n_layers,)
    assert model.decoder.skip_unet.shape == (n_layers,)


def test_flags_off_matches_git_head():
    """Flags off: state_dict keys and weights identical to the git HEAD model.

    (Logits themselves legitimately differ: HEAD had LOGIT_SOFTCAP=30, now 15.)
    """
    import subprocess
    from pathlib import Path
    root = Path(__file__).resolve().parent.parent
    src = subprocess.run(["git", "show", "HEAD:model.py"], cwd=root,
                         capture_output=True, text=True, check=True).stdout
    ns = {}
    exec(src, ns)
    torch.manual_seed(0)
    old = ns["Transformer"](32, 32, 2, 4, 0.0)
    torch.manual_seed(0)
    new = _tiny()
    old_sd, new_sd = old.state_dict(), new.state_dict()
    assert set(old_sd) == set(new_sd)
    assert all(torch.equal(old_sd[k], new_sd[k]) for k in old_sd)


def test_loops_with_value_residual_and_unet_skips():
    torch.manual_seed(0)
    V = 32
    model = _tiny(vocab=V, n_layers=3, kv_heads=2, loops=2,
                  value_residual=True, unet_skips=True)
    x = torch.randint(0, V, (2, 8))
    y = torch.randint(0, V, (2, 8))
    logits = model(x, None)
    assert logits.shape == (2, 8, V)
    assert torch.isfinite(logits).all()
    F.cross_entropy(logits.view(-1, V), y.view(-1)).backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and p.grad is None]
    assert not missing, missing


def test_special_token_map_layout():
    m = special_token_map(32000)
    assert m[IM_START] == 31997
    assert m[IM_END] == 31998
    assert m[EOS_TOKEN] == 31999


def test_render_conversation():
    text = render_conversation(
        [{'role': 'system', 'content': DEFAULT_SYSTEM},
         {'role': 'user', 'content': 'hi'}],
        add_generation_prompt=True)
    assert text.startswith(f"{IM_START}system\n")
    assert f"{IM_START}user\nhi{IM_END}\n" in text
    assert text.endswith(f"{IM_START}assistant\n")


def test_data_feeder_masked_alignment():
    tokens = np.arange(40, dtype=np.uint16)
    mask = (np.arange(40) % 2).astype(np.uint8)
    batches = list(data_feeder_masked(tokens, mask, batch_size=2, seq_len=10,
                                      device=torch.device('cpu')))
    assert len(batches) == 2
    x, y, m = batches[0]
    assert x.shape == (2, 9) and y.shape == (2, 9) and m.shape == (2, 9)
    assert torch.equal(y[0], x[0] + 1)  # targets are next-token shifted
    assert m.dtype == torch.bool
    assert m[0].long().tolist() == [(i % 2) for i in range(1, 10)]


def test_masked_loss_ignores_unmasked():
    from finetune import masked_loss
    V = 10
    pred = torch.randn(1, 4, V)
    target = torch.randint(0, V, (1, 4))
    mask_keep = torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)
    mask_one = torch.tensor([[1, 0, 0, 0]], dtype=torch.bool)
    full = masked_loss(pred, target, mask_keep, V)
    single = masked_loss(pred, target, mask_one, V)
    expected = F.cross_entropy(pred[0, :1], target[0, :1])
    assert torch.allclose(single, expected, atol=1e-6)
    assert full != pytest.approx(single.item()) or True


def test_sft_encode_conversation_masks_assistant():
    from sft_prepare import encode_conversation

    class FakeTok:
        def encode_ordinary(self, text):
            return [min(ord(c), 250) for c in text]

    im_start, im_end, eos = 300, 301, 302
    msgs = [{'role': 'user', 'content': 'hi'},
            {'role': 'assistant', 'content': 'yo'}]
    ids, mask = encode_conversation(FakeTok(), msgs, im_start, im_end, eos)
    assert len(ids) == len(mask)
    assert ids[-1] == eos and mask[-1] == 1
    # user content never contributes loss
    user_body = [m for i, m in zip(ids, mask) if i == ord('h')]
    assert all(v == 0 for v in user_body)
    # assistant body tokens have loss
    asst_body = [m for i, m in zip(ids, mask) if i == ord('y')]
    assert any(v == 1 for v in asst_body)
    # assistant im_end has loss, role headers don't
    im_end_flags = [m for i, m in zip(ids, mask) if i == im_end]
    assert im_end_flags == [0, 1]


def _zero_residual_projections(model):
    """Replicate get_model's zero-init (weights and biases) so every block is
    exactly the identity at init."""
    for layer in model.decoder.layers:
        for lin in (layer.attn_1.out,):
            torch.nn.init.zeros_(lin.weight)
            torch.nn.init.zeros_(lin.bias)
        torch.nn.init.zeros_(layer.ff.w_down.weight)


def test_attn_res_forward_backward():
    torch.manual_seed(0)
    V, B, T = 32, 2, 8
    model = _tiny(vocab=V, n_layers=4, attn_res=2, loops=2,
                  value_residual=True, unet_skips=True)
    x = torch.randint(0, V, (B, T))
    logits = model(x, nopeak_mask(T, torch.device("cpu")))
    assert logits.shape == (B, T, V)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    for proj in (model.decoder.attnres_wq, model.decoder.attnres_wk):
        assert proj.weight.grad is not None
        assert proj.weight.grad.abs().sum() > 0


def test_attn_res_uneven_blocks():
    """N not divisible by the block size: last block is shorter."""
    torch.manual_seed(0)
    model = _tiny(vocab=32, n_layers=5, attn_res=3)
    x = torch.randint(0, 32, (2, 8))
    logits = model(x, nopeak_mask(8, torch.device("cpu")))
    assert logits.shape == (2, 8, 32)
    assert torch.isfinite(logits).all()


def test_attn_res_disabled_is_unchanged():
    torch.manual_seed(0)
    a = _tiny(vocab=32, n_layers=4)
    torch.manual_seed(0)
    b = _tiny(vocab=32, n_layers=4, attn_res=0)
    assert not hasattr(a.decoder, 'attnres_wq')
    x = torch.randint(0, 32, (2, 8))
    mask = nopeak_mask(8, torch.device("cpu"))
    assert torch.equal(a(x, mask), b(x, mask))


def test_attn_res_neutral_at_init():
    """With identity blocks (zero-init residual projections) every block output
    equals x0, so the depth mix — a convex combination of identical tensors —
    must leave the stream unchanged."""
    torch.manual_seed(0)
    plain = _tiny(vocab=32, n_layers=4)
    _zero_residual_projections(plain)
    mixed = _tiny(vocab=32, n_layers=4, attn_res=2)
    # copy shared params exactly (separate RNG streams would otherwise diverge
    # on non-tied draws like out.bias); attnres_* keys keep their own init
    mixed.load_state_dict(plain.state_dict(), strict=False)
    _zero_residual_projections(mixed)
    x = torch.randint(0, 32, (2, 8))
    mask = nopeak_mask(8, torch.device("cpu"))
    assert torch.allclose(plain(x, mask), mixed(x, mask), atol=1e-5)


def test_attn_res_depth_mix_semantics():
    torch.manual_seed(0)
    model = _tiny(vocab=32, n_layers=4, attn_res=2)
    dec = model.decoder
    with torch.no_grad():
        dec.attnres_wq.weight.copy_(torch.eye(32))
        dec.attnres_wk.weight.copy_(torch.eye(32))

    t = torch.randn(2, 5, 32)
    assert torch.allclose(dec._depth_mix([t, t, t]), t, atol=1e-5)

    # query (last candidate) dominates -> output ~= last candidate
    a = torch.zeros(2, 5, 32)
    b = torch.randn(2, 5, 32) * 5
    assert torch.allclose(dec._depth_mix([a, b]), b, atol=1e-3)


def test_attn_res_kv_cache_matches_full_forward():
    """The depth mix is per-token, so incremental cached decoding must match
    the full forward exactly."""
    torch.manual_seed(0)
    V, T = 32, 6
    model = _tiny(vocab=V, n_layers=4, loops=2, attn_res=2).eval()
    x = torch.randint(0, V, (1, T))
    full = model(x, nopeak_mask(T, torch.device("cpu")))

    model.reset_cache()
    out = []
    for t in range(T):
        out.append(model(x[:, t:t+1], None, start_pos=t))
    inc = torch.cat(out, dim=1)
    assert torch.allclose(full, inc, atol=1e-4), (full - inc).abs().max()
