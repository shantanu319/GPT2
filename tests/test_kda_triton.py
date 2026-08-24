"""Parity of the fused KDA scan against the Python loop it replaces.

CUDA-only: triton ships no kernels for MPS or CPU, so core.kda falls back to
the loop everywhere else and these have nothing to compare.
"""
import pytest
import torch

from core import kda
from core.kda import kda_chunk, kda_recurrence

pytestmark = pytest.mark.skipif(
    kda.kda_triton is None or not torch.cuda.is_available(),
    reason="needs CUDA and triton")

SHAPE = dict(B=2, H=3, NT=5, BT=64, K=64, V=64)


def _scan_inputs(dtype=torch.float32, B=2, H=3, NT=5, BT=64, K=64, V=64):
    """Operands scaled so the state neither vanishes nor explodes over NT."""
    gen = torch.Generator(device='cuda').manual_seed(0)

    def r(*shape, scale=1.0):
        x = torch.randn(*shape, generator=gen, device='cuda', dtype=dtype)
        return (x * scale).detach().requires_grad_(True)

    return (r(B, H, NT, BT, V), r(B, H, NT, BT, K, scale=K ** -0.5),
            r(B, H, NT, BT, K, scale=K ** -0.5),
            r(B, H, NT, BT, BT, scale=BT ** -0.5),
            r(B, H, NT, BT, K, scale=K ** -0.5),
            torch.rand(B, H, NT, K, generator=gen, device='cuda', dtype=dtype)
            .detach().requires_grad_(True))


def _run(args, S0, fused, dtype=torch.float32):
    kda.kda_triton.ENABLED = fused
    try:
        cloned = [x.detach().to(dtype).clone().requires_grad_(True) for x in args]
        s0 = None if S0 is None else S0.detach().to(dtype).clone().requires_grad_(True)
        o, S = kda.chunk_scan(*cloned[:5], cloned[5], s0)
        (o.float().square().sum() + S.float().square().sum()).backward()
        return [o.double(), S.double()] + [x.grad.double() for x in cloned] \
            + ([] if s0 is None else [s0.grad.double()])
    finally:
        kda.kda_triton.ENABLED = True


NAMES = 'o S u w qg Aqk kg dec S0'.split()


@pytest.mark.parametrize('with_state', [False, True])
def test_fused_scan_is_as_accurate_as_the_loop(with_state):
    """Both paths are fp32 and neither is the truth, so the bar is float64:
    the fused scan has to land at least as close to it as the loop does.
    Comparing the two fp32 paths to each other only measures roundoff -- the
    gradients here run to ~1e4, where fp32 roundoff alone is ~1e-3."""
    args = _scan_inputs()
    S0 = (torch.randn(SHAPE['B'], SHAPE['H'], SHAPE['K'], SHAPE['V'],
                      device='cuda') * 0.1) if with_state else None
    exact = _run(args, S0, False, torch.float64)
    fused = _run(args, S0, True)
    loop = _run(args, S0, False)
    for name, e, f, l in zip(NAMES, exact, fused, loop):
        scale = e.abs().max().item()
        err_f = (f - e).abs().max().item()
        err_l = (l - e).abs().max().item()
        assert err_f <= 2e-6 * scale, f"{name}: {err_f:.3e} vs scale {scale:.3e}"
        assert err_f <= 2 * err_l + 1e-12, f"{name}: fused {err_f:.3e} > loop {err_l:.3e}"


def test_fused_scan_with_no_gradient_on_the_final_state():
    """The training path keeps only o and drops S, so the kernel is handed no
    gradient for the state it ends on -- the one case the parity test above,
    which uses both outputs, cannot reach."""
    args = _scan_inputs()
    out = []
    for fused in (True, False):
        kda.kda_triton.ENABLED = fused
        cloned = [x.detach().clone().requires_grad_(True) for x in args]
        kda.chunk_scan(*cloned[:5], cloned[5], None)[0].square().sum().backward()
        out.append([x.grad.double() for x in cloned])
    kda.kda_triton.ENABLED = True
    for name, f, l in zip(NAMES[2:], *out):
        scale = l.abs().max().item()
        assert (f - l).abs().max().item() <= 1e-5 * scale, name


def test_fused_scan_takes_the_triton_path():
    """Guard the dispatch itself: a silent fallback would make the rest of
    this file compare the loop against itself."""
    u, w = _scan_inputs()[:2]
    assert kda.kda_triton.supported(u, w)
    kda.kda_triton.ENABLED = False
    try:
        assert not kda.kda_triton.supported(u, w)
    finally:
        kda.kda_triton.ENABLED = True
    assert not kda.kda_triton.supported(u.cpu(), w.cpu())
    assert not kda.kda_triton.supported(u[..., :8], w[..., :8])  # tile < 16
    assert not kda.kda_triton.supported(u.double(), w.double())


def test_fused_scan_needs_no_grad_state_when_nothing_requires_grad():
    args = [x.detach() for x in _scan_inputs()]
    with torch.no_grad():
        o, S = kda.chunk_scan(*args[:5], args[5], None)
    assert torch.isfinite(o).all() and torch.isfinite(S).all()


def test_chunk_end_to_end_matches_the_recurrence():
    """The whole layer path, not just the scan: fused chunking still has to
    equal the sequential reference."""
    B, T, H, D = 2, 256, 4, 64
    gen = torch.Generator(device='cuda').manual_seed(1)
    q, k = (torch.nn.functional.normalize(
        torch.randn(B, T, H, D, generator=gen, device='cuda'), dim=-1)
        for _ in range(2))
    v = torch.randn(B, T, H, D, generator=gen, device='cuda')
    g = torch.nn.functional.logsigmoid(
        torch.randn(B, T, H, D, generator=gen, device='cuda'))
    beta = torch.rand(B, T, H, generator=gen, device='cuda')
    o_ref, S_ref = kda_recurrence(q, k, v, g, beta)
    o, S = kda_chunk(q, k, v, g, beta, chunk_size=64)
    torch.testing.assert_close(o, o_ref, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(S, S_ref, atol=2e-4, rtol=2e-4)


def test_chunk_with_segments_still_resets_at_boundaries():
    B, T, H, D = 2, 128, 4, 64
    gen = torch.Generator(device='cuda').manual_seed(2)
    q, k, v, g = (torch.randn(B, T, H, D, generator=gen, device='cuda')
                  for _ in range(4))
    q, k = torch.nn.functional.normalize(q, dim=-1), torch.nn.functional.normalize(k, dim=-1)
    g = torch.nn.functional.logsigmoid(g)
    beta = torch.rand(B, T, H, generator=gen, device='cuda')
    seg = (torch.arange(T, device='cuda') >= 40).long().expand(B, T).contiguous()
    o, S = kda_chunk(q, k, v, g, beta, chunk_size=64, seg_ids=seg)
    o0, _ = kda_recurrence(q[:, :40], k[:, :40], v[:, :40], g[:, :40], beta[:, :40])
    o1, S1 = kda_recurrence(q[:, 40:], k[:, 40:], v[:, 40:], g[:, 40:], beta[:, 40:])
    torch.testing.assert_close(o, torch.cat([o0, o1], dim=1), atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(S, S1, atol=2e-4, rtol=2e-4)


def test_bf16_operands_stay_within_bf16_roundoff():
    """Autocast hands the scan bf16, where the fused path also switches tl.dot
    to tf32 -- 10 mantissa bits against the operands' 8, so nothing is given
    up. The carried state is fp32 either way in the kernel, which is why S
    lands closer to exact than the loop's bf16 state does; the per-chunk
    gradients are a wash, both sitting at bf16 roundoff."""
    args = _scan_inputs()
    exact = _run(args, None, False, torch.float64)
    fused = _run(args, None, True, torch.bfloat16)
    loop = _run(args, None, False, torch.bfloat16)
    for name, e, f, l in zip(NAMES, exact, fused, loop):
        scale = e.abs().max().item()
        err_f, err_l = (f - e).abs().max().item(), (l - e).abs().max().item()
        assert err_f <= 1.5e-2 * scale, f"{name}: {err_f:.3e} vs scale {scale:.3e}"
        assert err_f <= 2 * err_l, f"{name}: fused {err_f:.3e} > loop {err_l:.3e}"
    assert (fused[1] - exact[1]).abs().max() < (loop[1] - exact[1]).abs().max()
