"""Measure the fused KDA scan against the Python loop it replaces.

  python -m bench.kda_kernel              # scan-only + whole-layer timings
  python -m bench.kda_kernel --sweep      # warps / tl.dot precision grid

Section 1 prices the scan in isolation, which is what the kernel changes.
Section 2 prices a whole kda_chunk call, eager and compiled, which is what
the training step actually sees -- the scan's share of that is the ceiling
on what any amount of tuning here can buy.
"""
import argparse
import time

import torch

from core import kda
from core.kda import kda_chunk


def timed(fn, iters=20, warmup=5):
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def scan_inputs(B, H, T, K, BT, dtype):
    NT = T // BT
    def leaf(last, scale=1.0):
        x = torch.randn(B, H, NT, BT, last, device='cuda', dtype=dtype) * scale
        return x.requires_grad_(True)
    return (leaf(K), leaf(K, K ** -0.5), leaf(K, K ** -0.5),
            leaf(BT, BT ** -0.5), leaf(K, K ** -0.5),
            torch.rand(B, H, NT, K, device='cuda', dtype=dtype, requires_grad=True))


def layer_inputs(B, H, T, K, dtype):
    n = lambda: torch.randn(B, T, H, K, device='cuda', dtype=dtype, requires_grad=True)
    g = torch.nn.functional.logsigmoid(n()).detach().requires_grad_(True)
    beta = torch.rand(B, T, H, device='cuda', dtype=dtype, requires_grad=True)
    return n(), n(), n(), g, beta


def fwd_bwd(fn, args, backward, autocast=False):
    def run():
        with torch.autocast('cuda', torch.bfloat16, enabled=autocast):
            out = fn(*args)[0]
        if backward:
            out.float().square().sum().backward()
            for a in args:
                if a is not None:
                    a.grad = None
    return run


def ab(label, make_fn, args, backward, autocast=False):
    """Time with the fused scan on and off; return (fused, loop) seconds.

    make_fn is rebuilt per setting: dynamo guards on the ENABLED global, so a
    shared compiled object would spend the measurement recompiling."""
    out = []
    for on in (True, False):
        kda.kda_triton.ENABLED = on
        out.append(timed(fwd_bwd(make_fn(), args, backward, autocast)))
    kda.kda_triton.ENABLED = True
    print(f"{label:>28} {out[0] * 1e3:>9.3f} {out[1] * 1e3:>9.3f} "
          f"{out[1] / out[0]:>7.2f}x", flush=True)
    return out


def parity(args):
    outs = []
    for on in (True, False):
        kda.kda_triton.ENABLED = on
        cloned = [x.detach().clone().requires_grad_(True) for x in args]
        o, S = kda.chunk_scan(*cloned[:5], cloned[5], None)
        (o.float().square().sum() + S.float().square().sum()).backward()
        outs.append([o.float(), S.float()] + [x.grad.float() for x in cloned])
    kda.kda_triton.ENABLED = True
    err = max((a - b).abs().max().item() / max(b.abs().max().item(), 1e-9)
              for a, b in zip(*outs))
    print(f"fused-vs-loop relative error (outputs and all grads): {err:.3e}\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seqlens', type=int, nargs='+', default=[1024, 2048, 4096])
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--head-dim', type=int, default=64)
    ap.add_argument('--chunk', type=int, default=64)
    ap.add_argument('--dtype', default='bf16', choices=['bf16', 'fp32'])
    ap.add_argument('--sweep', action='store_true')
    a = ap.parse_args()

    if kda.kda_triton is None or not torch.cuda.is_available():
        raise SystemExit('needs CUDA and triton')
    dt = torch.bfloat16 if a.dtype == 'bf16' else torch.float32
    B, H, K = a.batch, a.heads, a.head_dim
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__} | "
          f"B={B} H={H} K={K} chunk={a.chunk} {a.dtype}\n")

    parity(scan_inputs(B, H, 1024, K, a.chunk, torch.float32))

    if a.sweep:
        print(f"{'T':>6} {'warps':>10} {'prec':>6} {'ms':>9}")
        for T in a.seqlens:
            args = scan_inputs(B, H, T, K, a.chunk, dt) + (None,)
            kda.kda_triton.ENABLED = False
            base = timed(fwd_bwd(kda.chunk_scan, args, True)) * 1e3
            kda.kda_triton.ENABLED = True
            print(f"{T:>6} {'loop':>10} {'-':>6} {base:>9.3f}", flush=True)
            for prec in ('ieee', 'tf32'):
                for fw, bw in ((2, 4), (4, 4), (4, 8), (8, 8), (8, 16),
                               (16, 16), (16, 32)):
                    kda.kda_triton.WARPS = (fw, bw)
                    kda.kda_triton.PRECISION = prec
                    dtms = timed(fwd_bwd(kda.chunk_scan, args, True)) * 1e3
                    print(f"{T:>6} {f'{fw}/{bw}':>10} {prec:>6} {dtms:>9.3f}", flush=True)
            kda.kda_triton.WARPS = kda.kda_triton.PRECISION = None
        return

    print(f"{'case':>28} {'fused':>9} {'loop':>9} {'gain':>8}")
    for T in a.seqlens:
        args = scan_inputs(B, H, T, K, a.chunk, dt) + (None,)
        ab(f"scan T={T} fwd", lambda: kda.chunk_scan, args, False)
        ab(f"scan T={T} fwd+bwd", lambda: kda.chunk_scan, args, True)
    print()

    def fresh():
        torch._dynamo.reset()
        return torch.compile(kda_chunk)

    for T in a.seqlens:
        args = layer_inputs(B, H, T, K, dt)
        ab(f"kda_chunk T={T} eager", lambda: kda_chunk, args, True, autocast=True)
        ab(f"kda_chunk T={T} compiled", fresh, args, True, autocast=True)


if __name__ == '__main__':
    main()
