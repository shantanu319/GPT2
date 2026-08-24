"""Time the attention stack's two kernels in PyTorch/MPS against MLX.

Both are checked against the PyTorch implementation before anything is timed,
so a drifted port fails loudly instead of producing a fast wrong number.

  python -m bench.compare_mlx            # both benchmarks
  python -m bench.compare_mlx --kda      # just the KDA chunked path
"""
import argparse
import time

import numpy as np
import torch
import torch.nn.functional as F

from core.kda import kda_chunk
from core.model import window_attention, window_block_mask

WINDOW = 128


def _timeit(step, iters, warmup=3):
    for i in range(warmup + iters):
        if i == warmup:
            t0 = time.perf_counter()
        step()
    return (time.perf_counter() - t0) / iters


def _inputs(B, T, H, K, seed=0):
    rng = np.random.default_rng(seed)
    unit = lambda a: (a / np.linalg.norm(a, axis=-1, keepdims=True)).astype(np.float32)
    return (unit(rng.standard_normal((B, T, H, K))),
            unit(rng.standard_normal((B, T, H, K))),
            rng.standard_normal((B, T, H, K)).astype(np.float32),
            (-np.log1p(np.exp(rng.standard_normal((B, T, H, K))))).astype(np.float32),
            rng.random((B, T, H)).astype(np.float32))


def bench_kda(mx, mlx_ops, shapes):
    """kda_chunk forward and forward+backward, PyTorch/MPS vs MLX."""
    arrs = _inputs(2, 256, 4, 32)
    ref, _ = kda_chunk(*(torch.from_numpy(a) for a in arrs), chunk_size=64)
    got, _ = mlx_ops.kda_chunk(*(mx.array(a) for a in arrs), chunk_size=64)
    mx.eval(got)
    print(f"kda_chunk port vs core/kda.py: {np.abs(np.array(got) - ref.numpy()).max():.2e}\n")

    print(f"{'B':>3} {'T':>6} {'H':>3} {'pass':>8} {'torch/mps':>10} {'mlx':>9} "
          f"{'mlx+compile':>12} {'speedup':>8}", flush=True)
    for B, T, H, K in shapes:
        arrs = _inputs(B, T, H, K)
        for grad in (False, True):
            xs = [torch.tensor(a, device='mps', requires_grad=grad) for a in arrs]

            def torch_step():
                o, S = kda_chunk(*xs, chunk_size=64)
                if grad:
                    (o.mean() + S.mean()).backward()
                    for x in xs:
                        x.grad = None
                torch.mps.synchronize()

            ms = [mx.array(a) for a in arrs]

            def loss(*args):
                o, S = mlx_ops.kda_chunk(*args, chunk_size=64)
                return o.mean() + S.mean()

            fn = mx.value_and_grad(loss, argnums=tuple(range(5))) if grad else loss
            t = _timeit(torch_step, 5)
            m = _timeit(lambda: mx.eval(fn(*ms)), 5)
            compiled = mx.compile(fn)
            c = _timeit(lambda: mx.eval(compiled(*ms)), 5)
            print(f"{B:>3} {T:>6} {H:>3} {'fwd+bwd' if grad else 'fwd':>8} "
                  f"{t:>10.4f} {m:>9.4f} {c:>12.4f} {t / min(m, c):>7.2f}x", flush=True)


def bench_attention(mx, mlx_ops, seqlens):
    """Global causal vs sliding-window attention in both frameworks."""
    B, H, KV, D = 4, 8, 2, 64
    scale = D ** -0.5
    rng = np.random.default_rng(0)
    shapes = lambda T: [rng.standard_normal((B, h, T, D)).astype(np.float32)
                        for h in (H, KV, KV)]

    a = shapes(512)
    ref = window_attention(*(torch.tensor(x) for x in a), WINDOW,
                           window_block_mask(WINDOW, torch.device('cpu')),
                           enable_gqa=True)
    got = mlx_ops.window_attention(*(mx.array(x) for x in a), WINDOW,
                                   mlx_ops.window_masks(WINDOW), scale)
    mx.eval(got)
    print(f"window_attention port vs core/model.py: "
          f"{np.abs(np.array(got) - ref.numpy()).max():.2e}\n")

    print(f"{'T':>6} | {'torch global':>12} {'torch swa':>10} | {'mlx global':>10} "
          f"{'mlx swa':>9}", flush=True)
    for T in seqlens:
        a = shapes(T)
        tq, tk, tv = (torch.tensor(x, device='mps') for x in a)
        tm = window_block_mask(WINDOW, tq.device)
        mq, mk, mv = (mx.array(x) for x in a)
        mm = mlx_ops.window_masks(WINDOW)
        swa = mx.compile(lambda q, k, v: mlx_ops.window_attention(q, k, v, WINDOW,
                                                                  mm, scale))
        steps = [
            lambda: (F.scaled_dot_product_attention(tq, tk, tv, is_causal=True,
                                                    enable_gqa=True),
                     torch.mps.synchronize()),
            lambda: (window_attention(tq, tk, tv, WINDOW, tm, enable_gqa=True),
                     torch.mps.synchronize()),
            lambda: mx.eval(mx.fast.scaled_dot_product_attention(
                mq, mk, mv, scale=scale, mask="causal")),
            lambda: mx.eval(swa(mq, mk, mv)),
        ]
        out = []
        for step in steps:
            try:
                out.append(f"{_timeit(step, 8) * 1e3:.2f}m")
            except RuntimeError:      # MPS runs out of memory on the T^2 scores
                out.append("OOM")
                torch.mps.empty_cache()
        print(f"{T:>6} | {out[0]:>12} {out[1]:>10} | {out[2]:>10} {out[3]:>9}",
              flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--kda', action='store_true', help='only the KDA chunked path')
    ap.add_argument('--attn', action='store_true', help='only global vs windowed')
    args = ap.parse_args()

    import mlx.core as mx
    from bench import mlx_ops

    if not args.attn:
        bench_kda(mx, mlx_ops, [(4, 512, 6, 64), (4, 1024, 6, 64),
                                (2, 2048, 6, 64), (8, 1024, 8, 64)])
        print()
    if not args.kda:
        bench_attention(mx, mlx_ops, (512, 1024, 2048, 4096, 8192))


if __name__ == '__main__':
    main()
