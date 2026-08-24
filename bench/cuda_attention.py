"""Time the attention stack on CUDA, eager and under torch.compile.

The Apple Silicon numbers in bench/compare_mlx.py are launch-bound; this is
the same question asked on the hardware the repo actually trains on, where
compile fuses the KDA scan and flash SDPA changes what global attention costs.

  python -m bench.cuda_attention                 # full sweep
  python -m bench.cuda_attention --no-compile    # eager only
"""
import argparse
import time

import torch

from core.model import Transformer

CONFIGS = [(0, 0), (0, 128), (4, 0), (4, 128), (1, 0)]


def step_time(model, T, B, device, compile_trunk, iters=8, warmup=4):
    """Seconds per fwd+bwd step, doc-masked like a real training window."""
    torch.manual_seed(0)
    m = model.to(device).train()
    if compile_trunk:
        m.decoder = torch.compile(m.decoder)
    x = torch.randint(0, 4000, (B, T), device=device)
    seg = (torch.arange(T, device=device) // (T // 2)).expand(B, T).contiguous()
    for i in range(warmup + iters):
        if i == warmup:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = m(x, None, seg_ids=seg).float().mean()
        loss.backward()
        m.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--seqlens', type=int, nargs='+', default=[1024, 2048, 4096])
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--layers', type=int, default=12)
    ap.add_argument('--d-model', type=int, default=512)
    ap.add_argument('--heads', type=int, default=8)
    ap.add_argument('--kv-heads', type=int, default=2)
    ap.add_argument('--no-compile', action='store_true')
    a = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit('no CUDA device; this benchmark is CUDA-only')
    dev = torch.device('cuda')
    print(f"{torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print(f"B={a.batch} layers={a.layers} d_model={a.d_model} "
          f"heads={a.heads} ({a.kv_heads} kv) | bf16 autocast, doc-masked\n")

    modes = [False] if a.no_compile else [False, True]
    print(f"{'kda':>4} {'swa':>5} {'T':>6} {'compile':>8} {'s/step':>9} {'tok/s':>9}",
          flush=True)
    for kda, swa in CONFIGS:
        for T in a.seqlens:
            for comp in modes:
                model = Transformer(4096, a.d_model, a.layers, a.heads, 0.0,
                                    kv_heads=a.kv_heads, value_residual=True,
                                    unet_skips=True, kda=kda, swa=swa)
                try:
                    dt = step_time(model, T, a.batch, dev, comp)
                    print(f"{kda:>4} {swa:>5} {T:>6} {str(comp):>8} {dt:>9.4f} "
                          f"{a.batch * T / dt:>9.0f}", flush=True)
                except torch.OutOfMemoryError:
                    print(f"{kda:>4} {swa:>5} {T:>6} {str(comp):>8} {'OOM':>9}",
                          flush=True)
                except Exception as e:
                    print(f"{kda:>4} {swa:>5} {T:>6} {str(comp):>8}  "
                          f"{type(e).__name__}: {str(e)[:70]}", flush=True)
                del model
                torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
