"""Decode throughput, torch backend vs MLX backend, on the same checkpoint.

Times the decode loop the CLI actually runs rather than isolated ops, so
per-step launch overhead and the periodic token read-back are both in the
number -- on this model they are most of it.

  python -m inference.bench_decode --checkpoint saved/model/ckpt_final.pt \
      --data-dir data_cache/cosmopedia
"""
import argparse
import os
import statistics
import time

from core.model import load_checkpoint
from core.tokenizer import BPETokenizer
from inference.sample import (BACKENDS, build_backend, checkpoint_params,
                              decode_loop)


def timed_run(model, backend, ids, n_tokens, max_context, temperature, top_p):
    """(prefill seconds, decode seconds) for one generation from a cold cache."""
    model.reset_cache()
    context = list(ids)
    t0 = time.perf_counter()
    last = backend.prefill(model, context)
    backend.wait(last)
    t1 = time.perf_counter()
    # No stop ids: every run generates exactly n_tokens, so the runs compare.
    n, _ = decode_loop(model, last, context, len(context), n_tokens, temperature,
                       top_p, max_context, backend, set())
    t2 = time.perf_counter()
    assert n == n_tokens, f"generated {n} of {n_tokens} tokens"
    return t1 - t0, t2 - t1


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='data_cache/cosmopedia')
    parser.add_argument('--backends', nargs='+', default=['torch', 'mlx'],
                        choices=BACKENDS)
    parser.add_argument('--prompt', default='The history of the Roman empire is')
    parser.add_argument('--prompt-tokens', type=int, default=64,
                        help='Prompt is tiled/truncated to this many tokens')
    parser.add_argument('--tokens', type=int, default=128, help='Tokens to decode per run')
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--max-context', type=int, default=512)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(args.data_dir, 'tokenizer.json'))
    seed = tokenizer.encode(args.prompt)
    ids = (seed * (args.prompt_tokens // len(seed) + 1))[:args.prompt_tokens]

    ckpt = load_checkpoint(args.checkpoint)
    cfg = ckpt['config']
    print(f"{args.checkpoint}: {checkpoint_params(ckpt) / 1e6:.1f}M params, "
          f"d_model={cfg['d_model']} layers={cfg['n_layers']} heads={cfg['heads']} "
          f"kv_heads={cfg.get('kv_heads')} kda={cfg.get('kda', 0)}")
    print(f"prompt {len(ids)} tokens, decoding {args.tokens}, "
          f"{args.repeats} runs, max_context {args.max_context}\n")

    runners = []
    for name in args.backends:
        model, backend, label = build_backend(ckpt, name, args.no_cuda)
        backend.seed(args.seed)
        runners.append((label, lambda m=model, b=backend: timed_run(
            m, b, ids, args.tokens, args.max_context, args.temperature, args.top_p)))

    for _ in range(2):  # warm-up: kernel compilation and the first allocations
        for _, run in runners:
            run()
    # Round-robin rather than all of one backend then all of the other, so GPU
    # clock drift over the run lands on both of them equally.
    rows = {label: [] for label, _ in runners}
    for _ in range(args.repeats):
        for label, run in runners:
            rows[label].append(run())

    decode_rates = {}
    for label, _ in runners:
        prefill = [len(ids) / p for p, _ in rows[label]]
        decode = sorted(args.tokens / d for _, d in rows[label])
        decode_rates[label] = statistics.median(decode)
        print(f"{label:>6}  prefill {statistics.median(prefill):8.1f} tok/s   "
              f"decode {statistics.median(decode):7.2f} tok/s "
              f"[{decode[0]:.2f}-{decode[-1]:.2f}]")

    base, rate = next(iter(decode_rates.items()))
    rest = list(decode_rates.items())[1:]
    if rest:
        print(f"\ndecode speedup over {base}:   "
              + "   ".join(f"{label} {r / rate:.2f}x" for label, r in rest))


if __name__ == '__main__':
    main()
