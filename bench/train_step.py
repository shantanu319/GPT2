"""Price a whole training step, so a run can be sized to its budget.

  python -m bench.train_step --hours 9
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
      -m bench.train_step --sweep --hours 9

Runs the real step -- the same batch_loss, optimizers, autocast, clip and
gradient all-reduce as pretrain.train -- over synthetic tokens, so it works
before prepare has produced a corpus. The number that matters is the
--max-train-docs it prints: the WSD schedule anneals to total_steps, which
comes from the corpus size, so a corpus sized by guesswork either leaves the
final weights hot mid-schedule or ends training with budget unspent.

--sweep walks batch sizes and the memory/compute knobs smallest-first and
prints one table, which is the shape of an hour of cluster time.
"""
import argparse
import time

import numpy as np
import torch

from core import dist
from core.data import BIN_DTYPE, data_feeder
from core.model import get_model
from pretrain.config import parse_args
from pretrain.telemetry import PhaseTimer, format_phases
from pretrain.train import batch_loss, make_optimizers, resolve_device

# bf16 dense TFLOP/s, for turning measured throughput into an MFU.
PEAK_TFLOPS = {
    'A100': 312, 'H100': 989, 'H200': 989, 'B200': 2250,
    'L40S': 362, 'L40': 362, 'A40': 149, 'A16': 36, 'GH200': 989,
}


def peak_tflops(device):
    if device.type != 'cuda':
        return None
    name = torch.cuda.get_device_name(device)
    return next((value for key, value in PEAK_TFLOPS.items() if key in name), None)


def agreed(ok, device):
    """True only if every rank succeeded. Ranks run identical shapes, so an
    OOM hits all of them; this keeps the sweep in lockstep regardless."""
    if not dist.is_distributed():
        return ok
    dist.barrier()
    return dist.sum_across([1.0 if ok else 0.0], device)[0] == dist.world_size()


def measure(opt, steps, warmup, vocab_size):
    """One config: build, run warmup+steps, return metrics (None on OOM)."""
    grad_accum = max(1, opt.grad_accum)
    model = get_model(opt, vocab_size)
    if opt.device.type == 'cuda' and not opt.no_compile:
        model.decoder = torch.compile(model.decoder)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizers = make_optimizers(model, muon_lr=opt.muon_lr, embed_lr=opt.embed_lr,
                                 scalar_lr=opt.scalar_lr, muon_impl=opt.muon_impl,
                                 muon_per_head=bool(opt.muon_per_head))
    # Every rank takes its own shard and _window_order drops the ragged tail,
    # so the corpus must cover world_size ranks plus a batch of slack.
    batches = (warmup + steps) * grad_accum * dist.world_size() + dist.world_size()
    corpus = np.random.default_rng(0).integers(
        0, vocab_size, batches * opt.batchsize * opt.seqlen, dtype=BIN_DTYPE)

    if opt.device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(opt.device)
    timer = PhaseTimer(opt.device)
    model.train()
    step = micro = 0
    started = None
    try:
        for inX, out in data_feeder(corpus, opt.batchsize, opt.seqlen, opt.device,
                                    rank=dist.rank(), world=dist.world_size()):
            timer.mark('data')
            with torch.autocast(device_type=opt.device.type, dtype=torch.bfloat16):
                loss = batch_loss(model, inX, out, opt)
            timer.mark('fwd')
            (loss / grad_accum).backward()
            timer.mark('bwd')
            micro += 1
            if micro % grad_accum:
                continue
            dist.average_grads(model.parameters())
            timer.mark('reduce')
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.norm)
            for optimizer in optimizers:
                optimizer.step()
                optimizer.zero_grad()
            timer.mark('optim')
            step += 1
            if step == warmup:
                timer.drain()
                dist.barrier()
                started = time.perf_counter()
            elif step == warmup + steps:
                break
    except torch.OutOfMemoryError:
        del model, optimizers, corpus
        torch.cuda.empty_cache()
        agreed(False, opt.device)
        return None
    if not agreed(step == warmup + steps, opt.device):
        raise RuntimeError(f"feeder ran dry after {step} of {warmup + steps} steps")

    if opt.device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    phases = timer.drain()
    tokens_per_step = opt.batchsize * opt.seqlen * grad_accum * dist.world_size()
    rate = tokens_per_step * steps / elapsed
    memory = (torch.cuda.max_memory_allocated(opt.device) / 1024 ** 3
              if opt.device.type == 'cuda' else 0)
    del model, optimizers, corpus
    if opt.device.type == 'cuda':
        torch.cuda.empty_cache()
    return {'params': params, 'ms': elapsed * 1000 / steps, 'rate': rate,
            'memory': memory, 'phases': format_phases(phases, steps)}


def report(opt, result, args):
    peak = peak_tflops(opt.device)
    line = (f"{result['ms']:.0f} ms/step | {result['rate']:,.0f} tokens/s "
            f"({result['rate'] / dist.world_size():,.0f} per GPU)")
    if peak:
        achieved = 6 * result['params'] * result['rate'] / 1e12 / dist.world_size()
        line += f" | {achieved:.0f} of {peak} TFLOP/s = {achieved / peak:.0%} MFU"
    dist.printr(line)
    dist.printr(f"  {result['phases']} | {result['memory']:.1f} GiB peak")
    if args.hours:
        tokens = result['rate'] * args.hours * 3600
        dist.printr(f"  {args.hours:g}h buys {tokens / 1e9:.1f}B tokens "
                    f"({tokens / result['params']:.0f}/param) -> "
                    f"--max-train-docs {int(tokens / args.tokens_per_doc):,}")


def sweep(opt, args):
    """Smallest-first, so an OOM ends the sweep with every earlier row kept."""
    rows = []
    if not opt.no_compile:
        dist.printr("note: each row rebuilds the model, so torch.compile pays a "
                    "fresh compile per config — add -no_compile to sweep faster")
    for batchsize in [int(b) for b in args.sweep_batch.split(',')]:
        for ckpt in (0, 1):
            opt.batchsize, opt.grad_ckpt = batchsize, ckpt
            label = f"batch {batchsize:>4} ckpt {ckpt}"
            result = measure(opt, args.steps, args.warmup, args.vocab_size)
            if result is None:
                dist.printr(f"{label} | OOM")
                rows.append((label, None))
                continue
            rows.append((label, result))
            dist.printr(f"{label} | {result['ms']:>6.0f} ms | "
                        f"{result['rate']:>12,.0f} tok/s | "
                        f"{result['memory']:>5.1f} GiB | {result['phases']}")
    best = max((r for _, r in rows if r), key=lambda r: r['rate'], default=None)
    if best:
        dist.printr(f"\nbest: {best['rate']:,.0f} tokens/s")
        report(opt, best, args)


def main():
    # add_help=False: the training flags use a single dash, so argparse's -h
    # would swallow -heads. allow_abbrev=False keeps -kda etc. out of reach too.
    bench = argparse.ArgumentParser(description=__doc__, add_help=False,
                                    allow_abbrev=False)
    bench.add_argument('--help', action='help')
    bench.add_argument('--steps', type=int, default=20, help='timed optimizer steps')
    bench.add_argument('--warmup', type=int, default=5, help='steps before the clock starts')
    bench.add_argument('--hours', type=float, default=0, help='budget to size a corpus for')
    bench.add_argument('--vocab-size', type=int, default=32000)
    bench.add_argument('--tokens-per-doc', type=int, default=1290,
                       help='estimate for converting tokens into --max-train-docs')
    bench.add_argument('--sweep', action='store_true', help='walk --sweep-batch x grad_ckpt')
    bench.add_argument('--sweep-batch', default='32,64,128')
    args, _ = bench.parse_known_args()

    opt = parse_args()
    opt.device = resolve_device(opt.no_cuda)
    if opt.device.type == 'cuda':
        torch.cuda.set_device(opt.device)
    dist.init(opt.device)
    opt.vocab_size = args.vocab_size
    opt.eos_id = None

    dist.printr(f"{dist.world_size()} rank(s) | seqlen {opt.seqlen} | "
                f"accum {max(1, opt.grad_accum)} | kda {opt.kda} swa {opt.swa} "
                f"| compile {not opt.no_compile}")
    if args.sweep:
        sweep(opt, args)
    else:
        result = measure(opt, args.steps, args.warmup, args.vocab_size)
        if result is None:
            raise SystemExit(f"OOM at batch {opt.batchsize}/rank; lower -batchsize "
                             "or raise -grad_accum")
        dist.printr(f"{result['params']:,} params | batch {opt.batchsize}/rank")
        report(opt, result, args)
    dist.shutdown()


if __name__ == '__main__':
    main()
