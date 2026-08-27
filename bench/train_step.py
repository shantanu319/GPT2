"""Price a whole training step, so a run can be sized to its budget.

  python -m bench.train_step --hours 9
  python -m torch.distributed.run --standalone --nproc-per-node=8 \
      -m bench.train_step --hours 9 -batchsize 64 -grad_accum 2

Runs the real step -- the same batch_loss, optimizers, autocast, clip and
gradient all-reduce as pretrain.train -- over synthetic tokens, so it works
before prepare has produced a corpus. The number that matters is the
--max-train-docs it prints: the WSD schedule anneals to total_steps, which
comes from the corpus size, so a corpus sized by guesswork either leaves the
final weights hot mid-schedule or ends training with budget unspent.
"""
import argparse
import time

import numpy as np
import torch

from core import dist
from core.data import BIN_DTYPE, data_feeder
from pretrain.config import parse_args
from pretrain.train import batch_loss, make_optimizers, resolve_device
from core.model import get_model

# bf16 dense TFLOP/s, for turning measured throughput into an MFU.
PEAK_TFLOPS = {
    'A100': 312, 'H100': 989, 'H200': 989, 'B200': 2250,
    'L40S': 362, 'L40': 362, 'A40': 149, 'A16': 36, 'GH200': 989,
}


def peak_tflops(device):
    if device.type != 'cuda':
        return None
    name = torch.cuda.get_device_name(device)
    return next((v for k, v in PEAK_TFLOPS.items() if k in name), None)


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
    bench.add_argument('--tokens-per-doc', type=int, default=1000,
                       help='estimate for converting tokens into --max-train-docs')
    args, _ = bench.parse_known_args()

    opt = parse_args()
    opt.device = resolve_device(opt.no_cuda)
    if opt.device.type == 'cuda':
        torch.cuda.set_device(opt.device)
    dist.init(opt.device)
    opt.vocab_size = args.vocab_size
    opt.eos_id = None
    grad_accum = max(1, opt.grad_accum)

    model = get_model(opt, opt.vocab_size)
    if opt.device.type == 'cuda' and not opt.no_compile:
        model.decoder = torch.compile(model.decoder)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizers = make_optimizers(model, muon_lr=opt.muon_lr, embed_lr=opt.embed_lr,
                                 scalar_lr=opt.scalar_lr, muon_impl=opt.muon_impl,
                                 muon_per_head=bool(opt.muon_per_head))

    # Every rank consumes its own shard, and _window_order drops the ragged
    # tail, so the corpus has to cover world_size ranks plus a batch of slack.
    batches = (args.warmup + args.steps) * grad_accum * dist.world_size() + dist.world_size()
    corpus = np.random.default_rng(0).integers(
        0, opt.vocab_size, batches * opt.batchsize * opt.seqlen, dtype=BIN_DTYPE)

    def feeder():
        return data_feeder(corpus, opt.batchsize, opt.seqlen, opt.device,
                           rank=dist.rank(), world=dist.world_size())

    dist.printr(f"{params:,} params | {dist.world_size()} rank(s) | "
                f"batch {opt.batchsize}/rank x {opt.seqlen} x accum {grad_accum}")
    model.train()
    step = micro = 0
    started = None
    for inX, out in feeder():
        with torch.autocast(device_type=opt.device.type, dtype=torch.bfloat16):
            loss = batch_loss(model, inX, out, opt)
        (loss / grad_accum).backward()
        micro += 1
        if micro % grad_accum:
            continue
        dist.average_grads(model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.norm)
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()
        step += 1
        if step == args.warmup:
            if opt.device.type == 'cuda':
                torch.cuda.synchronize()
            dist.barrier()
            started = time.perf_counter()
        elif step == args.warmup + args.steps:
            break
    if opt.device.type == 'cuda':
        torch.cuda.synchronize()
    dist.barrier()
    if step != args.warmup + args.steps:
        raise RuntimeError(
            f"feeder ran dry after {step} of {args.warmup + args.steps} steps; "
            "the timing would be meaningless")
    elapsed = time.perf_counter() - started

    per_step = elapsed / args.steps
    tokens_per_step = opt.batchsize * opt.seqlen * grad_accum * dist.world_size()
    rate = tokens_per_step / per_step
    dist.printr(f"\n{per_step * 1000:.0f} ms/step | {tokens_per_step:,} tokens/step "
                f"| {rate:,.0f} tokens/s ({rate / dist.world_size():,.0f} per GPU)")

    peak = peak_tflops(opt.device)
    if peak:
        achieved = 6 * params * rate / 1e12 / dist.world_size()
        dist.printr(f"{achieved:.1f} TFLOP/s per GPU of {peak} peak bf16 "
                    f"= {achieved / peak:.0%} MFU")
    if args.hours:
        tokens = rate * args.hours * 3600
        dist.printr(f"\n{args.hours:g}h of this buys {tokens / 1e9:.1f}B tokens "
                    f"({tokens / params:.0f} per param)")
        dist.printr(f"  --max-train-docs {int(tokens / args.tokens_per_doc):,} "
                    f"at ~{args.tokens_per_doc} tokens/doc")
    dist.shutdown()


if __name__ == '__main__':
    main()
