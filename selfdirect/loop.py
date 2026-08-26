"""The self-directed continual loop: the model's own progress picks its reading.

Each round:
  1. probe every arm on its fixed held-out slice
  2. the director picks one arm from its exponential weights
  3. train a block of steps on that arm, resuming from that arm's own cursor
  4. probe every arm again -- the reward is the mean loss drop across ALL arms,
     so a domain that improves itself by wrecking the others scores near zero
  5. director update, journal the round, repeat

Round r's "after" probe is round r+1's "before", so the probe is paid for once
per round, not twice.

There is no fixed horizon and so no LR schedule past the warmup: a run meant
never to end cannot be annealed toward a finish line. Weights, optimizers,
cursors and director state all live in one checkpoint, which is also a plain
training checkpoint that inference/sample.py can read.

  python -m selfdirect.loop --checkpoint saved/model/ckpt_final.pt --rounds 200
"""
import argparse
import json
import os
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from core.data import data_feeder, load_bin
from core.model import LOGIT_SOFTCAP, get_model, load_checkpoint, model_from_config
from pretrain.train import (batch_loss, make_optimizers, resolve_device,
                            save_checkpoint)
from selfdirect.director import Director
from selfdirect.domains import load_manifest

STATE_FILE = 'state.pt'
JOURNAL_FILE = 'journal.jsonl'


@dataclass
class Arm:
    name: str
    train: np.ndarray
    probe: np.ndarray
    cursor: int = 0


def load_arms(data_dir, probe_tokens=0):
    """probe_tokens caps how much of each probe shard the loop reads. The probe
    is fixed, so it carries no sampling noise and only has to be big enough to
    be representative: on the 98M checkpoint a 4k-token probe put the measured
    round delta within 3% of the full 33k one, for a seventh of the time."""
    manifest = load_manifest(data_dir)
    arms = []
    for a in manifest['arms']:
        probe = load_bin(os.path.join(data_dir, a['name'], 'probe.bin'))
        arms.append(Arm(a['name'],
                        load_bin(os.path.join(data_dir, a['name'], 'train.bin')),
                        probe[:probe_tokens] if probe_tokens else probe))
    return arms, manifest


def take_block(arm, n_tokens):
    """The next n_tokens of an arm's train shard, wrapping at the end.

    The cursor is per-arm and survives rounds, so a domain the director keeps
    coming back to keeps reading forward instead of re-reading its opening
    pages every time."""
    size = len(arm.train)
    start, chunks, taken = arm.cursor % size, [], 0
    while taken < n_tokens:
        stop = min(start + n_tokens - taken, size)
        chunks.append(arm.train[start:stop])
        taken += stop - start
        start = 0 if stop == size else stop
    arm.cursor = start
    return chunks[0] if len(chunks) == 1 else np.concatenate(chunks)


def ce_sum(hidden, weight, bias, targets, chunk=2048):
    """Soft-capped cross-entropy summed over rows, in chunks so the (N, V)
    logits are never materialized -- pretrain.fused_ce only takes its chunked
    path on CUDA, and the probe has to run anywhere."""
    total = torch.zeros((), dtype=torch.float32, device=hidden.device)
    for lo in range(0, hidden.size(0), chunk):
        z = LOGIT_SOFTCAP * torch.tanh(
            F.linear(hidden[lo:lo + chunk], weight, bias).float() / LOGIT_SOFTCAP)
        picked = z.gather(1, targets[lo:lo + chunk, None]).squeeze(1)
        total += (torch.logsumexp(z, -1) - picked).sum()
    return total


@torch.no_grad()
def probe_all(model, arms, opt):
    """Mean loss on every arm's probe shard.

    fp32 and in eval mode. Measured against a bf16 autocast probe on MPS, fp32
    was both more precise (autocast moved the mean delta by 6e-5, which is a
    real fraction of a steady-state round) and faster: under no_grad autocast's
    cast cache is dead, so every weight is re-cast on every forward."""
    model.eval()
    losses = []
    for arm in arms:
        total, n = torch.zeros((), device=opt.device), 0
        for x, y, seg in data_feeder(arm.probe, opt.batchsize, opt.seqlen,
                                     opt.device, eos_id=opt.eos_id):
            hidden = model.decoder(x, None, seg_ids=seg)
            total += ce_sum(hidden.reshape(-1, hidden.size(-1)),
                            model.out.weight, model.out.bias, y.reshape(-1))
            n += y.numel()
        losses.append((total / max(n, 1)).item())
    model.train()
    return losses


def short_names(names, width=4):
    """Shortest prefix that still tells every arm apart, for the mix readout."""
    longest = max(len(n) for n in names)
    while width < longest and len({n[:width] for n in names}) < len(names):
        width += 1
    return [n[:width] for n in names]


def backoff(opt, history):
    """Halve the LR when the global probe loss is no better than it was
    lr_patience rounds ago.

    pretrain/train.py early-stops on a stalled validation loss; a run with no
    horizon has nothing to stop into, so the analogous move is to shorten the
    step. It reads the mean the director already computes, so it costs nothing,
    and it only ever goes down -- floored, so the loop keeps learning."""
    p = opt.lr_patience
    if not p or opt.round % p or len(history) <= p or opt.lr_scale <= opt.lr_floor:
        return
    if history[-1] >= history[0]:
        opt.lr_scale = max(opt.lr_floor, opt.lr_scale / 2)
        print(f"probe loss no better than {p} rounds ago -> LR x{opt.lr_scale:g}")


def set_lr(optimizers, factor):
    for o in optimizers:
        for group in o.param_groups:
            group['lr'] = group['peak_lr'] * factor


def train_block(model, arm, opt):
    """One block of optimizer steps on a single arm. Returns its mean loss."""
    block = take_block(arm, opt.block_tokens)
    # Accumulated on the device: an .item() per micro-batch is a sync per
    # micro-batch, and the mean is only wanted once at the end.
    total, n, micro = torch.zeros((), device=opt.device), 0, 0
    for o in opt.optimizers:
        o.zero_grad()
    for x, y, seg in data_feeder(block, opt.batchsize, opt.seqlen, opt.device,
                                 eos_id=opt.eos_id):
        with torch.autocast(device_type=opt.device.type, dtype=torch.bfloat16):
            loss = batch_loss(model, x, y, opt, seg=seg)
        (loss / opt.grad_accum).backward()
        total += loss.detach()
        n += 1
        micro += 1
        if micro % opt.grad_accum:
            continue
        set_lr(opt.optimizers,
               opt.lr_scale * min(1.0, (opt.step + 1) / max(opt.warmup_steps, 1)))
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=opt.norm)
        for o in opt.optimizers:
            o.step()
            o.zero_grad()
        opt.step += 1
    return (total / max(n, 1)).item()


def save_state(model, arms, director, opt, path):
    save_checkpoint(model, opt.optimizers, opt.step, path, config=opt.model_config,
                    extra={'selfdirect': {
                        'director': director.state_dict(),
                        'cursors': {a.name: a.cursor for a in arms},
                        'round': opt.round,
                        'best_probe': opt.best_probe,
                        'lr_scale': opt.lr_scale,
                        'probe_history': list(opt.probe_history),
                    }})


def load_state(model, arms, director, opt, path):
    """Optimizer, director and cursor state from a saved run; the weights and
    architecture already came out of the same file via build_model."""
    state = load_checkpoint(path)
    for o, s in zip(opt.optimizers, state['optimizers']):
        o.load_state_dict(s)
    sd = state['selfdirect']
    director.load_state_dict(sd['director'])
    for arm in arms:
        arm.cursor = sd['cursors'][arm.name]
    opt.step, opt.round, opt.best_probe = state['step'], sd['round'], sd['best_probe']
    opt.lr_scale = sd['lr_scale']
    opt.probe_history.extend(sd['probe_history'])
    print(f"resumed from {path}: round {opt.round}, step {opt.step}")


def run(model, arms, director, opt):
    names = [a.name for a in arms]
    labels = short_names(names)
    journal = os.path.join(opt.out, JOURNAL_FILE)
    before = probe_all(model, arms, opt)
    history = opt.probe_history

    for _ in range(opt.rounds):
        idx = director.choose()
        probs = director.probs()
        started = time.time()
        train_loss = train_block(model, arms[idx], opt)
        after = probe_all(model, arms, opt)

        # Reward: mean probe-loss drop over every arm, not just the one studied.
        # Absolute nats, not proportional: the objective is the loss a uniform
        # mixture of the probes would report, so a domain the model is already
        # good at has less left to give and should say so.
        reward = sum(b - a for b, a in zip(before, after)) / len(arms)
        scaled = director.update(idx, reward)
        # Folding `after` into the running best first makes the gap below the
        # amount given back since that arm's best, and never negative.
        opt.best_probe = [min(b, a) for b, a in zip(opt.best_probe, after)]
        forgetting = sum(a - b for a, b in zip(after, opt.best_probe))
        opt.round += 1
        history.append(sum(after) / len(after))
        backoff(opt, history)

        with open(journal, 'a') as f:
            f.write(json.dumps({
                'round': opt.round, 'step': opt.step, 'studied': names[idx],
                'train_loss': train_loss, 'reward': reward, 'scaled': scaled,
                'forgetting': forgetting, 'lr_scale': opt.lr_scale,
                'seconds': round(time.time() - started, 2),
                'probs': dict(zip(names, probs)),
                'probe_before': dict(zip(names, before)),
                'probe_after': dict(zip(names, after)),
            }) + '\n')
        mix = ' '.join(f'{n} {p*100:3.0f}' for n, p in zip(labels, probs))
        print(f"r{opt.round:>4} | {names[idx]:<14} | train {train_loss:5.3f} | "
              f"probe {sum(after)/len(after):6.4f} ({reward:+.4f}) | "
              f"forget {forgetting:.4f} | {mix}")

        before = after
        if opt.save_every and opt.round % opt.save_every == 0:
            save_state(model, arms, director, opt, os.path.join(opt.out, STATE_FILE))
    save_state(model, arms, director, opt, os.path.join(opt.out, STATE_FILE))


def build_model(opt, manifest):
    """From --checkpoint when given (its config wins), else a fresh model at the
    architecture the -d_model/-n_layers/-heads flags describe."""
    if opt.checkpoint:
        ckpt = load_checkpoint(opt.checkpoint)
        cfg = ckpt['config']
        if cfg is None:
            raise ValueError(f"{opt.checkpoint} has no config — retrain with train.py")
        if cfg['vocab_size'] != manifest['vocab_size']:
            raise ValueError(f"checkpoint vocab {cfg['vocab_size']} != arm vocab "
                             f"{manifest['vocab_size']} — different tokenizer")
        model = model_from_config(cfg, opt.device)
        model.load_state_dict(ckpt['model'])
        return model, cfg
    opt.loadname = None
    model = get_model(opt, manifest['vocab_size'])
    return model, {'vocab_size': manifest['vocab_size'], 'd_model': opt.d_model,
                   'n_layers': opt.n_layers, 'heads': opt.heads,
                   'kv_heads': opt.kv_heads or opt.heads, 'dropout': opt.dropout}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data-dir', default='data_cache/selfdirect',
                   help='Arm shards from selfdirect.domains')
    p.add_argument('--checkpoint', help='Pretrain checkpoint to continue (omit to '
                                        'start the curriculum from scratch)')
    p.add_argument('--out', default='saved/selfdirect')
    p.add_argument('--rounds', type=int, default=200)
    p.add_argument('--block-steps', type=int, default=25,
                   help='Optimizer steps per round on the chosen arm. Smaller '
                        'blocks give the director finer control and pay the '
                        'probe more often')
    p.add_argument('--batchsize', type=int, default=8)
    p.add_argument('--seqlen', type=int, default=512)
    p.add_argument('--grad-accum', type=int, default=1)
    p.add_argument('--probe-tokens', type=int, default=8192,
                   help='Probe tokens read per arm per round (0 = the whole '
                        'shard). The probe is fixed, so this trades how '
                        'representative the reward is against its cost')
    p.add_argument('--lr', type=float, default=3e-4, help='AdamW peak LR')
    p.add_argument('--muon-lr', type=float, default=0.01)
    p.add_argument('--embed-lr', type=float, default=1e-3)
    p.add_argument('--scalar-lr', type=float, default=0.005)
    p.add_argument('--muon-impl', choices=['local', 'torch'], default='torch',
                   help="stdlib torch.optim.Muon, as sft/finetune.py uses. "
                        "Measured 22%% faster per step than pretrain/muon.py "
                        "on MPS (1.33s vs 1.70s at 98M params)")
    p.add_argument('--warmup-steps', type=int, default=50,
                   help='Kept short: while the LR ramps, every round improves '
                        'on the last for reasons that have nothing to do with '
                        'which arm was picked')
    p.add_argument('--grad-clip', type=float, default=1.0)
    p.add_argument('--lr-patience', type=int, default=10,
                   help='Halve the LR when the mean probe loss is no better '
                        'than it was this many rounds ago (0 disables)')
    p.add_argument('--lr-floor', type=float, default=0.03,
                   help='Floor for that backoff, as a fraction of the peak LR')
    p.add_argument('--ce-chunk', type=int, default=16384)
    p.add_argument('--eta', type=float, default=0.08, help='Director step size')
    p.add_argument('--explore', type=float, default=0.1,
                   help='Total probability held back for exploration')
    p.add_argument('--decay', type=float, default=0.01,
                   help='Pull of the director weights back toward uniform')
    p.add_argument('--window', type=int, default=32,
                   help='Rounds of reward history the rank rescaling looks at')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--save-every', type=int, default=20, help='Rounds per save')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-cuda', action='store_true')
    # Architecture, used only when --checkpoint is omitted.
    p.add_argument('-d_model', type=int, default=256)
    p.add_argument('-n_layers', type=int, default=6)
    p.add_argument('-heads', type=int, default=4)
    p.add_argument('-kv_heads', type=int, default=0)
    p.add_argument('-dropout', type=float, default=0.0)
    return p.parse_args()


def main():
    opt = parse_args()
    opt.device = resolve_device(opt.no_cuda)
    opt.norm = opt.grad_clip
    os.makedirs(opt.out, exist_ok=True)

    arms, manifest = load_arms(opt.data_dir, opt.probe_tokens)
    opt.eos_id = manifest['eos_id']
    opt.vocab_size = manifest['vocab_size']
    opt.block_tokens = opt.block_steps * opt.grad_accum * opt.batchsize * opt.seqlen

    # A resume reads its architecture out of the state file, so the -d_model
    # flags a from-scratch run started with do not have to be repeated.
    state_path = os.path.join(opt.out, STATE_FILE)
    resuming = opt.resume and os.path.exists(state_path)
    if resuming:
        opt.checkpoint = state_path
    model, opt.model_config = build_model(opt, manifest)
    model.train()
    opt.optimizers = make_optimizers(model, muon_lr=opt.muon_lr, embed_lr=opt.embed_lr,
                                     scalar_lr=opt.scalar_lr, muon_impl=opt.muon_impl)
    opt.step, opt.round = 0, 0
    opt.best_probe = [float('inf')] * len(arms)
    opt.lr_scale = 1.0
    opt.probe_history = deque(maxlen=max(opt.lr_patience, 1) + 1)

    director = Director([a.name for a in arms], eta=opt.eta, explore=opt.explore,
                        decay=opt.decay, window=opt.window, seed=opt.seed)
    if resuming:
        load_state(model, arms, director, opt, state_path)

    params = sum(p.numel() for p in model.parameters())
    print(f"device {opt.device} | {params:,} params | arms: "
          f"{', '.join(a.name for a in arms)}")
    print(f"{opt.block_tokens:,} tokens per round "
          f"({opt.block_steps} steps x {opt.batchsize} x {opt.seqlen})")
    run(model, arms, director, opt)


if __name__ == '__main__':
    main()
