"""DPO: direct preference optimization on top of an SFT checkpoint.

Loads an SFT checkpoint as the policy (architecture from its embedded config)
plus a frozen copy as the reference model, and trains on the preference pairs
from dpo_prepare.py with the standard DPO objective:

  loss = -log sigmoid( beta * [(logp_pi(chosen) - logp_ref(chosen))
                              - (logp_pi(rejected) - logp_ref(rejected))] )

where logp is the summed log-probability of the completion tokens only
(prompt masked out per dpo_*_mask.bin). Writes checkpoints in the same
format as train.py so chat_server/sample/evaluate work unchanged.

Example:
  python dpo.py --checkpoint saved/sft/sft_final.pt \
      --data-dir data_cache/cosmopedia --epochs 1 --dir-name dpo
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from chat_format import EOS_TOKEN
from data import load_bin, load_bin_u8
from finetune import make_sft_optimizers
from model import Transformer
from tokenizer import BPETokenizer
from train import lr_factor, resolve_device, save_checkpoint

PAIRS_DTYPE = np.int32


def load_pairs(path):
    return np.memmap(path, dtype=PAIRS_DTYPE, mode='r').reshape(-1, 4)


def build_batch(tokens, masks, pairs, pair_ids, max_len, pad_id, device):
    """Build a padded batch: chosen sequences first, then rejected.

    Returns (ids_in, targets, loss_mask, attn_mask), all (2B, T) except
    attn_mask (2B, T, T) = causal AND not-padding. Returns None if every
    sequence in the batch is degenerate."""
    seqs = []
    for side in (0, 2):  # columns 0-1 = chosen off/len, 2-3 = rejected off/len
        for p in pair_ids:
            off, ln = int(pairs[p][side]), int(pairs[p][side + 1])
            ids = np.asarray(tokens[off:off + ln])[:max_len]
            msk = np.asarray(masks[off:off + ln])[:max_len]
            seqs.append((ids, msk))

    seqs = [(i, m) for i, m in seqs if len(i) >= 2]
    if not seqs:
        return None
    T = max(len(i) - 1 for i, _ in seqs)
    B = len(seqs)

    ids_in = np.full((B, T), pad_id, dtype=np.int64)
    targets = np.full((B, T), pad_id, dtype=np.int64)
    loss_mask = np.zeros((B, T), dtype=bool)
    keep = np.zeros((B, T), dtype=bool)
    for row, (ids, msk) in enumerate(seqs):
        n = len(ids) - 1
        ids_in[row, :n] = ids[:-1]
        targets[row, :n] = ids[1:]
        loss_mask[row, :n] = np.asarray(msk[1:], dtype=bool)
        keep[row, :n] = True

    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    attn = causal[None, :, :] & torch.tensor(keep, device=device)[:, None, :]
    return (torch.tensor(ids_in, device=device),
            torch.tensor(targets, device=device),
            torch.tensor(loss_mask, device=device),
            attn)


def sequence_logprobs(model, ids_in, targets, loss_mask, attn_mask, autocast_device):
    """Summed log p(target | prompt) over completion tokens, per sequence."""
    with torch.autocast(device_type=autocast_device.type, dtype=torch.bfloat16):
        logits = model(ids_in, attn_mask)
    logp = torch.log_softmax(logits.float(), dim=-1)
    tok_logp = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (tok_logp * loss_mask).sum(dim=-1)


def dpo_loss_and_metrics(policy_logps, ref_logps, beta):
    """policy_logps/ref_logps: (2B,) with chosen first, rejected second."""
    B = policy_logps.size(0) // 2
    pi_c, pi_r = policy_logps[:B], policy_logps[B:]
    ref_c, ref_r = ref_logps[:B], ref_logps[B:]
    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    loss = -F.logsigmoid(logits).mean()
    with torch.no_grad():
        reward_c = beta * (pi_c - ref_c)
        reward_r = beta * (pi_r - ref_r)
        margin = (reward_c - reward_r).mean()
        acc = (reward_c > reward_r).float().mean()
    return loss, margin.item(), acc.item()


@torch.no_grad()
def validate(policy, ref, tokens, masks, pairs, args, device, max_batches=50):
    policy.eval()
    total_loss, total_margin, total_acc, count = 0.0, 0.0, 0.0, 0
    num_batches = max(1, len(pairs) // args.batchsize)
    pad_id = args.pad_id
    for i in range(min(num_batches, max_batches)):
        pair_ids = range(i * args.batchsize, (i + 1) * args.batchsize)
        batch = build_batch(tokens, masks, pairs, pair_ids,
                            args.max_len, pad_id, device)
        if batch is None:
            continue
        pi = sequence_logprobs(policy, *batch, device)
        rf = sequence_logprobs(ref, *batch, device)
        loss, margin, acc = dpo_loss_and_metrics(pi, rf, args.beta)
        total_loss += loss.item()
        total_margin += margin
        total_acc += acc
        count += 1
    policy.train()
    avg_loss = total_loss / max(1, count)
    print(f"DPO val loss = {avg_loss:.4f} | margin = {total_margin / max(1, count):.4f} "
          f"| acc = {total_acc / max(1, count):.3f}")
    return avg_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True,
                        help='SFT checkpoint (policy init + reference model)')
    parser.add_argument('--data-dir', default='data_cache/cosmopedia')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batchsize', type=int, default=8,
                        help='Preference pairs per optimizer step')
    parser.add_argument('--beta', type=float, default=0.1,
                        help='DPO temperature (KL budget to the reference model)')
    parser.add_argument('--lr', type=float, default=1e-6, help='AdamW peak LR')
    parser.add_argument('--muon-lr', type=float, default=1e-4, help='Muon peak LR')
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--max-len', type=int, default=1024,
                        help='Truncate sequences to this many tokens')
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--val-every', type=int, default=500)
    parser.add_argument('--printevery', type=int, default=20)
    parser.add_argument('--dir-name', default='dpo')
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    device = resolve_device(args.no_cuda)
    print(f"device: {device}")

    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(args.data_dir, 'tokenizer.json'))
    if EOS_TOKEN not in tokenizer.special_tokens:
        raise ValueError("tokenizer lacks <|endoftext|> — rerun prepare.py")
    args.pad_id = tokenizer.special_tokens[EOS_TOKEN]

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']
    if cfg is None:
        raise ValueError("checkpoint lacks config — retrain with current train.py")

    def build():
        m = Transformer(
            vocab=cfg['vocab_size'], d_model=cfg['d_model'], N=cfg['n_layers'],
            heads=cfg['heads'], dropout=cfg.get('dropout', 0.0),
            kv_heads=cfg.get('kv_heads'), loops=cfg.get('loops', 1),
        ).to(device)
        m.load_state_dict(ckpt['model'])
        return m

    policy = build()
    ref = build()
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    print(f"loaded {sum(p.numel() for p in policy.parameters()):,} params "
          f"from {args.checkpoint} (policy + frozen reference)")

    train_tokens = load_bin(os.path.join(args.data_dir, 'dpo_train.bin'))
    train_masks = load_bin_u8(os.path.join(args.data_dir, 'dpo_train_mask.bin'))
    train_pairs = load_pairs(os.path.join(args.data_dir, 'dpo_train_pairs.bin'))
    val_tokens = load_bin(os.path.join(args.data_dir, 'dpo_val.bin'))
    val_masks = load_bin_u8(os.path.join(args.data_dir, 'dpo_val_mask.bin'))
    val_pairs = load_pairs(os.path.join(args.data_dir, 'dpo_val_pairs.bin'))

    optimizers = make_sft_optimizers(policy, muon_lr=args.muon_lr, adamw_lr=args.lr)
    batches_per_epoch = max(1, len(train_pairs) // args.batchsize)
    total_steps = max(1, args.epochs * batches_per_epoch)

    save_dir = os.path.join('saved', args.dir_name)
    os.makedirs(save_dir, exist_ok=True)

    policy.train()
    step = 0
    for epoch in range(args.epochs):
        for i in range(batches_per_epoch):
            pair_ids = range(i * args.batchsize, (i + 1) * args.batchsize)
            batch = build_batch(train_tokens, train_masks, train_pairs, pair_ids,
                                args.max_len, args.pad_id, device)
            if batch is None:
                continue

            factor = lr_factor(step, total_steps, warmup_steps=args.warmup_steps)
            for opt in optimizers:
                for group in opt.param_groups:
                    group['lr'] = group['peak_lr'] * factor

            pi = sequence_logprobs(policy, *batch, device)
            with torch.no_grad():
                rf = sequence_logprobs(ref, *batch, device)
            loss, margin, acc = dpo_loss_and_metrics(pi, rf, args.beta)

            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            for opt in optimizers:
                opt.step()

            if step % args.printevery == 0:
                print(f"epoch {epoch+1} | step {step}/{total_steps} | "
                      f"loss {loss.item():.4f} | margin {margin:.4f} | acc {acc:.3f}")
            step += 1
            if args.save_every and step % args.save_every == 0:
                path = os.path.join(save_dir, f'dpo_step{step}.pt')
                save_checkpoint(policy, optimizers, step, path, config=cfg)
                print(f"saved {path}")
            if args.val_every and step % args.val_every == 0:
                validate(policy, ref, val_tokens, val_masks, val_pairs, args, device)

    final = os.path.join(save_dir, 'dpo_final.pt')
    save_checkpoint(policy, optimizers, step, final, config=cfg)
    validate(policy, ref, val_tokens, val_masks, val_pairs, args, device)
    print(f"saved final DPO checkpoint: {final}")


if __name__ == '__main__':
    main()
