"""SFT: fine-tune a pretrained checkpoint on packed chat data with loss masking.

Loads ckpt (architecture comes from its embedded config), trains on
sft_train.bin with cross-entropy computed only on assistant tokens
(per sft_train_mask.bin), and writes checkpoints in the same format
as train.py so chat_server/sample work unchanged.

Example:
  python -m sft.finetune --checkpoint saved/model/ckpt_final.pt \
      --data-dir data_cache/cosmopedia --epochs 2 --dir-name sft
"""
import argparse
import math
import os

import torch
import torch.nn.functional as F

from core.data import data_feeder_masked, load_bin, load_bin_u8
from core.model import Transformer, nopeak_mask
from core.tokenizer import BPETokenizer
from pretrain.train import lr_factor, resolve_device, save_checkpoint


def masked_loss(pred, target, mask, vocab_size):
    loss = F.cross_entropy(pred.view(-1, vocab_size), target.reshape(-1),
                           reduction='none')
    mask = mask.reshape(-1).float()
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def make_sft_optimizers(model, muon_lr, adamw_lr):
    embedding_weight = model.decoder.embed.embed.weight
    muon_params, adamw_params = [], []
    for p in model.parameters():
        if not p.requires_grad:
            continue
        if p is embedding_weight or p.ndim < 2:
            adamw_params.append(p)
        else:
            muon_params.append(p)
    muon = torch.optim.Muon(muon_params, lr=muon_lr, weight_decay=0.01)
    adamw = torch.optim.AdamW(adamw_params, lr=adamw_lr,
                              weight_decay=0.1, betas=(0.9, 0.95))
    for opt in (muon, adamw):
        for group in opt.param_groups:
            group['peak_lr'] = group['lr']
    return [muon, adamw]


@torch.no_grad()
def validate(model, val, val_mask, opt, device, vocab_size, max_batches=100):
    model.eval()
    total, count = 0.0, 0
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        for i, (x, y, m) in enumerate(
                data_feeder_masked(val, val_mask, opt.batchsize, opt.seqlen, device)):
            if i >= max_batches:
                break
            mask = nopeak_mask(x.size(1), device)
            loss = masked_loss(model(x, mask), y, m, vocab_size)
            total += loss.item()
            count += 1
    model.train()
    avg = total / max(1, count)
    print(f"SFT val loss = {avg:.4f} | pplx = {math.exp(avg):.2f}")
    return avg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='data_cache/cosmopedia')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batchsize', type=int, default=32)
    parser.add_argument('--seqlen', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4, help='AdamW peak LR')
    parser.add_argument('--muon-lr', type=float, default=0.003, help='Muon peak LR')
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--save-every', type=int, default=1000)
    parser.add_argument('--val-every', type=int, default=500)
    parser.add_argument('--printevery', type=int, default=50)
    parser.add_argument('--dir-name', default='sft')
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    device = resolve_device(args.no_cuda)
    print(f"device: {device}")

    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(args.data_dir, 'tokenizer.json'))

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']
    if cfg is None:
        raise ValueError("checkpoint lacks config — retrain with current train.py")
    model = Transformer(
        vocab=cfg['vocab_size'], d_model=cfg['d_model'], N=cfg['n_layers'],
        heads=cfg['heads'], dropout=cfg.get('dropout', 0.0),
        kv_heads=cfg.get('kv_heads'), loops=cfg.get('loops', 1),
        value_residual=cfg.get('value_residual', False),
        unet_skips=cfg.get('unet_skips', False),
        attn_res=cfg.get('attn_res', 0),
        kda=cfg.get('kda', 0),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    vocab_size = cfg['vocab_size']
    print(f"loaded {sum(p.numel() for p in model.parameters()):,} params "
          f"from {args.checkpoint}")

    train = load_bin(os.path.join(args.data_dir, 'sft_train.bin'))
    train_mask = load_bin_u8(os.path.join(args.data_dir, 'sft_train_mask.bin'))
    val = load_bin(os.path.join(args.data_dir, 'sft_val.bin'))
    val_mask = load_bin_u8(os.path.join(args.data_dir, 'sft_val_mask.bin'))

    optimizers = make_sft_optimizers(model, muon_lr=args.muon_lr, adamw_lr=args.lr)
    batches_per_epoch = max(1, len(train) // (args.batchsize * args.seqlen))
    total_steps = max(1, args.epochs * batches_per_epoch)

    save_dir = os.path.join('saved', args.dir_name)
    os.makedirs(save_dir, exist_ok=True)

    model.train()
    step = 0
    for epoch in range(args.epochs):
        for x, y, m in data_feeder_masked(train, train_mask, args.batchsize,
                                          args.seqlen, device):
            factor = lr_factor(step, total_steps, warmup_steps=args.warmup_steps)
            for opt in optimizers:
                for group in opt.param_groups:
                    group['lr'] = group['peak_lr'] * factor

            mask = nopeak_mask(x.size(1), device)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                loss = masked_loss(model(x, mask), y, m, vocab_size)

            for opt in optimizers:
                opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            for opt in optimizers:
                opt.step()

            if step % args.printevery == 0:
                print(f"epoch {epoch+1} | step {step}/{total_steps} | "
                      f"loss {loss.item():.4f} | pplx {math.exp(loss.item()):.2f}")
            step += 1
            if args.save_every and step % args.save_every == 0:
                path = os.path.join(save_dir, f'sft_step{step}.pt')
                save_checkpoint(model, optimizers, step, path, config=cfg)
                print(f"saved {path}")
            if args.val_every and step % args.val_every == 0:
                validate(model, val, val_mask, args, device, vocab_size)

    final = os.path.join(save_dir, 'sft_final.pt')
    save_checkpoint(model, optimizers, step, final, config=cfg)
    validate(model, val, val_mask, args, device, vocab_size)
    print(f"saved final SFT checkpoint: {final}")


if __name__ == '__main__':
    main()
