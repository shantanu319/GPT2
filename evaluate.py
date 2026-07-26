"""Zero-shot multiple-choice eval for tiny chat checkpoints.

Scores each answer choice by total log-likelihood (acc) and per-token
log-likelihood (acc_norm) of the continuation, the lm-evaluation-harness
convention for ARC-Easy / HellaSwag / PIQA. At <100M params expect modest
but above-chance numbers; acc_norm is the smoother signal at this scale.

Examples:
  python evaluate.py --checkpoint vast_out/saved/vast_run_sft/sft_final.pt \
      --tokenizer vast_out/tokenizer.json --tasks arc_easy,piqa --limit 500
  python evaluate.py --checkpoint ... --chat        # wrap context in ChatML
"""
import argparse

import torch
import torch.nn.functional as F

from chat_format import DEFAULT_SYSTEM, IM_START, render_turn
from model import Transformer, nopeak_mask
from tokenizer import BPETokenizer


def _arc(split='test', name='ARC-Easy'):
    from datasets import load_dataset
    ds = load_dataset('allenai/ai2_arc', name, split=split)
    for row in ds:
        labels = row['choices']['label']
        if row['answerKey'] not in labels:
            continue
        yield {
            'context': f"Question: {row['question']}\nAnswer:",
            'choices': [f" {t}" for t in row['choices']['text']],
            'gold': labels.index(row['answerKey']),
        }


def _hellaswag(split='validation'):
    from datasets import load_dataset
    for row in load_dataset('Rowan/hellaswag', split=split):
        yield {
            'context': row['ctx'],
            'choices': [f" {t}" for t in row['endings']],
            'gold': int(row['label']),
        }


def _piqa(split='validation'):
    from datasets import load_dataset
    for row in load_dataset('ybisk/piqa', split=split, trust_remote_code=True):
        yield {
            'context': f"Question: {row['goal']}\nAnswer:",
            'choices': [f" {row['sol1']}", f" {row['sol2']}"],
            'gold': int(row['label']),
        }


TASKS = {
    'arc_easy': lambda: _arc(name='ARC-Easy'),
    'arc_challenge': lambda: _arc(name='ARC-Challenge'),
    'hellaswag': _hellaswag,
    'piqa': _piqa,
}


@torch.no_grad()
def choice_logprob(model, tokenizer, context, choice, device, max_len=1024):
    ctx_ids = tokenizer.encode(context)
    cho_ids = tokenizer.encode_ordinary(choice)
    ids = (ctx_ids + cho_ids)[-max_len:]
    n_cho = min(len(cho_ids), len(ids) - 1)
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device)
    mask = nopeak_mask(x.size(1), device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(x, mask)
    logp = F.log_softmax(logits.float()[0], dim=-1)
    tok_lp = logp[torch.arange(y.size(0)), y][-n_cho:]
    return tok_lp.sum().item(), tok_lp.mean().item()


def run_task(model, tokenizer, task, device, limit=None, chat=False):
    n = acc = acc_norm = 0
    for ex in TASKS[task]():
        if limit and n >= limit:
            break
        ctx = ex['context']
        if chat:
            ctx = (render_turn('system', DEFAULT_SYSTEM)
                   + render_turn('user', ex['context'])
                   + f"{IM_START}assistant\n")
        scores = [choice_logprob(model, tokenizer, ctx, c, device)
                  for c in ex['choices']]
        acc += max(range(len(scores)), key=lambda i: scores[i][0]) == ex['gold']
        acc_norm += max(range(len(scores)), key=lambda i: scores[i][1]) == ex['gold']
        n += 1
        if n % 100 == 0:
            print(f"  {task}: {n} examples | acc {acc/n:.3f} | acc_norm {acc_norm/n:.3f}")
    return n, acc / max(n, 1), acc_norm / max(n, 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--tokenizer', default='data_cache/cosmopedia/tokenizer.json')
    parser.add_argument('--tasks', default='arc_easy,hellaswag,piqa')
    parser.add_argument('--limit', type=int, default=500, help='Examples per task (0 = all)')
    parser.add_argument('--chat', action='store_true', help='Wrap context in ChatML template')
    parser.add_argument('--no-cuda', action='store_true')
    args = parser.parse_args()

    from train import resolve_device
    device = resolve_device(args.no_cuda)

    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt['config']
    model = Transformer(
        vocab=cfg['vocab_size'], d_model=cfg['d_model'], N=cfg['n_layers'],
        heads=cfg['heads'], dropout=cfg.get('dropout', 0.0),
        kv_heads=cfg.get('kv_heads'), loops=cfg.get('loops', 1),
        value_residual=cfg.get('value_residual', False),
        unet_skips=cfg.get('unet_skips', False),
    ).to(device).eval()
    model.load_state_dict(ckpt['model'])

    print(f"{'task':<14} {'n':>5} {'acc':>7} {'acc_norm':>9}")
    for task in args.tasks.split(','):
        n, acc, acc_norm = run_task(model, tokenizer, task.strip(), device,
                                    limit=args.limit or None, chat=args.chat)
        print(f"{task:<14} {n:>5} {acc:>7.3f} {acc_norm:>9.3f}")


if __name__ == '__main__':
    main()
