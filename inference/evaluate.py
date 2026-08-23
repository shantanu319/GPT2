"""Zero-shot multiple-choice eval for tiny chat checkpoints.

Scores each answer choice by total log-likelihood (acc) and per-token
log-likelihood (acc_norm) of the continuation, the lm-evaluation-harness
convention for ARC-Easy / HellaSwag / PIQA. At <100M params expect modest
but above-chance numbers; acc_norm is the smoother signal at this scale.

Examples:
  python -m inference.evaluate --checkpoint vast_out/saved/vast_run_sft/sft_final.pt \
      --tokenizer vast_out/tokenizer.json --tasks arc_easy,piqa --limit 500
  python -m inference.evaluate --checkpoint ... --chat        # wrap context in ChatML
"""
import argparse

import torch
import torch.nn.functional as F

from core.chat_format import DEFAULT_SYSTEM, IM_START, render_turn
from core.model import model_from_checkpoint, nopeak_mask
from core.tokenizer import BPETokenizer


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
def choice_logprobs(model, tokenizer, context, choices, device, max_len=1024):
    """Score every choice of one question in a single batched forward.

    The choices share a context, so it is tokenized once and the batch is
    right-padded, with the padding masked out of attention the same way
    dpo.build_batch does it. Returns (sum, mean) of the continuation's token
    log-probabilities per choice."""
    ctx_ids = tokenizer.encode(context)
    seqs, n_cho = [], []
    for choice in choices:
        cho_ids = tokenizer.encode_ordinary(choice)
        ids = (ctx_ids + cho_ids)[-max_len:]
        seqs.append(ids)
        n_cho.append(min(len(cho_ids), len(ids) - 1))

    B, T = len(seqs), max(len(s) for s in seqs) - 1
    x = torch.zeros(B, T, dtype=torch.long, device=device)
    y = torch.zeros(B, T, dtype=torch.long, device=device)
    keep = torch.zeros(B, T, dtype=torch.bool, device=device)
    for i, ids in enumerate(seqs):
        n = len(ids) - 1
        x[i, :n] = torch.tensor(ids[:-1], dtype=torch.long, device=device)
        y[i, :n] = torch.tensor(ids[1:], dtype=torch.long, device=device)
        keep[i, :n] = True

    logits = model(x, nopeak_mask(T, device) & keep[:, None, :])
    logp = F.log_softmax(logits.float(), dim=-1)
    tok_lp = logp.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    out = []
    for i, ids in enumerate(seqs):
        n = len(ids) - 1
        lp = tok_lp[i, n - n_cho[i]:n]
        out.append((lp.sum().item(), lp.mean().item()))
    return out


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
        scores = choice_logprobs(model, tokenizer, ctx, ex['choices'], device)
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

    from pretrain.train import resolve_device
    device = resolve_device(args.no_cuda)

    tokenizer = BPETokenizer()
    tokenizer.load(args.tokenizer)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = model_from_checkpoint(ckpt, device)

    print(f"{'task':<14} {'n':>5} {'acc':>7} {'acc_norm':>9}")
    for task in args.tasks.split(','):
        n, acc, acc_norm = run_task(model, tokenizer, task.strip(), device,
                                    limit=args.limit or None, chat=args.chat)
        print(f"{task:<14} {n:>5} {acc:>7.3f} {acc_norm:>9.3f}")


if __name__ == '__main__':
    main()
