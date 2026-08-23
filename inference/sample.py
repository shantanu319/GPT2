"""Generate text from a trained checkpoint using temperature + nucleus (top-p) sampling.

Example:
  python -m inference.sample --checkpoint saved/model/ckpt_final.pt --prompt "Once upon a time" \
                   --max-tokens 200 --temperature 0.5 --top-p 0.9
"""
import argparse
import os

import torch
import torch.nn.functional as F

from core.model import model_from_checkpoint, nopeak_mask
from core.tokenizer import BPETokenizer


def top_p_filter(probs, top_p):
    """Zero out tokens outside the smallest nucleus whose cumulative prob >= top_p.
    Always keeps at least the single highest-prob token."""
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    drop = cumsum >= top_p
    # Shift right so the token that first crosses the threshold is still kept.
    drop[..., 1:] = drop[..., :-1].clone()
    drop[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(0, sorted_idx, sorted_probs)
    return filtered


def _resolve_device(no_cuda):
    if not no_cuda:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _sample_next(logits, temperature, top_p):
    logits = logits.float() / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1).squeeze(0)
    if top_p < 1.0:
        probs = top_p_filter(probs, top_p)
        probs = probs / probs.sum()
    return torch.multinomial(probs, num_samples=1).item()


@torch.no_grad()
def generate(model, tokenizer, prompt, max_tokens, temperature, top_p,
             max_context, device, eos_id=None, stop_at_eos=True,
             use_kv_cache=True, stop_ids=None):
    model.eval()

    stop_ids = set(stop_ids or ())
    if stop_at_eos and eos_id is not None:
        stop_ids.add(eos_id)

    if prompt:
        ids = tokenizer.encode(prompt)
    elif eos_id is not None:
        ids = [eos_id]  # start a fresh document
    else:
        ids = [0]

    if not use_kv_cache:
        tokens = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
        for _ in range(max_tokens):
            context = tokens[:, -max_context:]
            mask = nopeak_mask(context.size(1), device)
            logits = model(context, mask)
            next_id = _sample_next(logits[:, -1, :], temperature, top_p)
            tokens = torch.cat(
                [tokens, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
            if next_id in stop_ids:
                break
        return tokenizer.decode(tokens[0].tolist())

    # KV-cache path: prefill the prompt, then decode one token at a time.
    model.reset_cache()
    all_ids = list(ids)

    def prefill(window):
        x = torch.tensor(window, dtype=torch.long, device=device).unsqueeze(0)
        mask = nopeak_mask(x.size(1), device)
        return model(x, mask, start_pos=0)[:, -1, :]

    last_logits = prefill(all_ids)
    cache_len = len(all_ids)

    for _ in range(max_tokens):
        next_id = _sample_next(last_logits, temperature, top_p)
        all_ids.append(next_id)
        if next_id in stop_ids:
            break

        if cache_len + 1 >= max_context:
            # Overflow: drop cache and re-prefill the last (max_context - 1) tokens.
            model.reset_cache()
            window = all_ids[-(max_context - 1):]
            last_logits = prefill(window)
            cache_len = len(window)
            continue

        x = torch.tensor([[next_id]], dtype=torch.long, device=device)
        last_logits = model(x, None, start_pos=cache_len)[:, -1, :]
        cache_len += 1

    return tokenizer.decode(all_ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', default='data_cache/cosmopedia')
    parser.add_argument('--prompt', default='')
    parser.add_argument('--max-tokens', type=int, default=200)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--num-samples', type=int, default=1)
    parser.add_argument('--max-context', type=int, default=512)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--no-kv-cache', action='store_true',
                        help='Disable KV cache (re-feed full context each step). '
                             'Temporary: used to verify KV-cache correctness.')
    parser.add_argument('--chat', action='store_true',
                        help='Wrap --prompt in the ChatML template (for SFT checkpoints)')
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = _resolve_device(args.no_cuda)

    tok_path = os.path.join(args.data_dir, 'tokenizer.json')
    if not os.path.exists(tok_path):
        raise FileNotFoundError(f"tokenizer not found at {tok_path}")
    tokenizer = BPETokenizer()
    tokenizer.load(tok_path)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'config' not in ckpt or ckpt['config'] is None:
        raise ValueError(
            f"checkpoint {args.checkpoint} lacks a 'config' field — "
            f"retrain with the current save_checkpoint to include model architecture"
        )
    model = model_from_checkpoint(ckpt, device)

    eos_id = tokenizer.special_tokens.get('<|endoftext|>')

    prompt = args.prompt
    stop_ids = set()
    if args.chat:
        from core.chat_format import DEFAULT_SYSTEM, IM_END, render_conversation
        prompt = render_conversation(
            [{'role': 'system', 'content': DEFAULT_SYSTEM},
             {'role': 'user', 'content': args.prompt}],
            add_generation_prompt=True,
        )
        if IM_END in tokenizer.special_tokens:
            stop_ids.add(tokenizer.special_tokens[IM_END])

    for i in range(args.num_samples):
        if args.num_samples > 1:
            print(f"\n--- sample {i+1}/{args.num_samples} ---")
        text = generate(
            model, tokenizer, prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            max_context=args.max_context,
            device=device,
            eos_id=eos_id,
            use_kv_cache=not args.no_kv_cache,
            stop_ids=stop_ids,
        )
        print(text)


if __name__ == '__main__':
    main()
