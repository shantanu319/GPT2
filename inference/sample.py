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


STOP_CHECK_EVERY = 8  # decode steps between reading sampled tokens back


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
    """Sample one token id, returned as a (1, 1) tensor on the logits' device.

    Gumbel-max -- argmax of log p plus Gumbel noise -- draws from the same
    distribution as torch.multinomial, which on MPS synchronises and stalls the
    decode pipeline for a full step. argmax is invariant to a constant scale on
    p, so the nucleus does not need renormalizing."""
    probs = F.softmax(logits.float() / max(temperature, 1e-6), dim=-1).squeeze(0)
    if top_p < 1.0:
        probs = top_p_filter(probs, top_p)
    u = torch.rand_like(probs).clamp_min_(1e-20)
    return (probs.log() - (-u.log()).log()).argmax(-1).view(1, 1)


def prefill_logits(model, ids, device, start_pos=0):
    """Feed `ids` into the KV cache at start_pos in one batched forward.
    Returns logits for the final token."""
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = nopeak_mask(x.size(1), device, start_pos=start_pos)
    return model(x, mask, start_pos=start_pos)[:, -1, :]


@torch.no_grad()
def decode_loop(model, last_logits, ids, cache_len, max_tokens, temperature,
                top_p, max_context, device, stop_ids):
    """Sample up to max_tokens tokens, appending them to `ids` in place.
    Returns (n_generated, cache_len).

    Sampled tokens stay on the device and are read back a block at a time:
    testing each one against stop_ids on the CPU costs a pipeline stall per
    step, which is worth more than the few tokens generated past a stop and
    thrown away. `ids` must end with every token the KV cache holds."""
    pending, stopped = [], False

    def flush():
        """Read the pending tokens into `ids` up to the first stop id. Sets
        `stopped`, and returns how many tokens past it are discarded."""
        nonlocal pending, stopped
        if not pending:
            return 0
        toks = torch.cat(pending, dim=1).view(-1).tolist()
        pending = []
        for i, tok in enumerate(toks):
            ids.append(tok)
            if tok in stop_ids:
                stopped = True
                return len(toks) - 1 - i
        return 0

    n = 0
    while n < max_tokens:
        if cache_len + 1 >= max_context:
            # Rebuilding the cache needs every sampled token back on the CPU.
            dropped = flush()
            n, cache_len = n - dropped, cache_len - dropped
            if stopped:
                break
            model.reset_cache()
            window = ids[-(max_context - 1):]
            last_logits = prefill_logits(model, window, device)
            cache_len = len(window)
        tok = _sample_next(last_logits, temperature, top_p)
        pending.append(tok)
        n += 1
        last_logits = model(tok, None, start_pos=cache_len)[:, -1, :]
        cache_len += 1
        if len(pending) == STOP_CHECK_EVERY or n == max_tokens:
            dropped = flush()
            n, cache_len = n - dropped, cache_len - dropped
            if stopped:
                break
    return n, cache_len


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
            tok = _sample_next(model(context, mask)[:, -1, :], temperature, top_p)
            tokens = torch.cat([tokens, tok], dim=1)
            if tok.item() in stop_ids:
                break
        return tokenizer.decode(tokens[0].tolist())

    # KV-cache path: prefill the prompt, then decode one token at a time.
    model.reset_cache()
    all_ids = list(ids)
    last_logits = prefill_logits(model, all_ids, device)
    decode_loop(model, last_logits, all_ids, len(all_ids), max_tokens,
                temperature, top_p, max_context, device, stop_ids)
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
