"""Long-running inference server. Reads JSON-line requests from stdin and
writes JSON-line responses to stdout. Intended to be spawned by the Rust
`chat` CLI as a child process — not invoked directly by humans.

Protocol:
  Server -> client (stdout):
    {"type": "ready"}                         sent once after warm-up
    {"type": "response", "text": "..."}       after each prompt
    {"type": "reset_ok"}                      after each reset
    {"type": "error", "error": "..."}         on any failure

  Client -> server (stdin):
    {"type": "prompt", "prompt": "..."}       generate a continuation
    {"type": "reset"}                         clear running token context
"""
import argparse
import json
import os
import sys

import torch

from core.chat_format import DEFAULT_SYSTEM, IM_END, IM_START, render_turn
from core.model import Transformer, nopeak_mask
from core.tokenizer import BPETokenizer
from inference.sample import _sample_next


def log(msg):
    # Non-protocol output goes to stderr so stdout stays machine-parseable.
    print(f"[chat_server] {msg}", file=sys.stderr, flush=True)


def _send(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _resolve_device(no_cuda):
    if not no_cuda:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def _prefill(model, ids, device):
    """Batched prefill from start_pos=0. Returns logits for the final token."""
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
    mask = nopeak_mask(x.size(1), device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(x, mask, start_pos=0)
    return logits[:, -1, :]


def _decode_one(model, tok_id, start_pos, device):
    """Run a single token through the model using the existing cache."""
    x = torch.tensor([[tok_id]], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        logits = model(x, None, start_pos=start_pos)
    return logits[:, -1, :]


@torch.no_grad()
def generate_into(context_ids, cache_len, new_prompt_ids, model, eos_id,
                  max_tokens, temperature, top_p, max_context, device,
                  stop_ids=None):
    """Extend context_ids with new_prompt_ids, generate up to max_tokens, append
    generated tokens in place. Returns (newly_generated_ids, new_cache_len).

    cache_len is the number of tokens currently represented in the model's KV cache
    (0 after a reset). The caller is responsible for calling model.reset_cache()
    when they want to clear state.
    """
    model.eval()
    context_ids.extend(new_prompt_ids)

    # If adding this prompt overflows the window, drop cache and re-prefill the tail.
    if cache_len + len(new_prompt_ids) > max_context:
        model.reset_cache()
        window = context_ids[-(max_context - 1):]
        last_logits = _prefill(model, window, device)
        cache_len = len(window)
    elif cache_len == 0:
        last_logits = _prefill(model, new_prompt_ids, device)
        cache_len = len(new_prompt_ids)
    else:
        # Multi-turn continuation: advance the cache one new-prompt token at a time.
        # Simpler than building a rectangular causal mask for batched prefill; cost
        # is len(new_prompt_ids) single-token forwards, bounded by prompt length.
        last_logits = None
        for tok in new_prompt_ids:
            last_logits = _decode_one(model, tok, cache_len, device)
            cache_len += 1

    stop_ids = set(stop_ids or ())
    if eos_id is not None:
        stop_ids.add(eos_id)

    generated = []
    for _ in range(max_tokens):
        next_id = _sample_next(last_logits, temperature, top_p)
        generated.append(next_id)
        context_ids.append(next_id)
        if next_id in stop_ids:
            break

        if cache_len + 1 > max_context:
            model.reset_cache()
            window = context_ids[-(max_context - 1):]
            last_logits = _prefill(model, window, device)
            cache_len = len(window)
            continue

        last_logits = _decode_one(model, next_id, cache_len, device)
        cache_len += 1

    return generated, cache_len


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--max-tokens', type=int, default=100)
    parser.add_argument('--temperature', type=float, default=0.5)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--max-context', type=int, default=512)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--raw', action='store_true',
                        help='Disable the chat template (raw LM continuation), '
                             'e.g. for pretrain-only checkpoints')
    parser.add_argument('--system', default=DEFAULT_SYSTEM,
                        help='System prompt used in chat-template mode')
    args = parser.parse_args()

    device = _resolve_device(args.no_cuda)
    log(f"device: {device}")

    tokenizer = BPETokenizer()
    tokenizer.load(os.path.join(args.data_dir, 'tokenizer.json'))
    log(f"tokenizer vocab_size: {tokenizer.vocab_size}")

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if 'config' not in ckpt or ckpt['config'] is None:
        _send({"type": "error", "error": "checkpoint missing 'config' field"})
        return
    cfg = ckpt['config']
    model = Transformer(
        vocab=cfg['vocab_size'],
        d_model=cfg['d_model'],
        N=cfg['n_layers'],
        heads=cfg['heads'],
        dropout=cfg['dropout'],
        kv_heads=cfg.get('kv_heads'),
        loops=cfg.get('loops', 1),
        value_residual=cfg.get('value_residual', False),
        unet_skips=cfg.get('unet_skips', False),
        attn_res=cfg.get('attn_res', 0),
    ).to(device)
    model.load_state_dict(ckpt['model'])
    log(f"model loaded: {sum(p.numel() for p in model.parameters()):,} params")

    eos_id = tokenizer.special_tokens.get('<|endoftext|>')
    im_end_id = tokenizer.special_tokens.get(IM_END)
    chat_mode = not args.raw and IM_START in tokenizer.special_tokens
    log(f"chat template: {'on' if chat_mode else 'off'}")
    stop_ids = {im_end_id} if chat_mode and im_end_id is not None else set()
    context_ids = []
    cache_len = 0
    first_turn = True

    _send({"type": "ready"})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError as e:
            _send({"type": "error", "error": f"invalid json: {e}"})
            continue

        kind = msg.get("type")
        if kind == "reset":
            context_ids = []
            cache_len = 0
            first_turn = True
            model.reset_cache()
            _send({"type": "reset_ok"})
        elif kind == "prompt":
            prompt = msg.get("prompt", "")
            try:
                if chat_mode:
                    # assistant stop token has no trailing newline; add one
                    text = '' if first_turn else '\n'
                    if first_turn and args.system:
                        text += render_turn('system', args.system)
                    text += render_turn('user', prompt) + f"{IM_START}assistant\n"
                    new_prompt_ids = tokenizer.encode(text)
                    first_turn = False
                else:
                    new_prompt_ids = tokenizer.encode(prompt)
                new_ids, cache_len = generate_into(
                    context_ids, cache_len, new_prompt_ids, model, eos_id,
                    max_tokens=msg.get("max_tokens", args.max_tokens),
                    temperature=msg.get("temperature", args.temperature),
                    top_p=msg.get("top_p", args.top_p),
                    max_context=args.max_context,
                    device=device,
                    stop_ids=stop_ids,
                )
                if chat_mode and new_ids and new_ids[-1] in stop_ids | {eos_id}:
                    new_ids = new_ids[:-1]  # don't print the stop token
                _send({"type": "response", "text": tokenizer.decode(new_ids)})
            except Exception as e:  # noqa: BLE001
                _send({"type": "error", "error": repr(e)})
        else:
            _send({"type": "error", "error": f"unknown type: {kind!r}"})


if __name__ == "__main__":
    main()
