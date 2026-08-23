import torch
import torch.nn.functional as F

from core.model import Transformer, nopeak_mask
from inference.evaluate import choice_logprobs


class _CharTokenizer:
    """Byte-level stand-in: enough of the BPETokenizer surface to score text."""

    def encode(self, text):
        return [b % 64 for b in text.encode('utf-8')]

    encode_ordinary = encode


def _one_at_a_time(model, tokenizer, context, choice, device, max_len=1024):
    """The pre-batching implementation, kept here as the reference."""
    ctx_ids = tokenizer.encode(context)
    cho_ids = tokenizer.encode_ordinary(choice)
    ids = (ctx_ids + cho_ids)[-max_len:]
    n_cho = min(len(cho_ids), len(ids) - 1)
    x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
    y = torch.tensor(ids[1:], dtype=torch.long, device=device)
    logits = model(x, nopeak_mask(x.size(1), device))
    logp = F.log_softmax(logits.float()[0], dim=-1)
    tok_lp = logp[torch.arange(y.size(0)), y][-n_cho:]
    return tok_lp.sum().item(), tok_lp.mean().item()


def test_batched_choice_scoring_matches_one_at_a_time():
    """Padding the choices into one forward must not change any score, even
    though they differ in length."""
    dev = torch.device('cpu')
    torch.manual_seed(0)
    model = Transformer(vocab=64, d_model=32, N=2, heads=2, dropout=0.0, kv_heads=1).eval()
    tok = _CharTokenizer()
    context = "Question: why is the sky blue?\nAnswer:"
    choices = [" scattering", " because of light scattering in air", " no", " it is"]

    with torch.no_grad():
        batched = choice_logprobs(model, tok, context, choices, dev)
        one = [_one_at_a_time(model, tok, context, c, dev) for c in choices]

    for (bs, bm), (os_, om) in zip(batched, one):
        assert abs(bs - os_) < 1e-4, (bs, os_)
        assert abs(bm - om) < 1e-5, (bm, om)
