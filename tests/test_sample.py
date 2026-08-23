import torch

from inference.sample import top_p_filter


def test_top_p_filter_zeros_out_tail():
    probs = torch.tensor([0.5, 0.2, 0.2, 0.05, 0.05])
    out = top_p_filter(probs, top_p=0.7)
    # Nucleus is {0.5, 0.2} (cumsum = 0.7), tail {0.2, 0.05, 0.05} is dropped
    assert out[0] > 0
    assert out[1] > 0
    assert out[2] == 0
    assert out[3] == 0
    assert out[4] == 0


def test_top_p_filter_always_keeps_top_token():
    # Even a very small top-p should keep at least the highest-prob token.
    probs = torch.tensor([1.0, 0.0, 0.0])
    out = top_p_filter(probs, top_p=0.01)
    assert out[0] > 0


def test_top_p_filter_at_1_is_passthrough():
    probs = torch.tensor([0.4, 0.3, 0.2, 0.1])
    out = top_p_filter(probs, top_p=1.0)
    assert torch.allclose(out, probs)


def test_top_p_filter_preserves_original_position():
    # The non-zero outputs should sit at the original indices of the sorted top-k.
    probs = torch.tensor([0.1, 0.6, 0.3])  # sorted order: idx 1 (0.6), idx 2 (0.3), idx 0 (0.1)
    out = top_p_filter(probs, top_p=0.8)
    # Nucleus = {0.6, 0.3} at original indices 1 and 2; idx 0 dropped.
    assert out[1] > 0
    assert out[2] > 0
    assert out[0] == 0


class _ScriptedModel:
    """Stand-in model that drives decode_loop through a fixed token sequence:
    every call returns one-hot logits for the next token of the script."""

    def __init__(self, script, vocab=64):
        self.script = script
        self.vocab = vocab
        self.calls = 0
        self.resets = 0
        self.prefills = 0   # calls fed more than one token, i.e. cache rebuilds

    def reset_cache(self):
        self.resets += 1

    def __call__(self, x, mask=None, start_pos=None):
        nxt = self.script[self.calls % len(self.script)]
        self.calls += 1
        if x.size(1) > 1:
            self.prefills += 1
        logits = torch.full((1, x.size(1), self.vocab), -30.0)
        logits[:, -1, nxt] = 30.0
        return logits


def _run_decode(stop_at, max_tokens=20, max_context=10 ** 6):
    from inference.sample import decode_loop, prefill_logits
    script = list(range(1, 40))
    model = _ScriptedModel(script)
    dev = torch.device('cpu')
    ids = [0]
    logits = prefill_logits(model, ids, dev)
    stop = {script[stop_at]} if stop_at is not None else set()
    n, cache_len = decode_loop(model, logits, ids, len(ids), max_tokens, 0.01,
                               1.0, max_context, dev, stop)
    return script, ids, n, cache_len


def test_decode_loop_stops_exactly_at_stop_token():
    # Cover stops before, on, and after the STOP_CHECK_EVERY block boundary:
    # tokens sampled past a stop must be dropped from both ids and cache_len.
    for stop_at in range(0, 12):
        script, ids, n, cache_len = _run_decode(stop_at)
        assert ids == [0] + script[:stop_at + 1], stop_at
        assert n == stop_at + 1, stop_at
        assert cache_len == len(ids), stop_at


def test_decode_loop_runs_to_max_tokens_without_a_stop():
    script, ids, n, cache_len = _run_decode(None, max_tokens=20)
    assert n == 20
    assert ids == [0] + script[:20]
    assert cache_len == len(ids)


def test_decode_loop_reprefills_on_context_overflow():
    script, ids, n, cache_len = _run_decode(None, max_tokens=30, max_context=12)
    assert n == 30
    assert len(ids) == 31
    assert cache_len <= 12  # window was rebuilt instead of growing past it


def test_sample_next_stays_in_the_nucleus():
    from inference.sample import _sample_next
    torch.manual_seed(0)
    logits = torch.tensor([[10.0, 9.0, -20.0, -20.0, -20.0]])
    for _ in range(50):
        tok = _sample_next(logits, temperature=1.0, top_p=0.9)
        assert tok.shape == (1, 1)
        assert tok.item() in (0, 1)


def test_decode_loop_leaves_room_after_a_context_overflow():
    """Rebuilding the cache right up to max_context leaves none: the next token
    overflows again, and a re-prefill costs as much as tens of decode steps. A
    long run past the window should rebuild a handful of times, not per token."""
    from inference.sample import decode_loop, prefill_logits

    dev = torch.device('cpu')
    max_context, n_tokens = 32, 40
    model = _ScriptedModel(list(range(1, 40)))
    ids = list(range(1, max_context - 6))
    logits = prefill_logits(model, ids, dev)
    model.prefills = 0

    n, cache_len = decode_loop(model, logits, ids, len(ids), n_tokens, 0.01, 1.0,
                               max_context, dev, set())
    assert n == n_tokens
    assert cache_len <= max_context
    assert model.prefills <= n_tokens // 4, model.prefills
