"""Tests for prepare.py — focus on parity between serial and parallel encode."""
import numpy as np

from data import BIN_DTYPE
from prepare import _iter_encoded, _tokenize_stream_three_bins, mixed_stream, Source
from tokenizer import BPETokenizer


CORPUS = (
    "the quick brown fox jumps over the lazy dog "
    "hello world hello there hello friends "
    "lorem ipsum dolor sit amet consectetur " * 30
)

TEXTS = [
    "the quick brown fox jumps over the lazy dog",
    "hello world",
    "lorem ipsum dolor sit amet",
    "the quick brown",
    "consectetur adipiscing elit",
    "",  # empty doc edge case
    "single",
    "punctuation, and -- some! numbers 42 too",
] * 6  # 48 docs


def _train_tokenizer(tmp_path):
    eos_id = 511
    tok = BPETokenizer(special_tokens={'<|endoftext|>': eos_id})
    tok.train(CORPUS, vocab_size=400)
    path = tmp_path / 'tok.json'
    tok.save(str(path))
    return tok, str(path), eos_id


def _rows():
    return list(TEXTS)


def test_iter_encoded_parallel_matches_serial(tmp_path):
    tok, tok_path, eos_id = _train_tokenizer(tmp_path)

    serial = [(i, arr.tolist()) for i, arr in
              _iter_encoded(tok, tok_path, _rows(), eos_id, max_docs=None, num_workers=1)]

    parallel = [(i, arr.tolist()) for i, arr in
                _iter_encoded(tok, tok_path, _rows(), eos_id, max_docs=None, num_workers=2)]

    assert serial == parallel
    assert len(serial) == len(TEXTS)
    # And every encoded doc ends with the EOS id.
    for _, ids in serial:
        assert ids[-1] == eos_id


def test_iter_encoded_respects_max_docs(tmp_path):
    tok, tok_path, eos_id = _train_tokenizer(tmp_path)
    out = list(_iter_encoded(tok, tok_path, _rows(), eos_id, max_docs=5, num_workers=2))
    assert [i for i, _ in out] == [0, 1, 2, 3, 4]


def test_three_bins_parallel_byte_equal_to_serial(tmp_path):
    tok, tok_path, eos_id = _train_tokenizer(tmp_path)
    holdout = 5  # forces several rows into val + test buckets

    s_train = tmp_path / 's_train.bin'
    s_val = tmp_path / 's_val.bin'
    s_test = tmp_path / 's_test.bin'
    s_n = _tokenize_stream_three_bins(
        tok, tok_path, _rows(), eos_id, str(s_train), str(s_val), str(s_test),
        holdout_period=holdout, num_workers=1,
    )

    p_train = tmp_path / 'p_train.bin'
    p_val = tmp_path / 'p_val.bin'
    p_test = tmp_path / 'p_test.bin'
    p_n = _tokenize_stream_three_bins(
        tok, tok_path, _rows(), eos_id, str(p_train), str(p_val), str(p_test),
        holdout_period=holdout, num_workers=4,
    )

    assert s_n == p_n
    assert s_train.read_bytes() == p_train.read_bytes()
    assert s_val.read_bytes() == p_val.read_bytes()
    assert s_test.read_bytes() == p_test.read_bytes()


def _fake_sources(monkeypatch, docs_by_name, weights):
    import prepare

    def fake_open(source):
        return iter(docs_by_name[source.name])

    monkeypatch.setattr(prepare, '_open_stream', fake_open)
    return [Source(name, 'path', None, w, lambda r: r)
            for name, w in weights.items()]


def test_mixed_stream_deterministic_and_complete(monkeypatch):
    docs = {'a': [f"a{i}" for i in range(50)], 'b': [f"b{i}" for i in range(10)]}
    sources = _fake_sources(monkeypatch, docs, {'a': 0.8, 'b': 0.2})

    out1 = list(mixed_stream(sources, seed=7))
    out2 = list(mixed_stream(sources, seed=7))
    assert out1 == out2  # deterministic for a fixed seed
    # every doc from every source is consumed exactly once
    assert sorted(out1) == sorted(docs['a'] + docs['b'])
    # per-source doc order is preserved
    assert [d for d in out1 if d.startswith('a')] == docs['a']


def test_mixed_stream_renormalizes_on_exhaustion(monkeypatch):
    docs = {'big': [f"x{i}" for i in range(200)], 'tiny': ['t0', 't1']}
    sources = _fake_sources(monkeypatch, docs, {'big': 0.5, 'tiny': 0.5})

    out = list(mixed_stream(sources, seed=3))
    # tiny exhausts early but the stream still drains the big source
    assert len(out) == 202
    assert out.count('t0') == 1 and out.count('t1') == 1
