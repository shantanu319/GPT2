"""Tests for the self-directed curriculum: arm shards and the director policy."""
import struct

import numpy as np

from core.data import load_bin
from core.tokenizer import BPETokenizer
from selfdirect.domains import build_arm, discover_arms

CORPUS = ("the quick brown fox jumps over the lazy dog "
          "hello world hello there hello friends " * 30)


def _tokenizer():
    tok = BPETokenizer(special_tokens={'<|endoftext|>': 511})
    tok.train(CORPUS, vocab_size=400)
    return tok, 511


def _write_fetch(path, texts):
    with open(path, 'wb') as f:
        for t in texts:
            b = t.encode('utf-8')
            f.write(struct.pack('<Q', len(b)))
            f.write(b)


def test_discover_arms_finds_fetch_caches(tmp_path):
    _write_fetch(tmp_path / 'fetch_alpha.bin', ['a'])
    _write_fetch(tmp_path / 'fetch_beta.bin', ['b'])
    (tmp_path / 'notes.txt').write_text('ignore me')
    assert sorted(discover_arms(str(tmp_path))) == ['alpha', 'beta']


def test_build_arm_routes_every_nth_doc_to_the_probe(tmp_path):
    tok, eos = _tokenizer()
    texts = [f"hello world number {i}" for i in range(10)]
    _write_fetch(tmp_path / 'fetch_x.bin', texts)

    n_train, n_probe = build_arm(str(tmp_path / 'fetch_x.bin'), str(tmp_path / 'x'),
                                 tok, eos, probe_tokens=10**6, probe_period=2)

    total = sum(len(tok.encode_ordinary(t)) + 1 for t in texts)
    assert n_train + n_probe == total
    # Docs 0,2,4,6,8 are held out — half the docs, so roughly half the tokens.
    assert n_probe == sum(len(tok.encode_ordinary(t)) + 1 for t in texts[::2])
    assert len(load_bin(str(tmp_path / 'x' / 'probe.bin'))) == n_probe
    assert len(load_bin(str(tmp_path / 'x' / 'train.bin'))) == n_train


def test_build_arm_stops_holding_out_once_the_probe_budget_is_met(tmp_path):
    tok, eos = _tokenizer()
    texts = [f"hello world number {i}" for i in range(20)]
    _write_fetch(tmp_path / 'fetch_x.bin', texts)

    n_train, n_probe = build_arm(str(tmp_path / 'fetch_x.bin'), str(tmp_path / 'x'),
                                 tok, eos, probe_tokens=6, probe_period=1)

    # probe_period=1 would hold out everything; the budget cuts it off after
    # the first doc that crosses 6 tokens.
    assert 6 <= n_probe < 6 + len(tok.encode_ordinary(texts[0])) + 1
    assert n_train > n_probe


def test_build_arm_max_docs_caps_the_read(tmp_path):
    tok, eos = _tokenizer()
    _write_fetch(tmp_path / 'fetch_x.bin', [f"doc {i}" for i in range(50)])

    n_train, n_probe = build_arm(str(tmp_path / 'fetch_x.bin'), str(tmp_path / 'x'),
                                 tok, eos, probe_tokens=0, probe_period=25, max_docs=3)

    assert n_probe == 0
    assert n_train == sum(len(tok.encode_ordinary(f"doc {i}")) + 1 for i in range(3))


def test_arm_shards_are_uint16_token_ids(tmp_path):
    tok, eos = _tokenizer()
    _write_fetch(tmp_path / 'fetch_x.bin', ["hello world", "the quick brown fox"])
    build_arm(str(tmp_path / 'fetch_x.bin'), str(tmp_path / 'x'), tok, eos,
              probe_tokens=10**6, probe_period=2)

    train = load_bin(str(tmp_path / 'x' / 'train.bin'))
    assert train.dtype == np.uint16
    assert train[-1] == eos  # every doc is EOS-terminated
