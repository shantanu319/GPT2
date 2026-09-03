"""Tests for prepare.py — focus on parity between serial and parallel encode."""
import json
import os

import numpy as np

from core.data import BIN_DTYPE, shard_paths
from core.tokenizer import BPETokenizer
from pretrain.prepare import (_iter_encoded, _read_manifest, _shard_path,
                              _tokenize_stream_three_bins, mixed_stream, Source)


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


def _shard_bytes(train_path):
    """Train is written as numbered shards; concatenate them to compare."""
    return b''.join(open(p, 'rb').read() for p in shard_paths(str(train_path)))


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
    assert _shard_bytes(s_train) == _shard_bytes(p_train)
    assert s_val.read_bytes() == p_val.read_bytes()
    assert s_test.read_bytes() == p_test.read_bytes()


def _fake_sources(monkeypatch, docs_by_name, weights):
    import pretrain.prepare as prepare

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


def test_local_fast_path_matches_serial(monkeypatch, tmp_path):
    """fetch_source_to_disk + local_mixed_stream must reproduce mixed_stream."""
    import pretrain.prepare as prepare
    docs = {'a': [f"a{i}" for i in range(200)], 'b': [f"b{i}" for i in range(60)]}

    def fake_load_dataset(path, config, split, streaming):
        name = 'a' if 'aaa' in path else 'b'
        yield from ({'text': t} for t in docs[name])

    monkeypatch.setattr(prepare, 'load_dataset', fake_load_dataset)
    sources = [Source('a', 'aaa_data', None, 0.8, lambda r: r['text']),
               Source('b', 'bbb_data', None, 0.2, lambda r: r['text'])]

    def fake_open(source):
        return iter(docs[source.name])

    monkeypatch.setattr(prepare, '_open_stream', fake_open)
    serial = list(mixed_stream(sources, seed=1337))

    paths = {s.name: prepare.fetch_source_to_disk(s, 10**9, str(tmp_path))
             for s in sources}
    local = list(prepare.local_mixed_stream(paths, sources, seed=1337))
    assert local == serial

    capped = list(prepare.local_mixed_stream(paths, sources, seed=1337,
                                             max_docs=100))
    assert capped == serial[:100]


def test_every_mix_sums_to_one_and_names_each_source_once():
    import pretrain.prepare as prepare
    assert set(prepare.MIXES) == {'pretrain', 'anneal'}
    for name, sources in prepare.MIXES.items():
        assert abs(sum(s.weight for s in sources) - 1.0) < 1e-9, name
        assert len({s.name for s in sources}) == len(sources), name
        assert all(callable(s.render) for s in sources), name


def test_anneal_mix_tilts_toward_math_and_code():
    """The decay-phase corpus is the whole point of -anneal_dir: less web,
    more of what SmolLM2 found pays off late in the schedule."""
    import pretrain.prepare as prepare
    weight = lambda sources, *names: sum(s.weight for s in sources if s.name in names)
    web = ('fineweb-edu', 'dclm')
    assert weight(prepare.ANNEAL_SOURCES, *web) < weight(prepare.SOURCES, *web)
    assert weight(prepare.ANNEAL_SOURCES, 'finemath', 'openmath', 'code-python') > \
        weight(prepare.SOURCES, 'finemath', 'openmath', 'code-python')


def _kill_after_last_shard(tmp_path):
    """A killed run never writes its final manifest, so the last one on disk
    is the mid-run entry with complete=False."""
    path = f"{tmp_path}/train_manifest.json"
    state = json.load(open(path))
    state['complete'] = False
    json.dump(state, open(path, 'w'))
    return state


def test_tokenize_resumes_from_the_last_sealed_shard(tmp_path):
    """A preempted run must not re-tokenize the corpus from scratch."""
    tok, tok_path, eos_id = _train_tokenizer(tmp_path)
    paths = [str(tmp_path / f'{n}.bin') for n in ('train', 'val', 'test')]

    whole = _tokenize_stream_three_bins(tok, tok_path, _rows(), eos_id, *paths,
                                        holdout_period=5, num_workers=1)
    reference = (_shard_bytes(paths[0]),
                 open(paths[1], 'rb').read(), open(paths[2], 'rb').read())

    for path in shard_paths(paths[0]) + paths[1:]:
        os.remove(path)
    os.remove(f"{tmp_path}/train_manifest.json")

    # Kill the run partway by capping it, then let it finish from the manifest.
    shard = 40  # bytes -> seals a shard every couple of docs
    _tokenize_stream_three_bins(tok, tok_path, _rows()[:20], eos_id, *paths,
                                holdout_period=5, num_workers=1,
                                shard_tokens=shard)
    state = _kill_after_last_shard(tmp_path)
    assert 0 < state['docs'] < len(TEXTS)

    resumed = _tokenize_stream_three_bins(tok, tok_path, _rows(), eos_id, *paths,
                                          holdout_period=5, num_workers=1,
                                          shard_tokens=shard)
    assert resumed == whole
    assert (_shard_bytes(paths[0]),
            open(paths[1], 'rb').read(), open(paths[2], 'rb').read()) == reference


def test_partial_shard_is_discarded_on_resume(tmp_path):
    """The in-flight shard is not a resume point; a torn write must not be
    mistaken for tokenized data."""
    tok, tok_path, eos_id = _train_tokenizer(tmp_path)
    paths = [str(tmp_path / f'{n}.bin') for n in ('train', 'val', 'test')]
    _tokenize_stream_three_bins(tok, tok_path, _rows(), eos_id, *paths,
                                holdout_period=5, num_workers=1, shard_tokens=40)
    state = _kill_after_last_shard(tmp_path)

    # Simulate a kill after the manifest was written: an extra shard and
    # trailing bytes in val that the manifest does not account for.
    with open(_shard_path(paths[0], state['shards']), 'wb') as handle:
        handle.write(b'\xff' * 64)
    with open(paths[1], 'ab') as handle:
        handle.write(b'\xff' * 8)

    recovered = _read_manifest(*paths)
    assert recovered == state
    assert len(shard_paths(paths[0])) == state['shards']
    assert os.path.getsize(paths[1]) == state['val_bytes']


def test_completed_prepare_is_not_redone(tmp_path):
    tok, tok_path, eos_id = _train_tokenizer(tmp_path)
    paths = [str(tmp_path / f'{n}.bin') for n in ('train', 'val', 'test')]
    first = _tokenize_stream_three_bins(tok, tok_path, _rows(), eos_id, *paths,
                                        holdout_period=5, num_workers=1)
    assert json.load(open(f"{tmp_path}/train_manifest.json"))['complete']
    # An empty stream would produce nothing if it actually re-tokenized.
    again = _tokenize_stream_three_bins(tok, tok_path, iter([]), eos_id, *paths,
                                        holdout_period=5, num_workers=1)
    assert again == first
