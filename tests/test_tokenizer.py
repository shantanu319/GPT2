from tokenizer import BPETokenizer


CORPUS = (
    "the quick brown fox jumps over the lazy dog " * 20
    + "hello world hello there hello friends " * 20
)


def test_base_vocab_is_all_bytes():
    t = BPETokenizer()
    assert len(t.vocab) == 256
    for i in range(256):
        assert t.vocab[i] == bytes([i])


def test_train_produces_exact_vocab_size():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=300)
    assert len(t.vocab) == 300
    assert len(t.merges) == 44  # 300 - 256


def test_train_merges_shorten_token_sequence():
    """Trained tokenizer should produce fewer tokens than raw bytes."""
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400)
    sample = "the quick brown fox"
    assert len(t.encode(sample)) < len(sample.encode('utf-8'))


def test_ascii_roundtrip():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400)
    text = "the quick brown fox jumps"
    assert t.decode(t.encode(text)) == text


def test_unicode_roundtrip():
    t = BPETokenizer()
    t.train("café résumé naïve " * 20, vocab_size=320)
    for text in ["café", "résumé", "naïve", "café résumé"]:
        assert t.decode(t.encode(text)) == text


def test_unseen_text_roundtrip():
    """Encode/decode should round-trip on text not seen during training."""
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400)
    unseen = "Zebras don't eat pizza, 42 times!"
    assert t.decode(t.encode(unseen)) == unseen


def test_special_tokens_tokenized_as_single_id():
    t = BPETokenizer(special_tokens={'<|endoftext|>': 500})
    t.train(CORPUS, vocab_size=400)
    ids = t.encode("hello <|endoftext|> world")
    assert 500 in ids
    assert t.decode(ids) == "hello <|endoftext|> world"


def test_train_equivalent_to_naive():
    """Incremental train must produce the exact same merges as the
    straightforward per-chunk algorithm — same merges, same ids, same order."""
    from collections import Counter
    from tokenizer import _get_pair_counts, _merge, _pick_top_pair, GPT4_SPLIT_PATTERN
    import re

    def naive_train(text, vocab_size):
        compiled = re.compile(GPT4_SPLIT_PATTERN)
        ids = [list(c.encode('utf-8')) for c in compiled.findall(text)]
        merges = {}
        for i in range(vocab_size - 256):
            counts = Counter()
            for chunk in ids:
                _get_pair_counts(chunk, counts)
            if not counts:
                break
            top = _pick_top_pair(counts)
            new_id = 256 + i
            merges[top] = new_id
            ids = [_merge(c, top, new_id) for c in ids]
        return merges

    for corpus, vocab in [
        (CORPUS, 400),
        ("café résumé naïve " * 20, 320),
        ("a" * 100 + " b" * 100 + " ab " * 50, 280),
    ]:
        t = BPETokenizer()
        t.train(corpus, vocab_size=vocab)
        assert t.merges == naive_train(corpus, vocab), \
            f"dedup train diverged from naive for vocab={vocab}"


def test_chunk_cache_populated_and_consistent():
    """Repeated encodes of the same text should hit the cache and return
    identical results to a fresh tokenizer with no cache."""
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400)
    assert t._chunk_cache == {}

    text = "the quick brown fox the quick brown fox"
    first = t.encode(text)
    assert len(t._chunk_cache) > 0
    cache_after_first = dict(t._chunk_cache)

    second = t.encode(text)
    assert second == first
    # Cache should not grow when re-encoding same text.
    assert t._chunk_cache == cache_after_first

    # Compare against a fresh, cache-free tokenizer.
    t2 = BPETokenizer()
    t2.train(CORPUS, vocab_size=400)
    assert t2.encode(text) == first


def test_chunk_cache_dropped_on_pickle():
    """Cache must not be shipped through pickle — workers should start fresh."""
    import pickle
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=400)
    t.encode("the quick brown fox")
    assert len(t._chunk_cache) > 0

    t2 = pickle.loads(pickle.dumps(t))
    assert t2._chunk_cache == {}
    assert t2.merges == t.merges
    assert t2.encode("the quick brown fox") == t.encode("the quick brown fox")


def test_save_load_roundtrip(tmp_path):
    t = BPETokenizer(special_tokens={'<|endoftext|>': 500})
    t.train(CORPUS, vocab_size=350)

    path = tmp_path / "bpe.json"
    t.save(str(path))

    t2 = BPETokenizer()
    t2.load(str(path))

    text = "the quick brown fox <|endoftext|>"
    assert t2.encode(text) == t.encode(text)
    assert t2.decode(t.encode(text)) == text
    assert t2.merges == t.merges
    assert t2.special_tokens == t.special_tokens


def test_chunk_cache_bounded_by_max_size():
    """Cache must not grow unbounded — it should evict once max_size is hit."""
    t = BPETokenizer(cache_size=8)
    t.train(CORPUS, vocab_size=300)

    # Encode far more unique chunks than the cache can hold.
    text = " ".join(f"unique_word_{i}" for i in range(200))
    t.encode(text)
    assert len(t._chunk_cache) <= 8


def test_contains_pair_helper():
    from tokenizer import _contains_pair
    assert _contains_pair((1, 2, 3, 4), (2, 3))
    assert _contains_pair([1, 2, 3, 4], (1, 2))
    assert not _contains_pair((1, 2, 3), (3, 4))
    assert not _contains_pair((), (1, 2))
    assert not _contains_pair((5,), (5, 5))


def test_overlapping_special_tokens_longest_wins():
    """Longer specials must take precedence when one is a prefix of another."""
    t = BPETokenizer(special_tokens={'<|end|>': 500, '<|endoftext|>': 501})
    t.train(CORPUS, vocab_size=300)
    ids = t.encode("hi <|endoftext|> bye")
    assert 501 in ids
    assert 500 not in ids


def test_load_refreshes_special_token_maps(tmp_path):
    """After load(), decode() and encode() must use the new specials, not stale ones."""
    t = BPETokenizer(special_tokens={'<|endoftext|>': 500})
    t.train(CORPUS, vocab_size=300)
    path = tmp_path / "bpe.json"
    t.save(str(path))

    t2 = BPETokenizer()  # starts with empty specials
    t2.load(str(path))
    assert t2.inv_specials == {500: '<|endoftext|>'}
    assert t2._specials_re is not None
    assert t2.decode([500]) == '<|endoftext|>'


def test_train_resets_cache():
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=300)
    t.encode("the quick brown fox")
    assert len(t._chunk_cache) > 0
    t.train(CORPUS, vocab_size=300)
    assert len(t._chunk_cache) == 0


def test_pretokenizer_chunks_digit_runs_to_three():
    """Digit runs must split into <=3-char pre-tokens (arXiv:2402.14903)."""
    t = BPETokenizer()
    assert t._compiled.findall("1234567") == ["123", "456", "7"]
    assert t._compiled.findall("see 1234567890!") == ["see", " 123", "456", "789", "0", "!"]


def test_pretokenizer_separates_letters_and_digits():
    """Letters and digits must never share a pre-token."""
    t = BPETokenizer()
    assert t._compiled.findall("abc123") == ["abc", "123"]
    assert t._compiled.findall("abc123def") == ["abc", "123", "def"]


def test_underscore_roundtrip():
    """'_' is a word char but not a letter — it must still be consumable."""
    t = BPETokenizer()
    t.train(CORPUS, vocab_size=300)
    for text in ["a_b", "snake_case_function(x, y)"]:
        assert t.decode(t.encode(text)) == text
