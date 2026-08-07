import json
import os
import re
from collections import Counter, OrderedDict


# GPT-4 style pre-tokenization regex (cl100k/o200k lineage), in stdlib-re form:
# letters and digits never share a pre-token, and digit runs are chunked to <=3
# chars so rare numbers don't fragment arbitrarily (arXiv:2402.14903). Applied
# before BPE so merges can't cross chunk boundaries. The punctuation class is
# written (?:[^\s\w]|_) because plain [^\s\w] excludes '_' — without it,
# underscores would match no alternative and be silently dropped.
GPT4_SPLIT_PATTERN = (r"'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD]"
                      r"| ?[^\W\d_]+| ?\d{1,3}| ?(?:[^\s\w]|_)+"
                      r"|\s+(?!\S)|\s+")

# Legacy GPT-2 pattern, kept for reference; tokenizer.json stores the pattern
# it was trained with, so old files still load with this one.
GPT2_SPLIT_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+"""

DEFAULT_CHUNK_CACHE_SIZE = 50_000


def _get_pair_counts(tokens, counts=None):
    counts = counts if counts is not None else Counter()
    for pair in zip(tokens, tokens[1:]):
        counts[pair] += 1
    return counts


def _merge(tokens, pair, new_id):
    result = []
    i = 0
    n = len(tokens)
    while i < n:
        if i < n - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
            result.append(new_id)
            i += 2
        else:
            result.append(tokens[i])
            i += 1
    return result


def _contains_pair(tokens, pair):
    a, b = pair
    return any(x == a and y == b for x, y in zip(tokens, tokens[1:]))


def _pick_top_pair(pair_counts):
    """Highest-count pair; ties broken by lex-smallest pair.

    Deterministic so the incremental trainer's output doesn't depend on the
    order in which pairs were first observed across merge steps.
    """
    return min(pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))[0]


class _LRUCache:
    """Small bounded LRU cache built on OrderedDict."""

    def __init__(self, max_size):
        self.max_size = max_size
        self._data = OrderedDict()

    def __len__(self):
        return len(self._data)

    def __contains__(self, key):
        return key in self._data

    def __getitem__(self, key):
        return self._data[key]

    def keys(self):
        return self._data.keys()

    def __eq__(self, other):
        if isinstance(other, _LRUCache):
            return self._data == other._data
        if isinstance(other, dict):
            return dict(self._data) == other
        return NotImplemented

    def get(self, key):
        val = self._data.get(key)
        if val is not None:
            self._data.move_to_end(key)
        return val

    def put(self, key, value):
        if key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.max_size:
            self._data.popitem(last=False)

    def clear(self):
        self._data.clear()


class BPETokenizer:
    def __init__(self, pattern=None, special_tokens=None, cache_size=DEFAULT_CHUNK_CACHE_SIZE):
        self.pattern = pattern or GPT4_SPLIT_PATTERN
        self._compiled = re.compile(self.pattern)
        self.special_tokens = dict(special_tokens or {})
        self.merges = {}  # (int, int) -> int
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.cache_size = cache_size
        self._chunk_cache = _LRUCache(cache_size)
        self._refresh_special_token_maps()

    def __getstate__(self):
        # Don't ship the chunk cache through pickle (e.g. multiprocessing) —
        # workers should accumulate their own cache.
        state = self.__dict__.copy()
        state['_chunk_cache'] = _LRUCache(self.cache_size)
        return state

    @property
    def vocab_size(self):
        return len(self.vocab) + len(self.special_tokens)

    def _compile_specials(self):
        if not self.special_tokens:
            self._specials_re = None
            return
        # Sort by descending length so longer tokens take precedence when one
        # special is a prefix of another.
        keys = sorted(self.special_tokens.keys(), key=len, reverse=True)
        pattern = '(' + '|'.join(re.escape(s) for s in keys) + ')'
        self._specials_re = re.compile(pattern)

    def _refresh_special_token_maps(self):
        self.inv_specials = {v: k for k, v in self.special_tokens.items()}
        self._compile_specials()

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256, "vocab_size must be at least 256 (one per byte)"
        num_merges = vocab_size - 256

        # Pre-tokenize and dedupe: identical chunks share a merge schedule, so
        # we only merge each unique chunk once per step, weighted by frequency.
        chunk_freq = Counter(self._compiled.findall(text))
        chunks = {tuple(c.encode('utf-8')): f for c, f in chunk_freq.items()}

        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._chunk_cache = _LRUCache(self.cache_size)

        # Incremental BPE: maintain a global pair_counts Counter and a
        # pair_to_chunks inverted index. Each merge step only revisits the
        # chunks that actually contain the selected pair, instead of rescanning
        # every chunk. Turns the per-step cost from O(total tokens across all
        # chunks) into O(tokens in affected chunks).
        pair_counts = Counter()
        pair_to_chunks = {}
        for ck, f in chunks.items():
            for p in zip(ck, ck[1:]):
                pair_counts[p] += f
                pair_to_chunks.setdefault(p, set()).add(ck)

        for i in range(num_merges):
            if not pair_counts:
                break
            top_pair = _pick_top_pair(pair_counts)
            new_id = 256 + i
            self.merges[top_pair] = new_id
            self.vocab[new_id] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]

            affected = pair_to_chunks.pop(top_pair, set())
            for ck in affected:
                f = chunks.pop(ck, None)
                if f is None:
                    continue  # already folded into another new_ck this step
                # subtract this chunk's contribution to every pair it had
                for p in zip(ck, ck[1:]):
                    pair_counts[p] -= f
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    s = pair_to_chunks.get(p)
                    if s is not None:
                        s.discard(ck)
                        if not s:
                            del pair_to_chunks[p]
                # apply the merge and re-register
                new_ck = tuple(_merge(ck, top_pair, new_id))
                if new_ck in chunks:
                    chunks[new_ck] += f
                else:
                    chunks[new_ck] = f
                for p in zip(new_ck, new_ck[1:]):
                    pair_counts[p] += f
                    pair_to_chunks.setdefault(p, set()).add(new_ck)

            if verbose and (i + 1) % 100 == 0:
                merged_bytes = self.vocab[new_id]
                try:
                    preview = merged_bytes.decode('utf-8')
                except UnicodeDecodeError:
                    preview = repr(merged_bytes)
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {new_id} ({preview!r})")

    def _encode_chunk(self, chunk):
        cached = self._chunk_cache.get(chunk)
        if cached is not None:
            return cached
        chunk_ids = list(chunk.encode('utf-8'))
        while len(chunk_ids) >= 2:
            pairs = set(zip(chunk_ids, chunk_ids[1:]))
            pair = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
            chunk_ids = _merge(chunk_ids, pair, self.merges[pair])
        result = tuple(chunk_ids)
        self._chunk_cache.put(chunk, result)
        return result

    def encode_ordinary(self, text):
        """Encode ignoring special tokens."""
        out = []
        for chunk in self._compiled.findall(text):
            out.extend(self._encode_chunk(chunk))
        return out

    def encode(self, text):
        if self._specials_re is None:
            return self.encode_ordinary(text)
        out = []
        for piece in self._specials_re.split(text):
            if piece in self.special_tokens:
                out.append(self.special_tokens[piece])
            elif piece:
                out.extend(self.encode_ordinary(piece))
        return out

    def decode(self, ids):
        inv_specials = self.inv_specials
        parts = []
        byte_buf = bytearray()
        for i in ids:
            if i in inv_specials:
                if byte_buf:
                    parts.append(bytes(byte_buf).decode('utf-8', errors='replace'))
                    byte_buf.clear()
                parts.append(inv_specials[i])
            else:
                byte_buf.extend(self.vocab[i])
        if byte_buf:
            parts.append(bytes(byte_buf).decode('utf-8', errors='replace'))
        return ''.join(parts)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        data = {
            'version': 'bpe-v1',
            'pattern': self.pattern,
            'special_tokens': self.special_tokens,
            'merges': [[a, b] for (a, b) in sorted(self.merges, key=self.merges.get)],
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        assert data.get('version') == 'bpe-v1', f"unexpected file version: {data.get('version')!r}"
        self.pattern = data['pattern']
        self._compiled = re.compile(self.pattern)
        self.special_tokens = dict(data.get('special_tokens') or {})
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self._chunk_cache = _LRUCache(self.cache_size)
        for i, (a, b) in enumerate(data['merges']):
            new_id = 256 + i
            self.merges[(a, b)] = new_id
            self.vocab[new_id] = self.vocab[a] + self.vocab[b]
        self._refresh_special_token_maps()
