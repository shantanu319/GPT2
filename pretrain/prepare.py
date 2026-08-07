"""Offline data prep: train a BPE tokenizer and emit .bin shards for a mixed corpus.

The corpus is a weighted interleave of several HuggingFace streams (see SOURCES):
real educational web (fineweb-edu-dedup, DCLM-baseline), synthetic textbooks
(cosmopedia-v2), educational Python code (codeparrot-clean) and math (FineMath-4+).

Run once to produce:
  {output_dir}/tokenizer.json
  {output_dir}/train.bin  (uint16 tokens, docs separated by <|endoftext|>)
  {output_dir}/val.bin
  {output_dir}/test.bin

Then train.py points at {output_dir} and mmaps the .bin files directly.
"""
import argparse
import multiprocessing as mp
import os
import random
import struct
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from datasets import load_dataset

from core.chat_format import EOS_TOKEN, special_token_map
from core.data import BIN_DTYPE
from core.tokenizer import BPETokenizer


def _render_text(row):
    return row['text']


def _render_content(row):
    return row['content']


def _render_problem_solution(row):
    return f"Problem:\n{row['problem']}\n\nSolution:\n{row['generated_solution']}"


def _render_camel(row):
    return f"Problem:\n{row['message_1']}\n\nSolution:\n{row['message_2']}"


@dataclass(frozen=True)
class Source:
    name: str
    path: str
    config: Optional[str]
    weight: float
    render: Callable


# Weighted pretraining mixture (SmolLM2-style: FineWeb-Edu + DCLM web backbone —
# FineWeb-Edu wins ARC/MMLU, DCLM wins HellaSwag/CSQA — plus a synthetic-textbook
# slice, educational Python code, and a pinch of math; arXiv:2502.02737,
# arXiv:2408.10914). openmathinstruct/camel-physics renders stay available
# above but are out of the default mix.
SOURCES = [
    Source('fineweb-edu', 'HuggingFaceTB/smollm-corpus',    'fineweb-edu-dedup', 0.42, _render_text),
    Source('dclm',        'mlfoundations/dclm-baseline-1.0', None,               0.28, _render_text),
    Source('cosmopedia',  'HuggingFaceTB/smollm-corpus',    'cosmopedia-v2',     0.15, _render_text),
    Source('code-python', 'codeparrot/codeparrot-clean',    None,                0.10, _render_content),
    Source('finemath',    'HuggingFaceTB/finemath',         'finemath-4plus',    0.05, _render_text),
]

# Worker-process state, populated once per worker by _init_worker.
_WORKER_TOKENIZER = None
_WORKER_EOS_ID = None


def _init_worker(tokenizer_path, eos_id):
    global _WORKER_TOKENIZER, _WORKER_EOS_ID
    _WORKER_TOKENIZER = BPETokenizer()
    _WORKER_TOKENIZER.load(tokenizer_path)
    _WORKER_EOS_ID = eos_id


def _worker_encode(text):
    ids = _WORKER_TOKENIZER.encode_ordinary(text)
    ids.append(_WORKER_EOS_ID)
    arr = np.array(ids, dtype=BIN_DTYPE)
    if arr.size and arr.max() >= np.iinfo(BIN_DTYPE).max:
        raise ValueError(f"token id {arr.max()} exceeds {BIN_DTYPE} range")
    return arr


def _open_stream(source):
    ds = load_dataset(source.path, source.config, split='train', streaming=True)
    for row in ds:
        yield source.render(row)


def mixed_stream(sources=SOURCES, seed=1337):
    """Yield rendered docs sampled across sources proportionally to weight.

    Deterministic for a fixed seed. When a source is exhausted it is dropped
    and the remaining weights are renormalized, so small sets (e.g. CAMEL
    physics, ~20k rows) mix in early and the stream keeps going."""
    rng = random.Random(seed)
    iters = [(s.name, _open_stream(s), s.weight) for s in sources]
    while iters:
        names, streams, weights = zip(*iters)
        idx = rng.choices(range(len(iters)), weights=weights)[0]
        try:
            yield next(streams[idx])
        except StopIteration:
            print(f"  source '{names[idx]}' exhausted — renormalizing mixture")
            iters.pop(idx)


# --------------------------------------------------------------------------
# Fast path: per-source parallel fetch to disk, then a local interleave.
#
# A single mixed_stream() round-robins five HTTP streams and runs at the pace
# of the slowest pick (~10 docs/s). Each source streams at thousands of docs/s
# on its own, so with --max-train-docs set we instead stream each source in a
# separate process into a length-prefixed cache file, then interleave locally
# with the SAME seeded weighted logic (byte-identical doc order to the serial
# path for the docs it would have produced).
# --------------------------------------------------------------------------

def _iter_source_rows(source):
    """Row iterator for a source. camel-ai/physics streams at ~1 doc/s (its
    generator crawls), but the whole dataset is one small zip on the hub —
    download it and iterate the JSONs in sorted order, which is byte-identical
    to the datasets streaming order (verified)."""
    if source.path == 'camel-ai/physics':
        import json
        import zipfile
        from huggingface_hub import hf_hub_download
        zpath = hf_hub_download(source.path, 'physics.zip', repo_type='dataset')
        with zipfile.ZipFile(zpath) as z:
            for name in sorted(n for n in z.namelist() if n.endswith('.json')):
                yield json.loads(z.read(name))
        return
    yield from load_dataset(source.path, source.config, split='train', streaming=True)


def fetch_source_to_disk(source, quota, cache_dir):
    """Stream up to `quota` rendered docs from one source into a cache file.

    Format: repeated (uint64 LE byte-length, utf-8 bytes). Resumable — an
    existing complete-enough cache file is reused."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"fetch_{source.name}.bin")
    if os.path.exists(path):
        print(f"  reusing cached fetch: {path}")
        return path
    n = 0
    tmp_path = path + '.tmp'
    with open(tmp_path, 'wb') as f:
        for row in _iter_source_rows(source):
            if n >= quota:
                break
            b = source.render(row).encode('utf-8')
            f.write(struct.pack('<Q', len(b)))
            f.write(b)
            n += 1
    os.replace(tmp_path, path)  # atomic: a killed fetch never leaves a reusable partial
    print(f"  fetched {n:,} docs from {source.name} -> {path}")
    return path


def _read_docs(path):
    with open(path, 'rb') as f:
        while True:
            head = f.read(8)
            if len(head) < 8:
                return
            (n,) = struct.unpack('<Q', head)
            yield f.read(n).decode('utf-8')


def local_mixed_stream(fetch_paths, sources, seed=1337, max_docs=None):
    """mixed_stream over on-disk fetch caches: same rng, same drop-on-exhaust."""
    rng = random.Random(seed)
    iters = [(s.name, _read_docs(fetch_paths[s.name]), s.weight) for s in sources]
    n = 0
    while iters:
        if max_docs is not None and n >= max_docs:
            return
        names, streams, weights = zip(*iters)
        idx = rng.choices(range(len(iters)), weights=weights)[0]
        try:
            yield next(streams[idx])
            n += 1
        except StopIteration:
            print(f"  source '{names[idx]}' exhausted — renormalizing mixture")
            iters.pop(idx)


def _fetch_one(job):
    source, quota, cache_dir = job
    return fetch_source_to_disk(source, quota, cache_dir)


def fetch_all(sources, max_docs, cache_dir, workers):
    """Fetch all sources in parallel; returns {name: cache_path}.

    Quota per source is its weighted share of max_docs (+ headroom for the
    stochastic interleave); sources that run dry simply produce less, exactly
    like the serial path."""
    total_w = sum(s.weight for s in sources)
    jobs = [(s, int(max_docs * s.weight / total_w * 1.05) + 1000, cache_dir)
            for s in sources]
    with mp.get_context('fork').Pool(processes=min(workers, len(jobs))) as pool:
        paths = pool.map(_fetch_one, jobs)
    return {s.name: p for s, p in zip(sources, paths)}


def _bpe_training_corpus(stream, num_docs):
    texts = []
    for i, text in enumerate(stream):
        if i >= num_docs:
            break
        texts.append(text)
    return '\n'.join(texts)


def _encode_text(tokenizer, text, eos_id):
    ids = tokenizer.encode_ordinary(text)
    ids.append(eos_id)
    arr = np.array(ids, dtype=BIN_DTYPE)
    if arr.size and arr.max() >= np.iinfo(BIN_DTYPE).max:
        raise ValueError(f"token id {arr.max()} exceeds {BIN_DTYPE} range")
    return arr


def _iter_encoded(tokenizer, tokenizer_path, stream, eos_id, max_docs, num_workers):
    """Yield (doc_index, encoded_array) preserving doc order.

    Uses a process pool when num_workers > 1; falls back to in-process
    serial encoding otherwise."""
    def texts():
        for i, text in enumerate(stream):
            if max_docs is not None and i >= max_docs:
                break
            yield text

    if num_workers <= 1:
        for i, text in enumerate(texts()):
            yield i, _encode_text(tokenizer, text, eos_id)
        return

    # Fork beats spawn here by ~3-4x (no Python re-import cost per worker, and
    # workers inherit the loaded tokenizer via copy-on-write). Spawn is so much
    # slower than serial it isn't worth using; on platforms without fork
    # (Windows) we fall back to a serial run.
    try:
        ctx = mp.get_context('fork')
    except ValueError:
        for i, text in enumerate(texts()):
            yield i, _encode_text(tokenizer, text, eos_id)
        return

    with ctx.Pool(processes=num_workers,
                  initializer=_init_worker,
                  initargs=(tokenizer_path, eos_id)) as pool:
        for i, arr in enumerate(pool.imap(_worker_encode, texts(), chunksize=64)):
            yield i, arr


def _tokenize_stream_three_bins(
    tokenizer, tokenizer_path, stream, eos_id, train_path, val_path, test_path,
    max_docs=None, holdout_period=500, num_workers=1,
):
    """Single-pass streaming tokenize into three .bin shards.

    Doc index i routes as: i % holdout_period == 0 -> val,
    == 1 -> test, else -> train. Deterministic and single-pass."""
    train_tokens = 0
    val_tokens = 0
    test_tokens = 0
    tmp_paths = [p + '.tmp' for p in (train_path, val_path, test_path)]
    with open(tmp_paths[0], 'wb') as trf, open(tmp_paths[1], 'wb') as vf, open(tmp_paths[2], 'wb') as tf:
        for i, arr in _iter_encoded(tokenizer, tokenizer_path, stream, eos_id,
                                     max_docs, num_workers):
            bucket = i % holdout_period
            if bucket == 0:
                vf.write(arr.tobytes())
                val_tokens += len(arr)
            elif bucket == 1:
                tf.write(arr.tobytes())
                test_tokens += len(arr)
            else:
                trf.write(arr.tobytes())
                train_tokens += len(arr)
            if (i + 1) % 10000 == 0:
                total = train_tokens + val_tokens + test_tokens
                print(f"  tokenized {i + 1} docs, {total:,} tokens total")
    # Atomic finalize: a killed run never leaves reusable partial shards.
    for tmp, final in zip(tmp_paths, (train_path, val_path, test_path)):
        os.replace(tmp, final)
    return train_tokens, val_tokens, test_tokens


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='data_cache/cosmopedia')
    parser.add_argument('--vocab-size', type=int, default=32000,
                        help='Total vocab size including special tokens')
    parser.add_argument('--bpe-train-docs', type=int, default=100000,
                        help='Number of docs (from the mixed stream) for BPE training')
    parser.add_argument('--max-train-docs', type=int, default=None,
                        help='Cap for tokenizing the mixed stream (default: full stream)')
    parser.add_argument('--holdout-period', type=int, default=500,
                        help='Reserve 1-in-N docs each for val and test from the train stream.')
    parser.add_argument('--seed', type=int, default=1337,
                        help='Seed for the weighted source interleave')
    args = parser.parse_args()

    assert args.vocab_size > 256, "vocab_size must leave room for base bytes"
    assert np.iinfo(BIN_DTYPE).max >= args.vocab_size - 1, \
        f"vocab_size too large for {BIN_DTYPE}"

    os.makedirs(args.output_dir, exist_ok=True)
    num_workers = os.cpu_count() or 1

    # Reserve chat specials (<|im_start|>, <|im_end|>) now so SFT later needs
    # no embedding resize; they're simply unused during pretraining.
    specials = special_token_map(args.vocab_size)
    eos_id = specials[EOS_TOKEN]

    mix_desc = ', '.join(f"{s.name}={s.weight:.0%}" for s in SOURCES)
    print(f"Mixture: {mix_desc}")

    if args.max_train_docs is not None:
        # Fast path: parallel per-source fetch, then local interleave (same
        # seeded order as the serial mixed_stream).
        cache_dir = os.path.join(args.output_dir, 'fetch_cache')
        print(f"Fetching up to {args.max_train_docs:,} docs "
              f"(per-source parallel streams)...")
        fetch_paths = fetch_all(SOURCES, args.max_train_docs, cache_dir,
                                workers=min(len(SOURCES), num_workers))

        def make_stream():
            return local_mixed_stream(fetch_paths, SOURCES, seed=args.seed,
                                      max_docs=args.max_train_docs)
    else:
        def make_stream():
            return mixed_stream(seed=args.seed)

    tok_path = os.path.join(args.output_dir, 'tokenizer.json')
    if os.path.exists(tok_path):
        # Resume support: BPE is the expensive phase, so a preempted/restarted
        # run reuses the saved tokenizer and goes straight to tokenizing.
        # Delete tokenizer.json to force a retrain.
        tokenizer = BPETokenizer()
        tokenizer.load(tok_path)
        assert tokenizer.vocab_size == args.vocab_size, \
            f"existing tokenizer vocab {tokenizer.vocab_size} != requested {args.vocab_size}"
        print(f"Reusing existing tokenizer: {tok_path}")
    else:
        tokenizer = BPETokenizer(special_tokens=specials)
        print(f"Loading {args.bpe_train_docs} mixed docs for BPE training...")
        bpe_corpus = _bpe_training_corpus(make_stream(), args.bpe_train_docs)
        print(f"  BPE training corpus: {len(bpe_corpus):,} chars")

        target_ordinary_vocab = args.vocab_size - len(tokenizer.special_tokens)
        print(f"Training BPE to {target_ordinary_vocab} ordinary tokens (+{len(tokenizer.special_tokens)} special)...")
        tokenizer.train(bpe_corpus, vocab_size=target_ordinary_vocab, verbose=True)

        tokenizer.save(tok_path)
        print(f"Saved tokenizer: {tok_path}")

    train_path = os.path.join(args.output_dir, 'train.bin')
    val_path = os.path.join(args.output_dir, 'val.bin')
    test_path = os.path.join(args.output_dir, 'test.bin')

    print(f"Tokenizing mixed stream into train/val/test "
          f"(holdout 2-in-{args.holdout_period}, num_workers={num_workers})...")
    n_train, n_val, n_test = _tokenize_stream_three_bins(
        tokenizer, tok_path, make_stream(), eos_id,
        train_path, val_path, test_path,
        max_docs=args.max_train_docs, holdout_period=args.holdout_period,
        num_workers=num_workers,
    )
    print(f"  wrote {n_train:,} tokens to {train_path}")
    print(f"  wrote {n_val:,} tokens to {val_path}")
    print(f"  wrote {n_test:,} tokens to {test_path}")


if __name__ == '__main__':
    main()
