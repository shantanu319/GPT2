import glob
import os

import numpy as np
import torch


BIN_DTYPE = np.uint16


class ShardedArray:
    """Read-only concatenation of memmapped shards, indexed as one array.

    The feeders only ever ask for len() and one contiguous window at a time,
    so shards are never materialized together — which is the point when the
    corpus is bigger than RAM.
    """

    def __init__(self, paths, dtype=BIN_DTYPE):
        self.parts = [np.memmap(p, dtype=dtype, mode='r') for p in paths]
        self.bounds = np.cumsum([0] + [len(part) for part in self.parts])
        self.dtype = dtype

    def __len__(self):
        return int(self.bounds[-1])

    def __getitem__(self, key):
        if not isinstance(key, slice):
            shard = int(np.searchsorted(self.bounds, key, side='right')) - 1
            return self.parts[shard][key - self.bounds[shard]]
        lo, hi, _ = key.indices(len(self))
        first = int(np.searchsorted(self.bounds, lo, side='right')) - 1
        pieces = []
        for i in range(first, len(self.parts)):
            start, end = self.bounds[i], self.bounds[i + 1]
            if start >= hi:
                break
            pieces.append(self.parts[i][max(lo, start) - start:min(hi, end) - start])
        if len(pieces) == 1:
            return pieces[0]
        return np.concatenate(pieces) if pieces else np.empty(0, dtype=self.dtype)


def shard_paths(path):
    """The shard set that stands in for `path`: train.bin -> train_00000.bin..."""
    stem, ext = os.path.splitext(path)
    return sorted(glob.glob(f"{stem}_[0-9]*{ext}"))


def load_bin(path):
    """A single .bin, or the numbered shards that replaced it."""
    if os.path.exists(path):
        return np.memmap(path, dtype=BIN_DTYPE, mode='r')
    shards = shard_paths(path)
    if not shards:
        raise FileNotFoundError(path)
    return ShardedArray(shards)


def bin_exists(path):
    return os.path.exists(path) or bool(shard_paths(path))


def load_bin_u8(path):
    return np.memmap(path, dtype=np.uint8, mode='r')


def read_corpus(filename, tokenizer):
    seq = []
    with open(filename, 'rt') as f:
        for line in f:
            line = line.replace('\n', '')
            tokens = tokenizer(line)
            for t in tokens['input_ids']:
                seq.append(t)
    return seq


class _PinnedRing:
    """Two pinned staging buffers per array + events, for 1-batch lookahead H2D.

    Each slot's event guards buffer reuse: re-filling a slot waits only on the
    copy issued from that same buffer two batches earlier (long done in practice).
    """

    def __init__(self, specs, device):
        # specs: list of (numel, torch dtype) — one pinned ring per array
        self.device = device
        self.bufs = [[torch.empty(numel, dtype=dtype, pin_memory=True)
                      for _ in range(2)] for numel, dtype in specs]
        self.events = [torch.cuda.Event() for _ in range(2)]

    def push(self, slot, arrays):
        self.events[slot].synchronize()
        out = []
        for ring, arr in zip(self.bufs, arrays):
            ring[slot].copy_(torch.from_numpy(arr))
            out.append(ring[slot].to(self.device, non_blocking=True))
        self.events[slot].record()
        return out


def _window_order(num_sequences, batch_size, shuffle, seed, rank=0, world=1):
    n_batches = num_sequences // batch_size
    if shuffle:
        order = list(np.random.default_rng(seed).permutation(n_batches))
    else:
        order = list(range(n_batches))
    if world > 1:
        # Truncate the ragged tail so every rank yields the same number of
        # batches; a short rank would deadlock the next gradient all-reduce.
        order = order[:len(order) - len(order) % world][rank::world]
    return order


def segment_ids_np(tokens2d, eos_id):
    """Per-token document segment ids for a (rows, T) token array.

    Derived from EOS positions, so no separate boundary sidecar is needed:
    EOS is the last token of its segment, the next token starts a new one
    (exclusive cumsum over EOS flags)."""
    is_eos = tokens2d == eos_id
    return np.cumsum(is_eos, axis=1, dtype=np.int64) - is_eos.astype(np.int64)


def segment_ids_torch(tokens2d, eos_id):
    """Torch twin of segment_ids_np (CPU feeder path)."""
    is_eos = tokens2d == eos_id
    return torch.cumsum(is_eos.long(), dim=1) - is_eos.long()


def _slice(data, b, batch_size, seq_len, dtype):
    lo, hi = b * batch_size * seq_len, (b + 1) * batch_size * seq_len
    return np.asarray(data[lo:hi]).astype(dtype, copy=False)


def data_feeder_masked(data, mask, batch_size, seq_len, device, eos_id=None,
                       rank=0, world=1):
    """Like data_feeder, but also yields a loss mask aligned with the targets.

    data: uint16 token memmap; mask: uint8 memmap of the same length
    (1 = compute loss on this token when it appears as a target).
    When eos_id is given, also yields per-token segment ids aligned with the
    inputs (documents/conversations end at EOS; see segment_ids_np)."""
    total = min(len(data), len(mask))
    num_sequences = total // seq_len
    order = _window_order(num_sequences, batch_size, False, 0, rank, world)
    if not order:
        return

    if device.type != 'cuda':
        for b in order:
            batch = torch.tensor(_slice(data, b, batch_size, seq_len, np.int64),
                                 device=device).view(batch_size, seq_len)
            mbatch = torch.tensor(np.asarray(mask[b * batch_size * seq_len:
                                                  (b + 1) * batch_size * seq_len]),
                                  dtype=torch.bool, device=device).view(batch_size, seq_len)
            if eos_id is None:
                yield batch[:, :-1], batch[:, 1:], mbatch[:, 1:]
            else:
                seg = segment_ids_torch(batch, eos_id)
                yield batch[:, :-1], batch[:, 1:], mbatch[:, 1:], seg[:, :-1]
        return

    n = batch_size * seq_len
    specs = [(n, torch.int64), (n, torch.bool)]
    if eos_id is not None:
        specs.append((n, torch.int64))
    ring = _PinnedRing(specs, device)

    def arrays(b):
        lo, hi = b * n, (b + 1) * n
        toks = _slice(data, b, batch_size, seq_len, np.int64)
        out = [toks, np.asarray(mask[lo:hi])]
        if eos_id is not None:
            out.append(segment_ids_np(toks.reshape(batch_size, seq_len),
                                      eos_id).reshape(-1))
        return out

    pending = ring.push(0, arrays(order[0]))
    for i, b in enumerate(order):
        arrays_out = pending
        if i + 1 < len(order):
            pending = ring.push((i + 1) % 2, arrays(order[i + 1]))
        batch = arrays_out[0].view(batch_size, seq_len)
        mbatch = arrays_out[1].view(batch_size, seq_len)
        if eos_id is None:
            yield batch[:, :-1], batch[:, 1:], mbatch[:, 1:]
        else:
            seg = arrays_out[2].view(batch_size, seq_len)
            yield batch[:, :-1], batch[:, 1:], mbatch[:, 1:], seg[:, :-1]


def data_feeder(data, batch_size, seq_len, device, shuffle=False, seed=42, eos_id=None,
                rank=0, world=1):
    """Yields (inputs, targets) windows of (batch_size, seq_len), shifted by one.

    shuffle=True serves the windows in a seeded-permuted order for the pass
    (reproducible given the same seed). On CUDA, H2D copies go through pinned
    staging buffers with a 1-batch lookahead so transfer overlaps compute.
    When eos_id is given, also yields per-token segment ids aligned with the
    inputs (documents end at EOS; see segment_ids_np)."""
    total = len(data)
    num_sequences = total // seq_len
    order = _window_order(num_sequences, batch_size, shuffle, seed, rank, world)
    if not order:
        return

    if device.type != 'cuda':
        for b in order:
            batch = torch.tensor(_slice(data, b, batch_size, seq_len, np.int64),
                                 device=device).view(batch_size, seq_len)
            if eos_id is None:
                yield batch[:, :-1], batch[:, 1:]
            else:
                seg = segment_ids_torch(batch, eos_id)
                yield batch[:, :-1], batch[:, 1:], seg[:, :-1]
        return

    n = batch_size * seq_len
    specs = [(n, torch.int64)]
    if eos_id is not None:
        specs.append((n, torch.int64))
    ring = _PinnedRing(specs, device)

    def arrays(b):
        toks = _slice(data, b, batch_size, seq_len, np.int64)
        out = [toks]
        if eos_id is not None:
            out.append(segment_ids_np(toks.reshape(batch_size, seq_len),
                                      eos_id).reshape(-1))
        return out

    pending = ring.push(0, arrays(order[0]))
    for i, b in enumerate(order):
        arrays_out = pending
        if i + 1 < len(order):
            pending = ring.push((i + 1) % 2, arrays(order[i + 1]))
        batch = arrays_out[0].view(batch_size, seq_len)
        if eos_id is None:
            yield batch[:, :-1], batch[:, 1:]
        else:
            seg = arrays_out[1].view(batch_size, seq_len)
            yield batch[:, :-1], batch[:, 1:], seg[:, :-1]
