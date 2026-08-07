import numpy as np
import torch


BIN_DTYPE = np.uint16


def load_bin(path):
    return np.memmap(path, dtype=BIN_DTYPE, mode='r')


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


def _window_order(num_sequences, batch_size, shuffle, seed):
    n_batches = num_sequences // batch_size
    if shuffle:
        return list(np.random.default_rng(seed).permutation(n_batches))
    return list(range(n_batches))


def _slice(data, b, batch_size, seq_len, dtype):
    lo, hi = b * batch_size * seq_len, (b + 1) * batch_size * seq_len
    return np.asarray(data[lo:hi]).astype(dtype, copy=False)


def data_feeder_masked(data, mask, batch_size, seq_len, device):
    """Like data_feeder, but also yields a loss mask aligned with the targets.

    data: uint16 token memmap; mask: uint8 memmap of the same length
    (1 = compute loss on this token when it appears as a target)."""
    total = min(len(data), len(mask))
    num_sequences = total // seq_len
    order = _window_order(num_sequences, batch_size, shuffle=False, seed=0)
    if not order:
        return

    if device.type != 'cuda':
        for b in order:
            batch = torch.tensor(_slice(data, b, batch_size, seq_len, np.int64),
                                 device=device).view(batch_size, seq_len)
            mbatch = torch.tensor(np.asarray(mask[b * batch_size * seq_len:
                                                  (b + 1) * batch_size * seq_len]),
                                  dtype=torch.bool, device=device).view(batch_size, seq_len)
            yield batch[:, :-1], batch[:, 1:], mbatch[:, 1:]
        return

    ring = _PinnedRing([(batch_size * seq_len, torch.int64),
                        (batch_size * seq_len, torch.bool)], device)
    n = batch_size * seq_len

    def arrays(b):
        lo, hi = b * n, (b + 1) * n
        return [_slice(data, b, batch_size, seq_len, np.int64),
                np.asarray(mask[lo:hi])]

    pending = ring.push(0, arrays(order[0]))
    for i, b in enumerate(order):
        batch, mbatch = pending
        if i + 1 < len(order):
            pending = ring.push((i + 1) % 2, arrays(order[i + 1]))
        batch = batch.view(batch_size, seq_len)
        mbatch = mbatch.view(batch_size, seq_len)
        yield batch[:, :-1], batch[:, 1:], mbatch[:, 1:]


def data_feeder(data, batch_size, seq_len, device, shuffle=False, seed=42):
    """Yields (inputs, targets) windows of (batch_size, seq_len), shifted by one.

    shuffle=True serves the windows in a seeded-permuted order for the pass
    (reproducible given the same seed). On CUDA, H2D copies go through pinned
    staging buffers with a 1-batch lookahead so transfer overlaps compute."""
    total = len(data)
    num_sequences = total // seq_len
    order = _window_order(num_sequences, batch_size, shuffle, seed)
    if not order:
        return

    if device.type != 'cuda':
        for b in order:
            batch = torch.tensor(_slice(data, b, batch_size, seq_len, np.int64),
                                 device=device).view(batch_size, seq_len)
            yield batch[:, :-1], batch[:, 1:]
        return

    ring = _PinnedRing([(batch_size * seq_len, torch.int64)], device)
    pending = ring.push(0, [_slice(data, order[0], batch_size, seq_len, np.int64)])
    for i, b in enumerate(order):
        (batch,) = pending
        if i + 1 < len(order):
            pending = ring.push((i + 1) % 2,
                                [_slice(data, order[i + 1], batch_size, seq_len, np.int64)])
        batch = batch.view(batch_size, seq_len)
        yield batch[:, :-1], batch[:, 1:]
