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


def data_feeder_masked(data, mask, batch_size, seq_len, device):
    """Like data_feeder, but also yields a loss mask aligned with the targets.

    data: uint16 token memmap; mask: uint8 memmap of the same length
    (1 = compute loss on this token when it appears as a target)."""
    total = min(len(data), len(mask))
    num_sequences = total // seq_len

    for start_seq in range(0, num_sequences, batch_size):
        end_seq = start_seq + batch_size
        if end_seq > num_sequences:
            break
        lo, hi = start_seq * seq_len, end_seq * seq_len
        batch = torch.tensor(np.asarray(data[lo:hi]), dtype=torch.long, device=device)
        batch = batch.view(batch_size, seq_len)
        mbatch = torch.tensor(np.asarray(mask[lo:hi]), dtype=torch.bool, device=device)
        mbatch = mbatch.view(batch_size, seq_len)
        yield batch[:, :-1], batch[:, 1:], mbatch[:, 1:]


def data_feeder(data, batch_size, seq_len, device):
    total = len(data)
    num_sequences = total // seq_len

    for start_seq in range(0, num_sequences, batch_size):
        end_seq = start_seq + batch_size
        if end_seq > num_sequences:
            break
        slice_data = data[start_seq * seq_len: end_seq * seq_len]
        batch = torch.tensor(np.asarray(slice_data), dtype=torch.long, device=device)
        batch = batch.view(batch_size, seq_len)
        yield batch[:, :-1], batch[:, 1:]
