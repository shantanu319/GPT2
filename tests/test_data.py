import numpy as np
import torch

from core.data import BIN_DTYPE, data_feeder, load_bin


def test_data_feeder_batch_shapes():
    data = list(range(100))
    batches = list(data_feeder(data, batch_size=2, seq_len=8, device=torch.device("cpu")))
    assert len(batches) > 0
    for x, y in batches:
        assert x.shape == (2, 7)
        assert y.shape == (2, 7)


def test_data_feeder_target_is_input_shifted_by_one():
    data = list(range(100))
    for x, y in data_feeder(data, batch_size=2, seq_len=8, device=torch.device("cpu")):
        # y at position t should equal x at position t+1 within the same sequence
        assert torch.equal(x[:, 1:], y[:, :-1])


def test_data_feeder_drops_incomplete_last_batch():
    # 13 sequences of length 8 with batch_size=4 → 3 full batches, last partial dropped
    data = list(range(13 * 8))
    batches = list(data_feeder(data, batch_size=4, seq_len=8, device=torch.device("cpu")))
    assert len(batches) == 3
    for x, _ in batches:
        assert x.size(0) == 4


def test_load_bin_roundtrip(tmp_path):
    # Write uint16 tokens to disk; load_bin should return them mmapped.
    path = tmp_path / "toy.bin"
    arr = np.arange(100, dtype=BIN_DTYPE)
    arr.tofile(path)

    loaded = load_bin(str(path))
    assert loaded.dtype == BIN_DTYPE
    assert loaded.shape == (100,)
    assert np.array_equal(loaded, arr)


def test_data_feeder_accepts_mmap(tmp_path):
    path = tmp_path / "toy.bin"
    arr = np.arange(100, dtype=BIN_DTYPE)
    arr.tofile(path)
    mmap_data = load_bin(str(path))

    batches = list(data_feeder(mmap_data, batch_size=2, seq_len=8, device=torch.device("cpu")))
    assert len(batches) > 0
    for x, y in batches:
        assert x.shape == (2, 7)
        assert y.shape == (2, 7)
        assert torch.equal(x[:, 1:], y[:, :-1])


def _window_keys(batches):
    # First input token of every served row, order-insensitive.
    return sorted(t for x, _ in batches for t in x[:, 0].tolist())


def test_data_feeder_shuffle_same_seed_same_order():
    data = list(range(24 * 8))
    a = list(data_feeder(data, 2, 8, torch.device("cpu"), shuffle=True, seed=7))
    b = list(data_feeder(data, 2, 8, torch.device("cpu"), shuffle=True, seed=7))
    assert len(a) == len(b) == 12
    assert all(torch.equal(xa, xb) and torch.equal(ya, yb) for (xa, ya), (xb, yb) in zip(a, b))


def test_data_feeder_shuffle_different_seed_different_order():
    data = list(range(24 * 8))
    a = list(data_feeder(data, 2, 8, torch.device("cpu"), shuffle=True, seed=1))
    b = list(data_feeder(data, 2, 8, torch.device("cpu"), shuffle=True, seed=2))
    assert not all(torch.equal(xa, xb) for (xa, _), (xb, _) in zip(a, b))


def test_data_feeder_shuffle_covers_all_windows():
    data = list(range(24 * 8))
    shuffled = list(data_feeder(data, 2, 8, torch.device("cpu"), shuffle=True, seed=3))
    sequential = list(data_feeder(data, 2, 8, torch.device("cpu")))
    assert len(shuffled) == len(sequential)
    assert _window_keys(shuffled) == _window_keys(sequential)
    # and the order actually changed
    assert not all(torch.equal(xa, xb) for (xa, _), (xb, _) in zip(shuffled, sequential))


def test_data_feeder_segment_ids():
    """eos_id makes the feeder also yield segment ids aligned with the inputs:
    EOS is the last token of its segment, the next token starts a new one."""
    eos = 99
    row = [5, 6, eos, 7, 8, eos, 9, 10]  # window segs: 0 0 0 1 1 1 2 2
    data = row * 4  # 4 identical windows
    batches = list(data_feeder(data, 2, 8, torch.device("cpu"), eos_id=eos))
    assert len(batches) == 2
    for x, y, seg in batches:
        assert x.shape == (2, 7) and seg.shape == (2, 7)
        assert torch.equal(x[:, 1:], y[:, :-1])
        assert seg.tolist() == [[0, 0, 0, 1, 1, 1, 2]] * 2  # inputs = window[:-1]


def test_data_feeder_segment_ids_no_eos_in_window():
    data = list(range(1, 33))  # no 99 anywhere
    for x, y, seg in data_feeder(data, 2, 8, torch.device("cpu"), eos_id=99):
        assert torch.all(seg == 0)


def test_data_feeder_no_eos_id_keeps_pair():
    data = list(range(100))
    for batch in data_feeder(data, 2, 8, torch.device("cpu")):
        assert len(batch) == 2


def test_data_feeder_masked_segment_ids():
    from core.data import data_feeder_masked
    eos = 99
    row = [5, eos, 7, 8, 9, eos, 11, 12]  # window segs: 0 0 1 1 1 1 2 2
    data = row * 4
    mask = [1] * len(data)
    batches = list(data_feeder_masked(data, mask, 2, 8, torch.device("cpu"), eos_id=eos))
    assert len(batches) == 2
    for x, y, m, seg in batches:
        assert x.shape == m.shape == seg.shape == (2, 7)
        assert seg.tolist() == [[0, 0, 1, 1, 1, 1, 2]] * 2


def test_data_feeder_masked_no_eos_id_keeps_triple():
    from core.data import data_feeder_masked
    data = list(range(100))
    mask = [1] * 100
    for batch in data_feeder_masked(data, mask, 2, 8, torch.device("cpu")):
        assert len(batch) == 3
