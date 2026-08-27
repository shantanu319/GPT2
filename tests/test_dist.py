import os

import pytest
import torch
import torch.multiprocessing as mp

from core.data import _window_order
from core import dist


def test_window_order_shards_are_disjoint_and_equal_length():
    shards = [_window_order(100, 1, False, 0, rank=r, world=4) for r in range(4)]
    assert len({len(s) for s in shards}) == 1
    flat = [b for s in shards for b in s]
    assert len(flat) == len(set(flat)) == 100


def test_window_order_drops_the_ragged_tail():
    # 10 batches over 4 ranks: the last 2 are dropped so no rank runs short.
    shards = [_window_order(10, 1, False, 0, rank=r, world=4) for r in range(4)]
    assert [len(s) for s in shards] == [2, 2, 2, 2]
    assert max(b for s in shards for b in s) < 8


def test_window_order_single_rank_is_unchanged():
    assert _window_order(10, 1, False, 0) == list(range(10))


def _run(rank, world, port, tmpdir, fn):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port),
                      RANK=str(rank), WORLD_SIZE=str(world), LOCAL_RANK=str(rank))
    device = torch.device("cpu")
    dist.init(device)
    fn(rank, world, device, tmpdir)
    dist.shutdown()


def _grad_mean(rank, world, device, tmpdir):
    p = torch.nn.Parameter(torch.zeros(4))
    p.grad = torch.full((4,), float(rank + 1))
    dist.average_grads([p])
    expected = sum(range(1, world + 1)) / world
    torch.testing.assert_close(p.grad, torch.full((4,), expected))


def _stays_in_lockstep(rank, world, device, tmpdir):
    """Different data per rank must still leave every rank's weights identical."""
    from pretrain.muon import Muon
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False),
                                torch.nn.Linear(8, 8, bias=False))
    opt = Muon(model.parameters(), lr=0.02)
    for step in range(3):
        x = torch.randn(4, 8, generator=torch.Generator().manual_seed(100 * rank + step))
        model(x).square().mean().backward()
        dist.average_grads(model.parameters())
        opt.step()
        opt.zero_grad()
    torch.save(model.state_dict(), os.path.join(tmpdir, f"rank{rank}.pt"))


@pytest.mark.parametrize("fn,name", [(_grad_mean, "mean"), (_stays_in_lockstep, "lockstep")])
def test_two_rank_gloo(fn, name, tmp_path):
    port = 29500 + abs(hash(name)) % 1000
    mp.spawn(_run, args=(2, port, str(tmp_path), fn), nprocs=2, join=True)
    if name == "lockstep":
        a = torch.load(tmp_path / "rank0.pt")
        b = torch.load(tmp_path / "rank1.pt")
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=0, atol=0)
