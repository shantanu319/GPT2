"""Single-node multi-GPU helpers. torchrun sets the environment; this reads it.

There is no DDP wrapper on purpose: train.py's fused-CE path calls
model.decoder directly and computes the loss against the tied output head
outside the module, which DDP's reducer never sees. Averaging gradients by
hand also skips the all-reduce on grad-accum micro-steps for free.
"""
import os

import torch
import torch.distributed as dist


def world_size():
    return int(os.environ.get("WORLD_SIZE", 1))


def rank():
    return int(os.environ.get("RANK", 0))


def local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))


def is_distributed():
    return world_size() > 1


def is_main():
    return rank() == 0


def printr(*args, **kwargs):
    """Print on rank 0 only."""
    if is_main():
        print(*args, **kwargs)


def init(device):
    """Join the process group. Call once, after the device is resolved."""
    if not is_distributed() or dist.is_initialized():
        return
    dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")


def shutdown():
    if dist.is_initialized():
        dist.destroy_process_group()


def barrier():
    if dist.is_initialized():
        dist.barrier()


def average_grads(params):
    """Average gradients across ranks in place, as one flat all-reduce.

    One call beats per-parameter reduces: the model's ~200 gradient tensors
    would otherwise cost more in NCCL launch latency than in transfer, and
    the scatter back is one multi-tensor kernel rather than ~200 copies.
    """
    if not is_distributed():
        return
    grads = [p.grad for p in params if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    flat /= world_size()
    views, offset = [], 0
    for g in grads:
        views.append(flat[offset:offset + g.numel()].view_as(g))
        offset += g.numel()
    torch._foreach_copy_(grads, views)


def sum_across(values, device):
    """Sum a list of floats across ranks. Loss totals and token counts reduce
    together, so a mean stays weighted by each rank's actual token count."""
    if not is_distributed():
        return list(values)
    t = torch.tensor(list(values), dtype=torch.float64, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.tolist()
