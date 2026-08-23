import torch
import torch.nn.functional as F


class _ChunkedCrossEntropy(torch.autograd.Function):
    """Soft-capped cross-entropy computed in row chunks, recomputed in backward.

    Never materializes the full (N, V) logits: forward and backward each loop
    over `chunk` rows at a time, so transient memory is one (chunk, V) block
    instead of the whole logits + tanh + fp32 softmax stack. Only hidden,
    weight, bias and targets are saved for backward.

    Backward runs outside the caller's autocast region, so the dtype autocast
    would have used is recorded in forward and applied by hand. Otherwise the
    recomputed logits come out in fp32 -- a different operating point from the
    loss the forward measured, and three (N, V) x (V, d) products at fp32
    matmul rates instead of bf16.
    """

    @staticmethod
    def forward(ctx, hidden, weight, bias, targets, softcap, chunk):
        n = hidden.size(0)
        with torch.no_grad():
            loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                logits = F.linear(hidden[lo:hi], weight, bias).float()
                z = softcap * torch.tanh(logits / softcap)
                picked = z.gather(1, targets[lo:hi].unsqueeze(1)).squeeze(1)
                loss_sum += (torch.logsumexp(z, dim=-1) - picked).sum()
        ctx.save_for_backward(hidden, weight, bias, targets)
        ctx.softcap = softcap
        ctx.chunk = chunk
        dev = hidden.device.type
        ctx.compute_dtype = (torch.get_autocast_dtype(dev)
                             if torch.is_autocast_enabled(dev) else hidden.dtype)
        return loss_sum / n

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, bias, targets = ctx.saved_tensors
        softcap, chunk = ctx.softcap, ctx.chunk
        dtype = ctx.compute_dtype
        n = hidden.size(0)
        w = weight.to(dtype)
        b = None if bias is None else bias.to(dtype)
        dhidden = torch.empty_like(hidden)
        dweight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        dbias = None if bias is None else torch.zeros(
            bias.shape, dtype=torch.float32, device=bias.device)
        scale = grad_out.float() / n
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            h = hidden[lo:hi].to(dtype)
            logits = F.linear(h, w, b).float()   # the head the forward actually ran
            t = torch.tanh(logits / softcap)
            p = torch.softmax(softcap * t, dim=-1)
            p[torch.arange(hi - lo, device=p.device), targets[lo:hi]] -= 1
            # d loss/d logits through z = softcap * tanh(logits / softcap)
            dz = (p * (1 - t * t) * scale).to(dtype)
            dhidden[lo:hi] = (dz @ w).to(hidden.dtype)
            # chunk products are low precision, but they accumulate in fp32
            dweight += (dz.T @ h).float()
            if dbias is not None:
                dbias += dz.float().sum(0)
        return (dhidden, dweight.to(weight.dtype),
                None if dbias is None else dbias.to(bias.dtype), None, None, None)


def _chunked_cross_entropy(hidden, weight, bias, targets, softcap, chunk):
    return _ChunkedCrossEntropy.apply(hidden, weight, bias, targets,
                                      float(softcap), int(chunk))


def _reference_cross_entropy(hidden, weight, bias, targets, softcap):
    z = softcap * torch.tanh(F.linear(hidden, weight, bias) / softcap)
    return F.cross_entropy(z.float(), targets)


def chunked_cross_entropy(hidden, weight, bias, targets, softcap, chunk):
    """Mean soft-capped cross-entropy over N rows of the LM head applied to hidden.

    Uses the chunked/recompute path only on CUDA with chunk > 0; anything else
    (CPU, MPS, chunk <= 0) falls back to the plain unfused computation.
    """
    if chunk > 0 and hidden.is_cuda:
        return _chunked_cross_entropy(hidden, weight, bias, targets, softcap, chunk)
    return _reference_cross_entropy(hidden, weight, bias, targets, softcap)
