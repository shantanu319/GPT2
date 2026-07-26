import torch
import torch.nn.functional as F


class _ChunkedCrossEntropy(torch.autograd.Function):
    """Soft-capped cross-entropy computed in row chunks, recomputed in backward.

    Never materializes the full (N, V) logits: forward and backward each loop
    over `chunk` rows at a time, so transient memory is one (chunk, V) block
    instead of the whole logits + tanh + fp32 softmax stack. Only hidden,
    weight and targets are saved for backward.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, softcap, chunk):
        n = hidden.size(0)
        with torch.no_grad():
            loss_sum = torch.zeros((), dtype=torch.float32, device=hidden.device)
            for lo in range(0, n, chunk):
                hi = min(lo + chunk, n)
                logits = (hidden[lo:hi] @ weight.T).float()
                z = softcap * torch.tanh(logits / softcap)
                picked = z.gather(1, targets[lo:hi].unsqueeze(1)).squeeze(1)
                loss_sum += (torch.logsumexp(z, dim=-1) - picked).sum()
        ctx.save_for_backward(hidden, weight, targets)
        ctx.softcap = softcap
        ctx.chunk = chunk
        return loss_sum / n

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, targets = ctx.saved_tensors
        softcap, chunk = ctx.softcap, ctx.chunk
        n = hidden.size(0)
        w_f = weight.float()
        dhidden = torch.empty_like(hidden)
        dweight = torch.zeros(weight.shape, dtype=torch.float32, device=weight.device)
        for lo in range(0, n, chunk):
            hi = min(lo + chunk, n)
            h_f = hidden[lo:hi].float()
            logits = h_f @ w_f.T
            t = torch.tanh(logits / softcap)
            z = softcap * t
            p = torch.softmax(z, dim=-1)
            p[torch.arange(hi - lo, device=p.device), targets[lo:hi]] -= 1
            # d loss/d logits through z = softcap * tanh(logits / softcap)
            dz = p * (1 - t * t) * (grad_out.float() / n)
            dhidden[lo:hi] = (dz @ w_f).to(hidden.dtype)
            dweight += dz.T @ h_f
        return dhidden, dweight.to(weight.dtype), None, None, None


def _chunked_cross_entropy(hidden, weight, targets, softcap, chunk):
    return _ChunkedCrossEntropy.apply(hidden, weight, targets, float(softcap), int(chunk))


def _reference_cross_entropy(hidden, weight, targets, softcap):
    logits = hidden @ weight.T
    z = softcap * torch.tanh(logits / softcap)
    return F.cross_entropy(z.float(), targets)


def chunked_cross_entropy(hidden, weight, targets, softcap, chunk):
    """Mean soft-capped cross-entropy over N rows of hidden @ weight.T.

    Uses the chunked/recompute path only on CUDA with chunk > 0; anything else
    (CPU, MPS, chunk <= 0) falls back to the plain unfused computation.
    """
    if chunk > 0 and hidden.is_cuda:
        return _chunked_cross_entropy(hidden, weight, targets, softcap, chunk)
    return _reference_cross_entropy(hidden, weight, targets, softcap)
