import torch


# Polar Express per-step quintic coefficients (arXiv:2505.16932) — one tuple
# per iteration; better orthogonalization than the classic single-coefficient
# Newton-Schulz at GPT-2 scale.
_POLAR_EXPRESS = [
    (8.1566, -22.4833, 15.8788),
    (4.0429, -2.8089, 0.5000),
    (3.8917, -2.7725, 0.5061),
    (3.2858, -2.3681, 0.4645),
    (2.3465, -1.7098, 0.4232),
]


@torch.no_grad()
def _polar_express(G, steps=5, eps=1e-7):
    # Approximates the orthogonal factor U @ V.T of the SVD G = U S V.T.
    # Runs in bf16 (inputs pre-normalized with a 1.02 safety factor so the
    # iteration stays in its convergence basin); operates on the smaller dim
    # by transposing tall matrices.
    assert G.ndim == 2
    X = G.to(torch.bfloat16)
    X = X / (X.norm() * 1.02 + eps)
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
    for i in range(steps):
        a, b, c = _POLAR_EXPRESS[i % len(_POLAR_EXPRESS)]
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz (Polar Express variant).

    Only supports 2D matrix parameters. Use AdamW for 1D parameters
    (biases, norm scales) and for embeddings / LM heads — see Keller
    Jordan's paper for the rationale.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']
            wd = group['weight_decay']
            for p in group['params']:
                if p.grad is None:
                    continue
                if p.ndim != 2:
                    raise RuntimeError(
                        f"Muon only supports 2D parameters; got shape {tuple(p.shape)}"
                    )
                g = p.grad

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if nesterov else buf

                update = _polar_express(update, steps=ns_steps)
                if wd:
                    # Decoupled weight decay, applied before the orthogonal update.
                    p.mul_(1 - lr * wd)
                # Equalize update magnitude across different matrix shapes.
                scale = max(1.0, update.shape[0] / update.shape[1]) ** 0.5
                p.add_(update, alpha=-lr * scale)
        return loss
