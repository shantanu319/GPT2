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
    # by transposing tall matrices. Accepts 2D (single matrix) or 3D input
    # (batched — e.g. one (d_head, fan_in) matrix per attention head).
    assert G.ndim in (2, 3)
    X = G.to(torch.bfloat16)
    if G.ndim == 3:
        X = X / (X.norm(dim=(-1, -2), keepdim=True) * 1.02 + eps)
    else:
        X = X / (X.norm() * 1.02 + eps)
    transposed = False
    if X.size(-2) > X.size(-1):
        X = X.mT
        transposed = True
    for i in range(steps):
        a, b, c = _POLAR_EXPRESS[i % len(_POLAR_EXPRESS)]
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    """MomentUm Orthogonalized by Newton-schulz (Polar Express variant).

    Only supports 2D matrix parameters. Use AdamW for 1D parameters
    (biases, norm scales) and for embeddings / LM heads — see Keller
    Jordan's paper for the rationale.

    Per-head mode (Kimi K3's "Per-Head Muon"): tag a parameter with
    `p.muon_head_split = (n_heads, axis)` and the orthogonalization runs
    independently on each attention head's slice instead of the fused
    matrix. axis=0 slices output rows (q/k/v projections: (h*d_k, fan_in)
    -> h matrices of (d_k, fan_in)); axis=1 slices input columns (the attn
    out projection: (fan_out, h*d_k) -> h matrices of (fan_out, d_k)).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize(update, head_split, ns_steps):
        if head_split is None:
            ortho = _polar_express(update, steps=ns_steps)
            scale = max(1.0, update.shape[0] / update.shape[1]) ** 0.5
            return ortho, scale
        heads, axis = head_split
        if axis == 0:
            if update.size(0) % heads:
                raise RuntimeError(
                    f"muon_head_split=({heads}, 0) does not divide {tuple(update.shape)}")
            ortho = _polar_express(
                update.view(heads, update.size(0) // heads, update.size(1)),
                steps=ns_steps).view(update.shape)
            scale = max(1.0, (update.size(0) // heads) / update.size(1)) ** 0.5
        else:
            if update.size(1) % heads:
                raise RuntimeError(
                    f"muon_head_split=({heads}, 1) does not divide {tuple(update.shape)}")
            ortho = _polar_express(
                update.view(update.size(0), heads, update.size(1) // heads)
                      .permute(1, 0, 2),
                steps=ns_steps).permute(1, 0, 2).reshape(update.shape)
            scale = max(1.0, update.size(0) / (update.size(1) // heads)) ** 0.5
        return ortho, scale

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

                update, scale = self._orthogonalize(
                    update, getattr(p, 'muon_head_split', None), ns_steps)
                if wd:
                    # Decoupled weight decay, applied before the orthogonal update.
                    p.mul_(1 - lr * wd)
                # Equalize update magnitude across different matrix shapes.
                p.add_(update, alpha=-lr * scale)
        return loss
