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
    # by transposing tall matrices. The last two dims are the matrix; any
    # leading dims batch it — one matrix per attention head, per parameter,
    # or both.
    assert G.ndim >= 2
    X = G.to(torch.bfloat16)
    X = X / (X.norm(dim=(-1, -2), keepdim=True) * 1.02 + eps)
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

    Parameters that share a shape are orthogonalized as one batch: a
    transformer has only a handful of distinct matrix shapes, so the whole
    stack's Newton-Schulz iteration collapses into that many batched matmuls
    instead of one chain per parameter.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 weight_decay=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _orthogonalize(updates, head_split, ns_steps):
        """updates: (n, rows, cols), n same-shaped parameter updates."""
        n, rows, cols = updates.shape
        if head_split is None:
            return _polar_express(updates, steps=ns_steps), max(1.0, rows / cols) ** 0.5
        heads, axis = head_split
        if axis == 0:
            if rows % heads:
                raise RuntimeError(
                    f"muon_head_split=({heads}, 0) does not divide {(rows, cols)}")
            ortho = _polar_express(updates.view(n, heads, rows // heads, cols),
                                   steps=ns_steps).view(updates.shape)
            scale = max(1.0, (rows // heads) / cols) ** 0.5
        else:
            if cols % heads:
                raise RuntimeError(
                    f"muon_head_split=({heads}, 1) does not divide {(rows, cols)}")
            ortho = _polar_express(updates.view(n, rows, heads, cols // heads)
                                          .permute(0, 2, 1, 3),
                                   steps=ns_steps).permute(0, 2, 1, 3).reshape(updates.shape)
            scale = max(1.0, rows / (cols // heads)) ** 0.5
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
            buckets = {}  # (shape, head_split) -> params and their updates
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

                key = (tuple(p.shape), getattr(p, 'muon_head_split', None))
                buckets.setdefault(key, ([], []))
                buckets[key][0].append(p)
                # bf16 up front: the iteration runs there anyway, so stacking in
                # bf16 halves the copy without changing a single output bit.
                buckets[key][1].append(update.to(torch.bfloat16))

            for (_, head_split), (params, updates) in buckets.items():
                ortho, scale = self._orthogonalize(
                    torch.stack(updates), head_split, ns_steps)
                for p, o in zip(params, ortho.unbind(0)):
                    if wd:
                        # Decoupled weight decay, before the orthogonal update.
                        p.mul_(1 - lr * wd)
                    # Equalize update magnitude across different matrix shapes.
                    p.add_(o, alpha=-lr * scale)
        return loss
