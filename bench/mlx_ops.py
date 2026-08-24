"""MLX ports of the two attention kernels in core/, for benchmarking.

These mirror core/kda.py's kda_chunk and core/model.py's window_attention
closely enough to be checked against them numerically -- which compare_mlx.py
does before it times anything, so the comparison cannot quietly drift into
measuring two different computations.
"""
import mlx.core as mx

SUB_CHUNK = 8  # matches core.kda._SUB_CHUNK


def _decay_scores(x, y, g, diagonals):
    """[S, ..., BT, K] -> [S, ..., BT, BT] lower-triangular decayed scores."""
    BT = x.shape[-2]
    BC = min(SUB_CHUNK, BT)
    NB = BT // BC
    sub = lambda t: t.reshape(*t.shape[:-2], NB, BC, t.shape[-1])
    xs, ys, gs = sub(x), sub(y), sub(g)

    diag = mx.stack([(xs * mx.exp(mx.minimum(gs - gs[..., j:j + 1, :], 0.0))
                      * ys[..., j:j + 1, :]).sum(-1) for j in range(BC)], axis=-1)
    i = mx.arange(BC)
    tri = mx.stack([i[None, :] - i[:, None] >= d for d in diagonals])
    diag = mx.where(tri.reshape(len(diagonals), *(1,) * (diag.ndim - 3), BC, BC),
                    0.0, diag)

    rows = []
    for b in range(NB):
        lo, hi = b * BC, (b + 1) * BC
        parts = []
        if b:
            ref = g[..., lo:lo + 1, :]  # route both decays through the block start
            left = x[..., lo:hi, :] * mx.exp(g[..., lo:hi, :] - ref)
            right = y[..., :lo, :] * mx.exp(ref - g[..., :lo, :])
            parts.append(left @ mx.swapaxes(right, -1, -2))
        parts.append(diag[..., b, :, :])
        if b < NB - 1:
            parts.append(mx.zeros((*diag.shape[:-3], BC, BT - hi), dtype=x.dtype))
        rows.append(mx.concatenate(parts, axis=-1))
    return mx.concatenate(rows, axis=-2)


def _unit_lower_inverse(L):
    """(I - L)^-1 for strictly lower-triangular L, by repeated squaring."""
    BT = L.shape[-1]
    BC = min(SUB_CHUNK, BT)
    NB = BT // BC
    eye = mx.eye(BC, dtype=L.dtype)

    M = mx.stack([L[..., b * BC:(b + 1) * BC, b * BC:(b + 1) * BC]
                  for b in range(NB)], axis=-3)
    blocks = eye + M
    for _ in range((BC - 1).bit_length() - 1):
        M = M @ M
        blocks = blocks + blocks @ M

    X = blocks[..., 0, :, :]      # growing top-left square of the inverse
    for b in range(1, NB):
        lo, hi = b * BC, (b + 1) * BC
        D = blocks[..., b, :, :]
        off = D @ (L[..., lo:hi, :lo] @ X)
        X = mx.concatenate([
            mx.concatenate([X, mx.zeros((*X.shape[:-1], BC), dtype=L.dtype)], axis=-1),
            mx.concatenate([off, D], axis=-1),
        ], axis=-2)
    return X


def kda_chunk(q, k, v, g, beta, chunk_size=64):
    """q, k, g: [B,T,H,K]; v: [B,T,H,V]; beta: [B,T,H]. -> (o, final state)."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT, NT = chunk_size, T // chunk_size
    q = q * K ** -0.5

    blk = lambda x: x.reshape(B, NT, BT, H, x.shape[-1]).transpose(0, 3, 1, 2, 4)
    q, k, v, g = blk(q), blk(k), blk(v), blk(g)
    beta = beta.reshape(B, NT, BT, H).transpose(0, 3, 1, 2)
    g = mx.cumsum(g, axis=-2)

    A, Aqk = _decay_scores(mx.stack([k, q]), k, g, (0, 1))
    A = _unit_lower_inverse(-(A * beta[..., None])) * beta[..., None, :]

    ge = mx.exp(g)
    w, u = A @ (ge * k), A @ v
    qg = q * ge
    kg = mx.swapaxes(mx.exp(g[:, :, :, -1:] - g) * k, -1, -2)
    dec = mx.exp(g[:, :, :, -1])[..., None]

    S = mx.zeros((B, H, K, V), dtype=q.dtype)
    o = []
    for i in range(NT):
        v_i = u[:, :, i] - w[:, :, i] @ S
        o.append(qg[:, :, i] @ S + Aqk[:, :, i] @ v_i)
        S = dec[:, :, i] * S + kg[:, :, i] @ v_i
    o = mx.stack(o, axis=2).transpose(0, 2, 3, 1, 4).reshape(B, T, H, V)
    return o, S


def window_masks(window):
    """(head, tail) masks; see core.model.window_block_mask. MLX broadcasts the
    tail mask over the block batch, so it stays (1, 1, W, 2W)."""
    i, j = mx.arange(window)[:, None], mx.arange(2 * window)
    tail = ((j > i) & (j <= i + window))[None, None]
    return tail[..., window:], tail


def window_attention(q, k, v, window, masks, scale):
    """Every query attends the `window` keys ending at itself, at O(T*W)."""
    B, H, T, D = q.shape
    head_mask, tail_mask = masks
    if T % window:
        pad = lambda x: mx.concatenate(
            [x, mx.zeros((*x.shape[:2], -T % window, D), dtype=x.dtype)], axis=2)
        q, k, v = pad(q), pad(k), pad(v)
    nb = q.shape[2] // window

    def cut(x, paired):
        # W-wide blocks; each query block past the first pairs with the block
        # before it, which is the window it reaches back into.
        b = x.reshape(*x.shape[:2], nb, window, D)
        b = mx.concatenate([b[:, :, :-1], b[:, :, 1:]], axis=-2) if paired \
            else b[:, :, 1:]
        return b.swapaxes(1, 2).reshape(B * (nb - 1), -1, b.shape[-2], D)

    sdpa = mx.fast.scaled_dot_product_attention
    first = sdpa(q[:, :, :window], k[:, :, :window], v[:, :, :window],
                 scale=scale, mask=head_mask)
    rest = sdpa(cut(q, False), cut(k, True), cut(v, True),
                scale=scale, mask=tail_mask)
    rest = rest.reshape(B, nb - 1, H, window, D).swapaxes(1, 2).reshape(B, H, -1, D)
    return mx.concatenate([first, rest], axis=2)[:, :, :T]
