"""KDA (arXiv:2510.26692) on MLX — the inference half of core/kda.py.

Same math, same parameter names; document-boundary segment masking is left out
because it only ever runs in a training window, which this backend does not
serve.
"""
import math

import mlx.core as mx
import mlx.nn as nn


def _l2_normalize(x):
    return x / mx.maximum(mx.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def kda_recurrence(q, k, v, g, beta, initial_state=None):
    """Sequential KDA scan in fp32 — single-token decode and the ragged tail of
    a cached prefill.

    q, k: [B, T, H, K]; v: [B, T, H, V]; g: [B, T, H, K] log-decay (<= 0);
    beta: [B, T, H]; initial_state: [B, H, K, V] or None.
    Returns (o [B, T, H, V], S [B, H, K, V])."""
    B, T, H, K = q.shape
    q, k, v, g, beta = (x.astype(mx.float32) for x in (q, k, v, g, beta))
    q = q * K ** -0.5
    S = mx.zeros((B, H, K, v.shape[-1]))
    if initial_state is not None:
        S = S + initial_state.astype(mx.float32)
    o = []
    for t in range(T):
        qt, kt, vt, gt, bt = q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t]
        S = S * mx.exp(gt)[..., None]
        err = vt - (kt[..., None] * S).sum(-2)
        S = S + (bt[..., None] * kt)[..., None] * err[..., None, :]
        o.append((qt[..., None] * S).sum(-2))
    return mx.stack(o, axis=1), S


_SUB_CHUNK = 16  # sub-block size inside a chunk for the decayed-score matrices


def _decay_scores(x, y, g, diagonal):
    """Lower-triangular decayed inner products M[c, j] = sum_d x[c,d] y[j,d]
    exp(g[c,d] - g[j,d]), zeroed on and above the `diagonal`-th diagonal.

    Sub-blocks below the diagonal factor the decay through their row block's
    first position, so both exponents stay <= 0 and the block is a single
    matmul. Only the diagonal sub-blocks keep the per-column form, where a
    positive exponent can appear but always lands in a masked-out entry."""
    lead = x.shape[:-2]
    BT, K = x.shape[-2], x.shape[-1]
    BC = min(_SUB_CHUNK, BT)
    assert BT % BC == 0, f"chunk_size={BT} must be a multiple of {BC}"
    NB = BT // BC
    xs, ys, gs = (t.reshape(*lead, NB, BC, K) for t in (x, y, g))

    # clamp: entries with a positive exponent are all above the diagonal and get
    # masked out below — clamping keeps them from overflowing to inf.
    cols = [(xs * mx.exp(mx.minimum(gs - gs[..., j:j + 1, :], 0.0))
             * ys[..., j:j + 1, :]).sum(-1) for j in range(BC)]
    r = mx.arange(BC)
    keep = (r[None, :] - r[:, None]) < diagonal
    diag = mx.where(keep, mx.stack(cols, axis=-1), 0.0)

    rows = []
    for b in range(NB):
        lo, hi = b * BC, (b + 1) * BC
        row = diag[..., b, :, :]
        if b:
            ref = g[..., lo:lo + 1, :]  # route both decays through the block start
            left = x[..., lo:hi, :] * mx.exp(g[..., lo:hi, :] - ref)
            right = y[..., :lo, :] * mx.exp(ref - g[..., :lo, :])
            row = mx.concatenate([left @ right.swapaxes(-1, -2), row], axis=-1)
        if hi < BT:
            row = mx.concatenate([row, mx.zeros((*lead, BC, BT - hi))], axis=-1)
        rows.append(row)
    return mx.concatenate(rows, axis=-2)


def _unit_lower_inverse(L):
    """(I - L)^{-1} for a strictly lower-triangular L: [..., BT, BT].

    Blocked forward substitution. The BC-wide diagonal blocks are inverted by
    the row recursion, but all of them at once, so it costs BC steps instead of
    BT; each remaining block row is then one matmul against the block rows
    already solved."""
    BT = L.shape[-1]
    BC = min(_SUB_CHUNK, BT)
    NB = BT // BC

    # Diagonal blocks -> [..., NB, BC, BC], inverted together by the recursion
    # X[i, :i] += X[i, :] @ X[:, :i] (the strictly-lower Neumann series).
    blocks = mx.stack([L[..., b * BC:(b + 1) * BC, b * BC:(b + 1) * BC]
                       for b in range(NB)], axis=-3)
    for i in range(1, BC):
        blocks[..., i, :i] = (blocks[..., i, :i]
                              + (blocks[..., i, :, None] * blocks[..., :, :i]).sum(-2))
    blocks = blocks + mx.eye(BC)

    X = blocks[..., 0, :, :]      # growing top-left square of the inverse
    for b in range(1, NB):
        lo, hi = b * BC, (b + 1) * BC
        D = blocks[..., b, :, :]
        off = D @ (L[..., lo:hi, :lo] @ X)
        X = mx.concatenate([
            mx.concatenate([X, mx.zeros((*X.shape[:-1], BC))], axis=-1),
            mx.concatenate([off, D], axis=-1),
        ], axis=-2)
    return X


def kda_chunk(q, k, v, g, beta, initial_state=None, chunk_size=64):
    """Chunked-parallel KDA in fp32: all positions of a chunk are processed with
    batched matmuls, only the inter-chunk scan is sequential. Requires
    T % chunk_size == 0; same signature as kda_recurrence."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    BT = chunk_size
    assert T % BT == 0, f"T={T} must be a multiple of chunk_size={BT}"
    NT = T // BT
    q, k, v, g, beta = (x.astype(mx.float32) for x in (q, k, v, g, beta))
    q = q * K ** -0.5

    # -> [B, H, NT, BT, ...]
    q, k, v, g = (x.reshape(B, NT, BT, H, x.shape[-1]).transpose(0, 3, 1, 2, 4)
                  for x in (q, k, v, g))
    beta = beta.reshape(B, NT, BT, H).transpose(0, 3, 1, 2)  # [B, H, NT, BT]
    g = mx.cumsum(g, axis=-2)  # within-chunk cumulative log-decay

    # UT transform for the delta rule.
    A = _decay_scores(k, k, g, diagonal=0) * beta[..., None]
    A = _unit_lower_inverse(-A) * beta[..., None, :]

    w = A @ (mx.exp(g) * k)       # decayed keys   [B,H,NT,BT,K]
    u = A @ v                     # pseudo-values  [B,H,NT,BT,V]

    S = mx.zeros((B, H, K, V))
    if initial_state is not None:
        S = S + initial_state.astype(mx.float32)
    # Intra-chunk query->key weights, decayed. Independent of the running state,
    # so every chunk's block is built up front instead of inside the scan.
    Aqk_all = _decay_scores(q, k, g, diagonal=1)
    o = []
    for i in range(NT):
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        v_i = u_i - w_i @ S       # subtract what the incoming state already knows
        o.append((q_i * mx.exp(g_i)) @ S + Aqk_all[:, :, i] @ v_i)
        S = S * mx.exp(g_i[:, :, -1])[..., None]
        S = S + ((mx.exp(g_i[:, :, -1:] - g_i) * k_i).swapaxes(-1, -2) @ v_i)
    o = mx.stack(o, axis=2)       # [B,H,NT,BT,V]
    return o.transpose(0, 2, 3, 1, 4).reshape(B, T, H, V), S


def kda_scan(q, k, v, g, beta, initial_state=None, chunk_size=64):
    """kda_chunk over the leading whole chunks, kda_recurrence for the ragged
    tail — the sequential part is the tail rather than every position."""
    T = q.shape[1]
    full = T - T % chunk_size
    if not full:
        return kda_recurrence(q, k, v, g, beta, initial_state=initial_state)
    head = (x[:, :full] for x in (q, k, v, g, beta))
    o, S = kda_chunk(*head, initial_state=initial_state, chunk_size=chunk_size)
    if full < T:
        tail = (x[:, full:] for x in (q, k, v, g, beta))
        o_tail, S = kda_recurrence(*tail, initial_state=S)
        o = mx.concatenate((o, o_tail), axis=1)
    return o, S


class KimiDeltaAttention(nn.Module):
    """KDA layer, parameter-for-parameter with core.kda.KimiDeltaAttention: the
    state matrix replaces the KV cache, and order rides the recurrence rather
    than a positional encoding."""

    def __init__(self, d_model, heads, chunk_size=64):
        super().__init__()
        self.d_model = d_model
        self.h = heads
        self.d_k = d_model // heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # Per-channel forget gate, low-rank: d_model -> d_k -> heads * d_k.
        self.f_proj = [nn.Linear(d_model, self.d_k, bias=False),
                       nn.Linear(self.d_k, d_model, bias=False)]
        self.b_proj = nn.Linear(d_model, heads, bias=False)
        self.A_log = mx.zeros((heads,))
        self.dt_bias = mx.zeros((d_model,))
        self.g_proj = [nn.Linear(d_model, self.d_k, bias=False),
                       nn.Linear(self.d_k, d_model, bias=True)]
        self.o_norm = nn.RMSNorm(self.d_k, eps=1e-5)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self._s_cache = {}  # recurrence states keyed by recurrence-pass index

    def reset_cache(self):
        self._s_cache = {}

    def __call__(self, x, start_pos=None, cache_idx=0):
        B, T, _ = x.shape
        H, D = self.h, self.d_k
        q = _l2_normalize(nn.silu(self.q_proj(x)).reshape(B, T, H, D))
        k = _l2_normalize(nn.silu(self.k_proj(x)).reshape(B, T, H, D))
        v = nn.silu(self.v_proj(x)).reshape(B, T, H, D)
        f = self.f_proj[1](self.f_proj[0](x)).reshape(B, T, H, D)
        # Log-space per-channel decay: always negative, so exp(g) lies in (0, 1).
        g = -(mx.exp(self.A_log)[:, None]
              * nn.softplus(f + self.dt_bias.reshape(1, 1, H, D)))
        beta = mx.sigmoid(self.b_proj(x))  # [B, T, H]

        if start_pos is None:
            fn = kda_chunk if T % self.chunk_size == 0 else kda_recurrence
            o, _ = fn(q, k, v, g, beta)
        else:
            o, S = kda_scan(q, k, v, g, beta, initial_state=self._s_cache.get(cache_idx),
                            chunk_size=self.chunk_size)
            self._s_cache[cache_idx] = S

        o = o.astype(x.dtype)
        gate = self.g_proj[1](self.g_proj[0](x)).reshape(B, T, H, D)
        o = self.o_norm(o) * mx.sigmoid(gate)
        return self.o_proj(o.reshape(B, T, H * D))
