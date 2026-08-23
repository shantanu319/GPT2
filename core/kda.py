
"""
KDA (arXiv:2510.26692) extends Gated DeltaNet's scalar per-head
forget gate with a per-key-dim gate, so each channel of the recurrent state
decays at its own learned rate. The per-step transition is Diagonal-Plus-Low-
Rank
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def kda_recurrence(q, k, v, g, beta, initial_state=None, seg_ids=None):
    """Sequential KDA scan in fp32 — the reference path, also used for cached
    inference (prefill chunks and single-token decode).

    q, k: [B, T, H, K]; v: [B, T, H, V]; g: [B, T, H, K] log-decay (<= 0);
    beta: [B, T, H]; initial_state: [B, H, K, V] or None.
    seg_ids: optional [B, T] document segment ids — the carried state is
    dropped at every segment start (document-boundary reset).
    Returns (o [B, T, H, V], S [B, H, K, V])."""
    B, T, H, K = q.shape
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q * K ** -0.5
    S = q.new_zeros(B, H, K, v.size(-1))
    if initial_state is not None:
        S = S + initial_state.float()
    o = torch.zeros_like(v)
    for t in range(T):
        qt, kt, vt, gt, bt = q[:, t], k[:, t], v[:, t], g[:, t], beta[:, t]
        decay = gt.exp()
        if seg_ids is not None and t > 0:
            reset = (seg_ids[:, t] != seg_ids[:, t - 1]).to(decay.dtype)
            decay = decay * (1 - reset).view(B, 1, 1)
        S = S * decay.unsqueeze(-1)
        err = vt - torch.einsum('bhk,bhkv->bhv', kt, S)
        S = S + torch.einsum('bhk,bhv->bhkv', bt.unsqueeze(-1) * kt, err)
        o[:, t] = torch.einsum('bhk,bhkv->bhv', qt, S)
    return o, S


_SUB_CHUNK = 16  # sub-block size inside a chunk for the decayed-score matrices


def _decay_scores(x, y, g, diagonals):
    """Lower-triangular decayed inner products M[s, c, j] = sum_d x[s,c,d]
    y[j,d] exp(g[c,d] - g[j,d]), zeroed on and above the diagonals[s]-th
    diagonal.

    x: [S, ..., BT, K] — score matrices sharing y and g are stacked so the
    column loop below, which is the expensive part, runs once for all of them.
    y, g: [..., BT, K], g non-increasing along BT (a cumsum of non-positive
    log-decays). Sub-blocks below the diagonal factor the decay through their
    row block's first position, so both exponents stay <= 0 and the block is a
    single matmul. Only the diagonal sub-blocks keep the per-column form, where
    a positive exponent can appear but always lands in a masked-out entry."""
    BT, K = x.shape[-2], x.shape[-1]
    BC = min(_SUB_CHUNK, BT)
    assert BT % BC == 0, f"chunk_size={BT} must be a multiple of {BC}"
    NB = BT // BC
    xs, ys, gs = (t.unflatten(-2, (NB, BC)) for t in (x, y, g))

    diag = x.new_zeros(*x.shape[:-2], NB, BC, BC)
    for j in range(BC):
        # clamp: entries with a positive exponent are all above the diagonal and
        # get masked out below — clamping keeps them from overflowing to inf,
        # which would otherwise poison the backward pass with 0 * inf.
        decay = (gs - gs[..., j:j + 1, :]).clamp(max=0).exp()
        diag[..., j] = (xs * decay * ys[..., j:j + 1, :]).sum(-1)
    tri = torch.stack([torch.triu(torch.ones(BC, BC, dtype=torch.bool,
                                             device=x.device), diagonal=d)
                       for d in diagonals])
    diag = diag.masked_fill(tri.view(len(diagonals), *(1,) * (diag.dim() - 3),
                                     BC, BC), 0)

    out = x.new_zeros(*x.shape[:-2], BT, BT)
    for b in range(NB):
        lo, hi = b * BC, (b + 1) * BC
        out[..., lo:hi, lo:hi] = diag[..., b, :, :]
        if b:
            ref = g[..., lo:lo + 1, :]  # route both decays through the block start
            left = x[..., lo:hi, :] * (g[..., lo:hi, :] - ref).exp()
            right = y[..., :lo, :] * (ref - g[..., :lo, :]).exp()
            out[..., lo:hi, :lo] = left @ right.mT
    return out


def _unit_lower_inverse(L):
    """(I - L)^{-1} for a strictly lower-triangular L: [..., BT, BT].

    Blocked forward substitution. The BC-wide diagonal blocks are inverted
    together by repeated squaring — (I+M)(I+M^2)(I+M^4)... telescopes to
    I + M + ... + M^(BC-1), and M^BC is zero — which is log2(BC) batched
    matmuls where the row recursion needs BC sequential steps. Each remaining
    block row is then one matmul against the block rows already solved."""
    BT = L.shape[-1]
    BC = min(_SUB_CHUNK, BT)
    NB = BT // BC
    eye = torch.eye(BC, dtype=L.dtype, device=L.device)

    M = torch.diagonal(L.unflatten(-1, (NB, BC)).unflatten(-3, (NB, BC)),
                       dim1=-4, dim2=-2).movedim(-1, -3)   # [..., NB, BC, BC]
    blocks = eye + M
    for _ in range((BC - 1).bit_length() - 1):
        M = M @ M
        blocks = blocks + blocks @ M

    X = blocks[..., 0, :, :]      # growing top-left square of the inverse
    for b in range(1, NB):
        lo, hi = b * BC, (b + 1) * BC
        D = blocks[..., b, :, :]
        off = D @ (L[..., lo:hi, :lo] @ X)
        X = torch.cat([
            torch.cat([X, X.new_zeros(*X.shape[:-1], BC)], dim=-1),
            torch.cat([off, D], dim=-1),
        ], dim=-2)
    return X


def kda_chunk(q, k, v, g, beta, initial_state=None, chunk_size=64, seg_ids=None):
    """Chunked-parallel KDA in fp32 — the training path: all positions of a
    chunk are processed with batched matmuls, only the inter-chunk scan is
    sequential. Requires T % chunk_size == 0; same signature as kda_recurrence.

    With seg_ids ([B, T] document segment ids) the scan is exactly equivalent
    to running the recurrence per segment: cross-segment pairs are removed
    from the intra-chunk solves, and the inter-chunk state read/carry/write
    is gated so no state crosses a boundary (segment starts read a zero
    state; a chunk passes on only its final segment's contributions)."""
    B, T, H, K = q.shape
    V = v.size(-1)
    BT = chunk_size
    assert T % BT == 0, f"T={T} must be a multiple of chunk_size={BT}"
    NT = T // BT
    q, k, v, g, beta = (x.float() for x in (q, k, v, g, beta))
    q = q * K ** -0.5

    # -> [B, H, NT, BT, ...]
    q, k, v, g = (x.view(B, NT, BT, H, x.size(-1)).permute(0, 3, 1, 2, 4)
                  for x in (q, k, v, g))
    beta = beta.view(B, NT, BT, H).permute(0, 3, 1, 2)  # [B, H, NT, BT]
    g = g.cumsum(-2)  # within-chunk cumulative log-decay

    if seg_ids is not None:
        seg = seg_ids.view(B, NT, BT)
        # Same-segment pair mask within each chunk: [B, 1, NT, BT, BT] (c, j).
        same_pair = (seg[:, :, :, None] == seg[:, :, None, :]).unsqueeze(1)
        # Segment of the position immediately preceding each chunk. Chunk 0
        # treats its own first segment as preceding (an initial_state, if any,
        # is assumed to belong to it).
        seg_prev = torch.cat([seg[:, 0:1, 0], seg[:, :-1, -1]], dim=1)  # [B, NT]
        # Positions whose segment reaches back before the chunk read the state.
        read_gate = (seg == seg_prev[:, :, None]).float()[:, None]      # [B,1,NT,BT]
        seg_last = seg[:, :, -1:]                                       # [B, NT, 1]
        # The decayed incoming state survives only if no boundary occurred.
        carry = (seg_last == seg_prev[:, :, None]).float()[:, None]     # [B,1,NT,1]
        # Only the chunk's final (suffix) segment contributes to the next state.
        write_gate = (seg == seg_last).float()[:, None]                 # [B,1,NT,BT]

    # UT transform for the delta rule, and the intra-chunk query->key weights.
    # Both are decayed scores against k under the same g, so they share a pass.
    # Aqk is independent of the running state, so every chunk's block is built
    # up front instead of inside the scan below.
    scores = _decay_scores(torch.stack((k, q)), k, g, (0, 1))
    if seg_ids is not None:
        # Cross-segment pairs drop out of the solve; the inverse stays
        # block-diagonal, i.e. each segment solves independently.
        scores = scores.masked_fill(~same_pair, 0)
    A, Aqk_all = scores
    A = A * beta.unsqueeze(-1)    # row c scaled by beta of the writing position
    A = _unit_lower_inverse(-A) * beta.unsqueeze(-2)

    w = A @ (g.exp() * k)         # decayed keys   [B,H,NT,BT,K]
    u = A @ v                     # pseudo-values  [B,H,NT,BT,V]

    S = q.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state.float()
    o = torch.zeros_like(v)       # [B,H,NT,BT,V]
    for i in range(NT):
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        Aqk = Aqk_all[:, :, i]
        if seg_ids is not None:
            rg = read_gate[:, :, i].unsqueeze(-1)   # [B,H,BT,1]
            v_i = u_i - rg * (w_i @ S)  # fresh segments ignore the incoming state
            o[:, :, i] = rg * ((q_i * g_i.exp()) @ S) + Aqk @ v_i
            S = carry[:, :, i].unsqueeze(-1) * S * g_i[:, :, -1].exp().unsqueeze(-1)
            S = S + ((write_gate[:, :, i].unsqueeze(-1) * (g_i[:, :, -1:] - g_i).exp())
                     * k_i).transpose(-1, -2) @ v_i
        else:
            v_i = u_i - w_i @ S         # subtract what the incoming state already knows
            o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
            S = S * g_i[:, :, -1].exp().unsqueeze(-1)
            S = S + ((g_i[:, :, -1:] - g_i).exp() * k_i).transpose(-1, -2) @ v_i
    o = o.permute(0, 2, 3, 1, 4).reshape(B, T, H, V)
    return o, S


def kda_scan(q, k, v, g, beta, initial_state=None, chunk_size=64):
    """kda_chunk over the leading whole chunks, kda_recurrence for the ragged
    tail. Same result as running the recurrence over the whole span, but the
    sequential part is the tail rather than every position -- which is what a
    cached prefill of T tokens would otherwise cost."""
    T = q.shape[1]
    full = T - T % chunk_size
    if not full:
        return kda_recurrence(q, k, v, g, beta, initial_state=initial_state)
    head = (x[:, :full] for x in (q, k, v, g, beta))
    o, S = kda_chunk(*head, initial_state=initial_state, chunk_size=chunk_size)
    if full < T:
        tail = (x[:, full:] for x in (q, k, v, g, beta))
        o_tail, S = kda_recurrence(*tail, initial_state=S)
        o = torch.cat((o, o_tail), dim=1)
    return o, S


class KimiDeltaAttention(nn.Module):
    """KDA layer. Same role as MultiHeadAttention but with a constant-size
    recurrent state instead of a growing KV cache, and no positional encoding
    (order is carried by the recurrence).

    Projections follow fla's KimiDeltaAttention: q/k/v get a plain SiLU (no
    short conv), q/k are L2-normalized per head; the forget gate is the
    low-rank f_proj + softplus(dt_bias) + (-exp(A_log)) parameterization with a
    mamba-style per-channel dt init; the write strength beta is a per-head
    sigmoid; the output is per-head RMSNorm'ed, sigmoid-gated by the low-rank
    g_proj, then projected out (zero-init at the model level, like attn out).
    """

    def __init__(self, d_model, heads, chunk_size=64):
        super().__init__()
        assert d_model % heads == 0
        self.d_model = d_model
        self.h = heads
        self.d_k = d_model // heads
        self.chunk_size = chunk_size

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        # Per-channel forget gate, low-rank: d_model -> d_k -> heads * d_k.
        self.f_proj = nn.Sequential(
            nn.Linear(d_model, self.d_k, bias=False),
            nn.Linear(self.d_k, d_model, bias=False),
        )
        self.b_proj = nn.Linear(d_model, heads, bias=False)
        # Per-head decay rate (fla: log U(1, 16)) and per-channel time-step bias.
        self.A_log = nn.Parameter(torch.log(torch.empty(heads).uniform_(1.0, 16.0)))
        dt = (torch.rand(d_model) * (math.log(0.1) - math.log(0.001))
              + math.log(0.001)).exp().clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        self.g_proj = nn.Sequential(
            nn.Linear(d_model, self.d_k, bias=False),
            nn.Linear(self.d_k, d_model, bias=True),
        )
        self.o_norm = nn.RMSNorm(self.d_k, eps=1e-5)  # eps follows fla's FusedRMSNormGated
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.s_cache = {}  # recurrence states keyed by recurrence-pass index

    def reset_cache(self):
        self.s_cache = {}

    def forward(self, x, start_pos=None, cache_idx=0, seg_ids=None):
        B, T, _ = x.shape
        H, D = self.h, self.d_k
        q = F.normalize(F.silu(self.q_proj(x)).view(B, T, H, D), dim=-1)
        k = F.normalize(F.silu(self.k_proj(x)).view(B, T, H, D), dim=-1)
        v = F.silu(self.v_proj(x)).view(B, T, H, D)
        # Log-space per-channel decay: always negative, so exp(g) lies in (0, 1).
        g = -(self.A_log.exp()[:, None]
              * F.softplus(self.f_proj(x).view(B, T, H, D) + self.dt_bias.view(1, 1, H, D)))
        beta = torch.sigmoid(self.b_proj(x))  # [B, T, H]

        if start_pos is None:
            if T % self.chunk_size == 0:
                o, _ = kda_chunk(q, k, v, g, beta, chunk_size=self.chunk_size,
                                 seg_ids=seg_ids)
            else:
                o, _ = kda_recurrence(q, k, v, g, beta, seg_ids=seg_ids)
        else:
            o, S = kda_scan(q, k, v, g, beta, initial_state=self.s_cache.get(cache_idx),
                            chunk_size=self.chunk_size)
            self.s_cache[cache_idx] = S

        o = o.to(x.dtype)
        # Per-head RMSNorm, then the sigmoid output gate (fla FusedRMSNormGated).
        o = self.o_norm(o) * torch.sigmoid(self.g_proj(x).view(B, T, H, D))
        return self.o_proj(o.reshape(B, T, H * D))
