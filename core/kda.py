
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


def kda_recurrence(q, k, v, g, beta, initial_state=None):
    """Sequential KDA scan in fp32 — the reference path, also used for cached
    inference (prefill chunks and single-token decode).

    q, k: [B, T, H, K]; v: [B, T, H, V]; g: [B, T, H, K] log-decay (<= 0);
    beta: [B, T, H]; initial_state: [B, H, K, V] or None.
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
        S = S * gt.exp().unsqueeze(-1)
        err = vt - torch.einsum('bhk,bhkv->bhv', kt, S)
        S = S + torch.einsum('bhk,bhv->bhkv', bt.unsqueeze(-1) * kt, err)
        o[:, t] = torch.einsum('bhk,bhkv->bhv', qt, S)
    return o, S


def kda_chunk(q, k, v, g, beta, initial_state=None, chunk_size=64):
    """Chunked-parallel KDA in fp32 — the training path: all positions of a
    chunk are processed with batched matmuls, only the inter-chunk scan is
    sequential. Requires T % chunk_size == 0; same signature as kda_recurrence."""
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

    # UT transform for the delta rule. exp(g_c - g_i) is formed per column so no
    # entry ever exponentiates a positive number (a cumsum of non-positive
    # log-decays is non-increasing, so only masked-out entries could overflow).
    mask_incl = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)
    A = q.new_zeros(B, H, NT, BT, BT)
    for i in range(BT):
        k_i = k[..., i, :]        # [B,H,NT,K]
        g_i = g[..., i:i+1, :]    # [B,H,NT,1,K]
        A[..., i] = torch.einsum('...cd,...d->...c', k * (g - g_i).exp(), k_i)
    A = A * beta.unsqueeze(-1)    # row c scaled by beta of the writing position
    A = -A.masked_fill(mask_incl, 0)
    for i in range(1, BT):        # forward substitution: (I + strictly-lower)^{-1}
        A[..., i, :i] = (A[..., i, :i].clone()
                         + (A[..., i, :, None].clone() * A[..., :, :i].clone()).sum(-2))
    A = (A + torch.eye(BT, dtype=torch.float, device=q.device)) * beta.unsqueeze(-2)

    w = A @ (g.exp() * k)         # decayed keys   [B,H,NT,BT,K]
    u = A @ v                     # pseudo-values  [B,H,NT,BT,V]

    S = q.new_zeros(B, H, K, V)
    if initial_state is not None:
        S = S + initial_state.float()
    o = torch.zeros_like(v)       # [B,H,NT,BT,V]
    mask_strict = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=1)
    for i in range(NT):
        q_i, k_i, u_i, g_i, w_i = q[:, :, i], k[:, :, i], u[:, :, i], g[:, :, i], w[:, :, i]
        Aqk = q.new_zeros(B, H, BT, BT)  # intra-chunk query->key weights, decayed
        for j in range(BT):
            k_j = k_i[:, :, j]           # [B,H,K]
            g_j = g_i[:, :, j:j+1]       # [B,H,1,K]
            Aqk[..., j] = torch.einsum('...cd,...d->...c', q_i * (g_i - g_j).exp(), k_j)
        Aqk = Aqk.masked_fill(mask_strict, 0)
        v_i = u_i - w_i @ S              # subtract what the incoming state already knows
        o[:, :, i] = (q_i * g_i.exp()) @ S + Aqk @ v_i
        S = S * g_i[:, :, -1].exp().unsqueeze(-1)
        S = S + ((g_i[:, :, -1:] - g_i).exp() * k_i).transpose(-1, -2) @ v_i
    o = o.permute(0, 2, 3, 1, 4).reshape(B, T, H, V)
    return o, S


class _RMSNorm(nn.Module):
    """Mirror of core.model.RMSNorm (duplicated to keep this module import-
    cycle-free); eps follows fla's FusedRMSNormGated."""

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * x * rms


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
        self.o_norm = _RMSNorm(self.d_k)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.s_cache = {}  # recurrence states keyed by recurrence-pass index

    def reset_cache(self):
        self.s_cache = {}

    def forward(self, x, start_pos=None, cache_idx=0):
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
                o, _ = kda_chunk(q, k, v, g, beta, chunk_size=self.chunk_size)
            else:
                o, _ = kda_recurrence(q, k, v, g, beta)
        else:
            S = self.s_cache.get(cache_idx)
            o, S = kda_recurrence(q, k, v, g, beta, initial_state=S)
            self.s_cache[cache_idx] = S

        o = o.to(x.dtype)
        # Per-head RMSNorm, then the sigmoid output gate (fla FusedRMSNormGated).
        o = self.o_norm(o) * torch.sigmoid(self.g_proj(x).view(B, T, H, D))
        return self.o_proj(o.reshape(B, T, H * D))
