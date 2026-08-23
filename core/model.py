import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from core.kda import KimiDeltaAttention


LOGIT_SOFTCAP = 15.0  # tanh soft-capping (Gemma-2 / modded-nanogpt record #18)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x.int())


def precompute_rope_freqs(head_dim, max_seq_len, base=10000.0):
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_seq_len).float()
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x, cos, sin):
    # x: (..., T, D) with D even; cos/sin: (T, D/2) or (B, T, D/2) for
    # per-token positions (RoPE reset at document boundaries).
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    if cos.dim() == 2:
        cos = cos[None, None, :, :]
        sin = sin[None, None, :, :]
    else:
        cos = cos[:, None, :, :]
        sin = sin[:, None, :, :]
    rotated_1 = x1 * cos - x2 * sin
    rotated_2 = x1 * sin + x2 * cos
    return torch.stack((rotated_1, rotated_2), dim=-1).flatten(-2)


def apply_partial_rope(x, cos, sin, rot_dim):
    """Rotate only the first rot_dim dims of each head (partial rotary).

    Parameter-golf leaderboard finding: rotating ~half the head dims and
    leaving the rest position-free helps small models."""
    if rot_dim >= x.size(-1):
        return apply_rope(x, cos, sin)
    x_rot, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    return torch.cat((apply_rope(x_rot, cos, sin), x_pass), dim=-1)


class RMSNorm(nn.RMSNorm):
    """torch's fused RMSNorm at this model's eps. The fused kernel also
    reduces in fp32, so it is both faster and more accurate under bf16 than
    computing the mean square in the input dtype."""

    def __init__(self, d_model, eps=1e-6):
        super().__init__(d_model, eps=eps)


def attention(q, k, v, mask=None, dropout_p=0.0, is_causal=False):
    # Wrap SDPA so we keep the (1, T, S) bool-mask convention used by nopeak_mask.
    if mask is not None and mask.dim() == 3:
        mask = mask.unsqueeze(1)
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=dropout_p,
                                          is_causal=is_causal)


def window_band_mask(T, S, offset, window, device):
    """(1, T, S) bool: query i, sitting at position offset+i among the S cached
    keys, attends the `window` keys ending at itself."""
    d = (torch.arange(T, device=device)[:, None] + offset
         - torch.arange(S, device=device)[None, :])
    return ((d >= 0) & (d < window)).unsqueeze(0)


def window_block_mask(window, device, seg_ids=None):
    """(head, tail) bool masks for blocked sliding-window attention.

    `tail` scores query blocks 1..nb-1 against the 2W keys starting one window
    before each block, so the W keys ending at any query are always in range;
    `head` is the first block, which has no earlier window and is plain causal
    -- exactly `tail`'s own-block half. Leading dims are 1 without seg_ids."""
    i = torch.arange(window, device=device)
    j = torch.arange(2 * window, device=device)
    # key j of a block sits `window` positions before query 0, so its offset
    # from query i is i - j + window; keep that offset in [0, window).
    tail = ((j > i[:, None]) & (j <= i[:, None] + window))[None, None]
    if seg_ids is None:
        return tail[..., window:], tail
    q_seg = seg_ids.unflatten(1, (-1, window))            # (B, nb, W)
    k_seg = seg_ids.unfold(1, 2 * window, window)         # (B, nb-1, 2W)
    head = tail[..., window:] & (q_seg[:, :1, :, None] == q_seg[:, :1, None, :])
    tail = tail & (q_seg[:, 1:, :, None] == k_seg[:, :, None, :])
    return head, tail.flatten(0, 1).unsqueeze(1)


def window_attention(q, k, v, window, masks, dropout_p=0.0, enable_gqa=False):
    """Sliding-window attention: every query attends the `window` keys ending
    at itself, at O(T*W) cost instead of O(T^2). Needs T a multiple of, and
    larger than, `window`.

    Queries are cut into W-wide blocks and each block reads one 2W-key span, so
    the whole thing is two batched SDPA calls over short sequences."""
    B, H, T, D = q.shape
    nb = T // window
    head_mask, tail_mask = masks

    def blocks(x, width, drop):
        # Overlapping `width`-key spans, one per query block past the first.
        x = x[:, :, drop:].unfold(2, width, window).movedim(-1, -2)
        return x.transpose(1, 2).reshape(B * (nb - 1), -1, width, D)

    first = F.scaled_dot_product_attention(
        q[:, :, :window], k[:, :, :window], v[:, :, :window], attn_mask=head_mask,
        dropout_p=dropout_p, enable_gqa=enable_gqa)
    rest = F.scaled_dot_product_attention(
        blocks(q, window, window), blocks(k, 2 * window, 0), blocks(v, 2 * window, 0),
        attn_mask=tail_mask, dropout_p=dropout_p, enable_gqa=enable_gqa)
    rest = rest.view(B, nb - 1, H, window, D).transpose(1, 2).reshape(B, H, -1, D)
    return torch.cat((first, rest), dim=2)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with GQA, QK-norm, and partial RoPE.

    kv_heads < heads shares each KV head across heads//kv_heads query heads
    (saves params + KV cache). QK-norm (RMSNorm on per-head q/k) stabilizes
    training and tolerates higher Muon LRs. rope_frac controls partial rotary.
    """

    def __init__(self, heads, d_model, kv_heads=None, max_seq_len=4096,
                 dropout=0.1, rope_frac=0.5, window=0):
        super().__init__()

        self.window = window
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.h_kv = kv_heads or heads
        assert heads % self.h_kv == 0, "heads must be divisible by kv_heads"
        self.groups = heads // self.h_kv

        # rotate an even number of dims per head
        self.rot_dim = max(2, int(self.d_k * rope_frac) // 2 * 2)

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, self.h_kv * self.d_k)
        self.v_linear = nn.Linear(d_model, self.h_kv * self.d_k)

        self.q_norm = RMSNorm(self.d_k)
        self.k_norm = RMSNorm(self.d_k)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

        cos, sin = precompute_rope_freqs(self.rot_dim, max_seq_len)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        # KV caches keyed by recurrence pass index (depth-looped layers are
        # distinct positions in the unrolled stack, so they need separate caches).
        self.k_cache = {}
        self.v_cache = {}

    def reset_cache(self):
        self.k_cache = {}
        self.v_cache = {}

    def _cached_kv(self, cache_idx, k, v, start_pos):
        """Fold this step's keys and values into the cache and return the whole
        span the queries at start_pos.. may attend.

        A windowed layer can never reach further than `window - 1` positions
        back, so its cache holds exactly that many rows instead of the whole
        context: decode is constant-memory however long the generation runs.
        Global layers keep the full context, written at its absolute position."""
        T = k.size(2)
        rows = self.window - 1 if self.window else self.rope_cos.size(0)
        if cache_idx not in self.k_cache:
            shape = (k.size(0), self.h_kv, rows, self.d_k)
            self.k_cache[cache_idx] = torch.zeros(shape, device=k.device, dtype=k.dtype)
            self.v_cache[cache_idx] = torch.zeros(shape, device=k.device, dtype=k.dtype)
        kc, vc = self.k_cache[cache_idx], self.v_cache[cache_idx]
        if not self.window:
            kc[:, :, start_pos:start_pos+T] = k
            vc[:, :, start_pos:start_pos+T] = v
            return kc[:, :, :start_pos+T], vc[:, :, :start_pos+T]
        n = min(start_pos, rows)
        if n:
            k, v = (torch.cat((c[:, :, :n], x), dim=2) for c, x in ((kc, k), (vc, v)))
        keep = min(rows, n + T)
        kc[:, :, :keep], vc[:, :, :keep] = k[:, :, -keep:], v[:, :, -keep:]
        return k, v

    def rope_tables(self, pos_ids):
        """cos/sin gathered at per-token positions. Every layer's table holds
        the same values, so the decoder gathers once for the whole stack."""
        return self.rope_cos[pos_ids], self.rope_sin[pos_ids]

    def forward(self, q, k, v, mask=None, start_pos=None, cache_idx=0, v1=None,
                rope=None):

        bs = q.size(0)

        # perform linear operation and split into heads (kv may have fewer)
        k = self.k_linear(k).view(bs, -1, self.h_kv, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h_kv, self.d_k)

        # transpose to get dimensions bs * heads * sl * d_k
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        T = q.size(2)
        if rope is not None:
            cos, sin = rope   # per-token positions, prepared by the decoder
        else:
            pos = start_pos if start_pos is not None else 0
            cos = self.rope_cos[pos:pos+T]
            sin = self.rope_sin[pos:pos+T]
        q = apply_partial_rope(q, cos, sin, self.rot_dim)
        k = apply_partial_rope(k, cos, sin, self.rot_dim)

        v_pre = v  # this layer's own V (kv-head space), for value-residual mixing
        if v1 is not None:
            # Value residual (ResFormer): blend in layer-0 values. Mix BEFORE the
            # cache append so the cache stores the mixed v used in training.
            gate = torch.sigmoid(self.vres)
            v = (1 - gate) * v + gate * v1

        if start_pos is not None:
            k, v = self._cached_kv(cache_idx, k, v, start_pos)
            if self.window:
                # A step reads at most `window` keys and every one of them is
                # inside the window, so decoding a single token needs no mask.
                mask = None if T == 1 else window_band_mask(
                    T, k.size(2), k.size(2) - T, self.window, q.device)

        dropout_p = self.dropout.p if self.training else 0.0
        if start_pos is None and self.window and T > self.window:
            scores = window_attention(q, k, v, self.window, mask, dropout_p,
                                      enable_gqa=self.groups > 1)
        elif mask is None and start_pos is None and q.device.type == 'cuda':
            # Training fast path: flash SDPA handles GQA without expanding K/V.
            scores = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p, enable_gqa=True)
        else:
            if self.groups > 1:
                k = k.repeat_interleave(self.groups, dim=1)
                v = v.repeat_interleave(self.groups, dim=1)
            # mask None + no cache means causal training; mask None + cache means
            # single-chunk decode attending over the whole cache (not causal).
            scores = attention(q, k, v, mask, dropout_p,
                               is_causal=mask is None and start_pos is None)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous() \
            .view(bs, -1, self.d_model)
        output = self.out(concat)

        return output, v_pre


def _round_to_multiple(x, multiple):
    return ((x + multiple - 1) // multiple) * multiple


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None, dropout=0.1):
        super().__init__()
        if d_ff is None:
            # Match a 4*d_model 2-matmul FFN param budget with 3 matmuls: 8/3*d_model.
            d_ff = _round_to_multiple(8 * d_model // 3, 64)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        gated = F.silu(self.w_gate(x)) * self.w_up(x)
        return self.w_down(self.dropout(gated))


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# build a decoder layer with two multi-head attention layers and
# one feed-forward layer
class DecoderLayer(nn.Module):  # deleted any reference to encoder outputs
    def __init__(self, d_model, heads, dropout=0.1, kv_heads=None, use_kda=False,
                 window=0):
        super().__init__()
        self.norm_1 = RMSNorm(d_model)
        self.norm_3 = RMSNorm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.is_kda = use_kda
        if use_kda:
            self.attn_1 = KimiDeltaAttention(d_model, heads)
        else:
            self.attn_1 = MultiHeadAttention(heads, d_model, kv_heads=kv_heads,
                                             dropout=dropout, window=window)
        self.ff = SwiGLU(d_model, dropout=dropout)

    def forward(self, x, mask, start_pos=None, cache_idx=0, v1=None, seg_ids=None,
                rope=None):
        x2 = self.norm_1(x)
        if self.is_kda:
            # KDA ignores the attention mask (the recurrence is inherently
            # causal; seg_ids resets its state at document boundaries) and does
            # not participate in value-residual mixing.
            att, v = self.attn_1(x2, start_pos=start_pos, cache_idx=cache_idx,
                                 seg_ids=seg_ids), None
        else:
            att, v = self.attn_1(x2, x2, x2, mask, start_pos=start_pos,
                                 cache_idx=cache_idx, v1=v1, rope=rope)
        x = x + self.dropout_1(att)
        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))
        return x, v


class Decoder(nn.Module):
    def __init__(self, vocab, d_model, N, heads, dropout, kv_heads=None, loops=1,
                 grad_ckpt=False, value_residual=False, unet_skips=False, attn_res=0,
                 kda=0, swa=0):
        super().__init__()
        self.N = N
        self.swa = swa
        self.loops = loops
        self.grad_ckpt = grad_ckpt
        self.value_residual = value_residual
        self.unet_skips = unet_skips
        self.attn_res = attn_res
        self.kda = kda
        self.embed = Embedder(vocab, d_model)
        if kda:
            # Every layer is KDA except each kda-th one, which keeps full SDPA
            # attention: kda=1 -> all KDA, kda=4 -> Kimi-style 3:1 hybrid.
            self.layers = nn.ModuleList([
                DecoderLayer(d_model, heads, dropout, kv_heads=kv_heads, window=swa,
                             use_kda=not (kda > 1 and i % kda == kda - 1))
                for i in range(N)
            ])
        else:
            self.layers = get_clones(DecoderLayer(d_model, heads, dropout,
                                                  kv_heads=kv_heads, window=swa), N)
        self.norm = RMSNorm(d_model)
        if value_residual:
            # Per-layer value-mix scalars init 0 (sigmoid=0.5). Only full-
            # attention layers participate; the first one stays unmixed
            # (it defines v1), so it gets no scalar.
            mha_ids = [i for i in range(N) if not self.layers[i].is_kda]
            for i in mha_ids[1:]:
                self.layers[i].attn_1.vres = nn.Parameter(torch.zeros(()))
        if unet_skips:
            # Skip gates init -1.5 (sigmoid ~0.18): x0 embedding shortcut for
            # every layer, mirror skips for the second half (modded-nanogpt #11).
            self.skip_x0 = nn.Parameter(torch.full((N,), -1.5))
            self.skip_unet = nn.Parameter(torch.full((N,), -1.5))
        if attn_res:
            # Attention Residuals (Moonshot/Kimi, arXiv:2603.15031), block variant:
            # at each boundary of `attn_res` layers, the running stream is replaced
            # by a per-token softmax-weighted mix over all previous block outputs
            # (queries from the current stream, keys from each candidate). Standard
            # residual adds still apply within blocks. With zero-init residual
            # out-projections every block starts as identity, so all candidates are
            # equal at init and the mix starts neutral.
            self.attnres_wq = nn.Linear(d_model, d_model, bias=False)
            self.attnres_wk = nn.Linear(d_model, d_model, bias=False)

    def _depth_mix(self, block_outs):
        """Softmax attention over depth: mix stacked block outputs (J, ...) into
        one stream, per token. The last candidate is always the current stream."""
        cands = torch.stack(block_outs, dim=0)                  # (J, B, T, d)
        q = self.attnres_wq(cands[-1])                          # (B, T, d)
        k = self.attnres_wk(cands)                              # (J, B, T, d)
        logits = torch.einsum('btd,jbtd->btj', q, k) / (q.size(-1) ** 0.5)
        w = torch.softmax(logits, dim=-1)
        return torch.einsum('btj,jbtd->btd', w, cands)

    def forward(self, trg, mask, start_pos=None, seg_ids=None):
        rope = None
        T = trg.size(1)
        # Sliding-window layers read a blocked mask instead of the dense causal
        # one; a window at least as wide as the sequence is just full attention.
        windowed = bool(self.swa) and start_pos is None and T > self.swa
        if seg_ids is not None:
            # Document-aware training: no attention across boundaries, RoPE
            # positions restart at each new segment (KDA gets seg_ids directly).
            # The gathered tables are the same for every layer, so they are
            # built once here instead of per layer -- at (B, T, rot_dim/2) each
            # they are the single largest thing the rotation holds on to.
            assert start_pos is None, "seg_ids is only supported in training windows"
            if not windowed:
                mask = segment_mask(seg_ids)
            pos_ids = segment_pos_ids(seg_ids)
            mha = next((l.attn_1 for l in self.layers if not l.is_kda), None)
            rope = mha.rope_tables(pos_ids) if mha is not None else None
        if windowed:
            assert T % self.swa == 0, f"seqlen {T} must be a multiple of swa={self.swa}"
            mask = window_block_mask(self.swa, trg.device, seg_ids)
        x0 = x = self.embed(trg)
        half = (self.N + 1) // 2
        # Block outputs feed AttnRes boundaries; accumulates across the whole
        # unrolled depth when loops > 1. h_0 is the embedding output.
        block_outs = [x0] if self.attn_res else None
        # Depth recurrence (parameter-golf): run the stack `loops` times for
        # loops*N effective layers with N layers' worth of params.
        for loop in range(self.loops):
            v1 = None  # fresh layer-0 values per unrolled pass
            outs = []  # per-pass layer outputs for mirror skips
            for i in range(self.N):
                if self.attn_res and i % self.attn_res == 0 and len(block_outs) > 1:
                    x = self._depth_mix(block_outs)
                x_in = x
                if self.unet_skips:
                    x_in = x + torch.sigmoid(self.skip_x0[i]) * x0
                    if i >= half:
                        x_in = x_in + torch.sigmoid(self.skip_unet[i]) * outs[self.N - 1 - i]
                v1_in = v1 if (self.value_residual and not self.layers[i].is_kda) else None
                if self.training and self.grad_ckpt:
                    x, v = checkpoint(self.layers[i], x_in, mask, start_pos, loop,
                                      v1_in, seg_ids, rope, use_reentrant=False)
                else:
                    x, v = self.layers[i](x_in, mask, start_pos=start_pos,
                                          cache_idx=loop, v1=v1_in, seg_ids=seg_ids,
                                          rope=rope)
                if v1 is None and v is not None:
                    v1 = v
                outs.append(x)
                if self.attn_res and (i % self.attn_res == self.attn_res - 1
                                      or i == self.N - 1):
                    block_outs.append(x)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab, d_model, N, heads, dropout, kv_heads=None, loops=1,
                 grad_ckpt=False, value_residual=False, unet_skips=False, attn_res=0,
                 kda=0, swa=0):
        super().__init__()
        self.decoder = Decoder(vocab, d_model, N, heads, dropout,
                               kv_heads=kv_heads, loops=loops, grad_ckpt=grad_ckpt,
                               value_residual=value_residual, unet_skips=unet_skips,
                               attn_res=attn_res, kda=kda, swa=swa)
        self.out = nn.Linear(d_model, vocab)
        self.out.weight = self.decoder.embed.embed.weight

    def forward(self, vocab, mask, start_pos=None, seg_ids=None):
        d_output = self.decoder(vocab, mask, start_pos=start_pos, seg_ids=seg_ids)
        output = self.out(d_output)
        # Soft-cap logits so no token can dominate early; keeps loss landscape smooth.
        output = LOGIT_SOFTCAP * torch.tanh(output / LOGIT_SOFTCAP)
        return output

    def reset_cache(self):
        for layer in self.decoder.layers:
            layer.attn_1.reset_cache()


def get_model(opt, vocab):

    assert opt.d_model % opt.heads == 0
    assert opt.dropout < 1

    kv_heads = getattr(opt, 'kv_heads', None) or opt.heads
    loops = getattr(opt, 'loops', 1) or 1
    grad_ckpt = bool(getattr(opt, 'grad_ckpt', False))
    value_residual = bool(getattr(opt, 'value_residual', False))
    unet_skips = bool(getattr(opt, 'unet_skips', False))
    attn_res = getattr(opt, 'attn_res', 0) or 0
    kda = getattr(opt, 'kda', 0) or 0
    swa = getattr(opt, 'swa', 0) or 0

    model = Transformer(vocab, opt.d_model, opt.n_layers, opt.heads, opt.dropout,
                        kv_heads=kv_heads, loops=loops, grad_ckpt=grad_ckpt,
                        value_residual=value_residual, unet_skips=unet_skips,
                        attn_res=attn_res, kda=kda, swa=swa)
    model.to(opt.device)

    if opt.loadname is not None:
        print("loading pretrained weights...")
        ckpt = load_checkpoint(opt.loadname)
        state = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
        model.load_state_dict(state)
    else:
        for name, p in model.named_parameters():
            if p.dim() > 1:
                # Zero-init residual-out projections (attn/KDA out + FFN down):
                # each block starts as identity, so depth costs nothing at init.
                if name.endswith(('attn_1.out.weight', 'attn_1.o_proj.weight',
                                  'ff.w_down.weight')):
                    nn.init.zeros_(p)
                else:
                    nn.init.xavier_uniform_(p)

    return model


def inference_dtype(device):
    """bf16 on accelerators, fp32 on CPU (no fast bf16 path there)."""
    return torch.bfloat16 if device.type in ('cuda', 'mps') else torch.float32


def load_checkpoint(path):
    """Read a checkpoint without pulling it onto the accelerator.

    map_location='cpu' with mmap keeps unpickling lazy; materializing every
    tensor on the device as it is unpickled costs far more than mapping the
    file and moving the assembled model once."""
    return torch.load(path, map_location='cpu', mmap=True, weights_only=False)


def model_from_checkpoint(ckpt, device, dtype=None):
    """Rebuild a Transformer from a checkpoint's saved config, weights loaded
    and ready for inference.

    The weights are converted once here rather than under torch.autocast:
    autocast's cast cache is only live while grad mode is on, so under
    no_grad it re-casts every weight on every forward, which dominates
    single-token decode."""
    cfg = ckpt['config']
    model = Transformer(
        vocab=cfg['vocab_size'], d_model=cfg['d_model'], N=cfg['n_layers'],
        heads=cfg['heads'], dropout=cfg.get('dropout', 0.0),
        kv_heads=cfg.get('kv_heads'), loops=cfg.get('loops', 1),
        value_residual=cfg.get('value_residual', False),
        unet_skips=cfg.get('unet_skips', False),
        attn_res=cfg.get('attn_res', 0),
        kda=cfg.get('kda', 0),
        swa=cfg.get('swa', 0),
    )
    dtype = inference_dtype(device) if dtype is None else dtype
    # assign hands the checkpoint's tensors straight to the module rather than
    # copying into the freshly initialized ones, which are discarded anyway --
    # but it also replaces the tied head weight, so re-tie it afterwards. The
    # final to() still carries dtype: the rope tables are non-persistent
    # buffers, so they are not in the state dict and nothing else converts them.
    model.load_state_dict({k: v.to(dtype) for k, v in ckpt['model'].items()},
                          assign=True)
    model.out.weight = model.decoder.embed.embed.weight
    return model.to(device=device, dtype=dtype).eval()


def nopeak_mask(size, device, start_pos=0):
    """(1, size, start_pos + size) bool mask (True = attend): query i may see
    every key up to start_pos + i. start_pos > 0 covers a chunk fed into an
    existing KV cache; start_pos = 0 is the square causal mask."""
    keys = torch.arange(start_pos + size, device=device)
    queries = torch.arange(size, device=device) + start_pos
    return (keys[None, :] <= queries[:, None]).unsqueeze(0)


def segment_mask(seg_ids):
    """(B, T) segment ids -> (B, T, T) bool mask: causal AND same-segment.

    Blocks attention across document/conversation boundaries in packed
    training windows (True = attend, SDPA convention)."""
    T = seg_ids.size(1)
    same = seg_ids[:, :, None] == seg_ids[:, None, :]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=seg_ids.device))
    return same & causal


def segment_pos_ids(seg_ids):
    """(B, T) segment ids -> (B, T) intra-segment position ids (RoPE reset).

    Position 0 of the window counts as a segment start — the first segment in
    a window may be a document continuation, which is the same assumption the
    windowing already makes."""
    T = seg_ids.size(1)
    arange = torch.arange(T, device=seg_ids.device)
    is_start = torch.ones_like(seg_ids, dtype=torch.bool)
    is_start[:, 1:] = seg_ids[:, 1:] != seg_ids[:, :-1]
    starts = torch.cummax(arange.unsqueeze(0) * is_start, dim=1).values
    return arange.unsqueeze(0) - starts
