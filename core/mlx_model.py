"""MLX inference backend: core/model.py's decoder stack, layer for layer.

The module tree and every parameter name match the torch model exactly, so a
checkpoint loads with no key rewriting and the two backends run the same
weights. Training-only machinery (dropout, document-segment masks, gradient
checkpointing) is left out.
"""
import mlx.core as mx
import mlx.nn as nn

from core.mlx_kda import KimiDeltaAttention
from core.model import LOGIT_SOFTCAP, _round_to_multiple

CACHE_STEP = 256    # KV cache grows in blocks of this many positions
ROPE_BASE = 10000.0
QUANT_GROUP = 64    # weights per shared scale when --backend mlx:4 / mlx:8


def _rms_norm(d_model):
    return nn.RMSNorm(d_model, eps=1e-6)


class Embedder(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)

    def __call__(self, x):
        return self.embed(x)


class MultiHeadAttention(nn.Module):
    """GQA + QK-norm + partial RoPE, with a KV cache keyed by recurrence pass
    (depth-looped layers are distinct positions in the unrolled stack, so they
    need separate caches)."""

    def __init__(self, heads, d_model, kv_heads=None, rope_frac=0.5):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        self.h_kv = kv_heads or heads
        self.scale = self.d_k ** -0.5
        # rotate an even number of dims per head
        self.rot_dim = max(2, int(self.d_k * rope_frac) // 2 * 2)

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, self.h_kv * self.d_k)
        self.v_linear = nn.Linear(d_model, self.h_kv * self.d_k)
        self.q_norm = _rms_norm(self.d_k)
        self.k_norm = _rms_norm(self.d_k)
        self.out = nn.Linear(d_model, d_model)

        self._k_cache = {}
        self._v_cache = {}

    def reset_cache(self):
        self._k_cache = {}
        self._v_cache = {}

    def _cache(self, idx, start_pos, k, v):
        """Append k/v at start_pos and return the cache up to that point.

        Grown a block at a time rather than preallocated to the context limit:
        a slice assignment costs the whole buffer, not the slice, so an
        oversized cache is paid for again on every decode step."""
        end = start_pos + k.shape[2]
        held = self._k_cache[idx].shape[2] if idx in self._k_cache else 0
        if held < end:
            shape = (k.shape[0], self.h_kv,
                     _round_to_multiple(end, CACHE_STEP) - held, self.d_k)
            for cache in (self._k_cache, self._v_cache):
                pad = mx.zeros(shape, dtype=k.dtype)
                cache[idx] = mx.concatenate([cache[idx], pad], axis=2) if held else pad
        self._k_cache[idx][:, :, start_pos:end] = k
        self._v_cache[idx][:, :, start_pos:end] = v
        return self._k_cache[idx][:, :, :end], self._v_cache[idx][:, :, :end]

    def __call__(self, x, mask=None, start_pos=None, cache_idx=0, v1=None):
        B, T, _ = x.shape
        q = self.q_norm(self.q_linear(x).reshape(B, T, self.h, self.d_k))
        k = self.k_norm(self.k_linear(x).reshape(B, T, self.h_kv, self.d_k))
        v = self.v_linear(x).reshape(B, T, self.h_kv, self.d_k)
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))

        offset = start_pos or 0
        q, k = (mx.fast.rope(t, self.rot_dim, traditional=True, base=ROPE_BASE,
                             scale=1.0, offset=offset) for t in (q, k))

        v_pre = v  # this layer's own V (kv-head space), for value-residual mixing
        if v1 is not None:
            # Value residual (ResFormer): blend in layer-0 values. Mix BEFORE the
            # cache append so the cache stores the mixed v used in training.
            gate = mx.sigmoid(self.vres)
            v = (1 - gate) * v + gate * v1

        if start_pos is not None:
            k, v = self._cache(cache_idx, start_pos, k, v)

        # SDPA handles GQA without expanding K/V to the query head count.
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out(o.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)), v_pre


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        if d_ff is None:
            d_ff = _round_to_multiple(8 * d_model // 3, 64)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def __call__(self, x):
        return self.w_down(nn.silu(self.w_gate(x)) * self.w_up(x))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, kv_heads=None, use_kda=False):
        super().__init__()
        self.norm_1 = _rms_norm(d_model)
        self.norm_3 = _rms_norm(d_model)
        self.is_kda = use_kda
        if use_kda:
            self.attn_1 = KimiDeltaAttention(d_model, heads)
        else:
            self.attn_1 = MultiHeadAttention(heads, d_model, kv_heads=kv_heads)
        self.ff = SwiGLU(d_model)

    def __call__(self, x, mask=None, start_pos=None, cache_idx=0, v1=None):
        x2 = self.norm_1(x)
        if self.is_kda:
            # KDA is inherently causal and does not participate in value-residual
            # mixing, so it takes neither the mask nor v1.
            att, v = self.attn_1(x2, start_pos=start_pos, cache_idx=cache_idx), None
        else:
            att, v = self.attn_1(x2, mask, start_pos=start_pos,
                                 cache_idx=cache_idx, v1=v1)
        x = x + att
        return x + self.ff(self.norm_3(x)), v


class Decoder(nn.Module):
    def __init__(self, vocab, d_model, N, heads, kv_heads=None, loops=1,
                 value_residual=False, unet_skips=False, attn_res=0, kda=0):
        super().__init__()
        self.N = N
        self.loops = loops
        self.value_residual = value_residual
        self.unet_skips = unet_skips
        self.attn_res = attn_res
        self.embed = Embedder(vocab, d_model)
        # Every layer is KDA except each kda-th one, which keeps full SDPA
        # attention: kda=1 -> all KDA, kda=4 -> Kimi-style 3:1 hybrid.
        self.layers = [
            DecoderLayer(d_model, heads, kv_heads=kv_heads,
                         use_kda=bool(kda) and not (kda > 1 and i % kda == kda - 1))
            for i in range(N)
        ]
        self.norm = _rms_norm(d_model)
        if value_residual:
            # Only full-attention layers participate; the first one stays
            # unmixed (it defines v1), so it gets no scalar.
            mha_ids = [i for i in range(N) if not self.layers[i].is_kda]
            for i in mha_ids[1:]:
                self.layers[i].attn_1.vres = mx.zeros(())
        if unet_skips:
            self.skip_x0 = mx.zeros((N,))
            self.skip_unet = mx.zeros((N,))
        if attn_res:
            self.attnres_wq = nn.Linear(d_model, d_model, bias=False)
            self.attnres_wk = nn.Linear(d_model, d_model, bias=False)

    def _depth_mix(self, block_outs):
        """Softmax attention over depth: mix stacked block outputs (J, ...) into
        one stream, per token. The last candidate is always the current stream."""
        cands = mx.stack(block_outs, axis=0)                    # (J, B, T, d)
        q = self.attnres_wq(cands[-1])                          # (B, T, d)
        k = self.attnres_wk(cands)                              # (J, B, T, d)
        logits = mx.einsum('btd,jbtd->btj', q, k) / (q.shape[-1] ** 0.5)
        return mx.einsum('btj,jbtd->btd', mx.softmax(logits, axis=-1), cands)

    def __call__(self, trg, start_pos=None):
        # A single-token step attends over the whole cache; anything wider is a
        # causal chunk, which SDPA aligns to the end of the keys.
        mask = None if trg.shape[1] == 1 and start_pos is not None else "causal"
        x0 = x = self.embed(trg)
        half = (self.N + 1) // 2
        block_outs = [x0] if self.attn_res else None
        for loop in range(self.loops):
            v1 = None  # fresh layer-0 values per unrolled pass
            outs = []  # per-pass layer outputs for mirror skips
            for i in range(self.N):
                if self.attn_res and i % self.attn_res == 0 and len(block_outs) > 1:
                    x = self._depth_mix(block_outs)
                x_in = x
                if self.unet_skips:
                    x_in = x + mx.sigmoid(self.skip_x0[i]) * x0
                    if i >= half:
                        x_in = x_in + mx.sigmoid(self.skip_unet[i]) * outs[self.N - 1 - i]
                v1_in = v1 if (self.value_residual and not self.layers[i].is_kda) else None
                x, v = self.layers[i](x_in, mask, start_pos=start_pos,
                                      cache_idx=loop, v1=v1_in)
                if v1 is None and v is not None:
                    v1 = v
                outs.append(x)
                if self.attn_res and (i % self.attn_res == self.attn_res - 1
                                      or i == self.N - 1):
                    block_outs.append(x)
        return self.norm(x)


class Transformer(nn.Module):
    def __init__(self, vocab, d_model, N, heads, kv_heads=None, loops=1,
                 value_residual=False, unet_skips=False, attn_res=0, kda=0):
        super().__init__()
        self.decoder = Decoder(vocab, d_model, N, heads, kv_heads=kv_heads,
                               loops=loops, value_residual=value_residual,
                               unet_skips=unet_skips, attn_res=attn_res, kda=kda)
        self.out = nn.Linear(d_model, vocab)

    def __call__(self, tokens, start_pos=None):
        logits = self.out(self.decoder(tokens, start_pos=start_pos))
        # Soft-cap logits so no token can dominate early; keeps loss landscape smooth.
        return LOGIT_SOFTCAP * mx.tanh(logits / LOGIT_SOFTCAP)

    def reset_cache(self):
        for layer in self.decoder.layers:
            layer.attn_1.reset_cache()


def _quantizable(_path, module):
    """Linear/Embedding whose rows hold a whole number of quantization groups.
    KDA's low-rank projections are narrower than one group and stay in dtype."""
    weight = getattr(module, 'weight', None)
    return (hasattr(module, 'to_quantized') and weight is not None
            and weight.shape[-1] % QUANT_GROUP == 0)


def model_from_checkpoint(ckpt, dtype=mx.bfloat16, quantize=0):
    """Rebuild an MLX Transformer from a torch checkpoint's saved config.

    The state dict is handed over under its own keys — the module tree is
    identical — and the tied head is re-pointed at the embedding table
    afterwards so the duplicate copy load_weights made is dropped.

    quantize (4 or 8) trades accuracy for a smaller, faster weight read."""
    cfg = ckpt['config']
    model = Transformer(
        vocab=cfg['vocab_size'], d_model=cfg['d_model'], N=cfg['n_layers'],
        heads=cfg['heads'], kv_heads=cfg.get('kv_heads'), loops=cfg.get('loops', 1),
        value_residual=cfg.get('value_residual', False),
        unet_skips=cfg.get('unet_skips', False),
        attn_res=cfg.get('attn_res', 0),
        kda=cfg.get('kda', 0),
    )
    model.load_weights([(k, mx.array(v.numpy()).astype(dtype))
                        for k, v in ckpt['model'].items()])
    model.out.weight = model.decoder.embed.embed.weight
    if quantize:
        nn.quantize(model, group_size=QUANT_GROUP, bits=quantize,
                    class_predicate=_quantizable)
    mx.eval(model.parameters())
    return model
