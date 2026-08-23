"""MLX half of the sampling stack: the primitives inference/sample.py's
decode_loop calls, so the loop, its stop-token accounting, and its KV-window
policy are shared with the torch backend rather than reimplemented."""
import mlx.core as mx


def top_p_filter(probs, top_p):
    """Zero out tokens outside the smallest nucleus whose cumulative prob >= top_p.
    Always keeps at least the single highest-prob token."""
    order = mx.argsort(-probs)
    sorted_probs = probs[order]
    drop = mx.cumsum(sorted_probs, axis=-1) >= top_p
    # Shift right so the token that first crosses the threshold is still kept.
    drop = mx.concatenate([mx.array([False]), drop[:-1]])
    return mx.put_along_axis(mx.zeros_like(probs), order,
                             mx.where(drop, 0.0, sorted_probs), axis=-1)


class MLXBackend:
    """Counterpart to inference.sample.TorchBackend. There is no device to move
    tensors to -- MLX runs on unified memory -- so the object carries no state."""

    def prefill(self, model, ids, start_pos=0):
        """Feed `ids` through the model in one batched forward, into the KV cache
        at start_pos (start_pos=None runs uncached). Logits for the final token."""
        return model(mx.array([ids]), start_pos=start_pos)[:, -1, :]

    def step(self, model, tok, start_pos):
        """Extend the cache by the single token `tok`; logits for what follows.

        The step is dispatched but not waited on: MLX is lazy, so without this
        the graph would grow until the next read-back, and with a blocking eval
        the CPU would idle through every kernel instead of running ahead to
        build the next step."""
        logits = model(tok, start_pos=start_pos)[:, -1, :]
        mx.async_eval(logits)
        return logits

    def sample(self, logits, temperature, top_p):
        """Sample one token id, returned as a (1, 1) array.

        Gumbel-max -- argmax of log p plus Gumbel noise -- draws from the same
        distribution as a multinomial without ever leaving the GPU. argmax is
        invariant to a constant scale on p, so the nucleus needs no renormalizing."""
        probs = mx.softmax(logits.astype(mx.float32) / max(temperature, 1e-6), axis=-1)[0]
        if top_p < 1.0:
            probs = top_p_filter(probs, top_p)
        u = mx.maximum(mx.random.uniform(shape=probs.shape), 1e-20)
        return mx.argmax(mx.log(probs) - mx.log(-mx.log(u)), axis=-1).reshape(1, 1)

    def token_ids(self, pending):
        """Read a block of sampled tokens back off the device -- the one point in
        the decode loop that waits on the queued work."""
        return mx.concatenate(pending, axis=1).reshape(-1).tolist()
