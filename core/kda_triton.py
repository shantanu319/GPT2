"""Fused CUDA kernel for the KDA inter-chunk scan.

core.kda.chunk_scan walks the chunks in Python: a 4k-token window at
chunk_size 64 is 64 iterations of four small matmuls, and the running state
makes a round trip to HBM on every one of them. Inductor cannot fuse across
the carried dependency, so the scan stays launch-bound however well the rest
of the layer compiles. Here the whole scan is one kernel per (batch, head)
with the state resident in registers for its entire length.

Loads are cast to fp32 on arrival: under autocast the operands arrive bf16,
and a state that is accumulated over NT chunks wants more mantissa than that.
"""
import torch
import triton
import triton.language as tl

ENABLED = True    # off switch for A/B benchmarking
PRECISION = None  # tl.dot input_precision; None reads it off the operands
WARPS = None      # (forward, backward) warp counts; None picks them by precision
STAGES = 1        # the loop is serially dependent, so there is nothing to pipeline

# Tensor-core matmuls want a few wide warps. The fp32 path has no tensor cores,
# and at that width the live tiles spill, so it wants them spread four times
# wider: measured on an A100, ieee at 4 warps is slower than the Python loop
# it replaces and at 16 warps is 3.4x faster.
_WARPS = {'tf32': (4, 8, 4), 'ieee': (16, 32, 16)}


def _plan(operands):
    """(tl.dot precision, warp counts) for this set of operands.

    Warps are (forward, reverse scan, per-chunk gradients).

    Autocast leaves the scan a mix: the matmul-derived operands arrive bf16
    while the elementwise ones stay fp32. One low-precision operand caps the
    whole product at 8 mantissa bits, which is fewer than tf32 carries, so
    tf32 costs nothing there. On operands that really are all fp32, defer to
    torch's own switch rather than quietly downgrading the user's matmuls."""
    prec = PRECISION
    if prec is None:
        low = any(x.dtype in (torch.bfloat16, torch.float16) for x in operands)
        prec = 'tf32' if low or torch.backends.cuda.matmul.allow_tf32 else 'ieee'
    return prec, WARPS or _WARPS[prec]


@triton.jit
def _scan_fwd(U, W, QG, A, KG, DEC, S0, O, SS, SN, NT,
              BT: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
              HAS_S0: tl.constexpr, SAVE: tl.constexpr, PREC: tl.constexpr):
    bh = tl.program_id(0).to(tl.int64)
    rt, rk, rv = tl.arange(0, BT), tl.arange(0, K), tl.arange(0, V)
    state = rk[:, None] * V + rv[None, :]

    if HAS_S0:
        S = tl.load(S0 + bh * K * V + state).to(tl.float32)
    else:
        S = tl.zeros((K, V), dtype=tl.float32)

    for i in range(NT):
        row = (bh * NT + i) * BT            # first row of chunk i
        sq = (bh * NT + i) * K              # first row of its [K, *] blocks
        if SAVE:
            tl.store(SS + sq * V + state, S)
        w = tl.load(W + (row + rt[:, None]) * K + rk[None, :]).to(tl.float32)
        u = tl.load(U + (row + rt[:, None]) * V + rv[None, :]).to(tl.float32)
        v = u - tl.dot(w, S, input_precision=PREC)
        qg = tl.load(QG + (row + rt[:, None]) * K + rk[None, :]).to(tl.float32)
        a = tl.load(A + (row + rt[:, None]) * BT + rt[None, :]).to(tl.float32)
        o = tl.dot(qg, S, input_precision=PREC) + tl.dot(a, v, input_precision=PREC)
        tl.store(O + (row + rt[:, None]) * V + rv[None, :], o)
        kg = tl.load(KG + (row + rt[:, None]) * K + rk[None, :]).to(tl.float32)
        dec = tl.load(DEC + sq + rk).to(tl.float32)
        S = dec[:, None] * S + tl.dot(tl.trans(kg), v, input_precision=PREC)

    tl.store(SN + bh * K * V + state, S)


@triton.jit
def _scan_bwd(W, QG, A, KG, DEC, SS, DO, DSN, DU, DW, DSK, DS0, NT,
              BT: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
              HAS_DSN: tl.constexpr, PREC: tl.constexpr):
    """Reverse scan: everything the carried gradient is on the path of.

    Only du, dw and the carry itself need to be here. The rest of the chunk's
    gradients depend on nothing that crosses a chunk boundary, so they are
    left to _scan_bwd_local, which has NT times the programs to spread them
    over -- this one is stuck at one program per (batch, head)."""
    bh = tl.program_id(0).to(tl.int64)
    rt, rk, rv = tl.arange(0, BT), tl.arange(0, K), tl.arange(0, V)
    state = rk[:, None] * V + rv[None, :]

    if HAS_DSN:
        dS = tl.load(DSN + bh * K * V + state).to(tl.float32)
    else:
        dS = tl.zeros((K, V), dtype=tl.float32)

    for j in range(NT):
        p = bh * NT + NT - 1 - j
        row_k = (p * BT + rt[:, None]) * K + rk[None, :]
        row_v = (p * BT + rt[:, None]) * V + rv[None, :]
        tl.store(DSK + p * K * V + state, dS)

        do = tl.load(DO + row_v).to(tl.float32)
        kg = tl.load(KG + row_k).to(tl.float32)
        a = tl.load(A + (p * BT + rt[:, None]) * BT + rt[None, :]).to(tl.float32)
        dv = (tl.dot(kg, dS, input_precision=PREC)
              + tl.dot(tl.trans(a), do, input_precision=PREC))
        tl.store(DU + row_v, dv)

        S = tl.load(SS + p * K * V + state)
        tl.store(DW + row_k, -tl.dot(dv, tl.trans(S), input_precision=PREC))
        w = tl.load(W + row_k).to(tl.float32)
        qg = tl.load(QG + row_k).to(tl.float32)
        dec = tl.load(DEC + p * K + rk).to(tl.float32)
        dS = (dec[:, None] * dS + tl.dot(tl.trans(qg), do, input_precision=PREC)
              - tl.dot(tl.trans(w), dv, input_precision=PREC))

    tl.store(DS0 + bh * K * V + state, dS)


@triton.jit
def _scan_bwd_local(U, W, SS, DO, DSK, DQG, DA, DKG, DDEC,
                    BT: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                    PREC: tl.constexpr):
    """The rest of the chunk's gradients, one program per chunk. v is rebuilt
    from the saved entry state rather than carried over from the forward."""
    p = tl.program_id(0).to(tl.int64)
    rt, rk, rv = tl.arange(0, BT), tl.arange(0, K), tl.arange(0, V)
    state = rk[:, None] * V + rv[None, :]
    row_k = (p * BT + rt[:, None]) * K + rk[None, :]
    row_v = (p * BT + rt[:, None]) * V + rv[None, :]

    S = tl.load(SS + p * K * V + state)
    dS = tl.load(DSK + p * K * V + state)
    w = tl.load(W + row_k).to(tl.float32)
    v = tl.load(U + row_v).to(tl.float32) - tl.dot(w, S, input_precision=PREC)
    do = tl.load(DO + row_v).to(tl.float32)

    tl.store(DDEC + p * K + rk, tl.sum(dS * S, 1))
    tl.store(DKG + row_k, tl.dot(v, tl.trans(dS), input_precision=PREC))
    tl.store(DQG + row_k, tl.dot(do, tl.trans(S), input_precision=PREC))
    tl.store(DA + (p * BT + rt[:, None]) * BT + rt[None, :],
             tl.dot(do, tl.trans(v), input_precision=PREC))


class _ChunkScan(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, w, qg, a, kg, dec, S0):
        B, H, NT, BT, V = u.shape
        K = w.size(-1)
        u, w, qg, a, kg, dec = (x.contiguous() for x in (u, w, qg, a, kg, dec))
        o = torch.empty_like(u)
        SN = u.new_empty(B, H, K, V, dtype=torch.float32)
        save = any(ctx.needs_input_grad)
        prec, (warps, _, _) = _plan((u, w, qg, a, kg, dec))
        SS = u.new_empty(B, H, NT, K, V, dtype=torch.float32) if save else o
        if S0 is not None:
            S0 = S0.contiguous()
        _scan_fwd[(B * H,)](u, w, qg, a, kg, dec, S0 if S0 is not None else u,
                            o, SS, SN, NT, BT=BT, K=K, V=V, HAS_S0=S0 is not None,
                            SAVE=save, PREC=prec, num_warps=warps,
                            num_stages=STAGES)
        if save:
            ctx.save_for_backward(u, w, qg, a, kg, dec, SS)
            ctx.wants_s0 = ctx.needs_input_grad[6]
        return o, SN

    @staticmethod
    def backward(ctx, do, dSN):
        u, w, qg, a, kg, dec, SS = ctx.saved_tensors
        B, H, NT, BT, V = u.shape
        K = w.size(-1)
        du, dw, dqg, da, dkg, ddec = (torch.empty_like(x)
                                      for x in (u, w, qg, a, kg, dec))
        dS0 = u.new_empty(B, H, K, V, dtype=torch.float32)
        dSK = torch.empty_like(SS)   # the carried gradient entering each chunk
        do = do.contiguous()
        prec, (_, scan_warps, local_warps) = _plan((u, w, qg, a, kg, dec))
        _scan_bwd[(B * H,)](w, qg, a, kg, dec, SS, do,
                            dSN.contiguous() if dSN is not None else u,
                            du, dw, dSK, dS0, NT, BT=BT, K=K, V=V,
                            HAS_DSN=dSN is not None, PREC=prec,
                            num_warps=scan_warps, num_stages=STAGES)
        _scan_bwd_local[(B * H * NT,)](u, w, SS, do, dSK, dqg, da, dkg, ddec,
                                       BT=BT, K=K, V=V, PREC=prec,
                                       num_warps=local_warps, num_stages=STAGES)
        return du, dw, dqg, da, dkg, ddec, dS0 if ctx.wants_s0 else None


def supported(u, w):
    """tl.dot wants power-of-two tiles of at least 16 in every dimension."""
    dims = (u.size(-2), u.size(-1), w.size(-1))
    return (ENABLED and u.is_cuda
            and all(d >= 16 and not d & (d - 1) for d in dims))


def chunk_scan(u, w, qg, a, kg, dec, initial_state):
    return _ChunkScan.apply(u, w, qg, a, kg, dec, initial_state)
