"""Draw the model architecture (model.py) in the style of Vaswani et al. Figure 1.

Decoder-only transformer: tied embedding -> N x [RMSNorm -> GQA -> + -> RMSNorm ->
SwiGLU -> +] -> final RMSNorm -> tied Linear -> logit soft-cap -> Softmax.
Side connections: gated x0 skip, gated mirror (U-net) skip, residuals, value residual.

This version expands the condensed blocks: attention is drawn as its full internal
pipeline (projections -> QK-RMSNorm -> partial RoPE -> value-residual mix -> masked
SDPA -> out projection) and the FFN as the actual SwiGLU dataflow (W_gate -> SiLU and
W_up -> elementwise multiply -> dropout -> W_down). Skip gates, tensor shapes, head
dims and init conventions are annotated, plus a color legend.

Output: architecture.png / architecture.pdf
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle

plt.rcParams["font.family"] = "serif"

PINK = "#FADBD8"      # embedding
YELLOW = "#FCF3CF"    # norms
ORANGE = "#FDEBD0"    # attention
BLUE = "#D6EAF8"      # feed-forward
PURPLE = "#E8DAEF"    # linear
TEAL = "#D1F2EB"      # soft-cap
GREEN = "#D5F5E3"     # softmax
GRAY = "#F4F6F6"      # stack background
WHITE = "#FFFFFF"     # sub-boxes inside containers

CX = 62  # main column center

fig, ax = plt.subplots(figsize=(15, 22))
ax.set_xlim(0, 150)
ax.set_ylim(-3, 216)
ax.axis("off")


def box(cx, y0, w, h, fc, lw=2.0, zorder=2, alpha=1.0):
    ax.add_patch(FancyBboxPatch((cx - w / 2, y0), w, h,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fc, ec="black", lw=lw, zorder=zorder, alpha=alpha))


def txt(x, y, s, fs=10, bold=False, italic=False, **kw):
    kw.setdefault("ha", "center")
    kw.setdefault("va", "center")
    ax.text(x, y, s, fontsize=fs, zorder=6,
            weight="bold" if bold else "normal",
            style="italic" if italic else "normal", **kw)


def arrow(p1, p2, lw=2.2, dashed=False):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=20,
                                 color="black", lw=lw, shrinkA=0, shrinkB=0,
                                 linestyle=(0, (4, 3)) if dashed else "-",
                                 zorder=4))


def path(points, dashed=False, lw=2.2):
    """Polyline through points with an arrowhead on the last segment."""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, color="black", lw=lw,
            ls=(0, (4, 3)) if dashed else "-", solid_capstyle="round",
            solid_joinstyle="round", zorder=3)
    arrow(points[-2], points[-1], lw=0.1, dashed=dashed)


def add_node(x, y):
    ax.add_patch(Circle((x, y), 1.6, fc="white", ec="black", lw=2.2, zorder=5))
    ax.text(x, y, "+", fontsize=15, weight="bold", ha="center", va="center", zorder=6)


def gate(x, y):
    ax.add_patch(Circle((x, y), 1.6, fc=TEAL, ec="black", lw=2.0, zorder=5))
    ax.text(x, y, "σ", fontsize=13, weight="bold", ha="center", va="center", zorder=6)


def mul_node(x, y):
    ax.add_patch(Circle((x, y), 1.6, fc="white", ec="black", lw=2.2, zorder=5))
    ax.text(x, y, "⊙", fontsize=14, weight="bold", ha="center", va="center", zorder=6)


# ---------------- title / caption ----------------
txt(70, 212, "Decoder-Only Transformer — Model Architecture", 14, bold=True)
txt(75, -1.8, "defaults: d_model=512 · N=30 layers · 8 Q heads / 2 KV heads (d_k=64) · d_ff=1408 · "
              "tied embeddings · zero-init attn-out & FFN-down projections", 9)

# ---------------- inputs / embedding ----------------
txt(CX, 2.5, "Inputs (token ids) — shape (B, T)", 10)
arrow((CX, 4), (CX, 6.5))
box(CX, 6.5, 38, 5.5, PINK)
txt(CX, 9.25, "Token Embedding\n(vocab × d_model, tied weights)", 10)

txt(122, 9.25, "no additive positional encoding —\npositions enter via partial RoPE\ninside attention",
    8, italic=True, ha="left")

# ---------------- decoder stack background (N x, looped L times) ----------------
ax.add_patch(FancyBboxPatch((24, 16), 96, 120,
                            boxstyle="round,pad=0,rounding_size=2.0",
                            fc=GRAY, ec="black", lw=2.5, zorder=1))
txt(14, 80, "N×", 16, bold=True)
txt(14, 72, "loop L×\n(depth recurrence,\nshared weights)", 8)

# ---------------- layer-input add node (x + gated skips) ----------------
arrow((CX, 12), (CX, 18.4))
add_node(CX, 20)

# gated x0 skip: embedding output -> every layer input (left)
path([(43, 9.25), (8, 9.25), (8, 20), (50, 20)])
gate(53, 20)
arrow((54.6, 20), (60.4, 20))
txt(53, 16.9, "x0", 6.5)
txt(5, 14.6, "gated x0 skip — embedding output → every layer input", 8, rotation=90)

# gated mirror (U-net) skip: first-half layer output -> second-half layer input (right)
path([(CX, 132), (132, 132), (132, 20), (73.4, 20)], dashed=True)
gate(70, 20)
arrow((68.4, 20), (63.6, 20))
txt(70, 16.9, "unet", 6.5)
txt(136, 90, "gated mirror (U-net) skip — second-half layer i ← output of layer N−1−i", 8, rotation=90)
txt(36, 13.8, "all skip gates: σ(g), g init −1.5 (σ ≈ 0.18)", 7.5, italic=True)

# ---------------- RMSNorm 1 ----------------
arrow((CX, 21.6), (CX, 23.5))
box(CX, 23.5, 36, 4.5, YELLOW)
txt(CX, 25.75, "RMSNorm (pre-norm)", 10)

# split into Q/K/V arrows like the original figure
arrow((CX, 28), (CX, 30.5))
ax.plot([CX, CX], [30.5, 31.5], color="black", lw=2.2, zorder=3)
ax.plot([48, 76], [31.5, 31.5], color="black", lw=2.2, zorder=3)
for x in (48, 62, 76):
    arrow((x, 31.5), (x, 33))
txt(49.8, 32.3, "Q", 7.5, ha="left")
txt(63.8, 32.3, "K", 7.5, ha="left")
txt(77.8, 32.3, "V", 7.5, ha="left")

# ---------------- attention sublayer (expanded pipeline) ----------------
box(CX, 33, 52, 42, ORANGE, lw=2.2, zorder=1.5, alpha=0.55)
txt(CX, 72.5, "Grouped-Query Attention (GQA)", 11, bold=True)

# projections
box(CX, 34.5, 44, 4, WHITE)
txt(CX, 36.5, "Linear projections\nQ: d→8×64 · K,V: d→2×64", 8.5)
arrow((CX, 38.5), (CX, 40), lw=1.6)

# QK norm
box(CX, 40, 44, 4, WHITE)
txt(CX, 42, "QK-RMSNorm (per-head, over d_k=64)", 8.5)
arrow((CX, 44), (CX, 45.5), lw=1.6)

# partial RoPE
box(CX, 45.5, 44, 4, WHITE)
txt(CX, 47.5, "Partial RoPE on Q, K — rotate 32 of 64 dims", 8.5)
arrow((CX, 49.5), (CX, 51), lw=1.6)

# value residual mix
box(CX, 51, 44, 4, WHITE)
txt(CX, 53, "Value residual (ResFormer): V ← (1−σ(θ))·V + σ(θ)·v₁\n(v₁ = layer-0 values · θ init 0 ⇒ σ=0.5)", 8)
path([(30, 53), (39.8, 53)], dashed=True)
txt(34.5, 49.8, "v₁ from layer 0", 6.5, italic=True)
arrow((CX, 55), (CX, 56.5), lw=1.6)

# scaled dot-product attention
box(CX, 56.5, 44, 7, WHITE)
txt(CX, 60, "Scaled Dot-Product Attention\nsoftmax(QKᵀ/√d_k)·V — causal mask\n1 KV head shared per 4 Q heads · flash SDPA", 8.5)
arrow((CX, 63.5), (CX, 65), lw=1.6)

# output projection
box(CX, 65, 44, 4, WHITE)
txt(CX, 67, "Concat heads → output projection (zero-init)", 8.5)

arrow((CX, 75), (CX, 77.4))
add_node(CX, 79)
txt(68, 82, "x ← x + dropout(attn)", 8, ha="left")

# residual 1 (around norm+attention), right side
path([(CX, 22.5), (100, 22.5), (100, 79), (63.8, 79)])

# ---------------- RMSNorm 2 ----------------
arrow((CX, 80.6), (CX, 83))
box(CX, 83, 36, 4.5, YELLOW)
txt(CX, 85.25, "RMSNorm (pre-norm)", 10)

# ---------------- SwiGLU FFN (expanded dataflow) ----------------
arrow((CX, 87.5), (CX, 90))
box(CX, 90, 52, 34, BLUE, lw=2.2, zorder=1.5, alpha=0.55)
txt(CX, 121.5, "SwiGLU Feed-Forward", 11, bold=True)

# branch split into W_gate / W_up
ax.plot([CX, CX], [90, 91.5], color="black", lw=2.2, zorder=3)
ax.plot([50, 74], [91.5, 91.5], color="black", lw=2.2, zorder=3)
arrow((50, 91.5), (50, 93))
arrow((74, 91.5), (74, 93))

box(50, 93, 17, 4.5, WHITE)
txt(50, 95.25, "W_gate\nd→d_ff", 8.5)
box(74, 93, 17, 4.5, WHITE)
txt(74, 95.25, "W_up\nd→d_ff", 8.5)

txt(92, 95.25, "d_ff = ⌈8/3·d_model⌉₆₄ = 1408\n(param-matched to a\n4·d 2-matmul FFN)", 7.5, italic=True, ha="left")

arrow((50, 97.5), (50, 99.5), lw=1.6)
box(50, 99.5, 17, 4, WHITE)
txt(50, 101.5, "SiLU", 8.5)

# elementwise multiply of SiLU(W_gate x) and (W_up x)
arrow((50, 103.5), (60, 105.5), lw=1.6)
path([(74, 97.5), (74, 106), (63.8, 106)], lw=1.6)
mul_node(CX, 106)
txt(66.5, 106, "elementwise\nmultiply", 7, ha="left")
arrow((CX, 107.6), (CX, 109.5), lw=1.6)

box(CX, 109.5, 44, 3.5, WHITE)
txt(CX, 111.25, "Dropout", 8.5)
arrow((CX, 113), (CX, 114.5), lw=1.6)

box(CX, 114.5, 44, 4, WHITE)
txt(CX, 116.5, "W_down: d_ff→d (zero-init)", 8.5)

arrow((CX, 124), (CX, 124.9))
add_node(CX, 126.5)
txt(56, 129.5, "x ← x + dropout(ffn)", 8, ha="right")

# residual 2 (around norm+FFN), left side
path([(CX, 81.8), (28, 81.8), (28, 126.5), (60.4, 126.5)])

# ---------------- trunk after the stack ----------------
arrow((CX, 128.1), (CX, 140))

box(CX, 140, 36, 4.5, YELLOW)
txt(CX, 142.25, "RMSNorm (final)", 10)
txt(84, 146, "(B, T, d_model)", 8, italic=True, ha="left")

arrow((CX, 144.5), (CX, 147))

box(CX, 147, 40, 5, PURPLE)
txt(CX, 149.5, "LM head: z = x·W_eᵀ\n(weights tied to embedding)", 9.5)

arrow((CX, 152), (CX, 154.5))

box(CX, 154.5, 40, 4.5, TEAL)
txt(CX, 156.75, "logit soft-cap: z ← 15·tanh(z/15)", 9.5)
txt(86, 159.5, "(B, T, vocab)", 8, italic=True, ha="left")

arrow((CX, 159), (CX, 161.5))

box(CX, 161.5, 40, 4.5, GREEN)
txt(CX, 163.75, "Softmax", 10)

arrow((CX, 166), (CX, 168.5))
txt(CX, 171, "Output Probabilities — p(next token | context)", 11)

# ---------------- legend ----------------
ax.add_patch(FancyBboxPatch((98, 174), 50, 15,
                            boxstyle="round,pad=0,rounding_size=1.5",
                            fc="white", ec="black", lw=1.5, zorder=2))
txt(123, 186.5, "Legend", 9, bold=True)
legend_items = [
    (101, 183.5, PINK, "Embedding"),
    (101, 180.5, YELLOW, "RMSNorm"),
    (101, 177.5, ORANGE, "Attention"),
    (101, 175.5, BLUE, "Feed-Forward"),
    (124, 183.5, PURPLE, "Linear"),
    (124, 180.5, TEAL, "Soft-cap / gates"),
    (124, 177.5, GREEN, "Softmax"),
]
for x, y, c, label in legend_items:
    ax.add_patch(FancyBboxPatch((x, y - 0.8), 1.8, 1.6,
                                boxstyle="round,pad=0,rounding_size=0.3",
                                fc=c, ec="black", lw=1.0, zorder=3))
    txt(x + 2.8, y, label, 7.5, ha="left")

out_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(out_dir, "architecture.png"), dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig(os.path.join(out_dir, "architecture.pdf"), bbox_inches="tight", facecolor="white")
print("wrote architecture.png and architecture.pdf")
