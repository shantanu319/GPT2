"""Draw the model architecture (model.py) in the style of Vaswani et al. Figure 1.

Decoder-only transformer: tied embedding -> N x [RMSNorm -> GQA -> + -> RMSNorm ->
SwiGLU -> +] -> final RMSNorm -> tied Linear -> logit soft-cap -> Softmax.
Side connections: gated x0 skip, gated mirror (U-net) skip, residuals.

Output: architecture.png / architecture.pdf
"""

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
GREEN = "#D5F5E3"     # softmax
TEAL = "#D1F2EB"      # soft-cap
GRAY = "#F4F6F6"      # stack background

CX = 46  # main column center

fig, ax = plt.subplots(figsize=(10, 15))
ax.set_xlim(0, 100)
ax.set_ylim(-2, 134)
ax.axis("off")


def box(cx, y0, w, h, fc, lw=2.2, zorder=2):
    ax.add_patch(FancyBboxPatch((cx - w / 2, y0), w, h,
                                boxstyle="round,pad=0,rounding_size=1.0",
                                fc=fc, ec="black", lw=lw, zorder=zorder))


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


# ---------------- title / caption ----------------
txt(50, 131, "Decoder-Only Transformer — Model Architecture", 13, bold=True)
txt(50, -1.2, "defaults: d_model=512 · N=30 layers · 8 Q heads / 2 KV heads · tied embeddings", 8.5)

# ---------------- inputs / embedding ----------------
txt(CX, 3, "Inputs (token ids)", 10)
arrow((CX, 4.5), (CX, 7))
box(CX, 7, 30, 5, PINK)
txt(CX, 9.5, "Token Embedding\n(tied weights)", 10)

txt(71, 10, "no additive positional encoding —\npositions enter via partial RoPE\ninside attention", 8, italic=True, ha="left")

# ---------------- decoder stack background (N x) ----------------
ax.add_patch(FancyBboxPatch((25, 18), 44, 72,
                            boxstyle="round,pad=0,rounding_size=2.0",
                            fc=GRAY, ec="black", lw=2.5, zorder=1))
txt(17, 57, "N×", 15, bold=True)
txt(17, 52, "loop L×\n(depth recurrence)", 8)

# ---------------- main flow into the stack ----------------
arrow((CX, 12), (CX, 20.5))

# RMSNorm 1
box(CX, 20.5, 30, 5, YELLOW)
txt(CX, 23, "RMSNorm", 10)

# split into Q/K/V arrows like the original figure
arrow((CX, 25.5), (CX, 28.5))
ax.plot([CX, CX], [28.5, 29.8], color="black", lw=2.2, zorder=3)
ax.plot([40, 52], [29.8, 29.8], color="black", lw=2.2, zorder=3)
for x in (40, 46, 52):
    arrow((x, 29.8), (x, 31))

# attention sublayer
box(CX, 31, 30, 16, ORANGE)
txt(CX, 44, "Grouped-Query Attention", 10.5, bold=True)
txt(CX, 41.2, "causal mask · 8 Q heads / 2 KV heads", 8.5)
txt(CX, 38.6, "QK-RMSNorm · partial RoPE (½ dims)", 8.5)
txt(CX, 36, "value residual: blend layer-0 V", 8.5)
txt(CX, 33.4, "(flash SDPA)", 8, italic=True)

arrow((CX, 47), (CX, 50))
add_node(CX, 51.5)

# residual 1 (around norm+attention), right side
path([(CX, 19.2), (64, 19.2), (64, 51.5), (47.8, 51.5)])

arrow((CX, 53.2), (CX, 55))

# RMSNorm 2
box(CX, 55, 30, 5, YELLOW)
txt(CX, 57.5, "RMSNorm", 10)

arrow((CX, 60), (CX, 63))

# SwiGLU FFN
box(CX, 63, 30, 12, BLUE)
txt(CX, 72, "SwiGLU Feed-Forward", 10.5, bold=True)
txt(CX, 69.3, "SiLU(x·W_gate) ⊙ (x·W_up)", 8.5)
txt(CX, 66.8, "· W_down   (d_ff = 8/3 · d_model)", 8.5)
txt(CX, 64.5, "output projections zero-init", 8, italic=True)

arrow((CX, 75), (CX, 78))
add_node(CX, 79.5)

# residual 2 (around norm+FFN), left side
path([(CX, 53.8), (28, 53.8), (28, 79.5), (44.2, 79.5)])

# ---------------- side connections ----------------
# gated x0 skip: embedding output -> every layer input (left)
path([(31, 9.5), (8, 9.5), (8, 22), (25, 22)])
ax.text(5.5, 15.8, "gated x0 skip\n(every layer input)", rotation=90,
        fontsize=8, ha="center", va="center")

# gated mirror (U-net) skip: first-half layer output -> second-half layer input (right, outside)
path([(69, 84), (76, 84), (76, 23), (69.4, 23)], dashed=True)
ax.text(79, 53.5, "gated mirror skip\n(second half ← mirrored\nfirst-half layer)",
        rotation=90, fontsize=8, ha="center", va="center")

# ---------------- trunk after the stack ----------------
arrow((CX, 81.2), (CX, 92))

box(CX, 92, 30, 5, YELLOW)
txt(CX, 94.5, "RMSNorm (final)", 10)

arrow((CX, 97), (CX, 100))

box(CX, 100, 30, 5, PURPLE)
txt(CX, 102.5, "Linear (tied to embedding)", 10)

arrow((CX, 105), (CX, 108))

box(CX, 108, 30, 5, TEAL)
txt(CX, 110.5, "logit soft-cap: 15·tanh(z/15)", 9.5)

arrow((CX, 113), (CX, 116))

box(CX, 116, 30, 5, GREEN)
txt(CX, 118.5, "Softmax", 10)

arrow((CX, 121), (CX, 124.5))
txt(CX, 126.8, "Output Probabilities", 11)

plt.savefig("architecture.png", dpi=300, bbox_inches="tight", facecolor="white")
plt.savefig("architecture.pdf", bbox_inches="tight", facecolor="white")
print("wrote architecture.png and architecture.pdf")
