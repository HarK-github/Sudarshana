"""Generate publication-grade pipeline flowchart for Harshit Kandpal's
Token-Level Detection and Structure-Aware Redaction Engine.
Matches the white canvas #ffffff and a16z color theme.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIGURES_DIR / "p4_pipeline_flowchart.png"

# Palette
C_WHITE = "#ffffff"
C_NAVY = "#173462"
C_BODY = "#2a3b5c"
C_TEAL = "#089582"
C_TEAL_BG = "#f0fdf4"
C_PURPLE = "#65327d"
C_PURPLE_BG = "#faf5ff"
C_ORANGE = "#d97706"
C_ORANGE_BG = "#fffbeb"
C_GREEN = "#16a34a"
C_GREEN_BG = "#f0fdf4"
C_RED = "#dc2626"
C_RED_BG = "#fef2f2"
C_GRAY_BOX = "#f8fafc"
C_BORDER_GRAY = "#cbd5e1"
C_ARROW = "#475569"


def draw_box(ax, x, y, w, h, title, subtitle="", bg=C_WHITE, border=C_BORDER_GRAY, title_col=C_NAVY, sub_col=C_BODY, lw=1.5, radius=0.08):
    box = patches.FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.01,rounding_size={radius}",
        facecolor=bg,
        edgecolor=border,
        linewidth=lw,
        zorder=2
    )
    ax.add_patch(box)
    
    cx = x + w / 2.0
    if subtitle:
        cy_title = y + h * 0.65
        cy_sub = y + h * 0.32
        ax.text(cx, cy_title, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=title_col, zorder=3)
        ax.text(cx, cy_sub, subtitle, ha="center", va="center", fontsize=8, color=sub_col, zorder=3)
    else:
        cy = y + h / 2.0
        ax.text(cx, cy, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=title_col, zorder=3)
    return box


def draw_arrow(ax, x1, y1, x2, y2, label="", col=C_ARROW):
    ax.annotate(
        "",
        xy=(x2, y2), xycoords="data",
        xytext=(x1, y1), textcoords="data",
        arrowprops=dict(
            arrowstyle="->,head_width=0.35,head_length=0.45",
            color=col,
            lw=1.5,
            shrinkA=2,
            shrinkB=2,
        ),
        zorder=1
    )
    if label:
        mx = (x1 + x2) / 2.0
        my = (y1 + y2) / 2.0
        ax.text(mx + 0.05, my, label, ha="left", va="center", fontsize=7.5, color=col, fontweight="bold", zorder=3)


def main():
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    ax.set_facecolor(C_WHITE)
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # 1. Input Box (Top Left)
    draw_box(ax, 0.4, 4.85, 2.3, 0.95, "Raw Retrieved Document", "Arbitrary token length (no truncation)", bg=C_GRAY_BOX, border=C_BORDER_GRAY)
    draw_arrow(ax, 2.7, 5.325, 3.2, 5.325)

    # 2. Sliding Windows Box
    draw_box(ax, 3.2, 4.85, 2.5, 0.95, "Sliding Windows & Stride", "256 tokens / 64-token overlap", bg=C_WHITE, border=C_PURPLE, title_col=C_PURPLE)
    draw_arrow(ax, 5.7, 5.325, 6.2, 5.325)

    # 3. DeBERTa Token Head Box
    draw_box(ax, 6.2, 4.85, 3.9, 0.95, "DeBERTa-v3 Token Classifier", "Head 384 → (O, B-INJ, I-INJ) · Class Weight w=[1,8,8]", bg=C_PURPLE_BG, border=C_PURPLE, title_col=C_PURPLE)

    # Arrow Down to Stage 2
    draw_arrow(ax, 8.15, 4.85, 8.15, 4.25)

    # 4. Sliding Window Max-Pooling
    draw_box(ax, 5.7, 3.3, 4.4, 0.95, "Sliding-Window Max-Pooling", "P(inj) = max(P(B-INJ) + P(I-INJ)) across windows", bg=C_WHITE, border=C_TEAL, title_col=C_TEAL)
    draw_arrow(ax, 5.7, 3.775, 5.2, 3.775)

    # 5. Gap Bridging Box
    draw_box(ax, 0.4, 3.3, 4.8, 0.95, "Threshold & Subword Gap-Bridging", "P ≥ 0.35 · Bridges unflagged connective tokens (G ≤ 2)", bg=C_TEAL_BG, border=C_TEAL, title_col=C_TEAL)

    # Arrow Down to Structure-Aware Expansion
    draw_arrow(ax, 2.8, 3.3, 2.8, 2.7)

    # 6. Center Branching Header for Structure-Aware Expansion
    draw_box(ax, 0.4, 2.05, 9.7, 0.65, "Hierarchical Structure-Aware Container Expander", "Detects enclosing syntactic container to prevent syntax mutilation (capped at max 2.5x raw span)", bg=C_ORANGE_BG, border=C_ORANGE, title_col=C_ORANGE)

    # Sub-containers below (4 boxes)
    draw_box(ax, 0.4, 1.15, 2.2, 0.70, "Markdown Tables", "Row Bounding [\\n, \\n]", bg=C_WHITE, border=C_ORANGE, title_col=C_NAVY)
    draw_box(ax, 2.9, 1.15, 2.3, 0.70, "Code & Docstrings", "Enclosing Block (```, ''', \"\"\")", bg=C_WHITE, border=C_ORANGE, title_col=C_NAVY)
    draw_box(ax, 5.5, 1.15, 2.2, 0.70, "Comments", "Contiguous # / // lines", bg=C_WHITE, border=C_ORANGE, title_col=C_NAVY)
    draw_box(ax, 8.0, 1.15, 2.1, 0.70, "Natural Prose", "Sentence in paragraph (\\n\\n)", bg=C_WHITE, border=C_ORANGE, title_col=C_NAVY)

    for sub_x in [1.5, 4.05, 6.6, 9.05]:
        draw_arrow(ax, sub_x, 2.05, sub_x, 1.85)

    # Convergence to Final Decision (Bottom)
    draw_arrow(ax, 5.25, 1.15, 5.25, 0.75)

    # Final Decision Box: Fallback vs Sanitized
    draw_box(ax, 0.4, 0.15, 4.6, 0.55, "Excised > 50% → [BLOCKED DOCUMENT]", "Safety fallback prevents returning gibberish", bg=C_RED_BG, border=C_RED, title_col=C_RED, radius=0.06)
    draw_box(ax, 5.5, 0.15, 4.6, 0.55, "Excised ≤ 50% → Sanitized Context to LLM", "Preserves 57.84% to 76.19% clean carrier text", bg=C_GREEN_BG, border=C_GREEN, title_col=C_GREEN, radius=0.06)

    plt.tight_layout()
    plt.savefig(str(OUT_PATH), dpi=300, facecolor=C_WHITE, edgecolor="none")
    plt.close()
    print(f"Flowchart image successfully created at: {OUT_PATH}")


if __name__ == "__main__":
    main()
