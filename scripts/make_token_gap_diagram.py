"""Generate publication-grade Token Classification & Gap-Bridging Visual Diagram.
Illustrates Harshit Kandpal's subword token head posterior and gap-bridging algorithm (G <= 2).
Matches a16z style (white canvas #ffffff, navy #173462, teal #089582, purple #65327d).
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIGURES_DIR = REPO_ROOT / "reports" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = FIGURES_DIR / "token_gap_bridging.png"

C_WHITE = "#ffffff"
C_NAVY = "#173462"
C_BODY = "#2a3b5c"
C_TEAL = "#089582"
C_TEAL_BG = "#f0fdf4"
C_PURPLE = "#65327d"
C_PURPLE_BG = "#faf5ff"
C_ORANGE = "#d97706"
C_GREEN = "#16a34a"
C_GREEN_BG = "#f0fdf4"
C_RED = "#dc2626"
C_RED_BG = "#fef2f2"
C_GRAY_BOX = "#f8fafc"
C_BORDER_GRAY = "#cbd5e1"
C_LINE_GRAY = "#e2e8f0"


def main():
    fig, ax = plt.subplots(figsize=(10.5, 6.2), dpi=300)
    ax.set_facecolor(C_WHITE)
    fig.patch.set_facecolor(C_WHITE)
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 6.2)
    ax.axis("off")

    # =========================================================================
    # SECTION 1: Subword Token Stream & Head Posteriors P(inj)
    # =========================================================================
    tokens = ["[CLS]", "In", "##ject", "drop", "to", "the", "table", "[SEP]"]
    probs = [0.00, 0.91, 0.96, 0.88, 0.16, 0.22, 0.94, 0.00]
    tags = ["[MASK]", "B-INJ", "I-INJ", "I-INJ", "[GAP]", "[GAP]", "I-INJ", "[MASK]"]
    
    n_tok = len(tokens)
    start_x = 0.5
    col_w = 1.15
    bar_base_y = 4.0
    max_bar_h = 1.4

    # Container outline for Section 1
    s1_box = patches.FancyBboxPatch(
        (0.35, 3.5), 9.8, 2.5,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=C_WHITE, edgecolor=C_BORDER_GRAY, linewidth=1.2, zorder=1
    )
    ax.add_patch(s1_box)

    ax.text(0.55, 5.75, "1. DeBERTa-v3 Token Classification Head Posterior: P(inj) = P(B-INJ) + P(I-INJ)",
            fontsize=10.5, fontweight="bold", color=C_NAVY)

    # Threshold line at P = 0.35
    thresh_y = bar_base_y + 0.35 * max_bar_h
    ax.plot([0.5, 9.9], [thresh_y, thresh_y], color=C_RED, linestyle="--", linewidth=1.5, zorder=3)
    ax.text(10.0, thresh_y, "Threshold τ = 0.35", color=C_RED, fontsize=8, va="center", fontweight="bold")

    for i in range(n_tok):
        x = start_x + i * col_w
        p = probs[i]
        bar_h = p * max_bar_h
        
        # Bar color
        if p >= 0.35:
            bar_col = C_TEAL
        elif p > 0:
            bar_col = C_ORANGE
        else:
            bar_col = "#e2e8f0"

        # Draw probability bar
        if bar_h > 0:
            bar = patches.Rectangle((x + 0.25, bar_base_y), 0.5, bar_h, facecolor=bar_col, edgecolor="none", zorder=2)
            ax.add_patch(bar)
            ax.text(x + 0.5, bar_base_y + bar_h + 0.08, f"{p:.2f}", ha="center", fontsize=8, fontweight="bold", color=C_NAVY)
        else:
            ax.text(x + 0.5, bar_base_y + 0.08, "0.00", ha="center", fontsize=8, color=C_BODY)

        # Token Box
        tok_bg = C_PURPLE_BG if i in [0, 7] else (C_TEAL_BG if p >= 0.35 else C_GRAY_BOX)
        tok_border = C_PURPLE if i in [0, 7] else (C_TEAL if p >= 0.35 else C_BORDER_GRAY)
        tok_patch = patches.FancyBboxPatch(
            (x + 0.08, 3.6), 0.84, 0.32,
            boxstyle="round,pad=0.01,rounding_size=0.03",
            facecolor=tok_bg, edgecolor=tok_border, linewidth=1.0, zorder=2
        )
        ax.add_patch(tok_patch)
        ax.text(x + 0.5, 3.76, tokens[i], ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_NAVY)

    # =========================================================================
    # SECTION 2: Subword Fragmentation (Without Gap-Bridging)
    # =========================================================================
    s2_box = patches.FancyBboxPatch(
        (0.35, 2.0), 9.8, 1.35,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor="#fef2f2", edgecolor="#fca5a5", linewidth=1.2, zorder=1
    )
    ax.add_patch(s2_box)

    ax.text(0.55, 3.08, "2. Subword Tokenizer Fragmentation (Without Gap-Bridging: Naive Thresholding)",
            fontsize=10, fontweight="bold", color=C_RED)

    # Span 1 Box
    span1 = patches.FancyBboxPatch((1.65, 2.3), 3.4, 0.55, boxstyle="round,pad=0.01,rounding_size=0.04",
                                  facecolor=C_WHITE, edgecolor=C_RED, linewidth=1.5, zorder=2)
    ax.add_patch(span1)
    ax.text(3.35, 2.58, "Malicious Fragment 1: ['In', '##ject', 'drop']", ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_NAVY)

    # Missing Gap Box
    gap_box = patches.FancyBboxPatch((5.25, 2.3), 2.1, 0.55, boxstyle="round,pad=0.01,rounding_size=0.04",
                                    facecolor="#fee2e2", edgecolor=C_RED, linewidth=1.5, linestyle=":", zorder=2)
    ax.add_patch(gap_box)
    ax.text(6.3, 2.58, "LEAK! ['to', 'the'] (G=2)", ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_RED)

    # Span 2 Box
    span2 = patches.FancyBboxPatch((7.55, 2.3), 1.6, 0.55, boxstyle="round,pad=0.01,rounding_size=0.04",
                                  facecolor=C_WHITE, edgecolor=C_RED, linewidth=1.5, zorder=2)
    ax.add_patch(span2)
    ax.text(8.35, 2.58, "Fragment 2: ['table']", ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_NAVY)

    # =========================================================================
    # SECTION 3: Harshit Kandpal Subword Gap-Bridging (G <= 2)
    # =========================================================================
    s3_box = patches.FancyBboxPatch(
        (0.35, 0.2), 9.8, 1.65,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=C_TEAL_BG, edgecolor=C_TEAL, linewidth=1.5, zorder=1
    )
    ax.add_patch(s3_box)

    ax.text(0.55, 1.58, "3. Harshit Kandpal's Subword Gap-Bridging Engine (G ≤ 2 Connective Tokens)",
            fontsize=10.5, fontweight="bold", color=C_TEAL)

    # Unified Span Box
    bridge_box = patches.FancyBboxPatch((1.65, 0.5), 7.5, 0.85, boxstyle="round,pad=0.01,rounding_size=0.05",
                                        facecolor=C_WHITE, edgecolor=C_TEAL, linewidth=2.0, zorder=2)
    ax.add_patch(bridge_box)

    ax.text(5.4, 1.05, "Synthesized Continuous Malicious Span: ['In', '##ject', 'drop', 'to', 'the', 'table']",
            ha="center", va="center", fontsize=10, fontweight="bold", color=C_PURPLE)
    ax.text(5.4, 0.72, "Exact Round-Trip Character Span: document[42:78]  ·  100% Attack Coverage  ·  Zero Fragmentation Leaks",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color=C_TEAL)

    plt.tight_layout()
    plt.savefig(str(OUT_PATH), dpi=300, facecolor=C_WHITE, edgecolor="none")
    plt.close()
    print(f"Token Gap-Bridging diagram saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
