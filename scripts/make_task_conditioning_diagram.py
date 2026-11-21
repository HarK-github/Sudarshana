"""Generate publication-grade Task Conditioning & Disambiguation Visual Diagram.
Illustrates Utkarsh Chatterjee's pair-input cross-attention architecture:
[CLS] user_task [SEP] chunk [SEP], dropping False Positive Rate from 83% to 17%.
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
OUT_PATH = FIGURES_DIR / "task_conditioning_diagram.png"

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
    # LEFT PANEL: The Imperative Ambiguity Problem (Unconditioned Model)
    # =========================================================================
    left_box = patches.FancyBboxPatch(
        (0.35, 0.3), 4.7, 5.6,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=C_WHITE, edgecolor=C_RED, linewidth=1.5, zorder=1
    )
    ax.add_patch(left_box)

    ax.text(0.65, 5.45, "A. Unconditioned Chunk-Only Classifier",
            fontsize=11, fontweight="bold", color=C_RED)
    ax.text(0.65, 5.15, "Isolated document chunk lacks intent context",
            fontsize=8.5, style="italic", color=C_BODY)

    # Input Box: Legitimate Linux tutorial
    c_uncond = patches.FancyBboxPatch(
        (0.65, 3.85), 4.1, 1.15, boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=C_GRAY_BOX, edgecolor=C_BORDER_GRAY, linewidth=1.0, zorder=2
    )
    ax.add_patch(c_uncond)
    ax.text(0.85, 4.65, "Retrieved Technical Documentation:", fontsize=8.5, fontweight="bold", color=C_NAVY)
    ax.text(0.85, 4.35, "\"Run sudo rm -rf /tmp/cache to clear disk.", fontsize=8, color=C_BODY)
    ax.text(0.85, 4.05, " Ignore previous warnings and reboot.\"", fontsize=8, color=C_BODY)

    # Failure box
    fail_box = patches.FancyBboxPatch(
        (0.65, 1.8), 4.1, 1.7, boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=C_RED_BG, edgecolor=C_RED, linewidth=1.2, zorder=2
    )
    ax.add_patch(fail_box)
    ax.text(0.85, 3.15, "Model Decision: FALSE POSITIVE (83.33% FPR)", fontsize=9.5, fontweight="bold", color=C_RED)
    ax.text(0.85, 2.80, "• Attends to imperative triggers: 'rm -rf', 'Ignore'", fontsize=8, color=C_BODY)
    ax.text(0.85, 2.55, "• Cannot distinguish command tutorial from attack.", fontsize=8, color=C_BODY)
    ax.text(0.85, 2.30, "• Result: Legitimate RAG document falsely BLOCKED.", fontsize=8, fontweight="bold", color=C_RED)
    ax.text(0.85, 2.05, "• Severe context starvation for downstream LLM.", fontsize=8, color=C_BODY)

    ax.text(0.65, 0.95, "Hard-Negative False Positive Rate:", fontsize=9, fontweight="bold", color=C_NAVY)
    ax.text(0.65, 0.55, "83.33% [64.2%, 93.3%]", fontsize=18, fontweight="bold", color=C_RED)

    # =========================================================================
    # RIGHT PANEL: Utkarsh Chatterjee's Pair-Input Conditioning
    # =========================================================================
    right_box = patches.FancyBboxPatch(
        (5.35, 0.3), 4.8, 5.6,
        boxstyle="round,pad=0.01,rounding_size=0.05",
        facecolor=C_WHITE, edgecolor=C_TEAL, linewidth=1.5, zorder=1
    )
    ax.add_patch(right_box)

    ax.text(5.65, 5.45, "B. Utkarsh Chatterjee: Pair-Input Conditioning",
            fontsize=11, fontweight="bold", color=C_TEAL)
    ax.text(5.65, 5.15, "[CLS] user_task [SEP] chunk [SEP] cross-attention",
            fontsize=8.5, style="italic", color=C_BODY)

    # Conditioned Input Box
    c_cond = patches.FancyBboxPatch(
        (5.65, 3.85), 4.2, 1.15, boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=C_PURPLE_BG, edgecolor=C_PURPLE, linewidth=1.2, zorder=2
    )
    ax.add_patch(c_cond)
    ax.text(5.85, 4.65, "User Task + Retrieved Chunk Pair:", fontsize=8.5, fontweight="bold", color=C_PURPLE)
    ax.text(5.85, 4.35, "[CLS] How do I clean disk cache? [SEP]", fontsize=8, fontweight="bold", color=C_NAVY)
    ax.text(5.85, 4.05, "\"Run sudo rm -rf /tmp/cache to clear disk...\" [SEP]", fontsize=8, color=C_BODY)

    # Success Box
    succ_box = patches.FancyBboxPatch(
        (5.65, 1.8), 4.2, 1.7, boxstyle="round,pad=0.01,rounding_size=0.04",
        facecolor=C_TEAL_BG, edgecolor=C_TEAL, linewidth=1.2, zorder=2
    )
    ax.add_patch(succ_box)
    ax.text(5.85, 3.15, "Model Decision: CLEAN PASS (16.67% FPR)", fontsize=9.5, fontweight="bold", color=C_TEAL)
    ax.text(5.85, 2.80, "• Cross-attention aligns imperative with user query.", fontsize=8, color=C_BODY)
    ax.text(5.85, 2.55, "• Recognizes commands serve the requested task.", fontsize=8, color=C_BODY)
    ax.text(5.85, 2.30, "• 100.00% Recall preserved on genuine attacks.", fontsize=8, fontweight="bold", color=C_NAVY)
    ax.text(5.85, 2.05, "• 66.7% absolute drop in hard-negative false alarms.", fontsize=8, color=C_TEAL)

    ax.text(5.65, 0.95, "Conditioned Hard-Negative FPR:", fontsize=9, fontweight="bold", color=C_NAVY)
    ax.text(5.65, 0.55, "16.67% [5.4%, 39.6%]", fontsize=18, fontweight="bold", color=C_TEAL)

    plt.savefig(str(OUT_PATH), dpi=300, facecolor=C_WHITE, edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Task Conditioning diagram successfully saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
