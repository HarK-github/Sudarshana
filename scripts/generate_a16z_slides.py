"""Generate publication-grade 'a16z Style' Presentation Deck for Project Sudarshana.
Unofficial homage to the State of Crypto report slides:
- Canvas: Pure White #ffffff
- Typography: 'Noto Sans' (Google Fonts)
  - Headlines: 700 at 27-30px (Cover: 64px)
  - Subtitles: 400 italic 12px
  - Body: 400 at 12-14px
  - Footnotes: 10px #5f6b80
- Palette:
  - Text: Navy #173462 headlines, #2a3b5c body, #5f6b80 notes/sources
  - Mint Panel: gradient #c5e9df to #fcf8d5 with 4px edge #3fb59a to #d6dc6c
  - Orchid Panel: gradient #ffe6f1 to #efd8f6 with 4px edge #e596c3 to #8a3fa0
  - Highlighter Blocks (strictly one per headline):
    - Lime #e2fe6b with navy #173462 text on mint
    - Magenta #ea2ce8 with white #ffffff text on orchid
  - Series Colors: Purple #65327d, Teal #089582, Green #62b33a, Orange #f0a656, Gray #b8bfcb, Lilac #b5a0bf
- Structure:
  - Slide 1: Cover (Mint panel left, title 64px with highlighted year, authors, ringed bubbles on white canvas)
  - Slide 2: [PLACEHOLDER] P1 Data Pipeline & Benchmark (Tanmay Jaiswal)
  - Slide 3: [PLACEHOLDER] P2 Pair-Input Conditioning (Utkarsh Chatterjee)
  - Slide 4: [PLACEHOLDER] P3 Multi-Task Model & Ablation Ladder (Sujal Som)
  - Slide 5: P4 Token Head & Subword Gap-Bridging (Harshit Kandpal - FULL CONTENT)
  - Slide 6: P4 Structure-Aware Redaction Engine (Harshit Kandpal - FULL CONTENT)
  - Slide 7: P4 Empirical Audit Scorecard & Utility Gains (Harshit Kandpal - FULL CONTENT)
  - Slide 8: P4 Error Taxonomy & Dataset Defense (Harshit Kandpal - FULL CONTENT)
  - Slide 9: [PLACEHOLDER] P5 Downstream Evaluation & UI (Kinshuk Gupta)
  - Slide 10: Conclusion & Defense Readiness (Unified Summary)
"""

import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# -----------------------------------------------------------------------------
# Color Constants
# -----------------------------------------------------------------------------
C_WHITE = RGBColor(255, 255, 255)
C_NAVY = RGBColor(23, 52, 98)         # #173462
C_BODY = RGBColor(42, 59, 92)         # #2a3b5c
C_NOTES = RGBColor(95, 107, 128)      # #5f6b80
C_FOOTER = RGBColor(138, 147, 163)    # #8a93a3
C_GRID = RGBColor(230, 233, 238)      # #e6e9ee

# Panel Fills & Borders
C_MINT_TOP = RGBColor(197, 233, 223)  # #c5e9df
C_MINT_EDGE = RGBColor(63, 181, 154)  # #3fb59a
C_LIME_HL = RGBColor(226, 254, 107)   # #e2fe6b

C_ORCHID_TOP = RGBColor(255, 230, 241)# #ffe6f1
C_ORCHID_EDGE = RGBColor(229, 150, 195)# #e596c3
C_MAGENTA_HL = RGBColor(234, 44, 232) # #ea2ce8

# Series Colors
C_PURPLE = RGBColor(101, 50, 125)     # #65327d
C_TEAL = RGBColor(8, 149, 130)        # #089582
C_GREEN = RGBColor(98, 179, 58)       # #62b33a
C_ORANGE = RGBColor(240, 166, 86)     # #f0a656
C_GRAY = RGBColor(184, 191, 203)      # #b8bfcb
C_LILAC = RGBColor(181, 160, 191)     # #b5a0bf


def create_pptx_deck():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_white_bg(slide):
        bg = slide.background
        fill = bg.fill
        fill.solid()
        fill.fore_color.rgb = C_WHITE

    def add_panel(slide, style="mint", width_in=3.6):
        # Panel Rect
        panel = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(width_in), Inches(7.5))
        panel.fill.solid()
        panel.fill.fore_color.rgb = C_MINT_TOP if style == "mint" else C_ORCHID_TOP
        panel.line.fill.background()

        # Gradient Edge (Separator line)
        edge = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(width_in), Inches(0), Inches(0.045), Inches(7.5))
        edge.fill.solid()
        edge.fill.fore_color.rgb = C_MINT_EDGE if style == "mint" else C_ORCHID_EDGE
        edge.line.fill.background()
        return panel

    def add_footer(slide, page_num, source_str, notes_str=""):
        # Footer text bottom-left: '©2026 Sudarshana'
        tb_c = slide.shapes.add_textbox(Inches(0.4), Inches(7.15), Inches(2.5), Inches(0.25))
        tf_c = tb_c.text_frame
        p_c = tf_c.paragraphs[0]
        p_c.text = "©2026  Sudarshana"
        p_c.font.name = "Noto Sans"
        p_c.font.size = Pt(9.5)
        p_c.font.color.rgb = C_FOOTER

        # Notes bottom-left of exhibit
        if notes_str:
            tb_n = slide.shapes.add_textbox(Inches(3.8), Inches(7.15), Inches(4.5), Inches(0.25))
            tf_n = tb_n.text_frame
            p_n = tf_n.paragraphs[0]
            p_n.text = notes_str
            p_n.font.name = "Noto Sans"
            p_n.font.size = Pt(9)
            p_n.font.color.rgb = C_NOTES

        # Sources bottom-right
        tb_s = slide.shapes.add_textbox(Inches(7.2), Inches(7.15), Inches(5.3), Inches(0.25))
        tf_s = tb_s.text_frame
        p_s = tf_s.paragraphs[0]
        p_s.text = source_str
        p_s.font.name = "Noto Sans"
        p_s.font.size = Pt(9)
        p_s.font.color.rgb = C_NOTES

        # Page number bottom-right
        tb_p = slide.shapes.add_textbox(Inches(12.6), Inches(7.15), Inches(0.5), Inches(0.25))
        tf_p = tb_p.text_frame
        p_p = tf_p.paragraphs[0]
        p_p.text = str(page_num)
        p_p.font.name = "Noto Sans"
        p_p.font.size = Pt(9.5)
        p_p.font.bold = True
        p_p.font.color.rgb = C_NAVY

    # =========================================================================
    # SLIDE 1: Cover Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_white_bg(s1)
    add_panel(s1, "mint", 4.2)

    # Panel text (Title, Year highlighted, Authors)
    tb1_p = s1.shapes.add_textbox(Inches(0.4), Inches(0.8), Inches(3.4), Inches(5.8))
    tf1_p = tb1_p.text_frame
    tf1_p.word_wrap = True

    p = tf1_p.paragraphs[0]
    p.text = "STATE OF RAG SECURITY"
    p.font.name = "Noto Sans"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(20)

    p = tf1_p.add_paragraph()
    p.text = "Sudarshana\n2026"
    p.font.name = "Noto Sans"
    p.font.size = Pt(46)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(20)

    p = tf1_p.add_paragraph()
    p.text = "Token-Level Indirect Prompt Injection Localization and Surgical Redaction."
    p.font.name = "Noto Sans"
    p.font.size = Pt(13)
    p.font.italic = True
    p.font.color.rgb = C_BODY
    p.space_after = Pt(28)

    p = tf1_p.add_paragraph()
    p.text = "PROJECT AUTHORS\nHarshit Kandpal (P4 Lead)\nTanmay Jaiswal (P1)\nUtkarsh Chatterjee (P2)\nSujal Som (P3)\nKinshuk Gupta (P5)"
    p.font.name = "Noto Sans"
    p.font.size = Pt(11)
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(14)

    p = tf1_p.add_paragraph()
    p.text = "November 2026 · IIT Bhilai"
    p.font.name = "Noto Sans"
    p.font.size = Pt(10)
    p.font.color.rgb = C_NOTES

    # Exhibit Area: Ringed Category Bubbles (White side)
    tb1_e = s1.shapes.add_textbox(Inches(4.6), Inches(0.8), Inches(8.3), Inches(5.8))
    tf1_e = tb1_e.text_frame
    tf1_e.word_wrap = True

    p = tf1_e.paragraphs[0]
    p.text = "Five defense layers unified against indirect prompt injection¹"
    p.font.name = "Noto Sans"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(24)

    categories = [
        ("Layer 1 · Normalizer & Encodings", "P1 & P2: NFKC, zero-width stripping, Base64/Hex decoders", C_GRAY),
        ("Layer 2 · Task-Pair Conditioning", "P2: [CLS] user_task [SEP] chunk [SEP] cross-attention", C_PURPLE),
        ("Layer 3 · Multi-Task DeBERTa Model", "P3: DeBERTa-v3 multi-task filter & 6-step ablation ladder", C_ORANGE),
        ("Layer 4 · Token Head & Surgical Redaction", "P4 (Harshit Kandpal): Subword gap-bridging & structure-aware expander", C_TEAL),
        ("Layer 5 · Downstream LLM Evaluation & UI", "P5: 98.40% Protected Success Rate & live web dashboard", C_GREEN),
    ]
    for cat_title, cat_sub, color in categories:
        p = tf1_e.add_paragraph()
        p.text = f"●  {cat_title}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = color
        p.space_after = Pt(2)

        p = tf1_e.add_paragraph()
        p.text = f"    {cat_sub}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(11)
        p.font.color.rgb = C_BODY
        p.space_after = Pt(14)

    add_footer(s1, 1, "Sources: 1/ Microsoft BIPIA Benchmark (50 Attack Types); Sudarshana architecture (2026).")

    # =========================================================================
    # Helper for Content Slides
    # =========================================================================
    def build_content_slide(slide_num, style, takeaway_plain, highlight_phrase, stat_number, stat_label,
                            chart_title, exhibit_bullets, source_str, notes_str="", is_placeholder=False,
                            lead_name="", role_tag=""):
        s = prs.slides.add_slide(blank_layout)
        set_white_bg(s)
        add_panel(s, style, 3.6)

        # Panel Content
        tb_p = s.shapes.add_textbox(Inches(0.35), Inches(0.8), Inches(3.0), Inches(6.0))
        tf_p = tb_p.text_frame
        tf_p.word_wrap = True

        # Role Tag / Category
        p = tf_p.paragraphs[0]
        p.text = f"{role_tag}  ·  {lead_name}" if lead_name else "TAKEAWAY"
        p.font.name = "Noto Sans"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = C_NAVY
        p.space_after = Pt(18)

        # Headline
        p = tf_p.add_paragraph()
        p.text = takeaway_plain
        p.font.name = "Noto Sans"
        p.font.size = Pt(18)
        p.font.bold = True
        p.font.color.rgb = C_NAVY
        p.space_after = Pt(18)

        # Stat Block
        if stat_number:
            p = tf_p.add_paragraph()
            p.text = stat_number
            p.font.name = "Noto Sans"
            p.font.size = Pt(36)
            p.font.bold = True
            p.font.color.rgb = C_PURPLE if style == "orchid" else C_TEAL
            p.space_after = Pt(4)

            p = tf_p.add_paragraph()
            p.text = stat_label
            p.font.name = "Noto Sans"
            p.font.size = Pt(11)
            p.font.italic = True
            p.font.color.rgb = C_BODY

        # Exhibit Area (White Canvas)
        tb_e = s.shapes.add_textbox(Inches(4.0), Inches(0.8), Inches(8.8), Inches(6.0))
        tf_e = tb_e.text_frame
        tf_e.word_wrap = True

        p = tf_e.paragraphs[0]
        p.text = chart_title
        p.font.name = "Noto Sans"
        p.font.size = Pt(16)
        p.font.bold = True
        p.font.color.rgb = C_NAVY
        p.space_after = Pt(20)

        for b_heading, b_body, b_col in exhibit_bullets:
            p = tf_e.add_paragraph()
            p.text = f"■  {b_heading}"
            p.font.name = "Noto Sans"
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = b_col
            p.space_after = Pt(2)

            p = tf_e.add_paragraph()
            p.text = f"    {b_body}"
            p.font.name = "Noto Sans"
            p.font.size = Pt(11)
            p.font.color.rgb = C_BODY
            p.space_after = Pt(14)

        add_footer(s, slide_num, source_str, notes_str)
        return s

    # =========================================================================
    # SLIDE 2: PLACEHOLDER P1 (Tanmay Jaiswal)
    # =========================================================================
    build_content_slide(
        slide_num=2,
        style="mint",
        takeaway_plain="Deterministic character span ground truth is the foundational prerequisite for subword sequence taggers.",
        highlight_phrase="subword sequence taggers",
        stat_number="50",
        stat_label="disjoint attack categories (25 train / 25 held-out test)",
        chart_title="P1: Unified Data Pipeline & Schema [PLACEHOLDER — Tanmay Jaiswal]",
        exhibit_bullets=[
            ("Designated Component Exhibit:", "This slide is reserved for Tanmay Jaiswal to present the data engineering pipeline.", C_PURPLE),
            ("Microsoft BIPIA Benchmark Integration:", "Spanning 5 diverse RAG tasks: Text QA, Code QA, Table QA, Email, and Summarization.", C_TEAL),
            ("Exact Character Offsets:", "Verified 100% round-trip character offset match (start_char, end_char) for token supervision.", C_GRAY),
            ("Instructional Hard Negatives:", "Curated real-world Linux tutorials, READMEs, and recipes with empty spans [].", C_BODY)
        ],
        source_str="Sources: 1/ data/splits/train.jsonl (n=386); 2/ data/splits/test.jsonl (n=206).",
        notes_str="Content to be finalized by Tanmay Jaiswal",
        is_placeholder=True,
        lead_name="Tanmay Jaiswal",
        role_tag="P1 DATA PIPELINE"
    )

    # =========================================================================
    # SLIDE 3: PLACEHOLDER P2 (Utkarsh Chatterjee)
    # =========================================================================
    build_content_slide(
        slide_num=3,
        style="orchid",
        takeaway_plain="Conditioning on user tasks eliminates the imperative ambiguity that causes false positive blocks.",
        highlight_phrase="false positive blocks",
        stat_number="83.3% → 16.7%",
        stat_label="hard-negative False Positive Rate drop under calibration",
        chart_title="P2: Pair-Input Conditioning & Disambiguation [PLACEHOLDER — Utkarsh Chatterjee]",
        exhibit_bullets=[
            ("Designated Component Exhibit:", "This slide is reserved for Utkarsh Chatterjee to present pair-input conditioning.", C_PURPLE),
            ("Cross-Attention Conditioning:", "Encoding [CLS] user_task [SEP] chunk [SEP] allows attention heads to contrast intent.", C_TEAL),
            ("Breaking Keyword Shortcuts:", "Legitimate technical documentation like 'run sudo rm -rf' is safely distinguished from attacks.", C_GRAY),
            ("Zero Recall Trade-Off:", "Maintains 100% recall on task-relevant and targeted indirect attacks.", C_BODY)
        ],
        source_str="Sources: 1/ data/splits/hard_negatives.jsonl; 2/ results/calibration_day6.json.",
        notes_str="Content to be finalized by Utkarsh Chatterjee",
        is_placeholder=True,
        lead_name="Utkarsh Chatterjee",
        role_tag="P2 PAIR INPUT"
    )

    # =========================================================================
    # SLIDE 4: PLACEHOLDER P3 (Sujal Som)
    # =========================================================================
    build_content_slide(
        slide_num=4,
        style="mint",
        takeaway_plain="Sliding-window chunking and adversarial augmentations drove test recall to 100.00% across all positions.",
        highlight_phrase="100.00% across all positions",
        stat_number="16.9% → 100%",
        stat_label="recall on end-of-document injections after 256-token chunking",
        chart_title="P3: Multi-Task Model & 6-Step Ablation Ladder [PLACEHOLDER — Sujal Som]",
        exhibit_bullets=[
            ("Designated Component Exhibit:", "This slide is reserved for Sujal Som to present the model and training ladder.", C_PURPLE),
            ("DeBERTa-v3 Multi-Task Filter:", "Shared encoder with joint detection and 3-class sequence classification head.", C_TEAL),
            ("The 6-Step Ablation Ladder:", "Step 1 (512 Baseline) to Step 6 (Adversarial Augmentations: Leetspeak, Homoglyphs, Base64).", C_GRAY),
            ("Validated Hypotheses (H1–H4):", "Chunking, normalizer, pair-input, and token-head hypotheses empirically validated.", C_BODY)
        ],
        source_str="Sources: 1/ reports/ablation_ladder_tables.json; 2/ models/final_step1..6/.",
        notes_str="Content to be finalized by Sujal Som",
        is_placeholder=True,
        lead_name="Sujal Som",
        role_tag="P3 MODEL & ABLATION"
    )

    # =========================================================================
    # SLIDE 5: P4 TOKEN HEAD & INFERENCE (HARSHIT KANDPAL - FULL CONTENT)
    # =========================================================================
    build_content_slide(
        slide_num=5,
        style="mint",
        takeaway_plain="Bridging subword token gaps merges fragmented attack detections into coherent malicious spans.",
        highlight_phrase="coherent malicious spans",
        stat_number="8.0x",
        stat_label="loss weight on positive injection tokens (O vs B-INJ/I-INJ)",
        chart_title="DeBERTa Token Classification Head¹ vs. Subword Gap-Bridging² (G ≤ 2 tokens)",
        exhibit_bullets=[
            ("Pretrained DeBERTa-v3 Backbone:", "microsoft/deberta-v3-xsmall (~22M active, 70.7M total parameters) operating efficiently on CPU.", C_PURPLE),
            ("3-Class BIO Sequence Projection:", "Linear projection (384 -> 3 classes: 0: O, 1: B-INJ, 2: I-INJ). Class-weighted CrossEntropy w = [1.0, 8.0, 8.0] overcomes severe 2% token sparsity.", C_TEAL),
            ("Sliding-Window Max-Pooling:", "256-token windows with 64 stride. Overlapping probabilities max-pooled: P(inj) = P(B-INJ) + P(I-INJ). Zero truncation blind spots.", C_ORANGE),
            ("Subword Gap-Bridging (G ≤ 2):", "Connective tokens ('to', 'the', '.') frequently dip below confidence. Gaps of <= 2 unflagged tokens between positive tokens are bridged into continuous spans.", C_GREEN)
        ],
        source_str="Sources: 1/ sudarshana/models/token_head.py; 2/ sudarshana/redaction/redact.py (predict_spans).",
        notes_str="P4 Lead: Harshit Kandpal",
        is_placeholder=False,
        lead_name="Harshit Kandpal",
        role_tag="P4 TOKEN HEAD"
    )

    # =========================================================================
    # SLIDE 6: P4 STRUCTURE-AWARE REDACTION (HARSHIT KANDPAL - FULL CONTENT)
    # =========================================================================
    build_content_slide(
        slide_num=6,
        style="orchid",
        takeaway_plain="Syntactic container expansion prevents prose sentence splitters from destroying markdown tables and code.",
        highlight_phrase="destroying markdown tables and code",
        stat_number="2.5x",
        stat_label="relative expansion cap: max(150 chars, 2.5x raw span)",
        chart_title="Naive sentence splitting (PySBD)¹ vs. Hierarchical syntactic container expansion²",
        exhibit_bullets=[
            ("The Collapse of PySBD on Structured Data:", "Standard sentence splitters treat markdown pipes and code syntax like prose, causing runaway deletions and 56.8% false document blocks.", C_PURPLE),
            ("Markdown Tables (Row Bounding):", "Bounded strictly to the single enclosing row [\\n, \\n]. Surrounding legitimate data rows remain 100% intact.", C_TEAL),
            ("Code Blocks & Python Docstrings:", "Expands to enclosing container (```, ''', \"\"\", <!-- -->), excising multi-line script loops completely.", C_ORANGE),
            ("Calibrated 50% Fallback Block Rule:", "If total excised characters exceed 50% of document, safely block the document to prevent returning gibberish.", C_GREEN)
        ],
        source_str="Sources: 1/ sudarshana/redaction/redact.py (expand_spans_structure_aware); 2/ reports/p4_token_head_report.md.",
        notes_str="P4 Lead: Harshit Kandpal",
        is_placeholder=False,
        lead_name="Harshit Kandpal",
        role_tag="P4 SURGICAL REDACTION"
    )

    # =========================================================================
    # SLIDE 7: P4 EMPIRICAL AUDIT SCORECARD (HARSHIT KANDPAL - FULL CONTENT)
    # =========================================================================
    build_content_slide(
        slide_num=7,
        style="mint",
        takeaway_plain="Structure-aware redaction cut unnecessary document blocks in half while preserving 57.84% of clean context.",
        highlight_phrase="57.84% of clean context",
        stat_number="68.0%",
        stat_label="selective redaction rate (up from 43.20% in baseline; +24.8% absolute gain)",
        chart_title="Baseline PySBD configuration¹ vs. P4 Structure-Aware Engine² on held-out test split (n=206)",
        exhibit_bullets=[
            ("Selective Redaction Rate: 43.20% -> 68.00% [59.4%, 75.5%]:", "+24.8% absolute gain; 85/125 attacked documents sanitized cleanly without triggering full block.", C_TEAL),
            ("Fallback Block Rate: 56.80% -> 32.00% [24.5%, 40.6%]:", "Unnecessary full blocks cut nearly in half by eliminating table/code runaway deletions.", C_PURPLE),
            ("Legitimate Content Retained: 38.55% -> 57.84% [50.5%, 64.5%]:", "+19.3% absolute gain in RAG context utility; 76.19% retained on clean counterfactuals.", C_GREEN),
            ("1-to-1 Span Recall: 99.20% [95.6%, 99.9%] | Character Coverage: 98.96%:", "124 out of 125 attacks localized across 25 unseen types; 98.96% of attack characters excised.", C_BODY)
        ],
        source_str="Sources: 1/ reports/metrics.json; 2/ reports/audit/raw_test_predictions.jsonl (n=206 total, 125 attacked).",
        notes_str="P4 Lead: Harshit Kandpal · Wilson & Bootstrap 95% CIs reported",
        is_placeholder=False,
        lead_name="Harshit Kandpal",
        role_tag="P4 AUDIT SCORECARD"
    )

    # =========================================================================
    # SLIDE 8: P4 DEFENSE & ERROR TAXONOMY (HARSHIT KANDPAL - FULL CONTENT)
    # =========================================================================
    build_content_slide(
        slide_num=8,
        style="orchid",
        takeaway_plain="Rigorous error triage reveals that 94.4% of attacks were neutralized with zero surviving characters.",
        highlight_phrase="zero surviving characters",
        stat_number=">450,000",
        stat_label="supervised token decisions across 386 documents (DeBERTa transfer learning)",
        chart_title="Harmless whitespace shifts¹ vs. Substantive code loop residuals² (7 cases out of 125)",
        exhibit_bullets=[
            ("Error Taxonomy (7 Leaking Cases):", "118/125 attacks (94.4%) had 0 residual chars. 3 cases were harmless trailing '\\n'. Only 4 cases had partial prompt words (13, 23, 42, 224 chars).", C_PURPLE),
            ("Defense: 'Why 386 Documents?':", "386 is the document count, not sample size. At 256-token windows, the model was trained on >450k supervised token classification decisions.", C_TEAL),
            ("Pretrained Transfer Learning:", "DeBERTa-v3 already possesses dense English syntax and code representations; token extraction requires only hundreds of annotated examples (like CoNLL-2003).", C_ORANGE),
            ("Zero-Leakage Empirical Proof:", "Achieving 99.20% span recall across 25 strictly held-out test attack categories proves robust out-of-distribution generalization.", C_GREEN)
        ],
        source_str="Sources: 1/ reports/p4_token_head_report.md Section 4; 2/ data/splits/DATA_CARD.md.",
        notes_str="P4 Lead: Harshit Kandpal",
        is_placeholder=False,
        lead_name="Harshit Kandpal",
        role_tag="P4 SCIENTIFIC RIGOR"
    )

    # =========================================================================
    # SLIDE 9: PLACEHOLDER P5 (Kinshuk Gupta)
    # =========================================================================
    build_content_slide(
        slide_num=9,
        style="mint",
        takeaway_plain="Sanitized carrier documents allow downstream LLMs to maintain a 98.40% protected success rate.",
        highlight_phrase="98.40% protected success rate",
        stat_number="0.871s",
        stat_label="median CPU inference latency per 1,362-token document",
        chart_title="P5: Downstream LLM Evaluation & UI Dashboard [PLACEHOLDER — Kinshuk Gupta]",
        exhibit_bullets=[
            ("Designated Component Exhibit:", "This slide is reserved for Kinshuk Gupta to present downstream LLM generation and UI.", C_PURPLE),
            ("Protected Success Rate (PSR = 98.40%):", "Under attack, downstream LLMs complete user tasks successfully with near-zero hijacking.", C_TEAL),
            ("Semantic Generation Utility (ROUGE-1):", "Preserving clean carrier context directly maintains downstream generation quality.", C_GRAY),
            ("Interactive Web Dashboard (ui/index.html):", "Real-time visualization of token heatmaps, boundary expansions, and live redaction.", C_BODY)
        ],
        source_str="Sources: 1/ results/p5/downstream.json; 2/ ui/server.py.",
        notes_str="Content to be finalized by Kinshuk Gupta",
        is_placeholder=True,
        lead_name="Kinshuk Gupta",
        role_tag="P5 DOWNSTREAM & UI"
    )

    # =========================================================================
    # SLIDE 10: Conclusion & Defense Readiness
    # =========================================================================
    build_content_slide(
        slide_num=10,
        style="mint",
        takeaway_plain="Project Sudarshana demonstrates that surgical token-level redaction is a production-ready reality.",
        highlight_phrase="production-ready reality",
        stat_number="78 / 78",
        stat_label="unit tests passing across all repository components (100%)",
        chart_title="Pre-redaction security vulnerability¹ vs. Post-redaction RAG context preservation²",
        exhibit_bullets=[
            ("Unified Modular Defense Pipeline:", "Seamless handoffs from P1 data/spans to P2 conditioning, P3 DeBERTa model, P4 surgical redaction, and P5 downstream evaluation.", C_PURPLE),
            ("P4 Core Breakthrough:", "Overcame the false-block crisis (cut blocks from 56.8% to 32.0%), boosted selective redaction to 68.0%, and retained 57.8% clean context.", C_TEAL),
            ("Scientific Rigor & Reproducibility:", "Strict train/val/test isolation, Wilson/Bootstrap 95% intervals, frozen checkpoints, and reproducible audit runner.", C_GREEN),
            ("Deployment Readiness:", "Sub-second CPU latency (<0.9s), zero GPU requirement, and interactive live demonstration UI.", C_BODY)
        ],
        source_str="Sources: 1/ reports/FINAL_PROJECT_REPORT.md; 2/ reports/FINAL_PROJECT_REPORT.pdf.",
        notes_str="Project Sudarshana · Academic Session 2026",
        is_placeholder=False,
        lead_name="All Authors",
        role_tag="CONCLUSION"
    )

    out_pptx = REPORTS_DIR / "Sudarshana_a16z_Presentation.pptx"
    prs.save(str(out_pptx))
    # Also overwrite the primary presentation pptx so user has it immediately
    prs.save(str(REPORTS_DIR / "Sudarshana_Project_Presentation.pptx"))
    print(f"PowerPoint Presentation (a16z Style) saved to: {out_pptx}")
    return out_pptx


def create_html_deck():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Project Sudarshana: a16z Style Defense Deck</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans:ital,wght@0,400;0,600;0,700;1,400&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    * { box-sizing: border-box; }
    body {
      font-family: 'Noto Sans', sans-serif;
      background: #ffffff;
      color: #2a3b5c;
      margin: 0;
      padding: 0;
      overflow-x: hidden;
    }
    .mono { font-family: 'JetBrains Mono', monospace; }

    /* a16z Palette */
    .color-navy { color: #173462; }
    .color-body { color: #2a3b5c; }
    .color-notes { color: #5f6b80; }
    .color-footer { color: #8a93a3; }

    /* Series Colors */
    .s-purple { color: #65327d; border-bottom: 2px solid #65327d; }
    .s-teal { color: #089582; border-bottom: 2px solid #089582; }
    .s-green { color: #62b33a; border-bottom: 2px solid #62b33a; }
    .s-orange { color: #f0a656; border-bottom: 2px solid #f0a656; }
    .s-gray { color: #b8bfcb; border-bottom: 2px solid #b8bfcb; }
    .s-lilac { color: #b5a0bf; border-bottom: 2px solid #b5a0bf; }

    /* Highlighter Blocks (Strictly one per headline) */
    .hl-lime {
      background-color: #e2fe6b;
      color: #173462;
      padding: 2px 6px;
      font-weight: 700;
      display: inline;
    }
    .hl-magenta {
      background-color: #ea2ce8;
      color: #ffffff;
      padding: 2px 6px;
      font-weight: 700;
      display: inline;
    }

    /* Panels & Separator Edges */
    .panel-mint {
      background: linear-gradient(180deg, #c5e9df 0%, #fcf8d5 100%);
      border-right: 4px solid #3fb59a;
    }
    .panel-orchid {
      background: linear-gradient(180deg, #ffe6f1 0%, #efd8f6 100%);
      border-right: 4px solid #e596c3;
    }

    /* Slide Structure */
    .slide {
      display: none;
      width: 100vw;
      min-height: 100vh;
      background: #ffffff;
    }
    .slide.active {
      display: flex;
    }
    .panel {
      width: 320px;
      min-width: 320px;
      padding: 2.5rem 2rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
    }
    .exhibit {
      flex: 1;
      padding: 2.5rem 3.5rem;
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      background: #ffffff;
    }

    /* Placeholder Tile */
    .placeholder-box {
      border: 2px dashed #b8bfcb;
      background: #f8fafc;
      border-radius: 8px;
    }
  </style>
</head>
<body>

  <!-- Top Slim Progress -->
  <div class="fixed top-0 left-0 w-full h-1 bg-[#e6e9ee] z-50">
    <div id="progress" class="h-full bg-[#089582] transition-all duration-300" style="width: 10%;"></div>
  </div>

  <!-- SLIDE 1: Cover Slide -->
  <div class="slide active" id="slide-1">
    <div class="panel panel-mint" style="width: 380px; min-width: 380px;">
      <div>
        <div class="text-[11px] font-bold tracking-wider text-[#173462] uppercase mb-8">State of RAG Security</div>
        <h1 class="text-[52px] font-bold text-[#173462] leading-[1.05] tracking-tight mb-4">
          Sudarshana<br>
          <span class="hl-lime">2026</span>
        </h1>
        <p class="text-[12px] italic text-[#2a3b5c] mt-4 leading-relaxed">
          Token-Level Indirect Prompt Injection Localization and Structure-Aware Redaction for Secure Retrieval-Augmented Generation.
        </p>
      </div>
      <div>
        <div class="text-[11px] font-bold text-[#173462] uppercase mb-2">Project Authors</div>
        <div class="text-[12px] text-[#173462] space-y-1">
          <div><strong class="text-[#089582]">Harshit Kandpal:</strong> P4 Lead (Token Head & Redaction)</div>
          <div><strong>Tanmay Jaiswal:</strong> P1 (Data Pipeline & Schema)</div>
          <div><strong>Utkarsh Chatterjee:</strong> P2 (Pair Input & Conditioning)</div>
          <div><strong>Sujal Som:</strong> P3 (Model Architecture & Ladder)</div>
          <div><strong>Kinshuk Gupta:</strong> P5 (Downstream Eval & UI)</div>
        </div>
        <div class="text-[10px] text-[#5f6b80] mt-4">November 2026 · IIT Bhilai</div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-2">
          Five defense layers unified against indirect prompt injection¹
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-8">
          A modular defense pipeline from character normalization to token-level surgical redaction.
        </p>

        <!-- Ringed Category Bubbles Exhibit -->
        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl">
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <div class="flex items-center gap-2 mb-1">
              <span class="w-3 h-3 rounded-full bg-[#b8bfcb]"></span>
              <strong class="text-[13px] text-[#173462]">Layer 1 · Normalizer & Encodings (P1)</strong>
            </div>
            <p class="text-[12px] text-[#2a3b5c] pl-5">NFKC normalization, zero-width stripping, and Base64/Hex decoders.</p>
          </div>
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <div class="flex items-center gap-2 mb-1">
              <span class="w-3 h-3 rounded-full bg-[#65327d]"></span>
              <strong class="text-[13px] text-[#173462]">Layer 2 · Task Conditioning (P2)</strong>
            </div>
            <p class="text-[12px] text-[#2a3b5c] pl-5"><code class="mono text-[11px] bg-slate-100 px-1 py-0.5 rounded">[CLS] user_task [SEP] chunk [SEP]</code> cross-attention.</p>
          </div>
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <div class="flex items-center gap-2 mb-1">
              <span class="w-3 h-3 rounded-full bg-[#f0a656]"></span>
              <strong class="text-[13px] text-[#173462]">Layer 3 · Multi-Task DeBERTa (P3)</strong>
            </div>
            <p class="text-[12px] text-[#2a3b5c] pl-5">DeBERTa-v3 multi-task filter & 6-step ablation ladder.</p>
          </div>
          <div class="p-4 border-2 border-[#089582] rounded-lg bg-[#f0fdf4]">
            <div class="flex items-center gap-2 mb-1">
              <span class="w-3 h-3 rounded-full bg-[#089582]"></span>
              <strong class="text-[13px] text-[#089582]">Layer 4 · Token Head & Redaction (P4 - Harshit Kandpal)</strong>
            </div>
            <p class="text-[12px] text-[#2a3b5c] pl-5">Subword gap-bridging & structure-aware boundary expander.</p>
          </div>
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white md:col-span-2">
            <div class="flex items-center gap-2 mb-1">
              <span class="w-3 h-3 rounded-full bg-[#62b33a]"></span>
              <strong class="text-[13px] text-[#173462]">Layer 5 · Downstream LLM Evaluation & UI (P5)</strong>
            </div>
            <p class="text-[12px] text-[#2a3b5c] pl-5">98.40% Protected Success Rate, semantic ROUGE-1 evaluation, and interactive web dashboard.</p>
          </div>
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 1 / 10 · Use [←] and [→] keys</span>
        <span>Sources: 1/ Microsoft BIPIA Benchmark (50 Attack Types); Sudarshana architecture (2026).</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 2: PLACEHOLDER P1 (Tanmay Jaiswal) -->
  <div class="slide" id="slide-2">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">P1 Data Pipeline · Tanmay Jaiswal</div>
        <h2 class="text-[27px] font-bold text-[#173462] leading-[1.25] tracking-tight mb-6">
          Deterministic character span ground truth is the foundational prerequisite for <span class="hl-lime">training subword sequence taggers</span>.
        </h2>
        <div class="mt-8">
          <div class="text-[48px] font-bold text-[#089582] leading-none mb-1">50</div>
          <div class="text-[11px] italic text-[#2a3b5c]">disjoint attack categories (25 train / 25 strictly held-out test)</div>
        </div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-1">
          P1: Unified Data Pipeline & Schema [PLACEHOLDER — Tanmay Jaiswal]
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">Designated component exhibit reserved for Tanmay Jaiswal.</p>

        <div class="placeholder-box p-6 space-y-4 max-w-3xl">
          <div class="text-[11px] font-bold text-[#f0a656] uppercase tracking-wider">[ PLACEHOLDER: P1 DATA PIPELINE ]</div>
          <div class="text-[15px] font-bold text-[#173462]">Content to be finalized by Tanmay Jaiswal</div>
          <ul class="space-y-2 text-[12px] text-[#2a3b5c]">
            <li>• Microsoft BIPIA benchmark across 5 application domains (Text QA, Code QA, Table QA, Email, Summarization).</li>
            <li>• Exact character offset verification (start_char, end_char) with 100% round-trip match.</li>
            <li>• Zero-leakage split strategy: 50 disjoint attack types (25 train / 25 strictly held-out test).</li>
            <li>• Curated instructional Hard Negatives dataset (Linux tutorials, READMEs, recipes) with empty spans [].</li>
          </ul>
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 2 / 10 · Placeholder</span>
        <span>Sources: 1/ data/splits/train.jsonl (n=386); 2/ data/splits/test.jsonl (n=206).</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 3: PLACEHOLDER P2 (Utkarsh Chatterjee) -->
  <div class="slide" id="slide-3">
    <div class="panel panel-orchid">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">P2 Pair Input · Utkarsh Chatterjee</div>
        <h2 class="text-[27px] font-bold text-[#173462] leading-[1.25] tracking-tight mb-6">
          Conditioning on user tasks eliminates the imperative ambiguity that causes <span class="hl-magenta">false positive blocks</span>.
        </h2>
        <div class="mt-8">
          <div class="text-[44px] font-bold text-[#65327d] leading-none mb-1">83% → 17%</div>
          <div class="text-[11px] italic text-[#2a3b5c]">hard-negative False Positive Rate drop under calibrated threshold</div>
        </div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-1">
          P2: Pair-Input Conditioning & Disambiguation [PLACEHOLDER — Utkarsh Chatterjee]
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">Designated component exhibit reserved for Utkarsh Chatterjee.</p>

        <div class="placeholder-box p-6 space-y-4 max-w-3xl">
          <div class="text-[11px] font-bold text-[#f0a656] uppercase tracking-wider">[ PLACEHOLDER: P2 PAIR INPUT ]</div>
          <div class="text-[15px] font-bold text-[#173462]">Content to be finalized by Utkarsh Chatterjee</div>
          <ul class="space-y-2 text-[12px] text-[#2a3b5c]">
            <li>• Conditioning architecture: <code class="mono text-[11px] bg-slate-200 px-1 py-0.5 rounded text-[#173462]">[CLS] user_task [SEP] document_chunk [SEP]</code>.</li>
            <li>• Disambiguating legitimate imperative commands from malicious prompt injections.</li>
            <li>• Hard-negative false positive rate reduction: Dropped from 83.33% to 16.67% under calibrated thresholds.</li>
            <li>• Cross-attention maps preserving recall on task-relevant attacks.</li>
          </ul>
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 3 / 10 · Placeholder</span>
        <span>Sources: 1/ data/splits/hard_negatives.jsonl; 2/ results/calibration_day6.json.</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 4: PLACEHOLDER P3 (Sujal Som) -->
  <div class="slide" id="slide-4">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">P3 Model & Ladder · Sujal Som</div>
        <h2 class="text-[27px] font-bold text-[#173462] leading-[1.25] tracking-tight mb-6">
          Sliding-window chunking and adversarial augmentations drove test recall to <span class="hl-lime">100.00% across all positions</span>.
        </h2>
        <div class="mt-8">
          <div class="text-[44px] font-bold text-[#089582] leading-none mb-1">17% → 100%</div>
          <div class="text-[11px] italic text-[#2a3b5c]">recall on end-of-document injections after 256-token chunking</div>
        </div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-1">
          P3: Multi-Task Model & 6-Step Ablation Ladder [PLACEHOLDER — Sujal Som]
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">Designated component exhibit reserved for Sujal Som.</p>

        <div class="placeholder-box p-6 space-y-4 max-w-3xl">
          <div class="text-[11px] font-bold text-[#f0a656] uppercase tracking-wider">[ PLACEHOLDER: P3 MODEL & ABLATION LADDER ]</div>
          <div class="text-[15px] font-bold text-[#173462]">Content to be finalized by Sujal Som</div>
          <ul class="space-y-2 text-[12px] text-[#2a3b5c]">
            <li>• DeBERTa-v3 multi-task filter architecture with decoupled heads and joint loss (<code class="mono text-[11px]">L_detect + 0.5 * L_tok</code>).</li>
            <li>• 6-step ablation ladder: Step 1 (512 Baseline) to Step 6 (Adversarial Augmentations).</li>
            <li>• Context preservation: 256-token chunking eliminating end-of-document truncation failures (100% end recall).</li>
            <li>• Obfuscation suite evaluation: Robustness against Leetspeak, Homoglyphs, Zero-Width chars, and Base64 wrappers.</li>
          </ul>
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 4 / 10 · Placeholder</span>
        <span>Sources: 1/ reports/ablation_ladder_tables.json; 2/ models/final_step1..6/.</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 5: P4 TOKEN HEAD & INFERENCE (HARSHIT KANDPAL - FULL CONTENT) -->
  <div class="slide" id="slide-5">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">P4 Token Head · Harshit Kandpal</div>
        <h2 class="text-[27px] font-bold text-[#173462] leading-[1.25] tracking-tight mb-6">
          Bridging subword token gaps merges fragmented attack detections into <span class="hl-lime">coherent malicious spans</span>.
        </h2>
        <div class="mt-8">
          <div class="text-[48px] font-bold text-[#089582] leading-none mb-1">8.0×</div>
          <div class="text-[11px] italic text-[#2a3b5c]">loss weighting on positive injection tokens (O vs B-INJ / I-INJ)</div>
        </div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-1">
          <span class="s-purple">Raw token confidence map¹</span> vs. <span class="s-teal">gap-bridged injection spans²</span> (subword level)
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">
          Overcoming tokenizer subword fragmentation across sliding-window max-pooling.
        </p>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl">
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <h3 class="text-[14px] font-bold text-[#65327d] mb-2">Sequence Tagging Architecture</h3>
            <ul class="space-y-2 text-[12px] text-[#2a3b5c]">
              <li>• <strong class="text-[#173462]">Pretrained Backbone:</strong> <code class="mono text-[11px] bg-slate-100 px-1 py-0.5 rounded">microsoft/deberta-v3-xsmall</code> (~22M active, 70.7M total parameters) operating on commodity CPU.</li>
              <li>• <strong class="text-[#173462]">3-Class Linear Head:</strong> Projects hidden dimension 384 → 3 classes (<code class="mono text-[11px]">0: O</code>, <code class="mono text-[11px]">1: B-INJ</code>, <code class="mono text-[11px]">2: I-INJ</code>).</li>
              <li>• <strong class="text-[#173462]">Weighted CrossEntropy:</strong> Positive injection tokens represent only ~2–4% of text. Loss is weighted with <code class="mono text-[11px] text-[#65327d] font-bold">w = [1.0, 8.0, 8.0]</code> to penalize missed attacks 8x more heavily.</li>
              <li>• <strong class="text-[#173462]">Special Token Masking:</strong> <code class="mono text-[11px]">[CLS]</code>, <code class="mono text-[11px]">[SEP]</code>, <code class="mono text-[11px]">[PAD]</code> and query tokens masked with <code class="mono text-[11px]">-100</code>.</li>
            </ul>
          </div>
          <div class="p-4 border-2 border-[#089582] rounded-lg bg-[#f0fdf4]">
            <h3 class="text-[14px] font-bold text-[#089582] mb-2">Inference & Gap-Bridging (G ≤ 2)</h3>
            <ul class="space-y-2 text-[12px] text-[#2a3b5c]">
              <li>• <strong class="text-[#173462]">Sliding-Window Max-Pooling:</strong> 256-token windows with 64 stride. Overlapping probabilities max-pooled: <code class="mono text-[11px]">P(inj) = P(B-INJ) + P(I-INJ)</code>.</li>
              <li>• <strong class="text-[#173462]">Confidence Threshold:</strong> Tokens with <code class="mono text-[11px]">P ≥ 0.35</code> flagged as malicious.</li>
              <li>• <strong class="text-[#173462]">Subword Gap-Bridging:</strong> Connective tokens ('to', 'the', '.') frequently dip below threshold. Unflagged gaps of ≤ 2 tokens between positive tokens are bridged into continuous spans.</li>
              <li>• <strong class="text-[#173462]">Round-Trip Offsets:</strong> Subwords mapped back with exact character precision to the raw document.</li>
            </ul>
          </div>
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 5 / 10 · Component P4 (Harshit Kandpal)</span>
        <span>Sources: 1/ sudarshana/models/token_head.py; 2/ sudarshana/redaction/redact.py (predict_spans).</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 6: P4 STRUCTURE-AWARE REDACTION (HARSHIT KANDPAL - FULL CONTENT) -->
  <div class="slide" id="slide-6">
    <div class="panel panel-orchid">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">P4 Redaction · Harshit Kandpal</div>
        <h2 class="text-[27px] font-bold text-[#173462] leading-[1.25] tracking-tight mb-6">
          Syntactic container expansion prevents prose sentence splitters from <span class="hl-magenta">destroying markdown tables and code</span>.
        </h2>
        <div class="mt-8">
          <div class="text-[44px] font-bold text-[#65327d] leading-none mb-1">2.5×</div>
          <div class="text-[11px] italic text-[#2a3b5c]">relative expansion cap: max(150 chars, 2.5x raw span)</div>
        </div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-1">
          <span class="s-purple">Naive sentence splitting (PySBD)¹</span> vs. <span class="s-teal">Hierarchical syntactic container expansion²</span>
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">
          Preserving syntax integrity across markdown tables, code fences, docstrings, and prose.
        </p>

        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-4xl">
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <div class="text-[13px] font-bold text-[#65327d] mb-1">Markdown Tables</div>
            <p class="text-[11px] text-[#2a3b5c]">Bounded strictly to the single enclosing row <code class="mono text-[10px]">[\\n, \\n]</code>. Surrounding legitimate data rows remain 100% intact.</p>
          </div>
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <div class="text-[13px] font-bold text-[#089582] mb-1">Code & Docstrings</div>
            <p class="text-[11px] text-[#2a3b5c]">Expands to the enclosing block delimiter (<code class="mono text-[10px]">```</code>, <code class="mono text-[10px]">'''</code>, <code class="mono text-[10px]">\"\"\"</code>, <code class="mono text-[10px]">&lt;!-- --&gt;</code>), fully excising multi-line script loops.</p>
          </div>
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <div class="text-[13px] font-bold text-[#62b33a] mb-1">Comments & Prose</div>
            <p class="text-[11px] text-[#2a3b5c]">Expands to contiguous <code class="mono text-[10px]">#</code> or <code class="mono text-[10px]">//</code> lines; prose sentences bounded strictly within paragraphs (<code class="mono text-[10px]">\\n\\n</code>).</p>
          </div>
          <div class="p-4 border-2 border-[#e596c3] rounded-lg bg-[#fdf2f8] md:col-span-3">
            <div class="text-[13px] font-bold text-[#ea2ce8] mb-1">Safety Fallback: The Calibrated 50% Block Rule</div>
            <p class="text-[12px] text-[#2a3b5c]">If total excised characters exceed 50% of the document, surgical redaction is safely aborted, returning <code class="mono text-[11px] bg-white px-1.5 py-0.5 rounded text-red-600 font-bold">[BLOCKED: PROMPT INJECTION DETECTED]</code>. (Baseline 35% caused unnecessary blocks; 50% calibrated on validation data cut blocks by half).</p>
          </div>
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 6 / 10 · Component P4 (Harshit Kandpal)</span>
        <span>Sources: 1/ sudarshana/redaction/redact.py (expand_spans_structure_aware); 2/ reports/p4_token_head_report.md.</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 7: P4 EMPIRICAL SCORECARD (HARSHIT KANDPAL - FULL CONTENT) -->
  <div class="slide" id="slide-7">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">P4 Scorecard · Harshit Kandpal</div>
        <h2 class="text-[27px] font-bold text-[#173462] leading-[1.25] tracking-tight mb-6">
          Structure-aware redaction cut unnecessary document blocks in half while preserving <span class="hl-lime">57.84% of clean context</span>.
        </h2>
        <div class="mt-8">
          <div class="text-[48px] font-bold text-[#089582] leading-none mb-1">68.0%</div>
          <div class="text-[11px] italic text-[#2a3b5c]">selective redaction rate (+24.8% gain over baseline)</div>
        </div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-1">
          <span class="s-purple">Baseline PySBD configuration¹</span> vs. <span class="s-teal">P4 Structure-Aware Engine²</span> on held-out test split (n=206)
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-5">
          Audited metrics on 206 test documents (125 attacked across 25 unseen types, 81 benign).
        </p>

        <div class="border border-[#e6e9ee] rounded-lg overflow-hidden max-w-4xl">
          <table class="w-full text-left text-[12px] border-collapse">
            <thead>
              <tr class="bg-[#f8fafc] border-b border-[#e6e9ee] text-[10px] font-bold uppercase text-[#173462]">
                <th class="py-2.5 px-4">Metric</th>
                <th class="py-2.5 px-4 text-[#65327d]">Baseline (PySBD 35%)</th>
                <th class="py-2.5 px-4 text-[#089582]">P4 Engine (Audited)</th>
                <th class="py-2.5 px-4 text-[#5f6b80]">95% Confidence Interval</th>
                <th class="py-2.5 px-4 text-[#62b33a]">Impact / Gain</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-[#e6e9ee] text-[#2a3b5c]">
              <tr>
                <td class="py-2.5 px-4 font-bold text-[#173462]">Selective Redaction Rate</td>
                <td class="py-2.5 px-4">43.20%</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">68.00%</td>
                <td class="py-2.5 px-4 mono text-[11px]">[59.40%, 75.50%]</td>
                <td class="py-2.5 px-4 font-bold text-[#62b33a]">+24.8% gain (85/125 sanitized)</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold text-[#173462]">Fallback Block Rate</td>
                <td class="py-2.5 px-4 text-red-600">56.80%</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">32.00%</td>
                <td class="py-2.5 px-4 mono text-[11px]">[24.50%, 40.60%]</td>
                <td class="py-2.5 px-4 font-bold text-[#62b33a]">Blocks cut nearly in half</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold text-[#173462]">Legitimate Content Retained</td>
                <td class="py-2.5 px-4">38.55%</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">57.84%</td>
                <td class="py-2.5 px-4 mono text-[11px]">[50.50%, 64.50%]</td>
                <td class="py-2.5 px-4 font-bold text-[#62b33a]">+19.3% absolute gain in utility</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold text-[#173462]">Span Detection Recall (1-to-1)</td>
                <td class="py-2.5 px-4">99.20%</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">99.20%</td>
                <td class="py-2.5 px-4 mono text-[11px]">[95.61%, 99.86%]</td>
                <td class="py-2.5 px-4">124 / 125 attacks localized</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold text-[#173462]">Gold Span Character Coverage</td>
                <td class="py-2.5 px-4">98.51%</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">98.96%</td>
                <td class="py-2.5 px-4 mono text-[11px]">[97.70%, 99.90%]</td>
                <td class="py-2.5 px-4">98.96% attack chars excised</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold text-[#173462]">Residual Leaking Samples</td>
                <td class="py-2.5 px-4 text-red-600">17.60% (22/125)</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">5.60% (7/125)</td>
                <td class="py-2.5 px-4 mono text-[11px]">[2.70%, 11.10%]</td>
                <td class="py-2.5 px-4 font-bold text-[#62b33a]">68% drop in leaking samples</td>
              </tr>
              <tr>
                <td class="py-2.5 px-4 font-bold text-[#173462]">CPU Inference Latency</td>
                <td class="py-2.5 px-4">—</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">0.871s median</td>
                <td class="py-2.5 px-4 mono text-[11px]">p95: 1.040s</td>
                <td class="py-2.5 px-4">~15–20 docs/sec throughput</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 7 / 10 · Component P4 (Harshit Kandpal)</span>
        <span>Sources: 1/ reports/metrics.json; 2/ reports/audit/raw_test_predictions.jsonl (n=206 total).</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 8: P4 DEFENSE & ERROR TAXONOMY (HARSHIT KANDPAL - FULL CONTENT) -->
  <div class="slide" id="slide-8">
    <div class="panel panel-orchid">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">P4 Defense · Harshit Kandpal</div>
        <h2 class="text-[27px] font-bold text-[#173462] leading-[1.25] tracking-tight mb-6">
          Rigorous error triage reveals that 94.4% of attacks were neutralized with <span class="hl-magenta">zero surviving characters</span>.
        </h2>
        <div class="mt-8">
          <div class="text-[44px] font-bold text-[#65327d] leading-none mb-1">>450k</div>
          <div class="text-[11px] italic text-[#2a3b5c]">supervised token decisions across 386 training documents</div>
        </div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-1">
          <span class="s-purple">Harmless whitespace shifts¹</span> vs. <span class="s-orange">Substantive code loop residuals²</span> (7 cases out of 125)
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">
          Categorized error taxonomy and defense of token-level dataset size.
        </p>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 max-w-4xl">
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <h3 class="text-[14px] font-bold text-[#65327d] mb-2">Error Taxonomy (7 Residual Cases)</h3>
            <ul class="space-y-2 text-[12px] text-[#2a3b5c]">
              <li>• <strong class="text-[#173462]">118 / 125 Attacks (94.4%):</strong> Neutralized cleanly with zero residual payload characters.</li>
              <li>• <strong class="text-[#173462]">3 Whitespace Shifts:</strong> Indices 31, 124, 199 leaked exactly 1 character ('\n') because the model cleanly stopped at the word boundary. Completely benign.</li>
              <li>• <strong class="text-[#173462]">1 Subtle Table Cell:</strong> Index 3 leaked 13 chars in a table cell where confidence hovered at 0.34.</li>
              <li>• <strong class="text-[#173462]">2 Multi-Line Loops:</strong> Indices 28 & 87 (scapy & multiprocessing loop) exceeded the 2.5x expansion cap.</li>
              <li>• <strong class="text-[#173462]">1 Multi-Sentence Prompt:</strong> Index 66 had 42 surviving characters from secondary sentence.</li>
            </ul>
          </div>
          <div class="p-4 border-2 border-[#089582] rounded-lg bg-[#f0fdf4]">
            <h3 class="text-[14px] font-bold text-[#089582] mb-2">Viva Defense: "Why 386 Documents?"</h3>
            <ul class="space-y-2 text-[12px] text-[#2a3b5c]">
              <li>• <strong class="text-[#173462]">Token Instances > 450,000:</strong> We trained a sequence token tagger, not a document classifier. Across 386 documents, the model received >450k supervised token loss gradients.</li>
              <li>• <strong class="text-[#173462]">Pretrained Transfer Learning:</strong> DeBERTa-v3 already possesses dense English and code syntax; token extraction requires only hundreds of annotated examples (like CoNLL-2003).</li>
              <li>• <strong class="text-[#173462]">Exact Ground Truth Spans:</strong> 99% of datasets only have binary 0/1 labels. BIPIA provides verified character offsets with 100% round-trip fidelity.</li>
              <li>• <strong class="text-[#173462]">Proven Generalization:</strong> Tested on 25 unseen attack categories; 99.20% recall proves it did not overfit.</li>
            </ul>
          </div>
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 8 / 10 · Component P4 (Harshit Kandpal)</span>
        <span>Sources: 1/ reports/p4_token_head_report.md Section 4; 2/ data/splits/DATA_CARD.md.</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 9: PLACEHOLDER P5 (Kinshuk Gupta) -->
  <div class="slide" id="slide-9">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">P5 Downstream · Kinshuk Gupta</div>
        <h2 class="text-[27px] font-bold text-[#173462] leading-[1.25] tracking-tight mb-6">
          Sanitized carrier documents allow downstream LLMs to maintain a <span class="hl-lime">98.40% protected success rate</span>.
        </h2>
        <div class="mt-8">
          <div class="text-[44px] font-bold text-[#089582] leading-none mb-1">0.871s</div>
          <div class="text-[11px] italic text-[#2a3b5c]">median CPU inference latency per 1,362-token document</div>
        </div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-1">
          P5: Downstream LLM Evaluation & UI Dashboard [PLACEHOLDER — Kinshuk Gupta]
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">Designated component exhibit reserved for Kinshuk Gupta.</p>

        <div class="placeholder-box p-6 space-y-4 max-w-3xl">
          <div class="text-[11px] font-bold text-[#f0a656] uppercase tracking-wider">[ PLACEHOLDER: P5 DOWNSTREAM & UI ]</div>
          <div class="text-[15px] font-bold text-[#173462]">Content to be finalized by Kinshuk Gupta</div>
          <ul class="space-y-2 text-[12px] text-[#2a3b5c]">
            <li>• Downstream generation harness: Protected Success Rate (<code class="mono text-[11px] text-[#089582] font-bold">PSR = 98.40%</code>) and semantic ROUGE-1 on clean queries.</li>
            <li>• Decision threshold calibration on validation data.</li>
            <li>• CPU latency profiling: Median 0.871s per 1,362-token document; throughput ~15-20 docs/sec.</li>
            <li>• Interactive web UI dashboard (<code class="mono text-[11px]">ui/index.html & ui/server.py</code>) demonstrating real-time token heatmaps and live redaction.</li>
          </ul>
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 9 / 10 · Placeholder</span>
        <span>Sources: 1/ results/p5/downstream.json; 2/ ui/server.py.</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 10: Conclusion & Unified Defense Readiness -->
  <div class="slide" id="slide-10">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">Conclusion & Defense</div>
        <h2 class="text-[27px] font-bold text-[#173462] leading-[1.25] tracking-tight mb-6">
          Project Sudarshana demonstrates that surgical token-level redaction is a <span class="hl-lime">production-ready reality</span>.
        </h2>
        <div class="mt-8">
          <div class="text-[44px] font-bold text-[#089582] leading-none mb-1">78 / 78</div>
          <div class="text-[11px] italic text-[#2a3b5c]">unit tests passing across all repository components (100%)</div>
        </div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-1">
          <span class="s-purple">Pre-redaction security vulnerability¹</span> vs. <span class="s-teal">Post-redaction RAG context preservation²</span>
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">
          Complete defense system summary and production readiness verification.
        </p>

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl">
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <h3 class="text-[13px] font-bold text-[#173462] mb-1">Unified Defense Pipeline</h3>
            <p class="text-[12px] text-[#2a3b5c]">Seamless integration from P1 character normalization to P2 task conditioning, P3 DeBERTa multi-task model, P4 surgical redaction, and P5 downstream evaluation.</p>
          </div>
          <div class="p-4 border-2 border-[#089582] rounded-lg bg-[#f0fdf4]">
            <h3 class="text-[13px] font-bold text-[#089582] mb-1">P4 Core Breakthrough</h3>
            <p class="text-[12px] text-[#2a3b5c]">Eliminated the false-block crisis (cut blocks from 56.8% to 32.0%), boosted selective redaction to 68.0%, and retained 57.8% clean context with 98.96% character coverage.</p>
          </div>
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <h3 class="text-[13px] font-bold text-[#173462] mb-1">Rigorous Methodology</h3>
            <p class="text-[12px] text-[#2a3b5c]">Zero evaluation leakage. Tuning strictly on validation data; evaluated once on held-out test data with 95% Wilson and Bootstrap confidence intervals.</p>
          </div>
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <h3 class="text-[13px] font-bold text-[#173462] mb-1">Production Viability</h3>
            <p class="text-[12px] text-[#2a3b5c]">Sub-second CPU latency (&lt;0.9s), zero GPU cluster requirement, all 78 repository tests passing (100%), and interactive web dashboard.</p>
          </div>
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 10 / 10 · Thank you! Open for Questions.</span>
        <span>Sources: 1/ reports/FINAL_PROJECT_REPORT.md; 2/ reports/FINAL_PROJECT_REPORT.pdf.</span>
      </div>
    </div>
  </div>

  <!-- Bottom Floating Navigation Controls -->
  <div class="fixed bottom-4 right-6 flex items-center gap-2 bg-white/95 border border-[#e6e9ee] px-3 py-1.5 rounded-full shadow-lg">
    <button id="prevBtn" class="text-[#5f6b80] hover:text-[#173462] px-2 py-1 text-sm font-bold">← Prev</button>
    <span id="slideIndicator" class="text-xs font-semibold text-[#089582] px-2">1 / 10</span>
    <button id="nextBtn" class="text-[#5f6b80] hover:text-[#173462] px-2 py-1 text-sm font-bold">Next →</button>
  </div>

  <script>
    let currentSlide = 1;
    const totalSlides = 10;

    function showSlide(n) {
      if (n < 1) n = 1;
      if (n > totalSlides) n = totalSlides;
      currentSlide = n;

      for (let i = 1; i <= totalSlides; i++) {
        const slide = document.getElementById(`slide-${i}`);
        if (slide) slide.classList.remove('active');
      }

      const activeSlide = document.getElementById(`slide-${currentSlide}`);
      if (activeSlide) activeSlide.classList.add('active');

      document.getElementById('slideIndicator').innerText = `${currentSlide} / ${totalSlides}`;
      document.getElementById('progress').style.width = `${(currentSlide / totalSlides) * 100}%`;
    }

    document.getElementById('prevBtn').addEventListener('click', () => showSlide(currentSlide - 1));
    document.getElementById('nextBtn').addEventListener('click', () => showSlide(currentSlide + 1));

    document.addEventListener('keydown', (e) => {
      if (e.key === 'ArrowRight' || e.key === ' ' || e.key === 'PageDown') showSlide(currentSlide + 1);
      else if (e.key === 'ArrowLeft' || e.key === 'PageUp') showSlide(currentSlide - 1);
    });
  </script>
</body>
</html>
"""
    out_html = REPORTS_DIR / "presentation_a16z.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    # Also overwrite presentation.html so user has it immediately
    with open(REPORTS_DIR / "presentation.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML Presentation (a16z Style) saved to: {out_html}")
    return out_html


def main():
    print("=== Generating 'a16z Style' Presentation Decks ===")
    out_pptx = create_pptx_deck()
    out_html = create_html_deck()
    print("\n✅ a16z Style Slide Decks Generated:")
    print(f"  1. PowerPoint File (.pptx): {out_pptx}")
    print(f"  2. Interactive HTML Deck:   {out_html}")


if __name__ == "__main__":
    main()
