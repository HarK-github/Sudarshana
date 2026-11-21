"""Generate publication-grade 'a16z Style' Presentation Deck for Project Sudarshana.
Unofficial homage to the State of Crypto report slides:
- Canvas: Pure White #ffffff
- Typography: 'Noto Sans' (Google Fonts)
  - Headlines: 700 at 27-30px (Cover: 52px)
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
- Strict Constraints:
  - NO mentions of P1, P2, P3, P4, P5, or PX. ONLY people's names!
  - Harshit Kandpal's work is fully detailed with pipeline flowchart, architecture, algorithms, and audit metrics.
  - Teammates' sections (Tanmay Jaiswal, Utkarsh Chatterjee, Sujal Som, Kinshuk Gupta) are clean, designated placeholders.
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
FLOWCHART_PATH = FIGURES_DIR / "p4_pipeline_flowchart.png"
TOKEN_GAP_FIG_PATH = FIGURES_DIR / "token_gap_bridging.png"
REDACTION_FIG_PATH = FIGURES_DIR / "rep_f5_redaction.png"
ERROR_TAXONOMY_FIG_PATH = FIGURES_DIR / "error_taxonomy_chart.png"

# Color Constants
C_WHITE = RGBColor(255, 255, 255)
C_NAVY = RGBColor(23, 52, 98)         # #173462
C_BODY = RGBColor(42, 59, 92)         # #2a3b5c
C_NOTES = RGBColor(95, 107, 128)      # #5f6b80
C_FOOTER = RGBColor(138, 147, 163)    # #8a93a3
C_GRID = RGBColor(230, 233, 238)      # #e6e9ee

C_MINT_TOP = RGBColor(197, 233, 223)  # #c5e9df
C_MINT_EDGE = RGBColor(63, 181, 154)  # #3fb59a
C_LIME_HL = RGBColor(226, 254, 107)   # #e2fe6b

C_ORCHID_TOP = RGBColor(255, 230, 241)# #ffe6f1
C_ORCHID_EDGE = RGBColor(229, 150, 195)# #e596c3
C_MAGENTA_HL = RGBColor(234, 44, 232) # #ea2ce8

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
        panel = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(width_in), Inches(7.5))
        panel.fill.solid()
        panel.fill.fore_color.rgb = C_MINT_TOP if style == "mint" else C_ORCHID_TOP
        panel.line.fill.background()

        edge = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(width_in), Inches(0), Inches(0.045), Inches(7.5))
        edge.fill.solid()
        edge.fill.fore_color.rgb = C_MINT_EDGE if style == "mint" else C_ORCHID_EDGE
        edge.line.fill.background()
        return panel

    def add_footer(slide, page_num, total_pages, source_str, notes_str=""):
        tb_c = slide.shapes.add_textbox(Inches(0.4), Inches(7.15), Inches(2.5), Inches(0.25))
        tf_c = tb_c.text_frame
        p_c = tf_c.paragraphs[0]
        p_c.text = "©2026  Sudarshana"
        p_c.font.name = "Noto Sans"
        p_c.font.size = Pt(9.5)
        p_c.font.color.rgb = C_FOOTER

        if notes_str:
            tb_n = slide.shapes.add_textbox(Inches(3.8), Inches(7.15), Inches(4.5), Inches(0.25))
            tf_n = tb_n.text_frame
            p_n = tf_n.paragraphs[0]
            p_n.text = notes_str
            p_n.font.name = "Noto Sans"
            p_n.font.size = Pt(9)
            p_n.font.color.rgb = C_NOTES

        tb_s = slide.shapes.add_textbox(Inches(7.2), Inches(7.15), Inches(5.1), Inches(0.25))
        tf_s = tb_s.text_frame
        p_s = tf_s.paragraphs[0]
        p_s.text = source_str
        p_s.font.name = "Noto Sans"
        p_s.font.size = Pt(8.5)
        p_s.font.color.rgb = C_NOTES

        tb_p = slide.shapes.add_textbox(Inches(12.5), Inches(7.15), Inches(0.6), Inches(0.25))
        tf_p = tb_p.text_frame
        p_p = tf_p.paragraphs[0]
        p_p.text = f"{page_num} / {total_pages}"
        p_p.font.name = "Noto Sans"
        p_p.font.size = Pt(9.5)
        p_p.font.bold = True
        p_p.font.color.rgb = C_NAVY

    def add_panel_content(slide, style, role_str, lead_name, headline_text, stat_num="", stat_lbl=""):
        tb_p = slide.shapes.add_textbox(Inches(0.35), Inches(0.75), Inches(3.0), Inches(6.1))
        tf_p = tb_p.text_frame
        tf_p.word_wrap = True

        p = tf_p.paragraphs[0]
        p.text = f"{role_str.upper()}  ·  {lead_name}" if lead_name else "TAKEAWAY"
        p.font.name = "Noto Sans"
        p.font.size = Pt(10)
        p.font.bold = True
        p.font.color.rgb = C_NAVY
        p.space_after = Pt(16)

        p = tf_p.add_paragraph()
        p.text = headline_text
        p.font.name = "Noto Sans"
        p.font.size = Pt(17)
        p.font.bold = True
        p.font.color.rgb = C_NAVY
        p.space_after = Pt(18)

        if stat_num:
            p = tf_p.add_paragraph()
            p.text = stat_num
            p.font.name = "Noto Sans"
            p.font.size = Pt(36)
            p.font.bold = True
            p.font.color.rgb = C_PURPLE if style == "orchid" else C_TEAL
            p.space_after = Pt(4)

            p = tf_p.add_paragraph()
            p.text = stat_lbl
            p.font.name = "Noto Sans"
            p.font.size = Pt(10.5)
            p.font.italic = True
            p.font.color.rgb = C_BODY

    def format_cell(cell, text, bold=False, font_size=9.5, text_color=C_NAVY, bg_color=None, align=None):
        if bg_color:
            cell.fill.solid()
            cell.fill.fore_color.rgb = bg_color
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.text = text
        p.font.name = "Noto Sans"
        p.font.size = Pt(font_size)
        p.font.bold = bold
        p.font.color.rgb = text_color
        if align:
            p.alignment = align

    # =========================================================================
    # SLIDE 1: Cover Slide
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_white_bg(s1)
    add_panel(s1, "mint", 4.2)

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
    p.text = "PROJECT AUTHORS\nHarshit Kandpal (Token Head & Redaction Engine)\nTanmay Jaiswal (Data Engineering & Schema)\nUtkarsh Chatterjee (Task Conditioning & Disambiguation)\nSujal Som (Model Architecture & Ladder)\nKinshuk Gupta (Downstream Evaluation & UI)"
    p.font.name = "Noto Sans"
    p.font.size = Pt(11)
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(14)

    p = tf1_p.add_paragraph()
    p.text = "November 2026 · IIT Bhilai"
    p.font.name = "Noto Sans"
    p.font.size = Pt(10)
    p.font.color.rgb = C_NOTES

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

    layers = [
        ("Layer 1 · Normalizer & Encodings", "Tanmay Jaiswal: NFKC, zero-width stripping, Base64/Hex decoders", C_GRAY),
        ("Layer 2 · Task-Pair Conditioning", "Utkarsh Chatterjee: [CLS] user_task [SEP] chunk [SEP] cross-attention", C_PURPLE),
        ("Layer 3 · Multi-Task DeBERTa Model", "Sujal Som: DeBERTa-v3 multi-task filter & 6-step ablation ladder", C_ORANGE),
        ("Layer 4 · Token Head & Surgical Redaction", "Harshit Kandpal: Subword gap-bridging & structure-aware expander", C_TEAL),
        ("Layer 5 · Downstream LLM Evaluation & UI", "Kinshuk Gupta: 98.40% Protected Success Rate & live web dashboard", C_GREEN),
    ]
    for cat_title, cat_sub, color in layers:
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

    add_footer(s1, 1, 11, "Sources: 1/ Microsoft BIPIA Benchmark (50 Attack Types); Sudarshana architecture (2026).")

    # =========================================================================
    # SLIDE 2: PLACEHOLDER (Tanmay Jaiswal)
    # =========================================================================
    s2 = prs.slides.add_slide(blank_layout)
    set_white_bg(s2)
    add_panel(s2, "mint", 3.6)
    add_panel_content(
        s2, "mint", "Data Engineering", "Tanmay Jaiswal",
        "Deterministic character span ground truth is the foundational prerequisite for training subword sequence taggers.",
        "50", "disjoint attack categories (25 train / 25 strictly held-out test)"
    )

    tb2_e = s2.shapes.add_textbox(Inches(4.0), Inches(0.8), Inches(8.8), Inches(6.0))
    tf2_e = tb2_e.text_frame
    tf2_e.word_wrap = True

    p = tf2_e.paragraphs[0]
    p.text = "Unified Data Pipeline & Schema [PLACEHOLDER — Tanmay Jaiswal]"
    p.font.name = "Noto Sans"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(20)

    tanmay_items = [
        ("Designated Component Exhibit:", "This slide is reserved for Tanmay Jaiswal to present the data engineering pipeline.", C_PURPLE),
        ("Microsoft BIPIA Benchmark Integration:", "Spanning 5 diverse RAG tasks: Text QA, Code QA, Table QA, Email, and Summarization.", C_TEAL),
        ("Exact Character Offsets:", "Verified 100% round-trip character offset match (start_char, end_char) for token supervision.", C_GRAY),
        ("Instructional Hard Negatives:", "Curated real-world Linux tutorials, READMEs, and recipes with empty spans [].", C_BODY)
    ]
    for h, b, col in tanmay_items:
        p = tf2_e.add_paragraph()
        p.text = f"■  {h}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(2)
        p = tf2_e.add_paragraph()
        p.text = f"    {b}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(11)
        p.font.color.rgb = C_BODY
        p.space_after = Pt(14)

    add_footer(s2, 2, 11, "Sources: 1/ data/splits/train.jsonl (n=386); 2/ data/splits/test.jsonl (n=206).", "Lead: Tanmay Jaiswal")

    # =========================================================================
    # SLIDE 3: PLACEHOLDER (Utkarsh Chatterjee)
    # =========================================================================
    s3 = prs.slides.add_slide(blank_layout)
    set_white_bg(s3)
    add_panel(s3, "orchid", 3.6)
    add_panel_content(
        s3, "orchid", "Task Conditioning", "Utkarsh Chatterjee",
        "Conditioning on user tasks eliminates the imperative ambiguity that causes false positive blocks.",
        "83% → 17%", "hard-negative False Positive Rate drop under calibrated threshold"
    )

    tb3_e = s3.shapes.add_textbox(Inches(4.0), Inches(0.8), Inches(8.8), Inches(6.0))
    tf3_e = tb3_e.text_frame
    tf3_e.word_wrap = True

    p = tf3_e.paragraphs[0]
    p.text = "Pair-Input Conditioning & Disambiguation [PLACEHOLDER — Utkarsh Chatterjee]"
    p.font.name = "Noto Sans"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(20)

    utkarsh_items = [
        ("Designated Component Exhibit:", "This slide is reserved for Utkarsh Chatterjee to present pair-input conditioning.", C_PURPLE),
        ("Cross-Attention Conditioning:", "Encoding [CLS] user_task [SEP] chunk [SEP] allows attention heads to contrast intent.", C_TEAL),
        ("Breaking Keyword Shortcuts:", "Legitimate technical documentation like 'run sudo rm -rf' is safely distinguished from attacks.", C_GRAY),
        ("Zero Recall Trade-Off:", "Maintains 100% recall on task-relevant and targeted indirect attacks.", C_BODY)
    ]
    for h, b, col in utkarsh_items:
        p = tf3_e.add_paragraph()
        p.text = f"■  {h}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(2)
        p = tf3_e.add_paragraph()
        p.text = f"    {b}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(11)
        p.font.color.rgb = C_BODY
        p.space_after = Pt(14)

    add_footer(s3, 3, 11, "Sources: 1/ data/splits/hard_negatives.jsonl; 2/ results/calibration_day6.json.", "Lead: Utkarsh Chatterjee")

    # =========================================================================
    # SLIDE 4: PLACEHOLDER (Sujal Som)
    # =========================================================================
    s4 = prs.slides.add_slide(blank_layout)
    set_white_bg(s4)
    add_panel(s4, "mint", 3.6)
    add_panel_content(
        s4, "mint", "Model & Training Ladder", "Sujal Som",
        "Sliding-window chunking and adversarial augmentations drove test recall to 100.00% across all positions.",
        "17% → 100%", "recall on end-of-document injections after 256-token chunking"
    )

    tb4_e = s4.shapes.add_textbox(Inches(4.0), Inches(0.8), Inches(8.8), Inches(6.0))
    tf4_e = tb4_e.text_frame
    tf4_e.word_wrap = True

    p = tf4_e.paragraphs[0]
    p.text = "Multi-Task Model & 6-Step Ablation Ladder [PLACEHOLDER — Sujal Som]"
    p.font.name = "Noto Sans"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(20)

    sujal_items = [
        ("Designated Component Exhibit:", "This slide is reserved for Sujal Som to present the model and training ladder.", C_PURPLE),
        ("DeBERTa-v3 Multi-Task Filter:", "Shared encoder with joint detection and 3-class sequence classification head.", C_TEAL),
        ("The 6-Step Ablation Ladder:", "Step 1 (512 Baseline) to Step 6 (Adversarial Augmentations: Leetspeak, Homoglyphs, Base64).", C_GRAY),
        ("Validated Hypotheses (H1–H4):", "Chunking, normalizer, pair-input, and token-head hypotheses empirically validated.", C_BODY)
    ]
    for h, b, col in sujal_items:
        p = tf4_e.add_paragraph()
        p.text = f"■  {h}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(2)
        p = tf4_e.add_paragraph()
        p.text = f"    {b}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(11)
        p.font.color.rgb = C_BODY
        p.space_after = Pt(14)

    add_footer(s4, 4, 11, "Sources: 1/ reports/ablation_ladder_tables.json; 2/ models/final_step1..6/.", "Lead: Sujal Som")

    # =========================================================================
    # SLIDE 5: HARSHIT KANDPAL - PIPELINE FLOWCHART (CHART-ONLY EXHIBIT)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_white_bg(s5)
    add_panel(s5, "mint", 3.6)
    add_panel_content(
        s5, "mint", "Redaction Architecture", "Harshit Kandpal",
        "Surgical redaction operates across seven sequential stages from raw windows to sanitized output.",
        "7 Stages", "end-to-end token detection and container expansion pipeline"
    )

    tb5_e = s5.shapes.add_textbox(Inches(3.8), Inches(0.55), Inches(9.2), Inches(0.5))
    tf5_e = tb5_e.text_frame
    p = tf5_e.paragraphs[0]
    p.text = "Token Classification Sequence Head¹ & Structure-Aware Redaction Pipeline²"
    p.font.name = "Noto Sans"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    if FLOWCHART_PATH.exists():
        s5.shapes.add_picture(str(FLOWCHART_PATH), Inches(3.8), Inches(1.15), width=Inches(9.2))

    add_footer(s5, 5, 11, "Sources: 1/ sudarshana/models/token_head.py; 2/ sudarshana/redaction/redact.py.", "Lead: Harshit Kandpal")

    # =========================================================================
    # SLIDE 6: HARSHIT KANDPAL - TOKEN HEAD & GAP BRIDGING (CHART-ONLY EXHIBIT)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_white_bg(s6)
    add_panel(s6, "mint", 3.6)
    add_panel_content(
        s6, "mint", "Token Localization", "Harshit Kandpal",
        "Bridging subword token gaps merges fragmented attack detections into coherent malicious spans.",
        "8.0×", "loss weighting on positive injection tokens (O vs B-INJ / I-INJ)"
    )

    tb6_e = s6.shapes.add_textbox(Inches(3.8), Inches(0.55), Inches(9.2), Inches(0.5))
    tf6_e = tb6_e.text_frame
    p = tf6_e.paragraphs[0]
    p.text = "Raw Subword Token Confidences¹ vs. Gap-Bridged Injection Spans² (G ≤ 2)"
    p.font.name = "Noto Sans"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    if TOKEN_GAP_FIG_PATH.exists():
        s6.shapes.add_picture(str(TOKEN_GAP_FIG_PATH), Inches(3.8), Inches(1.15), width=Inches(9.2))

    add_footer(s6, 6, 11, "Sources: 1/ sudarshana/models/token_head.py; 2/ sudarshana/redaction/redact.py (predict_spans).", "Lead: Harshit Kandpal")

    # =========================================================================
    # SLIDE 7: HARSHIT KANDPAL - STRUCTURE-AWARE REDACTION (CHART-ONLY EXHIBIT)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_white_bg(s7)
    add_panel(s7, "orchid", 3.6)
    add_panel_content(
        s7, "orchid", "Surgical Redaction", "Harshit Kandpal",
        "Syntactic container expansion prevents prose sentence splitters from destroying markdown tables and code.",
        "2.5×", "relative expansion cap: max(150 chars, 2.5x raw span)"
    )

    tb7_e = s7.shapes.add_textbox(Inches(3.8), Inches(0.55), Inches(9.2), Inches(0.5))
    tf7_e = tb7_e.text_frame
    p = tf7_e.paragraphs[0]
    p.text = "Baseline PySBD (35%)¹ vs. Hierarchical Structure-Aware Redaction (50%)²"
    p.font.name = "Noto Sans"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    if REDACTION_FIG_PATH.exists():
        s7.shapes.add_picture(str(REDACTION_FIG_PATH), Inches(3.8), Inches(1.15), width=Inches(9.2))

    add_footer(s7, 7, 11, "Sources: 1/ sudarshana/redaction/redact.py; 2/ results/model_redaction.json (N=125 attacks).", "Lead: Harshit Kandpal")

    # =========================================================================
    # SLIDE 8: HARSHIT KANDPAL - EMPIRICAL AUDIT SCORECARD (TABLE-ONLY EXHIBIT)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_white_bg(s8)
    add_panel(s8, "mint", 3.6)
    add_panel_content(
        s8, "mint", "Audit Scorecard", "Harshit Kandpal",
        "Structure-aware redaction cut unnecessary document blocks in half while preserving 57.84% of clean context.",
        "68.0%", "selective redaction rate (up from 43.20% in baseline; +24.8% absolute gain)"
    )

    tb8_e = s8.shapes.add_textbox(Inches(3.8), Inches(0.55), Inches(9.2), Inches(0.5))
    tf8_e = tb8_e.text_frame
    p = tf8_e.paragraphs[0]
    p.text = "Baseline PySBD configuration¹ vs. Structure-Aware Engine² on held-out test split (n=206)"
    p.font.name = "Noto Sans"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    rows, cols = 8, 5
    table_shape = s8.shapes.add_table(rows, cols, Inches(3.8), Inches(1.2), Inches(9.2), Inches(5.4))
    table = table_shape.table
    table.columns[0].width = Inches(2.6)
    table.columns[1].width = Inches(1.5)
    table.columns[2].width = Inches(1.8)
    table.columns[3].width = Inches(1.5)
    table.columns[4].width = Inches(1.8)

    headers = ["Metric", "Baseline (PySBD 35%)", "Structure-Aware Engine", "95% Confidence Interval", "Impact / Gain"]
    for c_idx, h_text in enumerate(headers):
        format_cell(table.cell(0, c_idx), h_text, bold=True, font_size=10, text_color=C_WHITE, bg_color=C_NAVY)

    scorecard_data = [
        ("Selective Redaction Rate", "43.20%", "68.00%", "[59.40%, 75.50%]", "+24.8% gain (85/125 sanitized)", C_TEAL),
        ("Fallback Block Rate", "56.80%", "32.00%", "[24.50%, 40.60%]", "Blocks cut nearly in half", C_PURPLE),
        ("Legitimate Content Retained", "38.55%", "57.84%", "[50.50%, 64.50%]", "+19.3% absolute utility gain", C_GREEN),
        ("Span Detection Recall (1-to-1)", "99.20%", "99.20%", "[95.61%, 99.86%]", "124 / 125 attacks localized", C_BODY),
        ("Gold Span Character Coverage", "98.51%", "98.96%", "[97.70%, 99.90%]", "98.96% attack chars excised", C_BODY),
        ("Residual Leaking Samples", "17.60% (22/125)", "5.60% (7/125)", "[2.70%, 11.10%]", "68% drop in leaking samples", C_GREEN),
        ("CPU Inference Latency", "—", "0.871s median", "p95: 1.040s", "~15–20 docs/sec throughput", C_BODY),
    ]

    for r_idx, (m_lbl, b_val, s_val, ci_val, imp_val, imp_col) in enumerate(scorecard_data, start=1):
        bg = RGBColor(248, 250, 252) if r_idx % 2 == 1 else C_WHITE
        format_cell(table.cell(r_idx, 0), m_lbl, bold=True, font_size=9, text_color=C_NAVY, bg_color=bg)
        format_cell(table.cell(r_idx, 1), b_val, bold=False, font_size=9, text_color=C_BODY, bg_color=bg)
        format_cell(table.cell(r_idx, 2), s_val, bold=True, font_size=9.5, text_color=C_TEAL, bg_color=bg)
        format_cell(table.cell(r_idx, 3), ci_val, bold=False, font_size=8.5, text_color=C_NOTES, bg_color=bg)
        format_cell(table.cell(r_idx, 4), imp_val, bold=True, font_size=9, text_color=imp_col, bg_color=bg)

    add_footer(s8, 8, 11, "Sources: 1/ reports/metrics.json; 2/ reports/audit/raw_test_predictions.jsonl (n=206 total).", "Lead: Harshit Kandpal · Wilson & Bootstrap 95% CIs reported")

    # =========================================================================
    # SLIDE 9: HARSHIT KANDPAL - DEFENSE & ERROR TAXONOMY (CHART-ONLY EXHIBIT)
    # =========================================================================
    s9 = prs.slides.add_slide(blank_layout)
    set_white_bg(s9)
    add_panel(s9, "orchid", 3.6)
    add_panel_content(
        s9, "orchid", "Scientific Rigor", "Harshit Kandpal",
        "Rigorous error triage reveals that 94.4% of attacks were neutralized with zero surviving characters.",
        ">450,000", "supervised token decisions across 386 documents (DeBERTa transfer learning)"
    )

    tb9_e = s9.shapes.add_textbox(Inches(3.8), Inches(0.55), Inches(9.2), Inches(0.5))
    tf9_e = tb9_e.text_frame
    p = tf9_e.paragraphs[0]
    p.text = "Attack Payload Neutralization (94.4%)¹ & Token-Level Sample Rigor² (>450k)"
    p.font.name = "Noto Sans"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY

    if ERROR_TAXONOMY_FIG_PATH.exists():
        s9.shapes.add_picture(str(ERROR_TAXONOMY_FIG_PATH), Inches(3.8), Inches(1.15), width=Inches(9.2))

    add_footer(s9, 9, 11, "Sources: 1/ reports/p4_token_head_report.md Section 4; 2/ data/splits/DATA_CARD.md.", "Lead: Harshit Kandpal")

    # =========================================================================
    # SLIDE 10: PLACEHOLDER (Kinshuk Gupta)
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_white_bg(s10)
    add_panel(s10, "mint", 3.6)
    add_panel_content(
        s10, "mint", "Downstream & UI", "Kinshuk Gupta",
        "Sanitized carrier documents allow downstream LLMs to maintain a 98.40% protected success rate.",
        "0.871s", "median CPU inference latency per 1,362-token document"
    )

    tb10_e = s10.shapes.add_textbox(Inches(4.0), Inches(0.8), Inches(8.8), Inches(6.0))
    tf10_e = tb10_e.text_frame
    tf10_e.word_wrap = True

    p = tf10_e.paragraphs[0]
    p.text = "Downstream LLM Evaluation & UI Dashboard [PLACEHOLDER — Kinshuk Gupta]"
    p.font.name = "Noto Sans"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(20)

    kinshuk_items = [
        ("Designated Component Exhibit:", "This slide is reserved for Kinshuk Gupta to present downstream LLM generation and UI.", C_PURPLE),
        ("Protected Success Rate (PSR = 98.40%):", "Under attack, downstream LLMs complete user tasks successfully with near-zero hijacking.", C_TEAL),
        ("Semantic Generation Utility (ROUGE-1):", "Preserving clean carrier context directly maintains downstream generation quality.", C_GRAY),
        ("Interactive Web Dashboard (ui/index.html):", "Real-time visualization of token heatmaps, boundary expansions, and live redaction.", C_BODY)
    ]
    for h, b, col in kinshuk_items:
        p = tf10_e.add_paragraph()
        p.text = f"■  {h}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(2)
        p = tf10_e.add_paragraph()
        p.text = f"    {b}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(11)
        p.font.color.rgb = C_BODY
        p.space_after = Pt(14)

    add_footer(s10, 10, 11, "Sources: 1/ results/downstream_evaluation.json; 2/ ui/server.py.", "Lead: Kinshuk Gupta")

    # =========================================================================
    # SLIDE 11: CONCLUSION & DEFENSE READINESS
    # =========================================================================
    s11 = prs.slides.add_slide(blank_layout)
    set_white_bg(s11)
    add_panel(s11, "mint", 3.6)
    add_panel_content(
        s11, "mint", "Conclusion & Defense", "All Authors",
        "Project Sudarshana demonstrates that surgical token-level redaction is a production-ready reality.",
        "78 / 78", "unit tests passing across all repository components (100%)"
    )

    tb11_e = s11.shapes.add_textbox(Inches(4.0), Inches(0.8), Inches(8.8), Inches(6.0))
    tf11_e = tb11_e.text_frame
    tf11_e.word_wrap = True

    p = tf11_e.paragraphs[0]
    p.text = "Pre-redaction security vulnerability¹ vs. Post-redaction RAG context preservation²"
    p.font.name = "Noto Sans"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_NAVY
    p.space_after = Pt(20)

    c_items = [
        ("Unified Modular Defense Pipeline:", "Seamless handoffs from Tanmay Jaiswal (data/spans) to Utkarsh Chatterjee (conditioning), Sujal Som (DeBERTa model), Harshit Kandpal (surgical redaction), and Kinshuk Gupta (downstream evaluation).", C_PURPLE),
        ("Harshit Kandpal Core Breakthrough:", "Eliminated the false-block crisis (cut blocks from 56.8% to 32.0%), boosted selective redaction to 68.0%, and retained 57.8% clean context.", C_TEAL),
        ("Scientific Rigor & Reproducibility:", "Strict train/val/test isolation, Wilson/Bootstrap 95% intervals, frozen checkpoints, and reproducible audit runner.", C_GREEN),
        ("Deployment Readiness:", "Sub-second CPU latency (<0.9s), zero GPU cluster requirement, and interactive live demonstration UI.", C_BODY)
    ]
    for h, b, col in c_items:
        p = tf11_e.add_paragraph()
        p.text = f"■  {h}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(13)
        p.font.bold = True
        p.font.color.rgb = col
        p.space_after = Pt(2)
        p = tf11_e.add_paragraph()
        p.text = f"    {b}"
        p.font.name = "Noto Sans"
        p.font.size = Pt(11)
        p.font.color.rgb = C_BODY
        p.space_after = Pt(14)

    add_footer(s11, 11, 11, "Sources: 1/ reports/FINAL_PROJECT_REPORT.md; 2/ reports/FINAL_PROJECT_REPORT.pdf.", "Project Sudarshana · Academic Session 2026")

    out_pptx = REPORTS_DIR / "Sudarshana_a16z_Presentation.pptx"
    prs.save(str(out_pptx))
    prs.save(str(REPORTS_DIR / "Sudarshana_Project_Presentation.pptx"))
    print(f"PowerPoint Presentation saved to: {out_pptx}")
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

    /* Series Colors */
    .s-purple { color: #65327d; border-bottom: 2px solid #65327d; }
    .s-teal { color: #089582; border-bottom: 2px solid #089582; }
    .s-green { color: #62b33a; border-bottom: 2px solid #62b33a; }
    .s-orange { color: #f0a656; border-bottom: 2px solid #f0a656; }
    .s-gray { color: #b8bfcb; border-bottom: 2px solid #b8bfcb; }

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

    /* Placeholder Box */
    .placeholder-box {
      border: 2px dashed #b8bfcb;
      background: #f8fafc;
      border-radius: 8px;
    }
  </style>
</head>
<body>

  <!-- Top Progress Bar -->
  <div class="fixed top-0 left-0 w-full h-1 bg-[#e6e9ee] z-50">
    <div id="progress" class="h-full bg-[#089582] transition-all duration-300" style="width: 9%;"></div>
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
          <div><strong class="text-[#089582]">Harshit Kandpal:</strong> Token Head & Redaction Engine</div>
          <div><strong>Tanmay Jaiswal:</strong> Data Pipeline & Schema</div>
          <div><strong>Utkarsh Chatterjee:</strong> Task Conditioning & Disambiguation</div>
          <div><strong>Sujal Som:</strong> Model Architecture & Training Ladder</div>
          <div><strong>Kinshuk Gupta:</strong> Downstream Evaluation & UI</div>
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

        <div class="grid grid-cols-1 md:grid-cols-2 gap-4 max-w-4xl">
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <div class="flex items-center gap-2 mb-1">
              <span class="w-3 h-3 rounded-full bg-[#b8bfcb]"></span>
              <strong class="text-[13px] text-[#173462]">Layer 1 · Normalizer & Encodings</strong>
            </div>
            <p class="text-[12px] text-[#2a3b5c] pl-5">Tanmay Jaiswal: NFKC normalization, zero-width stripping, and Base64/Hex decoders.</p>
          </div>
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <div class="flex items-center gap-2 mb-1">
              <span class="w-3 h-3 rounded-full bg-[#65327d]"></span>
              <strong class="text-[13px] text-[#173462]">Layer 2 · Task Conditioning</strong>
            </div>
            <p class="text-[12px] text-[#2a3b5c] pl-5">Utkarsh Chatterjee: <code class="mono text-[11px] bg-slate-100 px-1 py-0.5 rounded">[CLS] user_task [SEP] chunk [SEP]</code> cross-attention.</p>
          </div>
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white">
            <div class="flex items-center gap-2 mb-1">
              <span class="w-3 h-3 rounded-full bg-[#f0a656]"></span>
              <strong class="text-[13px] text-[#173462]">Layer 3 · Multi-Task DeBERTa Model</strong>
            </div>
            <p class="text-[12px] text-[#2a3b5c] pl-5">Sujal Som: DeBERTa-v3 multi-task filter & 6-step ablation ladder.</p>
          </div>
          <div class="p-4 border-2 border-[#089582] rounded-lg bg-[#f0fdf4]">
            <div class="flex items-center gap-2 mb-1">
              <span class="w-3 h-3 rounded-full bg-[#089582]"></span>
              <strong class="text-[13px] text-[#089582]">Layer 4 · Token Head & Surgical Redaction</strong>
            </div>
            <p class="text-[12px] text-[#2a3b5c] pl-5">Harshit Kandpal: Subword gap-bridging & structure-aware boundary expander.</p>
          </div>
          <div class="p-4 border border-[#e6e9ee] rounded-lg bg-white md:col-span-2">
            <div class="flex items-center gap-2 mb-1">
              <span class="w-3 h-3 rounded-full bg-[#62b33a]"></span>
              <strong class="text-[13px] text-[#173462]">Layer 5 · Downstream LLM Evaluation & UI</strong>
            </div>
            <p class="text-[12px] text-[#2a3b5c] pl-5">Kinshuk Gupta: 98.40% Protected Success Rate, semantic ROUGE-1 evaluation, and interactive web dashboard.</p>
          </div>
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 1 / 11 · Use [←] and [→] keys</span>
        <span>Sources: 1/ Microsoft BIPIA Benchmark (50 Attack Types); Sudarshana architecture (2026).</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 2: PLACEHOLDER (Tanmay Jaiswal) -->
  <div class="slide" id="slide-2">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">Data Engineering · Tanmay Jaiswal</div>
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
          Unified Data Pipeline & Schema [PLACEHOLDER — Tanmay Jaiswal]
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">Designated component exhibit reserved for Tanmay Jaiswal.</p>

        <div class="placeholder-box p-6 space-y-4 max-w-3xl">
          <div class="text-[11px] font-bold text-[#f0a656] uppercase tracking-wider">[ PLACEHOLDER: DATA PIPELINE ]</div>
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
        <span>Slide 2 / 11 · Placeholder</span>
        <span>Sources: 1/ data/splits/train.jsonl (n=386); 2/ data/splits/test.jsonl (n=206).</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 3: PLACEHOLDER (Utkarsh Chatterjee) -->
  <div class="slide" id="slide-3">
    <div class="panel panel-orchid">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">Task Conditioning · Utkarsh Chatterjee</div>
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
          Pair-Input Conditioning & Disambiguation [PLACEHOLDER — Utkarsh Chatterjee]
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">Designated component exhibit reserved for Utkarsh Chatterjee.</p>

        <div class="placeholder-box p-6 space-y-4 max-w-3xl">
          <div class="text-[11px] font-bold text-[#f0a656] uppercase tracking-wider">[ PLACEHOLDER: PAIR INPUT ]</div>
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
        <span>Slide 3 / 11 · Placeholder</span>
        <span>Sources: 1/ data/splits/hard_negatives.jsonl; 2/ results/calibration_day6.json.</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 4: PLACEHOLDER (Sujal Som) -->
  <div class="slide" id="slide-4">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">Model & Ladder · Sujal Som</div>
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
          Multi-Task Model & 6-Step Ablation Ladder [PLACEHOLDER — Sujal Som]
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">Designated component exhibit reserved for Sujal Som.</p>

        <div class="placeholder-box p-6 space-y-4 max-w-3xl">
          <div class="text-[11px] font-bold text-[#f0a656] uppercase tracking-wider">[ PLACEHOLDER: MODEL & ABLATION LADDER ]</div>
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
        <span>Slide 4 / 11 · Placeholder</span>
        <span>Sources: 1/ reports/ablation_ladder_tables.json; 2/ models/final_step1..6/.</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 5: HARSHIT KANDPAL - PIPELINE FLOWCHART (NEW!) -->
  <div class="slide" id="slide-5">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">Pipeline Architecture · Harshit Kandpal</div>
        <h2 class="text-[27px] font-bold text-[#173462] leading-[1.25] tracking-tight mb-6">
          Surgical redaction operates across seven sequential stages from raw windows to <span class="hl-lime">sanitized carrier output</span>.
        </h2>
        <div class="mt-8">
          <div class="text-[44px] font-bold text-[#089582] leading-none mb-1">7 Stages</div>
          <div class="text-[11px] italic text-[#2a3b5c]">end-to-end token detection and container expansion pipeline</div>
        </div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-1">
          Harshit Kandpal: Token Detection & Structure-Aware Redaction Pipeline¹
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-3">
          Sliding-Window Inference → Subword Gap-Bridging → Syntactic Container Expansion → Fallback Calibration.
  <!-- SLIDE 5: HARSHIT KANDPAL - PIPELINE FLOWCHART (CHART-ONLY EXHIBIT) -->
  <div class="slide" id="slide-5">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">Pipeline Architecture · Harshit Kandpal</div>
        <h2 class="text-[27px] font-bold text-[#173462] leading-[1.25] tracking-tight mb-6">
          Surgical redaction operates across seven sequential stages from raw windows to <span class="hl-lime">sanitized carrier output</span>.
        </h2>
        <div class="mt-8">
          <div class="text-[44px] font-bold text-[#089582] leading-none mb-1">7 Stages</div>
          <div class="text-[11px] italic text-[#2a3b5c]">end-to-end token detection and container expansion pipeline</div>
        </div>
      </div>
      <div class="text-[10px] text-[#8a93a3]">©2026  Sudarshana</div>
    </div>
    <div class="exhibit">
      <div>
        <h2 class="text-[18px] font-bold text-[#173462] mb-3">
          <span class="s-purple">Token Classification Sequence Head¹</span> & <span class="s-teal">Structure-Aware Redaction Pipeline²</span>
        </h2>

        <!-- High-Res Flowchart Image Embedded (Chart-only, zero lingering text) -->
        <div class="border border-[#e6e9ee] rounded-lg p-2 bg-white flex justify-center shadow-sm">
          <img src="figures/p4_pipeline_flowchart.png" alt="Pipeline Flowchart" class="max-h-[520px] w-auto object-contain">
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 5 / 11 · Harshit Kandpal (Flowchart)</span>
        <span>Sources: 1/ sudarshana/models/token_head.py; 2/ sudarshana/redaction/redact.py.</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 6: HARSHIT KANDPAL - TOKEN HEAD & GAP-BRIDGING (CHART-ONLY EXHIBIT) -->
  <div class="slide" id="slide-6">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">Token Localization · Harshit Kandpal</div>
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
        <h2 class="text-[18px] font-bold text-[#173462] mb-3">
          <span class="s-purple">Raw Subword Token Confidences¹</span> vs. <span class="s-teal">Gap-Bridged Injection Spans² (G ≤ 2)</span>
        </h2>

        <!-- High-Res Token Gap-Bridging Diagram Embedded (Chart-only, zero lingering text) -->
        <div class="border border-[#e6e9ee] rounded-lg p-2 bg-white flex justify-center shadow-sm">
          <img src="figures/token_gap_bridging.png" alt="Token Gap-Bridging Diagram" class="max-h-[520px] w-auto object-contain">
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 6 / 11 · Harshit Kandpal (Token Localization)</span>
        <span>Sources: 1/ sudarshana/models/token_head.py; 2/ sudarshana/redaction/redact.py (predict_spans).</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 7: HARSHIT KANDPAL - STRUCTURE-AWARE REDACTION (CHART-ONLY EXHIBIT) -->
  <div class="slide" id="slide-7">
    <div class="panel panel-orchid">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">Surgical Redaction · Harshit Kandpal</div>
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
        <h2 class="text-[18px] font-bold text-[#173462] mb-3">
          <span class="s-purple">Baseline PySBD (35%)¹</span> vs. <span class="s-teal">Hierarchical Structure-Aware Redaction (50%)²</span>
        </h2>

        <!-- High-Res Empirical Redaction Chart Embedded (Chart-only, zero lingering text) -->
        <div class="border border-[#e6e9ee] rounded-lg p-2 bg-white flex justify-center shadow-sm">
          <img src="figures/rep_f5_redaction.png" alt="Redaction Trade-Off Chart" class="max-h-[520px] w-auto object-contain">
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 7 / 11 · Harshit Kandpal (Redaction Engine)</span>
        <span>Sources: 1/ sudarshana/redaction/redact.py; 2/ results/model_redaction.json (N=125 attacks).</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 8: HARSHIT KANDPAL - EMPIRICAL AUDIT SCORECARD (TABLE-ONLY EXHIBIT) -->
  <div class="slide" id="slide-8">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">Audit Scorecard · Harshit Kandpal</div>
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
        <h2 class="text-[18px] font-bold text-[#173462] mb-3">
          <span class="s-purple">Baseline PySBD configuration¹</span> vs. <span class="s-teal">Structure-Aware Engine²</span> on held-out test split (n=206)
        </h2>

        <div class="border border-[#e6e9ee] rounded-lg overflow-hidden max-w-5xl shadow-sm">
          <table class="w-full text-left text-[12px] border-collapse">
            <thead>
              <tr class="bg-[#173462] text-white text-[11px] font-bold uppercase tracking-wider">
                <th class="py-3 px-4">Metric</th>
                <th class="py-3 px-4 text-[#ffe6f1]">Baseline (PySBD 35%)</th>
                <th class="py-3 px-4 text-[#c5e9df]">Structure-Aware Engine (Audited)</th>
                <th class="py-3 px-4 text-slate-300">95% Confidence Interval</th>
                <th class="py-3 px-4 text-[#e2fe6b]">Impact / Gain</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-[#e6e9ee] text-[#2a3b5c]">
              <tr class="bg-[#f8fafc]">
                <td class="py-2.5 px-4 font-bold text-[#173462]">Selective Redaction Rate</td>
                <td class="py-2.5 px-4">43.20%</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">68.00%</td>
                <td class="py-2.5 px-4 mono text-[11px]">[59.40%, 75.50%]</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">+24.8% gain (85/125 sanitized)</td>
              </tr>
              <tr class="bg-white">
                <td class="py-2.5 px-4 font-bold text-[#173462]">Fallback Block Rate</td>
                <td class="py-2.5 px-4 text-red-600">56.80%</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">32.00%</td>
                <td class="py-2.5 px-4 mono text-[11px]">[24.50%, 40.60%]</td>
                <td class="py-2.5 px-4 font-bold text-[#65327d]">Blocks cut nearly in half</td>
              </tr>
              <tr class="bg-[#f8fafc]">
                <td class="py-2.5 px-4 font-bold text-[#173462]">Legitimate Content Retained</td>
                <td class="py-2.5 px-4">38.55%</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">57.84%</td>
                <td class="py-2.5 px-4 mono text-[11px]">[50.50%, 64.50%]</td>
                <td class="py-2.5 px-4 font-bold text-[#62b33a]">+19.3% absolute utility gain</td>
              </tr>
              <tr class="bg-white">
                <td class="py-2.5 px-4 font-bold text-[#173462]">Span Detection Recall (1-to-1)</td>
                <td class="py-2.5 px-4">99.20%</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">99.20%</td>
                <td class="py-2.5 px-4 mono text-[11px]">[95.61%, 99.86%]</td>
                <td class="py-2.5 px-4">124 / 125 attacks localized</td>
              </tr>
              <tr class="bg-[#f8fafc]">
                <td class="py-2.5 px-4 font-bold text-[#173462]">Gold Span Character Coverage</td>
                <td class="py-2.5 px-4">98.51%</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">98.96%</td>
                <td class="py-2.5 px-4 mono text-[11px]">[97.70%, 99.90%]</td>
                <td class="py-2.5 px-4">98.96% attack chars excised</td>
              </tr>
              <tr class="bg-white">
                <td class="py-2.5 px-4 font-bold text-[#173462]">Residual Leaking Samples</td>
                <td class="py-2.5 px-4 text-red-600">17.60% (22/125)</td>
                <td class="py-2.5 px-4 font-bold text-[#089582]">5.60% (7/125)</td>
                <td class="py-2.5 px-4 mono text-[11px]">[2.70%, 11.10%]</td>
                <td class="py-2.5 px-4 font-bold text-[#62b33a]">68% drop in leaking samples</td>
              </tr>
              <tr class="bg-[#f8fafc]">
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
        <span>Slide 8 / 11 · Harshit Kandpal (Audit Scorecard)</span>
        <span>Sources: 1/ reports/metrics.json; 2/ reports/audit/raw_test_predictions.jsonl (n=206 total).</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 9: HARSHIT KANDPAL - DEFENSE & ERROR TAXONOMY (CHART-ONLY EXHIBIT) -->
  <div class="slide" id="slide-9">
    <div class="panel panel-orchid">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">Scientific Rigor · Harshit Kandpal</div>
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
        <h2 class="text-[18px] font-bold text-[#173462] mb-3">
          <span class="s-teal">Attack Payload Neutralization (94.4%)¹</span> & <span class="s-purple">Token-Level Sample Rigor² (>450k)</span>
        </h2>

        <!-- High-Res Error Taxonomy Chart Embedded (Chart-only, zero lingering text) -->
        <div class="border border-[#e6e9ee] rounded-lg p-2 bg-white flex justify-center shadow-sm">
          <img src="figures/error_taxonomy_chart.png" alt="Error Taxonomy & Sample Rigor Chart" class="max-h-[520px] w-auto object-contain">
        </div>
      </div>
      <div class="flex justify-between items-center text-[10px] text-[#5f6b80] pt-4 border-t border-[#e6e9ee]">
        <span>Slide 9 / 11 · Harshit Kandpal (Defense Rationale)</span>
        <span>Sources: 1/ reports/p4_token_head_report.md Section 4; 2/ data/splits/DATA_CARD.md.</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 10: PLACEHOLDER (Kinshuk Gupta) -->
  <div class="slide" id="slide-10">
    <div class="panel panel-mint">
      <div>
        <div class="text-[10px] font-bold text-[#173462] uppercase tracking-wider mb-4">Downstream & UI · Kinshuk Gupta</div>
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
          Downstream LLM Evaluation & UI Dashboard [PLACEHOLDER — Kinshuk Gupta]
        </h2>
        <p class="text-[12px] italic text-[#5f6b80] mb-6">Designated component exhibit reserved for Kinshuk Gupta.</p>

        <div class="placeholder-box p-6 space-y-4 max-w-3xl">
          <div class="text-[11px] font-bold text-[#f0a656] uppercase tracking-wider">[ PLACEHOLDER: DOWNSTREAM & UI ]</div>
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
        <span>Slide 10 / 11 · Placeholder</span>
        <span>Sources: 1/ results/downstream_evaluation.json; 2/ ui/server.py.</span>
      </div>
    </div>
  </div>

  <!-- SLIDE 11: CONCLUSION & DEFENSE READINESS -->
  <div class="slide" id="slide-11">
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
            <p class="text-[12px] text-[#2a3b5c]">Seamless integration from Tanmay Jaiswal (data/spans) to Utkarsh Chatterjee (conditioning), Sujal Som (DeBERTa model), Harshit Kandpal (surgical redaction), and Kinshuk Gupta (downstream evaluation).</p>
          </div>
          <div class="p-4 border-2 border-[#089582] rounded-lg bg-[#f0fdf4]">
            <h3 class="text-[13px] font-bold text-[#089582] mb-1">Harshit Kandpal Core Breakthrough</h3>
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
        <span>Slide 11 / 11 · Thank you! Open for Questions.</span>
        <span>Sources: 1/ reports/FINAL_PROJECT_REPORT.md; 2/ reports/FINAL_PROJECT_REPORT.pdf.</span>
      </div>
    </div>
  </div>

  <!-- Bottom Floating Navigation Controls -->
  <div class="fixed bottom-4 right-6 flex items-center gap-2 bg-white/95 border border-[#e6e9ee] px-3 py-1.5 rounded-full shadow-lg">
    <button id="prevBtn" class="text-[#5f6b80] hover:text-[#173462] px-2 py-1 text-sm font-bold">← Prev</button>
    <span id="slideIndicator" class="text-xs font-semibold text-[#089582] px-2">1 / 11</span>
    <button id="nextBtn" class="text-[#5f6b80] hover:text-[#173462] px-2 py-1 text-sm font-bold">Next →</button>
  </div>

  <script>
    let currentSlide = 1;
    const totalSlides = 11;

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
    with open(REPORTS_DIR / "presentation.html", "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML Presentation saved to: {out_html}")
    return out_html


def main():
    print("=== Generating 'a16z Style' Presentation Decks (Zero PX tags, Flowchart Embedded) ===")
    out_pptx = create_pptx_deck()
    out_html = create_html_deck()
    print("\n✅ a16z Style Slide Decks Generated:")
    print(f"  1. PowerPoint File (.pptx): {out_pptx}")
    print(f"  2. Interactive HTML Deck:   {out_html}")


if __name__ == "__main__":
    main()
