"""Generate professional PowerPoint (.pptx) and interactive HTML slide deck for Project Sudarshana.
P4 (Harshit Kandpal) content is fully fleshed out with detailed architecture, algorithms,
figures, exact audited test metrics, and oral defense talking points.
P1, P2, P3, and P5 are structured as clean, designated team placeholders.
"""

import os
import sys
from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

REPO_ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = REPO_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Color Palette (Dark High-Tech Theme)
C_BG = RGBColor(11, 17, 32)           # #0B1120 Deep Dark Slate
C_CARD = RGBColor(22, 30, 49)         # #161E31 Darker Card Slate
C_CARD_BORDER = RGBColor(38, 52, 79)   # #26344F Subtle Slate Border
C_CYAN = RGBColor(6, 182, 212)        # #06B6D4 Vibrant Cyan
C_TEAL = RGBColor(16, 185, 129)       # #10B981 Emerald / Teal
C_TEXT_WHITE = RGBColor(248, 250, 252) # #F8FAFC Bright Text
C_TEXT_MUTED = RGBColor(148, 163, 184) # #94A3B8 Slate Muted Text
C_ACCENT_AMBER = RGBColor(245, 158, 11)# #F59E0B Warning / Notice
C_PLACEHOLDER_BORDER = RGBColor(71, 85, 105) # #475569 Dashed-look Slate


def create_presentation():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    blank_layout = prs.slide_layouts[6]

    def set_slide_background(slide):
        background = slide.background
        fill = background.fill
        fill.solid()
        fill.fore_color.rgb = C_BG

    def add_header(slide, title_text, category_text="PROJECT SUDARSHANA"):
        tb_cat = slide.shapes.add_textbox(Inches(0.8), Inches(0.4), Inches(11.5), Inches(0.35))
        tf_cat = tb_cat.text_frame
        tf_cat.word_wrap = True
        tf_cat.margin_left = tf_cat.margin_right = tf_cat.margin_top = tf_cat.margin_bottom = 0
        p_cat = tf_cat.paragraphs[0]
        p_cat.text = category_text.upper()
        p_cat.font.size = Pt(11)
        p_cat.font.bold = True
        p_cat.font.color.rgb = C_CYAN

        tb_title = slide.shapes.add_textbox(Inches(0.8), Inches(0.75), Inches(11.5), Inches(0.7))
        tf_title = tb_title.text_frame
        tf_title.word_wrap = True
        tf_title.margin_left = tf_title.margin_right = tf_title.margin_top = tf_title.margin_bottom = 0
        p_title = tf_title.paragraphs[0]
        p_title.text = title_text
        p_title.font.size = Pt(24)
        p_title.font.bold = True
        p_title.font.color.rgb = C_TEXT_WHITE

    def add_card(slide, left, top, width, height, bg_color=C_CARD, border_color=C_CARD_BORDER):
        shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
        shape.fill.solid()
        shape.fill.fore_color.rgb = bg_color
        shape.line.color.rgb = border_color
        shape.line.width = Pt(1)
        return shape

    def add_placeholder_slide(title, member_name, role_name, description_points):
        s = prs.slides.add_slide(blank_layout)
        set_slide_background(s)
        add_header(s, title, f"TEAM COMPONENT: {role_name}")

        card = add_card(s, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2), C_CARD, C_PLACEHOLDER_BORDER)
        tb = s.shapes.add_textbox(Inches(1.2), Inches(2.0), Inches(11.0), Inches(4.4))
        tf = tb.text_frame
        tf.word_wrap = True

        p = tf.paragraphs[0]
        p.text = f"[ PLACEHOLDER: {role_name.upper()} ]"
        p.font.size = Pt(14)
        p.font.bold = True
        p.font.color.rgb = C_ACCENT_AMBER
        p.space_after = Pt(10)

        p = tf.add_paragraph()
        p.text = f"Lead: {member_name}"
        p.font.size = Pt(22)
        p.font.bold = True
        p.font.color.rgb = C_TEXT_WHITE
        p.space_after = Pt(16)

        p = tf.add_paragraph()
        p.text = "This slide is designated for the component owner to insert their specific architectural diagrams, methodology, and experimental results:"
        p.font.size = Pt(13)
        p.font.color.rgb = C_TEXT_MUTED
        p.space_after = Pt(14)

        for pt in description_points:
            p = tf.add_paragraph()
            p.text = f"• {pt}"
            p.font.size = Pt(13)
            p.font.color.rgb = C_TEXT_WHITE
            p.space_after = Pt(8)

        p = tf.add_paragraph()
        p.text = f"(Content to be finalized by {member_name})"
        p.font.size = Pt(11)
        p.font.italic = True
        p.font.color.rgb = C_TEXT_MUTED

        notes = s.notes_slide.notes_text_frame
        notes.text = f"Placeholder slide for {member_name} ({role_name})."
        return s

    # =========================================================================
    # SLIDE 1: Title Slide (Full Team Listed)
    # =========================================================================
    s1 = prs.slides.add_slide(blank_layout)
    set_slide_background(s1)
    add_card(s1, Inches(0.8), Inches(1.0), Inches(11.733), Inches(5.5), C_CARD, C_CYAN)

    tb1 = s1.shapes.add_textbox(Inches(1.4), Inches(1.5), Inches(10.5), Inches(4.5))
    tf1 = tb1.text_frame
    tf1.word_wrap = True

    p = tf1.paragraphs[0]
    p.text = "PROJECT SUDARSHANA"
    p.font.size = Pt(14)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(14)

    p = tf1.add_paragraph()
    p.text = "Indirect Prompt Injection Detection and\nSurgical Redaction Pipeline for Secure RAG"
    p.font.size = Pt(32)
    p.font.bold = True
    p.font.color.rgb = C_TEXT_WHITE
    p.space_after = Pt(18)

    p = tf1.add_paragraph()
    p.text = "Overcoming Binary All-or-Nothing Blocking Through Structure-Aware Span Localization"
    p.font.size = Pt(16)
    p.font.color.rgb = C_TEXT_MUTED
    p.space_after = Pt(36)

    p = tf1.add_paragraph()
    p.text = "Team Members & Component Distribution:"
    p.font.size = Pt(13)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    p.space_after = Pt(8)

    p = tf1.add_paragraph()
    p.text = "• Tanmay Jaiswal (P1: Data Pipeline & Preprocessing)\n• Utkarsh Chatterjee (P2: Pair Input & Disambiguation)\n• Sujal Som (P3: Model Architecture & Ablation Ladder)\n• Harshit Kandpal (P4: Token Head & Surgical Redaction Engine)\n• Kinshuk Gupta (P5: Downstream LLM Evaluation & Reporting)\nIIT Bhilai — Academic Session 2026"
    p.font.size = Pt(12)
    p.font.color.rgb = C_TEXT_WHITE

    s1.notes_slide.notes_text_frame.text = "Good morning, Sir. Today we present Project Sudarshana, our modular defense pipeline for detecting indirect prompt injections in RAG systems and performing surgical redaction to preserve clean context for downstream LLMs."

    # =========================================================================
    # SLIDE 2: PLACEHOLDER P1 (Tanmay Jaiswal)
    # =========================================================================
    add_placeholder_slide(
        title="P1: Unified Data Pipeline & Schema",
        member_name="Tanmay Jaiswal",
        role_name="P1 (Data Engineering Lead)",
        description_points=[
            "Microsoft BIPIA benchmark integration across 5 application domains (Text QA, Code QA, Table QA, Email, Summarization).",
            "Exact character offset verification (start_char, end_char) with 100% round-trip match.",
            "Zero-leakage split strategy: 50 disjoint attack types (25 train / 25 strictly held-out test).",
            "Curated instructional Hard Negatives dataset (Linux tutorials, READMEs, recipes)."
        ]
    )

    # =========================================================================
    # SLIDE 3: PLACEHOLDER P2 (Utkarsh Chatterjee)
    # =========================================================================
    add_placeholder_slide(
        title="P2: Pair-Input Conditioning & Disambiguation",
        member_name="Utkarsh Chatterjee",
        role_name="P2 (Input Conditioning Lead)",
        description_points=[
            "Conditioning architecture: [CLS] user_task [SEP] document_chunk [SEP].",
            "Disambiguating legitimate imperative commands from malicious prompt injections.",
            "Hard-negative false positive rate reduction: Dropped from 83.33% to 16.67% under calibrated thresholds.",
            "Cross-attention attention maps preserving recall on task-relevant attacks."
        ]
    )

    # =========================================================================
    # SLIDE 4: PLACEHOLDER P3 (Sujal Som)
    # =========================================================================
    add_placeholder_slide(
        title="P3: Multi-Task Model & 6-Step Ablation Ladder",
        member_name="Sujal Som",
        role_name="P3 (Model Architecture & Training Lead)",
        description_points=[
            "DeBERTa-v3 multi-task filter architecture with decoupled heads and joint loss (L_detect + 0.5 * L_tok).",
            "6-step ablation ladder: Step 1 (512 Baseline) to Step 6 (Adversarial Augmentations).",
            "Context preservation: 256-token chunking eliminating end-of-document truncation failures (100% end recall).",
            "Obfuscation suite evaluation: Robustness against Leetspeak, Homoglyphs, Zero-Width chars, and Base64 wrappers."
        ]
    )

    # =========================================================================
    # SLIDE 5: P4 TOKEN HEAD & INFERENCE (HARSHIT KANDPAL - FULL CONTENT)
    # =========================================================================
    s5 = prs.slides.add_slide(blank_layout)
    set_slide_background(s5)
    add_header(s5, "P4: Token Classification Head & Inference Mechanics", "P4: HARSHIT KANDPAL")

    add_card(s5, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), C_CARD, C_CYAN)
    tb5_1 = s5.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf5_1 = tb5_1.text_frame
    tf5_1.word_wrap = True

    p = tf5_1.paragraphs[0]
    p.text = "Sequence Tagging Architecture"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(12)

    p4_1_bullets = [
        ("Pretrained Backbone:", "microsoft/deberta-v3-xsmall (~22M active, 70.7M total parameters). Fast and efficient on commodity CPU."),
        ("3-Class Linear Head:", "TokenClassificationHead projecting hidden states (384 dim) -> 3 BIO labels (0: O, 1: B-INJ, 2: I-INJ)."),
        ("Solving Class Imbalance:", "Positive attack tokens represent only ~2%–4% of tokens. Loss is weighted with w = [1.0, 8.0, 8.0] to penalize missed attacks 8x more heavily:"),
        ("Loss Formulation:", "L_tok = - (1/N) ∑ w_yi log P(yi | xi). Joint loss hook: L_total = L_detect + 0.5 * L_tok.")
    ]
    for b_title, b_desc in p4_1_bullets:
        p = tf5_1.add_paragraph()
        p.text = f"• {b_title} {b_desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_WHITE
        p.space_after = Pt(8)

    add_card(s5, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.2), C_CARD, C_TEAL)
    tb5_2 = s5.shapes.add_textbox(Inches(7.2), Inches(1.8), Inches(5.0), Inches(4.8))
    tf5_2 = tb5_2.text_frame
    tf5_2.word_wrap = True

    p = tf5_2.paragraphs[0]
    p.text = "Inference & Subword Gap-Bridging"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    p.space_after = Pt(12)

    p4_2_bullets = [
        ("Sliding Windows & Max-Pooling:", "256-token windows with 64 stride. Overlapping probabilities max-pooled: P(inj) = P(B-INJ) + P(I-INJ). Completely eliminates truncation blind spots."),
        ("Decision Threshold:", "Tokens with P(inj) >= 0.35 flagged as malicious."),
        ("Subword Gap-Bridging (G <= 2):", "Connective tokens ('to', 'the', '.') frequently dip below threshold, breaking attacks into fragments. Unflagged gaps of <= 2 connective tokens are bridged into continuous spans."),
        ("Round-Trip Character Offsets:", "Token subwords mapped back with exact character precision to the raw document.")
    ]
    for b_title, b_desc in p4_2_bullets:
        p = tf5_2.add_paragraph()
        p.text = f"• {b_title} {b_desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_WHITE
        p.space_after = Pt(8)

    s5.notes_slide.notes_text_frame.text = "In my component, P4, I built the token sequence head mapping DeBERTa hidden states to BIO labels. To address 2% token rarity, I applied 8x loss weighting. To stop attacks from fragmenting, my gap-bridger connects subwords separated by 2 or fewer connective tokens."

    # =========================================================================
    # SLIDE 6: P4 STRUCTURE-AWARE REDACTION ENGINE (HARSHIT KANDPAL - FULL)
    # =========================================================================
    s6 = prs.slides.add_slide(blank_layout)
    set_slide_background(s6)
    add_header(s6, "P4: Hierarchical Structure-Aware Redaction Engine", "P4: HARSHIT KANDPAL")

    f5_path = FIGURES_DIR / "rep_f5_redaction.png"
    if f5_path.exists():
        s6.shapes.add_picture(str(f5_path), Inches(0.8), Inches(1.6), width=Inches(6.6))
    else:
        add_card(s6, Inches(0.8), Inches(1.6), Inches(6.6), Inches(5.2), C_CARD, C_CYAN)

    add_card(s6, Inches(7.7), Inches(1.6), Inches(4.8), Inches(5.2), C_CARD, C_CARD_BORDER)
    tb6 = s6.shapes.add_textbox(Inches(7.9), Inches(1.8), Inches(4.4), Inches(4.8))
    tf6 = tb6.text_frame
    tf6.word_wrap = True

    p = tf6.paragraphs[0]
    p.text = "Syntactic Container Expansion"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(10)

    p4_3_bullets = [
        ("The Failure of PySBD:", "Standard sentence segmentation collapsed on code and tables, causing 56.8% false document blocks."),
        ("Markdown Tables:", "Bounded strictly to the single enclosing row [\\n, \\n]. Surrounding legitimate data rows remain 100% intact."),
        ("Code & Python Docstrings:", "Expands to the enclosing block delimiter (```, ''', \"\"\", <!-- -->), fully excising multi-line script loops."),
        ("Comments & Prose:", "Expands across contiguous # or // comment lines; prose sentences strictly bounded within paragraph breaks (\\n\\n)."),
        ("Relative Expansion Cap:", "Bounded by max(150 chars, 2.5x raw span) to stop isolated false-positive tokens from deleting entire sections."),
        ("Calibrated 50% Fallback Block Rule:", "If total excised characters exceed 50% of document, safely block the document to prevent returning gibberish.")
    ]
    for b_title, b_desc in p4_3_bullets:
        p = tf6.add_paragraph()
        p.text = f"• {b_title} {b_desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_WHITE
        p.space_after = Pt(5)

    s6.notes_slide.notes_text_frame.text = "My main innovation in P4 is structure-aware boundary expansion. PySBD treats code and tables like English prose, mutilating syntax. My engine identifies the syntactic container: markdown tables expand only to the row, and code fences expand to the full block."

    # =========================================================================
    # SLIDE 7: P4 EMPIRICAL AUDIT SCORECARD (HARSHIT KANDPAL - FULL)
    # =========================================================================
    s7 = prs.slides.add_slide(blank_layout)
    set_slide_background(s7)
    add_header(s7, "P4: Empirical Audit Scorecard (Held-Out Test Set)", "P4: HARSHIT KANDPAL")

    add_card(s7, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2), C_CARD, C_CYAN)
    tb7 = s7.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(11.1), Inches(4.8))
    tf7 = tb7.text_frame
    tf7.word_wrap = True

    p = tf7.paragraphs[0]
    p.text = "Audited Performance on Held-Out Test Set (n=206 total, 125 attacked across 25 unseen types)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(12)

    scorecard_rows = [
        ("Selective Redaction Rate", "43.20%", "68.00% [59.4%, 75.5%]", "+24.8% absolute gain; 85/125 sanitized cleanly"),
        ("Fallback Block Rate", "56.80%", "32.00% [24.5%, 40.6%]", "Unnecessary full blocks cut nearly in half"),
        ("Legitimate Content Retained", "38.55%", "57.84% [50.5%, 64.5%]", "+19.3% absolute gain in RAG context utility"),
        ("Span Detection Recall (1-to-1)", "99.20%", "99.20% [95.6%, 99.9%]", "124 / 125 attacks localized across 25 unseen types"),
        ("Gold Span Character Coverage", "98.51%", "98.96% [97.7%, 99.9%]", "Share of malicious characters successfully excised"),
        ("Residual Payload Leakage", "17.60% (22 docs)", "5.60% (7 docs)", "68% drop in leaking samples (3 were just trailing \\n)"),
        ("Clean Counterfactuals (n=125)", "17.60% blocked", "11.20% blocked", "Clean content preserved: 76.19% [70.7%, 81.3%]"),
        ("CPU Inference Latency", "—", "0.871s median", "p95: 1.040s; ~15-20 docs/sec on AMD Ryzen 7 CPU")
    ]

    p = tf7.add_paragraph()
    p.text = f"{'Metric':<32} | {'Baseline (PySBD)':<18} | {'P4 Engine (Audited)':<25} | {'Impact / Key Takeaway'}"
    p.font.size = Pt(11)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    p.space_after = Pt(6)

    for m_name, m_base, m_v2, m_impact in scorecard_rows:
        p = tf7.add_paragraph()
        p.text = f"• {m_name:<30}: {m_base:<16} -> {m_v2:<23} ({m_impact})"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_WHITE
        p.space_after = Pt(3)

    s7.notes_slide.notes_text_frame.text = "Sir, here is my audited scorecard on 206 held-out test documents. Fallback blocks were cut from 56.8% down to 32%, selective redaction jumped to 68%, and legitimate context retention rose from 38.5% to 57.8% while maintaining 99.2% recall."

    # =========================================================================
    # SLIDE 8: P4 DEFENSE & ERROR TAXONOMY (HARSHIT KANDPAL - FULL)
    # =========================================================================
    s8 = prs.slides.add_slide(blank_layout)
    set_slide_background(s8)
    add_header(s8, "P4: Error Analysis & Scientific Rigor Defense", "P4: HARSHIT KANDPAL")

    add_card(s8, Inches(0.8), Inches(1.6), Inches(5.6), Inches(5.2), C_CARD, C_CYAN)
    tb8_1 = s8.shapes.add_textbox(Inches(1.1), Inches(1.8), Inches(5.0), Inches(4.8))
    tf8_1 = tb8_1.text_frame
    tf8_1.word_wrap = True

    p = tf8_1.paragraphs[0]
    p.text = "Error Taxonomy (7 Leaking Cases)"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN
    p.space_after = Pt(12)

    p8_1_bullets = [
        ("118 of 125 Attacks Sanitized (94.4%):", "Excised cleanly with zero residual payload surviving."),
        ("Whitespace Boundary Leaks (3 samples):", "Indices 31, 124, 199 leaked exactly 1 character ('\\n') where the detector cleanly stopped at the word boundary. Completely benign."),
        ("Subtle Table Prompt (1 sample):", "Index 3 leaked 13 chars in a table cell where confidence hovered at 0.34."),
        ("Multi-line Code Loops (2 samples):", "Indices 28 and 87 (scapy and multiprocessing loop). Directive was excised, but trailing loop lines exceeded the 2.5x expansion cap."),
        ("Multi-Sentence Feedback (1 sample):", "Index 66 had 42 surviving characters from a secondary sentence in a long feedback attack.")
    ]
    for b_title, b_desc in p8_1_bullets:
        p = tf8_1.add_paragraph()
        p.text = f"• {b_title} {b_desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_WHITE
        p.space_after = Pt(8)

    add_card(s8, Inches(6.9), Inches(1.6), Inches(5.6), Inches(5.2), C_CARD, C_TEAL)
    tb8_2 = s8.shapes.add_textbox(Inches(7.2), Inches(1.8), Inches(5.0), Inches(4.8))
    tf8_2 = tb8_2.text_frame
    tf8_2.word_wrap = True

    p = tf8_2.paragraphs[0]
    p.text = "Defense: 'Why 386 Documents?'"
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    p.space_after = Pt(12)

    p8_2_bullets = [
        ("Token Decisions > 450,000:", "We trained a token-level sequence tagger, not a document classifier. Across 386 docs, the model trained on >450k supervised token instances."),
        ("Pretrained Transfer Learning:", "DeBERTa-v3 already possesses dense English syntax and code representations; token extraction requires only hundreds of annotated examples (like CoNLL-2003)."),
        ("Exact Span Ground Truth:", "99% of prompt injection datasets only provide binary 0/1 labels. BIPIA provides verified character offsets with 100% round-trip fidelity."),
        ("Proven Generalization:", "Tested on 25 unseen attack categories. Achieving 99.20% recall proves the model generalized without memorization.")
    ]
    for b_title, b_desc in p8_2_bullets:
        p = tf8_2.add_paragraph()
        p.text = f"• {b_title} {b_desc}"
        p.font.size = Pt(11)
        p.font.color.rgb = C_TEXT_WHITE
        p.space_after = Pt(8)

    s8.notes_slide.notes_text_frame.text = "If asked about the 7 residual leaks: 3 were harmless trailing newlines, and 4 were trailing code loops. If asked about dataset size: 386 documents equals over 450,000 supervised token decisions, and achieving 99.2% recall on 25 unseen attack categories proves out-of-distribution generalization."

    # =========================================================================
    # SLIDE 9: PLACEHOLDER P5 (Kinshuk Gupta)
    # =========================================================================
    add_placeholder_slide(
        title="P5: Downstream LLM Evaluation & UI Dashboard",
        member_name="Kinshuk Gupta",
        role_name="P5 (Downstream Evaluation & Reporting Lead)",
        description_points=[
            "Downstream generation harness: Evaluating Protected Success Rate (PSR = 98.40%) and semantic ROUGE-1 on clean queries.",
            "Decision threshold calibration on validation data.",
            "CPU latency profiling: Median 0.871s per 1,362-token document; throughput ~15-20 docs/sec.",
            "Interactive web UI dashboard (ui/index.html & ui/server.py) demonstrating real-time token heatmaps."
        ]
    )

    # =========================================================================
    # SLIDE 10: Conclusion & Defense Readiness
    # =========================================================================
    s10 = prs.slides.add_slide(blank_layout)
    set_slide_background(s10)
    add_header(s10, "Conclusion & Project Defense Readiness", "PROJECT COMPLETION")

    add_card(s10, Inches(0.8), Inches(1.6), Inches(11.733), Inches(5.2), C_CARD, C_TEAL)
    tb10 = s10.shapes.add_textbox(Inches(1.2), Inches(1.8), Inches(11.0), Inches(4.8))
    tf10 = tb10.text_frame
    tf10.word_wrap = True

    p = tf10.paragraphs[0]
    p.text = "Unified Sudarshana Deliverables: 100% Complete & Verified"
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = C_TEAL
    p.space_after = Pt(14)

    conclusion_points = [
        ("Modular Pipeline:", "Clean handoffs between P1 (data/spans), P2 (task conditioning), P3 (DeBERTa model), P4 (surgical redaction), and P5 (downstream evaluation)."),
        ("P4 Contribution:", "Eliminated the false-block crisis (cut from 56.8% to 32.0%), boosted selective redaction to 68.0%, and retained 57.8% clean context with 98.96% character coverage."),
        ("Scientific Integrity:", "Strict train/val/test isolation; 95% Wilson and Bootstrap confidence intervals; frozen checkpoints and reproducible audit runner."),
        ("Production Viability:", "Sub-second CPU latency (<0.9s), all 78 repository tests passing (100%), and interactive web dashboard.")
    ]
    for c_title, c_desc in conclusion_points:
        p = tf10.add_paragraph()
        p.text = f"• {c_title} {c_desc}"
        p.font.size = Pt(13)
        p.font.color.rgb = C_TEXT_WHITE
        p.space_after = Pt(12)

    p = tf10.add_paragraph()
    p.text = "Thank you! Open for Questions."
    p.font.size = Pt(16)
    p.font.bold = True
    p.font.color.rgb = C_CYAN

    s10.notes_slide.notes_text_frame.text = "In conclusion, Sudarshana bridges the gap between security and utility. My component, P4, demonstrated that surgical redaction is viable and effective without discarding entire documents. All 78 tests pass, and we are ready for your questions."

    out_pptx = REPORTS_DIR / "Sudarshana_Project_Presentation.pptx"
    prs.save(str(out_pptx))
    print(f"PowerPoint Presentation successfully saved to: {out_pptx}")
    return out_pptx


def create_html_presentation():
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Project Sudarshana: Defense & Presentation Deck</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    body { font-family: 'Inter', sans-serif; background: #0B1120; color: #F8FAFC; overflow-x: hidden; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .slide { display: none; min-height: 100vh; padding: 2.5rem 4rem; }
    .slide.active { display: flex; flex-direction: column; justify-content: space-between; }
    .card { background: #161E31; border: 1px solid #26344F; border-radius: 1rem; }
    .placeholder-card { background: #111827; border: 2px dashed #475569; border-radius: 1rem; }
    .glow-cyan { box-shadow: 0 0 25px rgba(6, 182, 212, 0.15); border-color: rgba(6, 182, 212, 0.4); }
    .glow-teal { box-shadow: 0 0 25px rgba(16, 185, 129, 0.15); border-color: rgba(16, 185, 129, 0.4); }
  </style>
</head>
<body class="bg-[#0B1120]">

  <!-- Top Progress Bar -->
  <div class="fixed top-0 left-0 w-full h-1.5 bg-slate-800 z-50">
    <div id="progress" class="h-full bg-cyan-400 transition-all duration-300" style="width: 10%;"></div>
  </div>

  <!-- SLIDE 1: Title -->
  <div class="slide active" id="slide-1">
    <div class="flex items-center justify-between text-xs font-semibold tracking-wider text-cyan-400 uppercase">
      <span>Project Sudarshana</span>
      <span>Presentation Deck</span>
    </div>
    <div class="my-auto max-w-5xl">
      <div class="inline-block px-3 py-1 mb-4 text-xs font-semibold tracking-wider text-cyan-400 uppercase bg-cyan-950/60 border border-cyan-800/60 rounded-full">Secure RAG Defense</div>
      <h1 class="text-5xl font-extrabold text-white leading-tight mb-4">
        Indirect Prompt Injection Detection and<br>
        <span class="text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 via-teal-300 to-emerald-400">Surgical Redaction Pipeline</span> for Secure RAG
      </h1>
      <p class="text-xl text-slate-400 mb-8 max-w-3xl">
        Overcoming Binary All-or-Nothing Blocking Through Structure-Aware Span Localization.
      </p>
      <div class="card p-5 bg-slate-900/80 border-slate-700/60">
        <div class="text-xs font-bold text-teal-400 uppercase tracking-wider mb-2">Team Members & Component Distribution:</div>
        <div class="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm text-slate-300">
          <div><strong class="text-white">Tanmay Jaiswal:</strong> P1 (Data Pipeline & Schema)</div>
          <div><strong class="text-white">Utkarsh Chatterjee:</strong> P2 (Pair Input & Conditioning)</div>
          <div><strong class="text-white">Sujal Som:</strong> P3 (Model Architecture & Ladder)</div>
          <div><strong class="text-cyan-400 font-bold">Harshit Kandpal:</strong> P4 (Token Head & Redaction)</div>
          <div><strong class="text-white">Kinshuk Gupta:</strong> P5 (Downstream Eval & UI)</div>
          <div class="text-slate-400 italic">IIT Bhilai — Academic Session 2026</div>
        </div>
      </div>
    </div>
    <div class="text-xs text-slate-500 flex justify-between">
      <span>Use [←] and [→] keys to navigate</span>
      <span>Slide 1 / 10</span>
    </div>
  </div>

  <!-- SLIDE 2: PLACEHOLDER P1 (Tanmay Jaiswal) -->
  <div class="slide" id="slide-2">
    <div>
      <div class="text-xs font-semibold tracking-wider text-amber-400 uppercase">Team Component: P1</div>
      <h2 class="text-3xl font-bold text-white mt-1">P1: Unified Data Pipeline & Schema</h2>
    </div>
    <div class="my-auto max-w-4xl mx-auto w-full">
      <div class="placeholder-card p-8 space-y-4">
        <div class="text-amber-400 font-bold text-sm tracking-wider uppercase">[ PLACEHOLDER: P1 DATA PIPELINE ]</div>
        <h3 class="text-2xl font-bold text-white">Lead: Tanmay Jaiswal</h3>
        <p class="text-slate-400 text-sm">This slide is designated for Tanmay Jaiswal to present the data engineering pipeline:</p>
        <ul class="space-y-2 text-slate-300 text-sm">
          <li>• Microsoft BIPIA benchmark across 5 application domains (Text QA, Code QA, Table QA, Email, Summarization).</li>
          <li>• Exact character offset verification (start_char, end_char) with 100% round-trip match.</li>
          <li>• Zero-leakage split strategy: 50 disjoint attack types (25 train / 25 strictly held-out test).</li>
          <li>• Curated instructional Hard Negatives dataset (Linux tutorials, READMEs, recipes).</li>
        </ul>
        <div class="text-xs text-slate-500 italic pt-2">(Content to be finalized by Tanmay Jaiswal)</div>
      </div>
    </div>
    <div class="text-xs text-slate-500 flex justify-between">
      <span>Placeholder: P1 Component</span>
      <span>Slide 2 / 10</span>
    </div>
  </div>

  <!-- SLIDE 3: PLACEHOLDER P2 (Utkarsh Chatterjee) -->
  <div class="slide" id="slide-3">
    <div>
      <div class="text-xs font-semibold tracking-wider text-amber-400 uppercase">Team Component: P2</div>
      <h2 class="text-3xl font-bold text-white mt-1">P2: Pair-Input Conditioning & Disambiguation</h2>
    </div>
    <div class="my-auto max-w-4xl mx-auto w-full">
      <div class="placeholder-card p-8 space-y-4">
        <div class="text-amber-400 font-bold text-sm tracking-wider uppercase">[ PLACEHOLDER: P2 PAIR INPUT ]</div>
        <h3 class="text-2xl font-bold text-white">Lead: Utkarsh Chatterjee</h3>
        <p class="text-slate-400 text-sm">This slide is designated for Utkarsh Chatterjee to present the pair-input cross-attention mechanism:</p>
        <ul class="space-y-2 text-slate-300 text-sm">
          <li>• Conditioning architecture: <code class="mono text-xs bg-slate-800 px-1 py-0.5 rounded text-amber-300">[CLS] user_task [SEP] document_chunk [SEP]</code>.</li>
          <li>• Disambiguating legitimate imperative commands from malicious prompt injections.</li>
          <li>• Hard-negative false positive rate reduction: Dropped from 83.33% to 16.67% under calibrated thresholds.</li>
          <li>• Cross-attention maps preserving recall on task-relevant attacks.</li>
        </ul>
        <div class="text-xs text-slate-500 italic pt-2">(Content to be finalized by Utkarsh Chatterjee)</div>
      </div>
    </div>
    <div class="text-xs text-slate-500 flex justify-between">
      <span>Placeholder: P2 Component</span>
      <span>Slide 3 / 10</span>
    </div>
  </div>

  <!-- SLIDE 4: PLACEHOLDER P3 (Sujal Som) -->
  <div class="slide" id="slide-4">
    <div>
      <div class="text-xs font-semibold tracking-wider text-amber-400 uppercase">Team Component: P3</div>
      <h2 class="text-3xl font-bold text-white mt-1">P3: Multi-Task Model & 6-Step Ablation Ladder</h2>
    </div>
    <div class="my-auto max-w-4xl mx-auto w-full">
      <div class="placeholder-card p-8 space-y-4">
        <div class="text-amber-400 font-bold text-sm tracking-wider uppercase">[ PLACEHOLDER: P3 MODEL & ABLATION LADDER ]</div>
        <h3 class="text-2xl font-bold text-white">Lead: Sujal Som</h3>
        <p class="text-slate-400 text-sm">This slide is designated for Sujal Som to present the model architecture and ablation ladder:</p>
        <ul class="space-y-2 text-slate-300 text-sm">
          <li>• DeBERTa-v3 multi-task filter architecture with decoupled heads and joint loss (<code class="mono text-xs">L_detect + 0.5 * L_tok</code>).</li>
          <li>• 6-step ablation ladder: Step 1 (512 Baseline) to Step 6 (Adversarial Augmentations).</li>
          <li>• Context preservation: 256-token chunking eliminating end-of-document truncation failures (100% end recall).</li>
          <li>• Obfuscation suite evaluation: Robustness against Leetspeak, Homoglyphs, Zero-Width chars, and Base64 wrappers.</li>
        </ul>
        <div class="text-xs text-slate-500 italic pt-2">(Content to be finalized by Sujal Som)</div>
      </div>
    </div>
    <div class="text-xs text-slate-500 flex justify-between">
      <span>Placeholder: P3 Component</span>
      <span>Slide 4 / 10</span>
    </div>
  </div>

  <!-- SLIDE 5: P4 TOKEN HEAD & INFERENCE (HARSHIT KANDPAL - FULL CONTENT) -->
  <div class="slide" id="slide-5">
    <div>
      <div class="text-xs font-semibold tracking-wider text-cyan-400 uppercase">P4: Token Localization</div>
      <h2 class="text-3xl font-bold text-white mt-1">P4: Token Classification Head & Inference Mechanics (Harshit Kandpal)</h2>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-8 my-auto">
      <div class="card p-6 glow-cyan">
        <h3 class="text-lg font-bold text-cyan-400 mb-3">Sequence Tagging Architecture</h3>
        <ul class="space-y-3 text-slate-300 text-sm">
          <li>• <strong class="text-white">Shared Backbone:</strong> DeBERTa-v3-xsmall (~22M active / 70.7M total parameters) operating on commodity CPU.</li>
          <li>• <strong class="text-white">3-Class Linear Head:</strong> Projects hidden dimension 384 → 3 BIO labels (<code class="mono text-xs text-cyan-300">0: O</code>, <code class="mono text-xs text-cyan-300">1: B-INJ</code>, <code class="mono text-xs text-cyan-300">2: I-INJ</code>).</li>
          <li>• <strong class="text-white">Class Weighting:</strong> Injection tokens make up only ~2%–4% of text. Loss is weighted with <code class="mono text-xs text-amber-300">w = [1.0, 8.0, 8.0]</code> to penalize missed attacks 8x more heavily.</li>
          <li>• <strong class="text-white">Token Masking:</strong> Special tokens (<code class="mono text-xs">[CLS]</code>, <code class="mono text-xs">[SEP]</code>, <code class="mono text-xs">[PAD]</code>) and query tokens masked with <code class="mono text-xs">-100</code>.</li>
        </ul>
      </div>
      <div class="card p-6 glow-teal">
        <h3 class="text-lg font-bold text-emerald-400 mb-3">Inference Mechanics & Gap-Bridging</h3>
        <ul class="space-y-3 text-slate-300 text-sm">
          <li>• <strong class="text-white">Sliding-Window Max-Pooling:</strong> 256-token windows with 64 stride. Overlapping probabilities max-pooled: <code class="mono text-xs text-teal-300">P(inj) = P(B-INJ) + P(I-INJ)</code>. Eliminates truncation limits.</li>
          <li>• <strong class="text-white">Decision Threshold:</strong> Tokens with <code class="mono text-xs">P ≥ 0.35</code> are flagged as malicious.</li>
          <li>• <strong class="text-white">Subword Gap-Bridging (G ≤ 2):</strong> Connective tokens ('to', 'the', '.') often dip below threshold. Unflagged gaps of ≤ 2 tokens between positive tokens are bridged into continuous spans.</li>
          <li>• <strong class="text-white">100% Round-Trip Character Offsets:</strong> Subwords mapped back to original document character positions.</li>
        </ul>
      </div>
    </div>
    <div class="text-xs text-slate-500 flex justify-between">
      <span>Weighted CrossEntropy solves class rarity; gap-bridging prevents attack fragmentation.</span>
      <span>Slide 5 / 10</span>
    </div>
  </div>

  <!-- SLIDE 6: P4 STRUCTURE-AWARE REDACTION (HARSHIT KANDPAL - FULL CONTENT) -->
  <div class="slide" id="slide-6">
    <div>
      <div class="text-xs font-semibold tracking-wider text-cyan-400 uppercase">P4: Surgical Redaction</div>
      <h2 class="text-3xl font-bold text-white mt-1">P4: Hierarchical Structure-Aware Redaction (Harshit Kandpal)</h2>
    </div>
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center my-auto">
      <div class="lg:col-span-7 flex justify-center">
        <img src="figures/rep_f5_redaction.png" alt="Redaction Trade-off" class="rounded-xl border border-slate-700 max-h-[460px] object-contain shadow-2xl">
      </div>
      <div class="lg:col-span-5 space-y-2.5 text-xs">
        <div class="card p-3">
          <strong class="text-red-400">The Problem with Naive Sentence Splitting (PySBD)</strong>
          <p class="text-slate-300 mt-0.5">In markdown tables and code blocks, naive splitting breaks syntax, causing 56.8% of documents to be completely blocked.</p>
        </div>
        <div class="card p-3">
          <strong class="text-cyan-400">Markdown Tables</strong>
          <p class="text-slate-300 mt-0.5">Bounded strictly to the single enclosing row <code class="mono">[\\n, \\n]</code>. Surrounding rows remain 100% intact.</p>
        </div>
        <div class="card p-3">
          <strong class="text-cyan-400">Code Blocks & Python Docstrings</strong>
          <p class="text-slate-300 mt-0.5">Expands to the enclosing block delimiter (<code class="mono">```</code>, <code class="mono">'''</code>, <code class="mono">\"\"\"</code>, <code class="mono">&lt;!-- --&gt;</code>), fully excising multi-line script loops.</p>
        </div>
        <div class="card p-3">
          <strong class="text-teal-400">Comments & Prose</strong>
          <p class="text-slate-300 mt-0.5">Expands to contiguous <code class="mono">#</code> or <code class="mono">//</code> lines; prose sentences bounded strictly within paragraphs (<code class="mono">\\n\\n</code>).</p>
        </div>
        <div class="card p-3">
          <strong class="text-amber-400">Relative Expansion Cap</strong>
          <p class="text-slate-300 mt-0.5">Bounded by <code class="mono">max(150 chars, 2.5x raw span)</code> to prevent isolated false-positive tokens from deleting entire paragraphs.</p>
        </div>
        <div class="card p-3">
          <strong class="text-emerald-400">Calibrated 50% Fallback Block Rule</strong>
          <p class="text-slate-300 mt-0.5">If total excised characters exceed 50%, safely block the entire document to prevent returning gibberish.</p>
        </div>
      </div>
    </div>
    <div class="text-xs text-slate-500 flex justify-between">
      <span>Syntactic containers preserve tables and code; 50% fallback ensures safety on short documents.</span>
      <span>Slide 6 / 10</span>
    </div>
  </div>

  <!-- SLIDE 7: P4 EMPIRICAL SCORECARD (HARSHIT KANDPAL - FULL CONTENT) -->
  <div class="slide" id="slide-7">
    <div>
      <div class="text-xs font-semibold tracking-wider text-cyan-400 uppercase">P4: Empirical Verification</div>
      <h2 class="text-3xl font-bold text-white mt-1">P4: Measured Audit Results on Held-Out Test Set (Harshit Kandpal)</h2>
    </div>
    <div class="my-auto">
      <div class="card p-6 glow-cyan overflow-x-auto">
        <table class="w-full text-left text-sm border-collapse">
          <thead>
            <tr class="border-b border-slate-700 text-xs font-semibold uppercase text-cyan-400">
              <th class="py-3 px-4">Metric</th>
              <th class="py-3 px-4">Baseline (PySBD 35%)</th>
              <th class="py-3 px-4">P4 Structure-Aware (50%)</th>
              <th class="py-3 px-4">95% Confidence Interval</th>
              <th class="py-3 px-4">Impact / Key Takeaway</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-slate-800 text-slate-200 text-xs">
            <tr>
              <td class="py-3 px-4 font-semibold text-white">Selective Redaction Rate</td>
              <td class="py-3 px-4 text-slate-400">43.20%</td>
              <td class="py-3 px-4 font-bold text-emerald-400">68.00%</td>
              <td class="py-3 px-4 mono text-slate-400">[59.40%, 75.50%]</td>
              <td class="py-3 px-4 text-emerald-300 font-semibold">+24.8% gain (85/125 sanitized cleanly)</td>
            </tr>
            <tr>
              <td class="py-3 px-4 font-semibold text-white">Fallback Block Rate</td>
              <td class="py-3 px-4 text-red-400">56.80%</td>
              <td class="py-3 px-4 font-bold text-emerald-400">32.00%</td>
              <td class="py-3 px-4 mono text-slate-400">[24.50%, 40.60%]</td>
              <td class="py-3 px-4 text-emerald-300 font-semibold">Unnecessary blocks cut nearly in half</td>
            </tr>
            <tr>
              <td class="py-3 px-4 font-semibold text-white">Legitimate Content Retained</td>
              <td class="py-3 px-4 text-slate-400">38.55%</td>
              <td class="py-3 px-4 font-bold text-emerald-400">57.84%</td>
              <td class="py-3 px-4 mono text-slate-400">[50.50%, 64.50%]</td>
              <td class="py-3 px-4 text-emerald-300 font-semibold">+19.3% absolute gain in RAG context utility</td>
            </tr>
            <tr>
              <td class="py-3 px-4 font-semibold text-white">Span Detection Recall (1-to-1)</td>
              <td class="py-3 px-4 text-slate-400">99.20%</td>
              <td class="py-3 px-4 font-bold text-emerald-400">99.20%</td>
              <td class="py-3 px-4 mono text-slate-400">[95.61%, 99.86%]</td>
              <td class="py-3 px-4 text-slate-300">124 / 125 attacks localized across 25 unseen types</td>
            </tr>
            <tr>
              <td class="py-3 px-4 font-semibold text-white">Gold Span Character Coverage</td>
              <td class="py-3 px-4 text-slate-400">98.51%</td>
              <td class="py-3 px-4 font-bold text-emerald-400">98.96%</td>
              <td class="py-3 px-4 mono text-slate-400">[97.70%, 99.90%]</td>
              <td class="py-3 px-4 text-slate-300">Share of malicious attack characters excised</td>
            </tr>
            <tr>
              <td class="py-3 px-4 font-semibold text-white">Samples with Residual Leaks</td>
              <td class="py-3 px-4 text-red-400">17.60% (22/125)</td>
              <td class="py-3 px-4 font-bold text-emerald-400">5.60% (7/125)</td>
              <td class="py-3 px-4 mono text-slate-400">[2.70%, 11.10%]</td>
              <td class="py-3 px-4 text-emerald-300 font-semibold">68% drop in leaks (3 were just trailing \n)</td>
            </tr>
            <tr>
              <td class="py-3 px-4 font-semibold text-white">Clean Counterfactuals (n=125)</td>
              <td class="py-3 px-4 text-slate-400">17.60% blocked</td>
              <td class="py-3 px-4 font-bold text-emerald-400">11.20% blocked</td>
              <td class="py-3 px-4 mono text-slate-400">[6.79%, 17.92%]</td>
              <td class="py-3 px-4 text-emerald-300 font-semibold">Clean content preserved: 76.19%</td>
            </tr>
            <tr>
              <td class="py-3 px-4 font-semibold text-white">CPU Inference Latency</td>
              <td class="py-3 px-4 text-slate-400">—</td>
              <td class="py-3 px-4 font-bold text-cyan-400">0.871s median</td>
              <td class="py-3 px-4 mono text-slate-400">p95: 1.040s</td>
              <td class="py-3 px-4 text-slate-300">Normalized 1.5k tokens: 0.959s; ~15-20 docs/sec</td>
            </tr>
          </tbody>
        </table>
      </div>
    </div>
    <div class="text-xs text-slate-500 flex justify-between">
      <span>Evaluated on 206 held-out test documents; zero test-set hyperparameter tuning.</span>
      <span>Slide 7 / 10</span>
    </div>
  </div>

  <!-- SLIDE 8: P4 DEFENSE & ERROR TAXONOMY (HARSHIT KANDPAL - FULL CONTENT) -->
  <div class="slide" id="slide-8">
    <div>
      <div class="text-xs font-semibold tracking-wider text-cyan-400 uppercase">P4: Scientific Rigor & Defense</div>
      <h2 class="text-3xl font-bold text-white mt-1">P4: Error Taxonomy & Dataset Defense (Harshit Kandpal)</h2>
    </div>
    <div class="grid grid-cols-1 md:grid-cols-2 gap-8 my-auto">
      <div class="card p-6 glow-cyan">
        <h3 class="text-lg font-bold text-cyan-400 mb-3">Error Taxonomy (7 Leaking Cases)</h3>
        <ul class="space-y-3 text-slate-300 text-sm">
          <li>• <strong class="text-white">118 of 125 Attacks Sanitized (94.4%):</strong> Excised cleanly with zero residual payload surviving.</li>
          <li>• <strong class="text-white">Whitespace Boundary Leaks (3 samples):</strong> Indices 31, 124, 199 leaked exactly 1 character ('\n') where the detector cleanly stopped at the word boundary. Completely benign.</li>
          <li>• <strong class="text-white">Subtle Table Prompt (1 sample):</strong> Index 3 leaked 13 chars in a table cell where confidence hovered at 0.34.</li>
          <li>• <strong class="text-white">Multi-line Code Loops (2 samples):</strong> Indices 28 and 87 (scapy and multiprocessing loop). Directive was excised, but trailing loop lines exceeded the 2.5x expansion cap.</li>
          <li>• <strong class="text-white">Multi-Sentence Feedback (1 sample):</strong> Index 66 had 42 surviving characters from a secondary sentence in a long feedback attack.</li>
        </ul>
      </div>
      <div class="card p-6 glow-teal">
        <h3 class="text-lg font-bold text-emerald-400 mb-3">Viva Defense: "Why 386 Documents?"</h3>
        <ul class="space-y-3 text-slate-300 text-sm">
          <li>• <strong class="text-white">Token Decisions > 450,000:</strong> We trained a token sequence tagger, not a document classifier. Across 386 docs, the model trained on >450k supervised token instances.</li>
          <li>• <strong class="text-white">Pretrained Transfer Learning:</strong> DeBERTa-v3 already possesses dense English syntax and code representations; token extraction requires only hundreds of annotated examples (like CoNLL-2003).</li>
          <li>• <strong class="text-white">Exact Span Ground Truth:</strong> 99% of prompt injection datasets only provide binary 0/1 labels. BIPIA provides verified character offsets with 100% round-trip fidelity.</li>
          <li>• <strong class="text-white">Proven Generalization:</strong> Tested on 25 unseen attack categories. Achieving 99.20% recall proves the model generalized without memorization.</li>
        </ul>
      </div>
    </div>
    <div class="text-xs text-slate-500 flex justify-between">
      <span>Clear analysis of surviving leaks and grounded scientific defense for viva questions.</span>
      <span>Slide 8 / 10</span>
    </div>
  </div>

  <!-- SLIDE 9: PLACEHOLDER P5 (Kinshuk Gupta) -->
  <div class="slide" id="slide-9">
    <div>
      <div class="text-xs font-semibold tracking-wider text-amber-400 uppercase">Team Component: P5</div>
      <h2 class="text-3xl font-bold text-white mt-1">P5: Downstream LLM Evaluation & UI Dashboard</h2>
    </div>
    <div class="my-auto max-w-4xl mx-auto w-full">
      <div class="placeholder-card p-8 space-y-4">
        <div class="text-amber-400 font-bold text-sm tracking-wider uppercase">[ PLACEHOLDER: P5 DOWNSTREAM & UI ]</div>
        <h3 class="text-2xl font-bold text-white">Lead: Kinshuk Gupta</h3>
        <p class="text-slate-400 text-sm">This slide is designated for Kinshuk Gupta to present the downstream LLM evaluation and UI:</p>
        <ul class="space-y-2 text-slate-300 text-sm">
          <li>• Downstream generation harness: Protected Success Rate (<code class="mono text-xs text-emerald-400">PSR = 98.40%</code>) and semantic ROUGE-1 on clean queries.</li>
          <li>• Decision threshold calibration on validation data.</li>
          <li>• CPU latency profiling: Median 0.871s per 1,362-token document; throughput ~15-20 docs/sec.</li>
          <li>• Interactive web UI dashboard (<code class="mono text-xs">ui/index.html & ui/server.py</code>) demonstrating real-time token heatmaps and live redaction.</li>
        </ul>
        <div class="text-xs text-slate-500 italic pt-2">(Content to be finalized by Kinshuk Gupta)</div>
      </div>
    </div>
    <div class="text-xs text-slate-500 flex justify-between">
      <span>Placeholder: P5 Component</span>
      <span>Slide 9 / 10</span>
    </div>
  </div>

  <!-- SLIDE 10: Conclusion -->
  <div class="slide" id="slide-10">
    <div>
      <div class="text-xs font-semibold tracking-wider text-teal-400 uppercase">Conclusion & Handoff</div>
      <h2 class="text-3xl font-bold text-white mt-1">Project Sudarshana: Conclusion & Defense Readiness</h2>
    </div>
    <div class="my-auto max-w-5xl">
      <div class="card p-8 glow-teal bg-slate-900/90 space-y-6">
        <h3 class="text-2xl font-bold text-white flex items-center gap-3">
          <span class="text-emerald-400">🛡️</span> Unified Defense Deliverables: 100% Complete & Verified
        </h3>
        <div class="grid grid-cols-1 md:grid-cols-2 gap-6 text-sm text-slate-300">
          <div class="space-y-3">
            <div>• <strong class="text-white">Modular Pipeline:</strong> Clean handoffs between P1 (data/spans), P2 (task conditioning), P3 (DeBERTa model), P4 (surgical redaction), and P5 (downstream evaluation).</div>
            <div>• <strong class="text-white">P4 Contribution:</strong> Eliminated the false-block crisis (cut from 56.8% to 32.0%), boosted selective redaction to 68.0%, and retained 57.8% clean context with 98.96% character coverage.</div>
          </div>
          <div class="space-y-3">
            <div>• <strong class="text-white">Scientific Integrity:</strong> Strict train/val/test isolation; 95% Wilson and Bootstrap confidence intervals; frozen checkpoints and reproducible audit runner.</div>
            <div>• <strong class="text-white">Production Viability:</strong> Sub-second CPU latency (&lt;0.9s), all 78 repository tests passing (100%), and interactive web dashboard.</div>
          </div>
        </div>
        <div class="pt-4 border-t border-slate-800 flex items-center justify-between">
          <div class="text-sm font-semibold text-cyan-400">Project Sudarshana — IIT Bhilai</div>
          <div class="text-sm font-bold text-emerald-400">Thank you! Open for Questions.</div>
        </div>
      </div>
    </div>
    <div class="text-xs text-slate-500 flex justify-between">
      <span>End of Presentation</span>
      <span>Slide 10 / 10</span>
    </div>
  </div>

  <!-- Bottom Floating Navigation Controls -->
  <div class="fixed bottom-4 right-6 flex items-center gap-2 bg-slate-900/90 border border-slate-700/80 px-3 py-1.5 rounded-full shadow-xl">
    <button id="prevBtn" class="text-slate-400 hover:text-white px-2 py-1 text-sm font-bold">← Prev</button>
    <span id="slideIndicator" class="text-xs font-semibold text-cyan-400 px-2">1 / 10</span>
    <button id="nextBtn" class="text-slate-400 hover:text-white px-2 py-1 text-sm font-bold">Next →</button>
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
    out_html = REPORTS_DIR / "presentation.html"
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"HTML Presentation successfully saved to: {out_html}")
    return out_html


def main():
    print("=== Generating Project Sudarshana Slide Decks (P4 Full / Others Placeholders) ===")
    out_pptx = create_presentation()
    out_html = create_html_presentation()
    print("\n✅ Slide Decks Generated:")
    print(f"  1. PowerPoint File (.pptx): {out_pptx}")
    print(f"  2. Interactive HTML Deck:   {out_html}")


if __name__ == "__main__":
    main()
