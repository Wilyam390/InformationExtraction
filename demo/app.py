"""
demo/app.py
===========
Gradio web interface for the Document Classifier + Invoice Extractor.

Launch:
    python3 demo/app.py
    # → opens http://localhost:7860
"""

import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from predict import predict, _load_models

# Pre-load models so first inference is instant
_load_models()

# ─────────────────────────────────────────────────────────────────────────────
# EXAMPLE DOCUMENTS
# ─────────────────────────────────────────────────────────────────────────────

EXAMPLES = {
    "Invoice (standard)": """INVOICE
Invoice Number: INV-4821
Invoice Date:   2024-03-15
Due Date:       2024-04-15

From: Acme Corp
To:   John Smith

Web Development        1    USD  3,500.00
Tax (20%)                   USD    700.00
──────────────────────────────────────────
TOTAL                       USD  4,200.00

Payment terms: Net 30 days""",

    "Invoice (freelancer)": """Freelance Invoice

Bill To: Maria García
From:    BlueSky Solutions

Invoice #: INV-0042
Date:      12 Jan 2024
Due:       11 Feb 2024

Services Rendered:
  - Consulting Services

Subtotal:   EUR 2,400.00
Tax:        EUR   480.00
─────────────────
Total Due:  EUR 2,880.00

Thank you for your business!""",

    "Email": """From: alex.johnson@techcorp.com
To: team@techcorp.com
Date: Mon, 08 Apr 2024 09:32:11
Subject: Q2 Performance Review – Action Required

Hi team,

Please complete your self-assessment forms by end of day Friday.
The annual performance review cycle kicks off next Monday and your
manager will need time to review before the 1-on-1s.

Link to the form: https://hr.techcorp.com/review

Any questions, reach out to HR directly.

Best,
Alex""",

    "Scientific Report": """Title: Attention Mechanisms for Document Classification

Abstract
We propose a hybrid attention-based model for multi-class document
classification. Our approach combines word-level and character-level
representations with a bidirectional attention pooling layer. Evaluated
on four public benchmarks, the model achieves 97.3% macro F1, outperforming
the previous best by 2.1 percentage points.

1. Introduction
Document classification remains a core task in NLP pipelines. Prior work
has relied on TF-IDF baselines (Zhang et al., 2015) or deep neural
architectures (Kim, 2014). However, the interplay between lexical and
structural features has not been thoroughly explored.

2. Methodology
Let X ∈ ℝ^(n×d) denote the input token embeddings. We apply a
multi-head self-attention layer: Attn(Q,K,V) = softmax(QK^T/√d)V.
The resulting context vectors are pooled and passed to a linear classifier.

3. Results
Table 1 shows ablation results across four datasets. Character-level
features contribute +1.4% F1 when combined with word-level attention.""",

    "Letter (tenancy)": """TENANCY AGREEMENT LETTER

Date: 15 April 2024
Landlord: Crestline Properties Ltd
Tenant:   Sophie Dubois

Property: 42 King Road, London, UK

This letter confirms the tenancy agreement for the above property
commencing 01 May 2024 for a fixed term of 12 months.

Monthly Rent: GBP 1,800
Deposit:      GBP 3,600

Both parties agree to the terms and conditions set out in the full
tenancy agreement dated 15 April 2024.

Signed: _________________________    Date: ___________
        Landlord / Agent

Signed: _________________________    Date: ___________
        Tenant""",
}

# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

LABEL_EMOJI = {
    "invoice":           "Invoice",
    "email":             "Email",
    "scientific_report": "Scientific Report",
    "letter":            "Letter",
}
LABEL_COLOR = {
    "invoice":           "#2563eb",
    "email":             "#16a34a",
    "scientific_report": "#9333ea",
    "letter":            "#d97706",
}

def run_pipeline(text: str, file):
    """Gradio callback — returns (label_html, confidence_html, extraction_html)."""
    # Prefer file upload
    if file is not None:
        inp = file.name
    elif text.strip():
        inp = text
    else:
        return (
            "<p style='color:red'>Please paste text or upload a file.</p>",
            "", ""
        )

    result = predict(inp)

    if "error" in result:
        return f"<p style='color:red'>{result['error']}</p>", "", ""

    label      = result["label"]
    color      = LABEL_COLOR.get(label, "#333")
    emoji_label = LABEL_EMOJI.get(label, label)
    ocr_used   = result.get("ocr_used", False)

    # ── Label card ──────────────────────────────────────────────────────────
    ocr_badge = (
        "<div style='margin-top:8px;font-size:0.6em;font-weight:normal;"
        "background:rgba(255,255,255,0.25);display:inline-block;"
        "padding:2px 10px;border-radius:20px;'>OCR</div>"
        if ocr_used else ""
    )
    label_html = f"""
    <div style='
        background:{color}; color:white; padding:20px 30px;
        border-radius:12px; font-size:1.5em; font-weight:bold;
        text-align:center; letter-spacing:0.03em;
    '>
        {emoji_label}{ocr_badge}
    </div>
    """

    # ── Confidence bars ─────────────────────────────────────────────────────
    conf = result["confidence"]
    max_score = max(conf.values()) if conf else 1
    min_score = min(conf.values()) if conf else 0
    score_range = max(max_score - min_score, 0.01)

    bars = ""
    for lbl, score in sorted(conf.items(), key=lambda x: -x[1]):
        pct   = int(((score - min_score) / score_range) * 100)
        c     = LABEL_COLOR.get(lbl, "#999")
        bold  = "font-weight:bold;" if lbl == label else ""
        bars += f"""
        <div style='margin:6px 0;{bold}'>
          <span style='display:inline-block;width:180px;font-size:0.9em;'>{LABEL_EMOJI.get(lbl,lbl)}</span>
          <div style='display:inline-block;width:{pct}%;min-width:4px;height:18px;
               background:{c};border-radius:4px;vertical-align:middle;'></div>
          <span style='margin-left:8px;font-size:0.85em;color:#555'>{score:+.2f}</span>
        </div>"""

    conf_html = f"<div style='padding:10px 0'>{bars}</div>"

    # ── Extraction table (invoices only) ────────────────────────────────────
    ext_html = ""
    if "extraction" in result:
        fields = result["extraction"]
        rows = ""
        for k, v in fields.items():
            display_k = k.replace("_", " ").title()
            status    = "Yes" if v else "No"
            val_str   = v if v else "<em style='color:#aaa'>not found</em>"
            rows += f"""
            <tr>
              <td style='padding:8px 12px;border-bottom:1px solid #eee;
                         font-weight:600;color:#444;width:180px;'>{display_k}</td>
              <td style='padding:8px 12px;border-bottom:1px solid #eee;'>{val_str}</td>
              <td style='padding:8px 12px;border-bottom:1px solid #eee;
                         text-align:center;'>{status}</td>
            </tr>"""

        ext_html = f"""
        <div style='margin-top:10px;'>
          <h4 style='color:#2563eb;margin-bottom:8px;'>Extracted Invoice Fields</h4>
          <table style='width:100%;border-collapse:collapse;
                        font-size:0.95em;border:1px solid #ddd;border-radius:8px;
                        overflow:hidden;'>
            <thead>
              <tr style='background:#eff6ff;'>
                <th style='padding:10px 12px;text-align:left;color:#1e3a8a;'>Field</th>
                <th style='padding:10px 12px;text-align:left;color:#1e3a8a;'>Value</th>
                <th style='padding:10px 12px;text-align:center;color:#1e3a8a;'>Status</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    return label_html, conf_html, ext_html


# ─────────────────────────────────────────────────────────────────────────────
# UI LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
#title  { text-align: center; }
#subtitle { text-align: center; color: #666; font-size: 0.95em; }
.example-btn { font-size: 0.82em !important; }
"""

with gr.Blocks(css=CSS, title="DocClassify — IE University") as demo:
    gr.HTML("""
    <h1 id='title'>DocClassify</h1>
    <p id='subtitle'>
      Document Classification &amp; Invoice Extraction &nbsp;|&nbsp;
      IE University · AI: Statistical Learning and Prediction
    </p>
    <hr/>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            text_in = gr.Textbox(
                label="Paste document text",
                lines=14,
                placeholder="Paste invoice, email, scientific report, or letter text here…",
            )
            file_in = gr.File(
                label="Or upload a file (.txt / .pdf / .png / .jpg / .tiff)",
                file_types=[".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"],
            )
            with gr.Row():
                submit_btn = gr.Button("Classify", variant="primary")
                clear_btn  = gr.Button("Clear")

            gr.Markdown("#### Quick Examples")
            for name, txt in EXAMPLES.items():
                gr.Button(name, elem_classes=["example-btn"]).click(
                    fn=lambda t=txt: t,
                    outputs=text_in,
                )

        with gr.Column(scale=1):
            gr.Markdown("### Results")
            label_out = gr.HTML(label="Predicted Category")
            conf_out  = gr.HTML(label="Confidence Scores")
            ext_out   = gr.HTML(label="Extracted Fields")

    submit_btn.click(
        fn=run_pipeline,
        inputs=[text_in, file_in],
        outputs=[label_out, conf_out, ext_out],
    )
    clear_btn.click(
        fn=lambda: ("", None, "", "", ""),
        outputs=[text_in, file_in, label_out, conf_out, ext_out],
    )

    gr.HTML("<hr/><p style='text-align:center;color:#aaa;font-size:0.8em;'>"
            "TF-IDF + LinearSVC · No generative AI · Traditional ML only</p>")


if __name__ == "__main__":
    demo.launch(share=False, show_error=True)
