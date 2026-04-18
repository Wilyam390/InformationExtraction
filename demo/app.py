"""
demo/app.py
===========
Gradio web interface for the Document Classifier + Invoice Extractor.
Redesigned UI — minimal, dark purple / black aesthetic.

Launch:
    python3 demo/app.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import gradio as gr
from predict import predict, _load_models

_load_models()

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

LABEL_DISPLAY = {
    "invoice":           "Invoice",
    "email":             "Email",
    "scientific_report": "Scientific Report",
    "letter":            "Letter",
}

LABEL_ACCENT = {
    "invoice":           ("#60a5fa", "rgba(96,165,250,0.08)",  "rgba(96,165,250,0.22)"),
    "email":             ("#34d399", "rgba(52,211,153,0.08)",   "rgba(52,211,153,0.22)"),
    "scientific_report": ("#c084fc", "rgba(192,132,252,0.08)",  "rgba(192,132,252,0.22)"),
    "letter":            ("#fbbf24", "rgba(251,191,36,0.08)",   "rgba(251,191,36,0.22)"),
}
BAR_COLORS = {
    "invoice": "#60a5fa", "email": "#34d399",
    "scientific_report": "#a855f7", "letter": "#fbbf24",
}

# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(text: str, file):
    if file is not None:
        inp = file.name
    elif text.strip():
        inp = text
    else:
        return "<p style='color:#f87171;font-size:.9em;'>Please paste text or upload a file.</p>", "", ""

    result = predict(inp)

    if "error" in result:
        return f"<p style='color:#f87171;font-size:.9em;'>{result['error']}</p>", "", ""

    label    = result["label"]
    ocr_used = result.get("ocr_used", False)
    display  = LABEL_DISPLAY.get(label, label)
    color, bg, border = LABEL_ACCENT.get(label, ("#e8e6f0", "rgba(255,255,255,0.04)", "rgba(255,255,255,0.12)"))

    ocr_badge = (
        "<span style='font-size:.6em;font-weight:600;letter-spacing:.08em;"
        "text-transform:uppercase;background:rgba(255,255,255,0.12);"
        "padding:2px 8px;border-radius:20px;margin-left:10px;'>OCR</span>"
        if ocr_used else ""
    )

    label_html = f"""
    <div style='background:{bg};border:1px solid {border};color:{color};
        padding:14px 18px;border-radius:12px;font-size:1.15em;font-weight:700;
        letter-spacing:-.02em;display:flex;align-items:center;justify-content:space-between;
        font-family:Geist,system-ui,sans-serif;margin-bottom:4px;'>
      <span style='display:flex;align-items:center;gap:10px;'>
        <span style='width:8px;height:8px;border-radius:50%;background:{color};
                     box-shadow:0 0 8px {color};display:inline-block;'></span>
        {display}{ocr_badge}
      </span>
    </div>"""

    conf        = result["confidence"]
    max_s       = max(conf.values()) if conf else 1
    min_s       = min(conf.values()) if conf else 0
    score_range = max(max_s - min_s, 0.01)

    bars = "<div style='display:flex;flex-direction:column;gap:9px;padding:4px 0;'>"
    for lbl, score in sorted(conf.items(), key=lambda x: -x[1]):
        pct  = int(((score - min_s) / score_range) * 100)
        c    = BAR_COLORS.get(lbl, "#888")
        name = LABEL_DISPLAY.get(lbl, lbl)
        bold = "font-weight:600;color:#e8e6f0;" if lbl == label else "color:#8a8799;"
        bars += f"""
        <div style='display:grid;grid-template-columns:138px 1fr 52px;align-items:center;gap:10px;'>
          <span style='font-size:.8em;font-family:Geist,system-ui,sans-serif;{bold}'>{name}</span>
          <div style='background:rgba(255,255,255,.06);border-radius:99px;height:5px;overflow:hidden;'>
            <div style='width:{pct}%;height:100%;background:{c};border-radius:99px;'></div>
          </div>
          <span style='font-size:.76em;color:#4a4858;text-align:right;font-family:Geist Mono,monospace;'>{score:+.2f}</span>
        </div>"""
    bars += "</div>"

    ext_html = ""
    if "extraction" in result:
        fields = result["extraction"]
        rows = ""
        for k, v in fields.items():
            dk    = k.replace("_", " ").title()
            found = bool(v)
            val_s = v if v else "<em style='color:#4a4858;'>not found</em>"
            stat  = (
                "<span style='font-size:.72em;font-weight:600;letter-spacing:.05em;"
                "text-transform:uppercase;padding:2px 7px;border-radius:5px;"
                "background:rgba(52,211,153,.1);color:#34d399;'>Found</span>"
                if found else
                "<span style='font-size:.72em;font-weight:600;letter-spacing:.05em;"
                "text-transform:uppercase;padding:2px 7px;border-radius:5px;"
                "background:rgba(255,255,255,.04);color:#4a4858;'>—</span>"
            )
            rows += f"""
            <tr>
              <td style='padding:9px 11px;border-bottom:1px solid rgba(255,255,255,.05);
                         font-weight:500;color:#e8e6f0;width:155px;font-size:.85em;'>{dk}</td>
              <td style='padding:9px 11px;border-bottom:1px solid rgba(255,255,255,.05);
                         color:#8a8799;font-size:.85em;'>{val_s}</td>
              <td style='padding:9px 11px;border-bottom:1px solid rgba(255,255,255,.05);
                         text-align:center;'>{stat}</td>
            </tr>"""

        ext_html = f"""
        <div style='margin-top:14px;'>
          <div style='font-size:.72em;font-weight:600;letter-spacing:.1em;text-transform:uppercase;
                      color:#a855f7;margin-bottom:9px;'>Extracted Invoice Fields</div>
          <table style='width:100%;border-collapse:collapse;font-family:Geist,system-ui,sans-serif;
                        border:1px solid rgba(160,120,255,.10);border-radius:10px;overflow:hidden;'>
            <thead>
              <tr style='background:rgba(168,85,247,.055);'>
                <th style='padding:9px 11px;text-align:left;font-size:.72em;letter-spacing:.07em;
                           text-transform:uppercase;color:#4a4858;font-weight:600;'>Field</th>
                <th style='padding:9px 11px;text-align:left;font-size:.72em;letter-spacing:.07em;
                           text-transform:uppercase;color:#4a4858;font-weight:600;'>Value</th>
                <th style='padding:9px 11px;text-align:center;font-size:.72em;letter-spacing:.07em;
                           text-transform:uppercase;color:#4a4858;font-weight:600;'>Status</th>
              </tr>
            </thead>
            <tbody>{rows}</tbody>
          </table>
        </div>"""

    return label_html, bars, ext_html


# ─────────────────────────────────────────────────────────────────────────────
# THEME
# ─────────────────────────────────────────────────────────────────────────────

theme = gr.themes.Base(
    primary_hue=gr.themes.Color(
        c50="#faf5ff", c100="#f3e8ff", c200="#e9d5ff", c300="#d8b4fe",
        c400="#c084fc", c500="#a855f7", c600="#9333ea", c700="#7c3aed",
        c800="#6d28d9", c900="#5b21b6", c950="#4c1d95",
    ),
    neutral_hue=gr.themes.Color(
        c50="#fafafa", c100="#f4f4f5", c200="#e4e4e7", c300="#d1d5db",
        c400="#9ca3af", c500="#6b7280", c600="#4b5563", c700="#374151",
        c800="#1f2937", c900="#111827", c950="#0a0a0f",
    ),
    font=[gr.themes.GoogleFont("Geist"), "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("Geist Mono"), "monospace"],
).set(
    body_background_fill="#0a0a0f",
    body_background_fill_dark="#0a0a0f",
    block_background_fill="#0f0f18",
    block_background_fill_dark="#0f0f18",
    input_background_fill="#141420",
    input_background_fill_dark="#141420",
    block_border_color="rgba(160,120,255,0.10)",
    block_border_color_dark="rgba(160,120,255,0.10)",
    block_border_width="1px",
    input_border_color="rgba(160,120,255,0.12)",
    input_border_color_dark="rgba(160,120,255,0.12)",
    input_border_color_focus="rgba(124,58,237,0.60)",
    input_border_color_focus_dark="rgba(124,58,237,0.60)",
    button_primary_background_fill="#7c3aed",
    button_primary_background_fill_dark="#7c3aed",
    button_primary_background_fill_hover="#a855f7",
    button_primary_background_fill_hover_dark="#a855f7",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#141420",
    button_secondary_background_fill_dark="#141420",
    button_secondary_background_fill_hover="#1a1a28",
    button_secondary_border_color="rgba(160,120,255,0.18)",
    body_text_color="#e8e6f0",
    body_text_color_dark="#e8e6f0",
    body_text_color_subdued="#8a8799",
    block_label_text_color="#8a8799",
    block_radius="12px",
    input_radius="10px",
    button_large_radius="8px",
    button_small_radius="6px",
    block_shadow="0 4px 20px rgba(0,0,0,0.50)",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Geist:wght@300..700&family=Geist+Mono:wght@400;500&display=swap');

body, .gradio-container { background: #0a0a0f !important; }
.gradio-container { font-family: 'Geist', system-ui, sans-serif !important; }

#app-header { padding: 18px 0 6px; border-bottom: 1px solid rgba(160,120,255,0.08); margin-bottom: 6px; }
#app-title  { font-size: 1.55em !important; font-weight: 700 !important; letter-spacing: -0.04em !important; color: #e8e6f0 !important; margin: 0 !important; }
#app-title .accent { color: #a855f7 !important; }
#app-subtitle { font-size: 0.8em !important; color: #4a4858 !important; margin-top: 4px !important; }

.label-section { font-size: 0.7em !important; font-weight: 600 !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; color: #a855f7 !important; }

textarea {
  font-family: 'Geist Mono', monospace !important;
  font-size: 0.8em !important; line-height: 1.75 !important;
  background: #141420 !important; border: 1px solid rgba(160,120,255,0.12) !important;
  color: #e8e6f0 !important; border-radius: 10px !important;
}
textarea:focus { border-color: rgba(124,58,237,0.55) !important; box-shadow: 0 0 0 3px rgba(109,40,217,0.18) !important; outline: none !important; }
textarea::placeholder { color: #4a4858 !important; }

.file-preview, .upload-container {
  background: rgba(168,85,247,0.03) !important;
  border: 1px dashed rgba(160,120,255,0.22) !important;
  border-radius: 10px !important;
}
.file-preview:hover, .upload-container:hover {
  border-color: rgba(168,85,247,0.5) !important;
  background: rgba(168,85,247,0.07) !important;
}

#classify-btn { background: #7c3aed !important; color: #fff !important; font-weight: 600 !important; border: none !important; transition: all 180ms cubic-bezier(0.16,1,0.3,1) !important; }
#classify-btn:hover { background: #a855f7 !important; box-shadow: 0 0 18px rgba(168,85,247,0.35) !important; }

#clear-btn { background: transparent !important; color: #8a8799 !important; border: 1px solid rgba(160,120,255,0.15) !important; }
#clear-btn:hover { color: #e8e6f0 !important; border-color: rgba(160,120,255,0.28) !important; background: #141420 !important; }

.example-btn button {
  font-size: 0.76em !important; padding: 4px 11px !important;
  border-radius: 99px !important; background: #141420 !important;
  border: 1px solid rgba(160,120,255,0.15) !important; color: #8a8799 !important;
  transition: all 180ms cubic-bezier(0.16,1,0.3,1) !important;
}
.example-btn button:hover { border-color: rgba(168,85,247,0.45) !important; color: #c084fc !important; background: rgba(168,85,247,0.07) !important; }

#app-footer { margin-top: 20px; padding: 14px 0 6px; border-top: 1px solid rgba(255,255,255,0.05); text-align: center; font-size: 0.7em; color: #4a4858; letter-spacing: 0.04em; }

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: rgba(160,120,255,0.18); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(168,85,247,0.38); }
"""

# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

with gr.Blocks(css=CSS, theme=theme, title="DocClassify — IE University") as demo:

    gr.HTML("""
    <div id="app-header">
      <h1 id="app-title">Doc<span class="accent">Classify</span></h1>
      <p id="app-subtitle">
        Document Classification &amp; Invoice Extraction &nbsp;·&nbsp;
        IE University · AI: Statistical Learning and Prediction
      </p>
    </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<div class='label-section'>Input</div>")
            text_in = gr.Textbox(label="Paste document text", lines=14,
                placeholder="Paste an invoice, email, scientific report, or letter here…")
            file_in = gr.File(label="Or upload a file",
                file_types=[".txt", ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif"])
            with gr.Row():
                submit_btn = gr.Button("▶  Classify", variant="primary", elem_id="classify-btn")
                clear_btn  = gr.Button("Clear", elem_id="clear-btn")

            gr.HTML("<div style='font-size:.7em;font-weight:600;letter-spacing:.09em;"
                    "text-transform:uppercase;color:#4a4858;margin:16px 0 7px;'>Quick Examples</div>")
            for name, txt in EXAMPLES.items():
                gr.Button(name, elem_classes=["example-btn"]).click(
                    fn=lambda t=txt: t, outputs=text_in)

        with gr.Column(scale=1):
            gr.HTML("<div class='label-section'>Results</div>")
            label_out = gr.HTML(label="Predicted Category")
            conf_out  = gr.HTML(label="Confidence Scores")
            ext_out   = gr.HTML(label="Extracted Fields")

    submit_btn.click(fn=run_pipeline, inputs=[text_in, file_in], outputs=[label_out, conf_out, ext_out])
    clear_btn.click(fn=lambda: ("", None, "", "", ""), outputs=[text_in, file_in, label_out, conf_out, ext_out])

    gr.HTML("<div id='app-footer'>TF-IDF &nbsp;·&nbsp; LinearSVC &nbsp;·&nbsp; No generative AI &nbsp;·&nbsp; Traditional ML only</div>")


if __name__ == "__main__":
    demo.launch(share=False, show_error=True)
