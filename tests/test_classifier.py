"""
tests/test_classifier.py
========================
Unit tests for the document classifier and full prediction pipeline.
Includes tests with real documents found online.

Run:  python3 -m pytest tests/ -v
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from predict import classify, predict


# ─────────────────────────────────────────────────────────────────────────────
# Model loading & structure
# ─────────────────────────────────────────────────────────────────────────────

class TestModelLoading:
    def test_returns_label(self):
        result = classify("This is a test document.")
        assert result["label"] in ["invoice", "email", "scientific_report", "letter"]

    def test_returns_four_confidence_scores(self):
        result = classify("This is a test document.")
        assert len(result["confidence"]) == 4


# ─────────────────────────────────────────────────────────────────────────────
# Classification with clear-cut examples
# ─────────────────────────────────────────────────────────────────────────────

SAMPLE_INVOICE = """INVOICE
Invoice Number: INV-9021
From: TechServ Ltd
To: Jane Doe
Consulting Services   USD  5,000.00
Tax                   USD  1,000.00
TOTAL                 USD  6,000.00
"""

SAMPLE_EMAIL = """From: sarah@company.com
To: team@company.com
Subject: Meeting Tomorrow

Hi everyone,
Just a reminder that we have a team meeting tomorrow at 10am.
Please bring your project updates.
Thanks, Sarah
"""

SAMPLE_REPORT = """Title: Deep Learning Approaches for Image Segmentation

Abstract
This paper presents a novel convolutional neural network architecture
for semantic image segmentation. We evaluate our method on the PASCAL VOC
and COCO datasets, achieving state-of-the-art results with a mean IoU
of 82.4%. Our approach uses dilated convolutions and skip connections.

1. Introduction
Image segmentation is a fundamental task in computer vision...

References
[1] Long, J., Shelhamer, E., & Darrell, T. (2015). CVPR.
"""

SAMPLE_LETTER = """AGREEMENT AND GENERAL RELEASE

This Agreement is entered into by and between the Company and the Employee.
In consideration of the mutual promises herein, the parties agree:
1. The Employee agrees to resign effective immediately.
2. The Company shall pay severance equal to six months salary.
3. Both parties agree to maintain confidentiality.

Signed:
Employee                    Authorized Representative
"""


class TestClassification:
    def test_invoice(self):
        assert classify(SAMPLE_INVOICE)["label"] == "invoice"

    def test_email(self):
        assert classify(SAMPLE_EMAIL)["label"] == "email"

    def test_scientific_report(self):
        assert classify(SAMPLE_REPORT)["label"] == "scientific_report"

    def test_letter(self):
        assert classify(SAMPLE_LETTER)["label"] == "letter"


# ─────────────────────────────────────────────────────────────────────────────
# Full predict() pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestPredictPipeline:
    def test_invoice_triggers_extraction(self):
        result = predict(SAMPLE_INVOICE)
        assert result["label"] == "invoice"
        assert "extraction" in result
        assert result["extraction"]["invoice_number"] == "INV-9021"

    def test_non_invoice_no_extraction(self):
        result = predict(SAMPLE_EMAIL)
        assert "extraction" not in result

    def test_empty_input_returns_error(self):
        result = predict("   ")
        assert "error" in result


# ─────────────────────────────────────────────────────────────────────────────
# Real documents found online (all 4 categories)
# ─────────────────────────────────────────────────────────────────────────────

REAL_INVOICE = """Invoice 2022435
Tax invoice
Your Business Name
BILL TO
Your Client
Issue date:    19/7/2022
Due date:      3/8/2022
Invoice No.  2022435
Total (AUD):  $2,510.00
"""

REAL_INVOICE_2 = """INVOICE
Date Issued: 01 January 2030
Invoice No: 01234
Issued to: Sacha Dubois
GRAND TOTAL   $300
"""

REAL_SCI_REPORT = """The Formal Scientific Research Report

A formal scientific research report is a piece of professional writing addressed to
other professionals who are interested in the investigation you conducted. Research
reports usually follow a standard five-part format: (1) introduction, (2) methods,
(3) results, (4) discussion of results, and (5) conclusions and recommendations.

Introduction. Here you explain briefly the purpose of your investigation.
Methods. This is a cookbook section detailing how you did your investigation.
Results. This section presents the empirical results of your investigation.
Discussion of results. Here you explain the significance of your findings.
Conclusions and recommendations. You focus on the main things you learned.
"""

REAL_EMAIL_NETWORKING = """Subject: Dartmouth alumni / content marketing

Hi Garett,
My name is Kristen and I found you while searching on LinkedIn for fellow
Dartmouth alums who are in the marketing field. Would you have time for a
phone or Zoom conversation with me this week?
Thanks!
Kristen
"""

REAL_EMAIL_SALES = """Subject: Ad budget optimization made easy

Hi Brianna,
I noticed you recently downloaded our PPC Pro Tips Guide.
If this resonates with you, AdPro may be a great fit for you.
Are you free this week to talk about your PPC goals?
Best,
Kristen
"""


class TestRealDocuments:
    """Tests with real documents found online across all 4 categories."""

    # ── Invoices ──
    def test_real_aus_invoice(self):
        assert classify(REAL_INVOICE)["label"] == "invoice"

    def test_real_simple_invoice(self):
        assert classify(REAL_INVOICE_2)["label"] == "invoice"

    # ── Scientific reports ──
    def test_real_sci_report(self):
        assert classify(REAL_SCI_REPORT)["label"] == "scientific_report"

    # ── Emails ──
    def test_real_networking_email(self):
        assert classify(REAL_EMAIL_NETWORKING)["label"] == "email"

    def test_real_sales_email(self):
        assert classify(REAL_EMAIL_SALES)["label"] == "email"

    # ── Full pipeline with real invoice ──
    def test_real_invoice_pipeline(self):
        result = predict(REAL_INVOICE)
        assert result["label"] == "invoice"
        assert "extraction" in result
        assert result["extraction"]["invoice_date"] == "2022-07-19"
