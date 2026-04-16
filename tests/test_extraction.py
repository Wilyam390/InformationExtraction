"""
tests/test_extraction.py
========================
Unit tests for the regex-based invoice field extractor.
Tests each extracted field with realistic invoice formats found online.

Run:  python3 -m pytest tests/ -v
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from extract import extract_invoice_fields, _normalise_date, _normalise_amount


# ─────────────────────────────────────────────────────────────────────────────
# Sample invoices (hand-written for controlled testing)
# ─────────────────────────────────────────────────────────────────────────────

STANDARD_INVOICE = """INVOICE
Invoice Number: INV-4821
Invoice Date:   2024-03-15
Due Date:       2024-04-15

From: Acme Corp
To:   John Smith

Web Development        1   USD  3,500.00
Tax (20%)                  USD    700.00
TOTAL                      USD  4,200.00
"""

FREELANCER_INVOICE = """Freelance Invoice

Bill To: Maria García
From:    BlueSky Solutions

Invoice #: INV-0042
Date:      12 Jan 2024
Due:       11 Feb 2024

Subtotal:   EUR 2,400.00
Tax:        EUR   480.00
Total Due:  EUR 2,880.00
"""

UTILITY_INVOICE = """COASTAL POWER & GAS
UTILITY BILL / INVOICE

Invoice No:      INV-7711
Billing Date:    03/15/2024
Payment Due:     04/14/2024

Customer: Li Wei

Electricity Bill     USD  245.00
Taxes & Fees         USD   24.50
Amount Due           USD  269.50
"""

# ─────────────────────────────────────────────────────────────────────────────
# Real invoices found online
# ─────────────────────────────────────────────────────────────────────────────

REAL_AUS_INVOICE = """Invoice 2022435
Tax invoice

Your Business Name

BILL TO
Your Client
100 Harris St
Sydney NSW NSW 2009

Issue date:    19/7/2022
Due date:      3/8/2022

Invoice No.    Issue date    Due date    Total due (AUD)
2022435        19/7/2022     3/8/2022    $2,510.00

Subtotal:                                              $2,100.00
Total (AUD):                                           $2,510.00
"""

REAL_SIMPLE_INVOICE = """INVOICE

Date Issued:
01 January 2030
Invoice No:
01234

Issued to:
Sacha Dubois

GRAND TOTAL   $300
"""

REAL_ZYLKER_INVOICE = """Zylker Electronics Hub
14B, Northern Street
New York New York 10001

INVOICE

Invoice#        INV-000001
Invoice Date    05 Aug 2024
Due Date        05 Aug 2024

Bill To
Ms. Mary D. Dunton

Sub Total   $2,227.00
Total       $2,338.35
Balance Due $2,338.35
"""


# ─────────────────────────────────────────────────────────────────────────────
# Invoice number extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestInvoiceNumber:
    def test_standard(self):
        assert extract_invoice_fields(STANDARD_INVOICE)["invoice_number"] == "INV-4821"

    def test_freelancer(self):
        assert extract_invoice_fields(FREELANCER_INVOICE)["invoice_number"] == "INV-0042"

    def test_utility(self):
        assert extract_invoice_fields(UTILITY_INVOICE)["invoice_number"] == "INV-7711"

    def test_real_simple(self):
        assert extract_invoice_fields(REAL_SIMPLE_INVOICE)["invoice_number"] == "01234"

    def test_real_zylker(self):
        assert extract_invoice_fields(REAL_ZYLKER_INVOICE)["invoice_number"] == "INV-000001"


# ─────────────────────────────────────────────────────────────────────────────
# Date extraction
# ─────────────────────────────────────────────────────────────────────────────

class TestDates:
    def test_iso_date(self):
        assert extract_invoice_fields(STANDARD_INVOICE)["invoice_date"] == "2024-03-15"

    def test_written_date(self):
        assert extract_invoice_fields(FREELANCER_INVOICE)["invoice_date"] == "2024-01-12"

    def test_due_date(self):
        assert extract_invoice_fields(STANDARD_INVOICE)["due_date"] == "2024-04-15"

    def test_real_aus_date(self):
        assert extract_invoice_fields(REAL_AUS_INVOICE)["invoice_date"] == "2022-07-19"

    def test_real_aus_due_date(self):
        assert extract_invoice_fields(REAL_AUS_INVOICE)["due_date"] == "2022-08-03"

    def test_real_zylker_date(self):
        assert extract_invoice_fields(REAL_ZYLKER_INVOICE)["invoice_date"] == "2024-08-05"

    def test_real_simple_date(self):
        assert extract_invoice_fields(REAL_SIMPLE_INVOICE)["invoice_date"] == "2030-01-01"


# ─────────────────────────────────────────────────────────────────────────────
# Names & amounts
# ─────────────────────────────────────────────────────────────────────────────

class TestNamesAndAmounts:
    def test_issuer(self):
        assert extract_invoice_fields(STANDARD_INVOICE)["issuer_name"] == "Acme Corp"

    def test_recipient(self):
        assert extract_invoice_fields(STANDARD_INVOICE)["recipient_name"] == "John Smith"

    def test_real_recipient(self):
        assert "Sacha Dubois" in extract_invoice_fields(REAL_SIMPLE_INVOICE)["recipient_name"]

    def test_standard_total(self):
        r = extract_invoice_fields(STANDARD_INVOICE)["total_amount"]
        assert "4200" in r or "4,200" in r

    def test_freelancer_total(self):
        assert "2880" in extract_invoice_fields(FREELANCER_INVOICE)["total_amount"]

    def test_utility_total(self):
        assert "269" in extract_invoice_fields(UTILITY_INVOICE)["total_amount"]

    def test_real_zylker_total(self):
        assert "2338" in extract_invoice_fields(REAL_ZYLKER_INVOICE)["total_amount"]

    def test_real_simple_total(self):
        assert "300" in extract_invoice_fields(REAL_SIMPLE_INVOICE)["total_amount"]


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions (date & amount normalisation)
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_iso_unchanged(self):
        assert _normalise_date("2024-03-15") == "2024-03-15"

    def test_written_month(self):
        assert _normalise_date("15 March 2024") == "2024-03-15"

    def test_month_first(self):
        assert _normalise_date("March 15, 2024") == "2024-03-15"

    def test_abbreviated(self):
        assert _normalise_date("12 Jan 2024") == "2024-01-12"

    def test_dollar(self):
        assert _normalise_amount("$1,234.56") == "1234.56"

    def test_eur(self):
        assert _normalise_amount("EUR 2,880.00") == "2880.00"

    def test_european_format(self):
        assert _normalise_amount("1.234,56") == "1234.56"


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_string(self):
        result = extract_invoice_fields("")
        assert result["invoice_number"] is None
        assert result["invoice_date"] is None
        assert result["total_amount"] is None

    def test_random_text(self):
        result = extract_invoice_fields("Hello, this is just a random sentence.")
        assert result["invoice_number"] is None

    def test_all_keys_present(self):
        result = extract_invoice_fields(REAL_AUS_INVOICE)
        for key in ["invoice_number", "invoice_date", "due_date",
                     "issuer_name", "recipient_name", "total_amount"]:
            assert key in result
