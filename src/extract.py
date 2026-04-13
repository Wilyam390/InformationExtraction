"""
extract.py
==========
Regex-based structured information extractor for invoice documents.

Extracts six fields:
  - invoice_number
  - invoice_date
  - due_date
  - issuer_name
  - recipient_name
  - total_amount

Design philosophy
-----------------
Multiple patterns per field, ordered from most specific to most general.
The first match wins.  All patterns are case-insensitive.

Usage
-----
    from src.extract import extract_invoice_fields

    fields = extract_invoice_fields(text)
    # → {"invoice_number": "INV-1234", "invoice_date": "2024-03-15", ...}
"""

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# DATE NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

_MONTH_MAP = {
    "jan": "01", "january": "01",
    "feb": "02", "february": "02",
    "mar": "03", "march": "03",
    "apr": "04", "april": "04",
    "may": "05",
    "jun": "06", "june": "06",
    "jul": "07", "july": "07",
    "aug": "08", "august": "08",
    "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10",
    "nov": "11", "november": "11",
    "dec": "12", "december": "12",
}

def _normalise_date(raw: str) -> str:
    """
    Attempt to normalise a raw date string to YYYY-MM-DD.
    Falls back to returning the cleaned raw string if parsing fails.
    """
    raw = raw.strip()

    # Already ISO  2024-03-15
    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw

    # DD/MM/YYYY or MM/DD/YYYY
    # If second group > 12, it must be a day → MM/DD/YYYY; otherwise assume DD/MM/YYYY
    m = re.match(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$", raw)
    if m:
        a, b, y = m.group(1), m.group(2), m.group(3)
        if int(b) > 12:          # b is day  → a=month, b=day  (MM/DD/YYYY)
            return f"{y}-{a.zfill(2)}-{b.zfill(2)}"
        elif int(a) > 12:        # a is day  → a=day, b=month  (DD/MM/YYYY)
            return f"{y}-{b.zfill(2)}-{a.zfill(2)}"
        else:                    # ambiguous → assume DD/MM/YYYY (international)
            return f"{y}-{b.zfill(2)}-{a.zfill(2)}"

    # MM/DD/YY or DD/MM/YY
    m = re.match(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})$", raw)
    if m:
        a, b, y2 = m.group(1), m.group(2), m.group(3)
        year = "20" + y2 if int(y2) < 50 else "19" + y2
        return f"{year}-{b.zfill(2)}-{a.zfill(2)}"

    # 15 March 2024 / March 15, 2024 / 15 Mar 2024
    m = re.match(
        r"^(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{4})$", raw
    ) or re.match(
        r"^([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(\d{4})$", raw
    )
    if m:
        groups = m.groups()
        if groups[0].isdigit():
            day, mon_str, year = groups
        else:
            mon_str, day, year = groups
        mon = _MONTH_MAP.get(mon_str.lower().rstrip("."))
        if mon:
            return f"{year}-{mon}-{str(day).zfill(2)}"

    return raw  # return as-is if no pattern matched


# ─────────────────────────────────────────────────────────────────────────────
# AMOUNT NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_amount(raw: str) -> str:
    """Strip currency symbols and normalise decimal separators."""
    raw = raw.strip()
    # Remove currency symbols / codes
    raw = re.sub(r"[£$€]|USD|EUR|GBP", "", raw).strip()
    # European format: 1.234,56 → 1234.56
    if re.search(r"\d{1,3}(?:\.\d{3})+,\d{2}$", raw):
        raw = raw.replace(".", "").replace(",", ".")
    else:
        # Remove thousands separator commas:  1,234.56 → 1234.56
        raw = re.sub(r",(?=\d{3}(?:[.,]|$))", "", raw)
    return raw.strip()


# ─────────────────────────────────────────────────────────────────────────────
# FIELD PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

# ── Invoice Number ────────────────────────────────────────────────────────────
# Notes:
#   • Max length raised to 30 to cover compound IDs like "9BF0758D-702530"
#   • [:\-]? made optional with surrounding \s* so "Invoice number  9BF0758D"
#     (no colon, multiple spaces) is handled
#   • Leading character broadened to [\w] to catch numeric-first IDs
_INV_NUMBER_PATTERNS = [
    r"invoice\s*(?:number|no\.?|#|num\.?)\s*[:\-]?\s*([\w][\w\-/\.]{2,30})",
    r"inv(?:oice)?\s*[#\-no\.]*\s*[:\-]?\s*([\w][\w\-/\.]{2,30})",
    r"bill\s*(?:number|no\.?|#)\s*[:\-]?\s*([\w][\w\-/\.]{2,30})",
    r"(?:^|\b)(?:INV|BILL|REF|SIN|TAX|FACTURA)[/\-]([\w\-]{3,20})",
    r"receipt\s*(?:number|no\.?|#)\s*[:\-]?\s*([\w][\w\-/\.]{2,30})",
]

# ── Date (generic helper) ─────────────────────────────────────────────────────
_DATE_VALUE = (
    r"(\d{4}[\/\-\.]\d{1,2}[\/\-\.]\d{1,2}"       # 2024-03-15
    r"|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}"      # 15/03/2024 or 15/03/24
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4}"         # 15 March 2024
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})"      # March 15, 2024
)

_INV_DATE_PATTERNS = [
    rf"invoice\s*date\s*[:\-]?\s*{_DATE_VALUE}",
    rf"date\s*of\s*(?:invoice|issue)\s*[:\-]?\s*{_DATE_VALUE}",  # "Date of issue"
    rf"(?:bill|billing)\s*date\s*[:\-]?\s*{_DATE_VALUE}",
    rf"issue\s*date\s*[:\-]?\s*{_DATE_VALUE}",
    rf"issued\s*(?:on|date)?\s*[:\-]?\s*{_DATE_VALUE}",
    rf"(?:^|\b)date\s*[:\-]\s*{_DATE_VALUE}",
]


# ── Issuer / Recipient ────────────────────────────────────────────────────────
# Issuer: explicit keyword labels first, then heuristic first-company-name.
# The first-line heuristic skips generic single-word headers like "INVOICE",
# "RECEIPT", "BILL", "STATEMENT" so we don't grab the document title.
_GENERIC_HEADERS = {
    "invoice", "receipt", "bill", "statement", "quotation", "quote",
    "estimate", "proforma", "purchase", "order",
}

_ISSUER_PATTERNS = [
    r"(?:from|issued\s*by|seller|vendor|supplier|biller|company)\s*[:\-]\s*([^\n\r]{3,60})",
    # Multi-word line ending with a legal suffix (catches "Anthropic Ireland, Limited")
    r"^([A-Z][A-Za-z &\.,'\-]{8,60}(?:Ltd|LLC|Inc|GmbH|SL|SA|Limited|Corp|BV|AG|PLC)\.?)\s*$",
    # All-caps company name on its own line (e.g. "COASTAL POWER & GAS")
    r"^([A-Z][A-Z &\'\-\.]{5,50})\s*\n",
]

_RECIPIENT_PATTERNS = [
    r"(?:bill(?:ed)?\s*to|sold\s*to|ship(?:ped)?\s*to|recipient|buyer)\s*[:\-]\s*([^\n\r]{3,60})",
    r"(?:dear|attn\.?|attention)\s*[:\-]?\s*([^\n\r]{3,60})",
    r"(?:customer|client)\s*[:\-]\s*([^\n\r]{3,60})",
    r"(?:^|\b)to\s*[:\-]\s*([^\n\r]{3,60})",
]

# ── Due Date (fix: "Payment Due:" without "date" keyword) ────────────────────
_DUE_DATE_PATTERNS = [
    rf"(?:payment\s*)?due\s*date\s*[:\-]?\s*{_DATE_VALUE}",
    rf"date\s*due\s*[:\-]?\s*{_DATE_VALUE}",            # "Date due" (reversed)
    rf"due\s*by\s*[:\-]?\s*{_DATE_VALUE}",
    rf"payment\s*due\s*[:\-]?\s*{_DATE_VALUE}",
    rf"due\s*[:\-]\s*{_DATE_VALUE}",
    rf"pay(?:ment)?\s*(?:by|before|on)\s*[:\-]?\s*{_DATE_VALUE}",
    rf"payable\s*(?:by|on)\s*[:\-]?\s*{_DATE_VALUE}",
    # "€21.78 due April 12, 2026" — amount followed by "due" followed by date
    rf"due\s+{_DATE_VALUE}",
]

# ── Total Amount ──────────────────────────────────────────────────────────────
# Handles: "USD 4,200.00", "EUR 2,880.00", "$4,200.00", "4,200.00 USD"
_AMOUNT_VALUE = (
    r"((?:[£$€]|USD|EUR|GBP)\s*[\d]{1,3}(?:[,\.\s]\d{3})*(?:[,\.]\d{2})?"
    r"|[\d]{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?\s*(?:[£$€]|USD|EUR|GBP)?)"
)

# Priority order matters:
#   1. "Amount due" / "Balance due"  — most explicit
#   2. "Grand total" / "Total due" / "Total payable"
#   3. Plain "Total" — BUT with a negative lookahead to skip
#      "Total excluding tax" / "Total excl." / "Total before tax"
_TOTAL_PATTERNS = [
    rf"amount\s*due\s*[:\-]?\s*{_AMOUNT_VALUE}",
    rf"(?:balance|outstanding)\s*due\s*[:\-]?\s*{_AMOUNT_VALUE}",
    rf"grand\s*total\s*[:\-]?\s*{_AMOUNT_VALUE}",
    rf"total\s+(?:amount\s+)?(?:due|payable|owed)\s*[:\-]?\s*{_AMOUNT_VALUE}",
    rf"total\s*(?:amount)?\s*[:\-]\s*{_AMOUNT_VALUE}",
    # Plain "Total" — skip lines that say "excluding", "excl", "before", "net"
    rf"(?m)^[ \t]*total(?!\s+(?:exclu|excl|before|net|tax))[^\n\d]*{_AMOUNT_VALUE}",
]


# ─────────────────────────────────────────────────────────────────────────────
# CORE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def _first_match(text: str, patterns: list) -> Optional[str]:
    """Try each pattern; return the first captured group that matches."""
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).strip().strip(".,;:")
            if val:
                return val
    return None


def extract_invoice_fields(text: str) -> dict:
    """
    Extract structured fields from invoice text.

    Parameters
    ----------
    text : str
        Raw text of the document (already classified as invoice).

    Returns
    -------
    dict with keys:
        invoice_number, invoice_date, due_date,
        issuer_name, recipient_name, total_amount
    Each value is a string or None if not found.
    """
    # Pre-process: normalise whitespace but preserve line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    inv_number  = _first_match(text, _INV_NUMBER_PATTERNS)
    inv_date_raw = _first_match(text, _INV_DATE_PATTERNS)
    due_date_raw = _first_match(text, _DUE_DATE_PATTERNS)
    issuer      = _first_match(text, _ISSUER_PATTERNS)
    recipient   = _first_match(text, _RECIPIENT_PATTERNS)
    total_raw   = _first_match(text, _TOTAL_PATTERNS)

    # Last-resort fallback: if keyword patterns found nothing, the largest
    # monetary value in the document is very likely the total amount.
    if not total_raw:
        all_amounts = re.findall(
            r"(?:[£$€]|USD|EUR|GBP)?\s*([\d]{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2}))",
            text, re.IGNORECASE
        )
        if all_amounts:
            def _to_float(s):
                s = re.sub(r",(?=\d{3})", "", s)   # remove thousands commas
                return float(s.replace(",", "."))
            try:
                total_raw = max(all_amounts, key=_to_float)
            except (ValueError, AttributeError):
                pass

    # Normalise dates and amounts
    inv_date = _normalise_date(inv_date_raw) if inv_date_raw else None
    due_date = _normalise_date(due_date_raw) if due_date_raw else None
    total    = _normalise_amount(total_raw)  if total_raw   else None

    # Trim issuer / recipient to first line if multi-line leaked through
    if issuer:
        issuer = issuer.splitlines()[0].strip().strip(".,;:")
    if recipient:
        recipient = recipient.splitlines()[0].strip().strip(".,;:")

    return {
        "invoice_number":  inv_number,
        "invoice_date":    inv_date,
        "due_date":        due_date,
        "issuer_name":     issuer,
        "recipient_name":  recipient,
        "total_amount":    total,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        # Standard format
        """INVOICE
Invoice Number: INV-4821
Invoice Date:   2024-03-15
Due Date:       2024-04-15

From: Acme Corp
To:   John Smith

Web Development        1   USD  3,500.00   USD  3,500.00
Tax (20%)                                  USD    700.00
──────────────────────────────────────────────────────
TOTAL                                      USD  4,200.00
""",
        # Freelancer format
        """Freelance Invoice

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
""",
        # Utility bill
        """COASTAL POWER & GAS
UTILITY BILL / INVOICE

Account Number:  AC-884421
Invoice No:      INV-7711
Billing Date:    03/15/2024
Payment Due:     04/14/2024

Customer: Li Wei

Charges:
  Electricity Bill     USD  245.00
  Taxes & Fees         USD   24.50
                       ──────────
  Amount Due           USD  269.50
""",
    ]

    for i, s in enumerate(samples, 1):
        print(f"{'─'*55}")
        print(f"Sample {i}")
        print(f"{'─'*55}")
        fields = extract_invoice_fields(s)
        for k, v in fields.items():
            status = "✓" if v else "✗"
            print(f"  {status} {k:<20} {v or 'NOT FOUND'}")
        print()
