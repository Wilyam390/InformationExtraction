"""
extract.py
==========

Regex + layout-aware structured information extractor for invoice documents.

Extracts:
  - invoice_number
  - invoice_date
  - due_date
  - issuer_name
  - recipient_name
  - total_amount
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
    raw = raw.strip()
    raw = re.sub(r"\s*([/\-.])\s*", r"\1", raw)

    if re.match(r"^\d{4}-\d{2}-\d{2}$", raw):
        return raw

    m = re.match(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{4})$", raw)
    if m:
        a, b, y = m.group(1), m.group(2), m.group(3)
        if int(b) > 12:
            return f"{y}-{a.zfill(2)}-{b.zfill(2)}"
        elif int(a) > 12:
            return f"{y}-{b.zfill(2)}-{a.zfill(2)}"
        else:
            return f"{y}-{b.zfill(2)}-{a.zfill(2)}"

    m = re.match(r"^(\d{1,2})[/\-\.](\d{1,2})[/\-\.](\d{2})$", raw)
    if m:
        a, b, y2 = m.group(1), m.group(2), m.group(3)
        year = "20" + y2 if int(y2) < 50 else "19" + y2
        return f"{year}-{b.zfill(2)}-{a.zfill(2)}"

    m = re.match(r"^(\d{1,2})\s+([A-Za-z]+)\.?\s+(\d{4})$", raw) or re.match(
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

    return raw

def _strip_trailing_field_labels(s: str) -> str:
    """
    Remove invoice-field text that OCR glued onto the same line
    as a party name or address.
    """
    if not s:
        return s

    s = re.split(
        r"\b(?:invoice\s*number|invoice\s*date|payment\s*due|p\.?\s*0?\.?\s*/\s*s\.?\s*0?\.?\s*number|"
        r"p\.?\s*o\.?\s*/\s*s\.?\s*o\.?\s*number|po\s*/\s*so\s*number|"
        r"amount\s*due|subtotal|total|tax)\b",
        s,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]

    return s.strip(" ,;:")

def _extract_parties_from_collapsed_top_ocr(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Handles OCR where the left/right columns collapse into one stream.

    Example:
        ths Lid, Kaisaniemis 00100 Due date 16.03.2017
        Bering Catering Our reference Mare MillerMarc Miller ...
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = lines[:12]

    issuer = None
    recipient = None

    for ln in head:
        m = re.search(r"^(.*?)\bDue date\b", ln, re.IGNORECASE)
        if m:
            cand = m.group(1).strip(" ,;:_-")
            cand = re.sub(r"\b(invoice number|invoice date)\b.*$", "", cand, flags=re.I).strip(" ,;:")
            cand = re.sub(r"\s+", " ", cand)

            # OCR cleanup for this family of errors
            cand = cand.replace(" Lid", " Ltd")
            cand = cand.replace("ths ", "")
            cand = cand.strip(" ,;:")

            if cand:
                issuer = cand
                break

    for ln in head:
        m = re.search(r"^(.*?)\bOur reference\b", ln, re.IGNORECASE)
        if m:
            cand = m.group(1).strip(" ,;:_-")
            cand = re.sub(r"\s+", " ", cand)

            # take the first company/person-looking chunk
            m2 = re.match(r"([A-Z][A-Za-z&'\-]+(?:\s+[A-Z][A-Za-z&'\-]+){0,2})", cand)
            if m2:
                recipient = m2.group(1).strip(" ,;:")
                break

    return issuer, recipient

def _extract_recipient_from_billto_ocr(text: str) -> Optional[str]:
    """
    Handle OCR like:

        BILLTO Invoice Number: 14
        Jiro Doi P.0./S.0. Number: AD29094
        1954 Bloor Street West Invoice Date: 2018-09-25

    We locate BILLTO/BILL TO, then inspect the next few lines and
    return the first plausible person/company name after cleaning.
    """
    lines = [ln.strip() for ln in text.splitlines()]

    for i, line in enumerate(lines):
        compact = re.sub(r"[\s:]+", "", line.lower())

        if not (compact.startswith("billto") or compact.startswith("billedto") or compact.startswith("invoiceto")):
            continue

        # check next few lines for recipient candidate
        for j in range(i + 1, min(i + 5, len(lines))):
            cand = lines[j].strip()
            if not cand:
                continue

            cand = _strip_trailing_field_labels(cand)
            cleaned = _cleanup_party_value(cand)
            if not cleaned:
                continue

            # skip obvious address/email/phone lines
            if "@" in cleaned:
                continue
            if re.search(r"\b\d{2,}\b", cleaned) and not re.search(_LEGAL_SUFFIX, cleaned, re.I):
                continue

            # prefer short human/company name-like lines
            words = cleaned.split()
            if 1 <= len(words) <= 5:
                return cleaned

    return None

def _is_metadata_line(line: str) -> bool:
    line = line.strip()
    return bool(re.search(
        r"\b(invoice\s*number|invoice\s*date|due\s*date|delivery\s*date|payment\s*terms|"
        r"our\s*reference|your\s*reference|buyer'?s\s*order\s*number|penalty\s*interest|"
        r"customer'?s\s*business\s*id|customer\s*number|vat|subtotal|total|amount due)\b",
        line,
        re.I,
    ))
    
def _is_addressish_line(line: str) -> bool:
    line = line.strip()
    if not line:
        return False

    if "@" in line:
        return True
    if re.search(r"\b\d{3,}[-\s]?\d*\b", line):
        return True
    if re.search(r"\b(?:street|st|road|rd|avenue|ave|lane|ln|apt|suite|floor|helsinki|toronto|canada|finland)\b", line, re.I):
        return True
    if re.search(r"\b\d{5}\b", line):
        return True

    return False

def _extract_top_left_blocks(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract issuer and recipient from unlabeled top-left blocks, e.g.

        SpiceImporter Ltd., Kaisaniemenkatu 6A, 00100 Helsinki

        Bering Catering
        Marc Miller
        Bulevardi 15
        00180 Helsinki

    before the right-side invoice metadata starts.
    """
    lines = [ln.strip() for ln in text.splitlines()]

    # Keep only the top section before item table / totals
    top = []
    for ln in lines[:25]:
        if not ln:
            top.append("")
            continue
        if re.search(r"\border delivered\b|\bproduct no\b|\bdescription\b|\bunit price\b|\bqty\b|\bvat\b|\btotal to pay\b", ln, re.I):
            break
        top.append(ln)

    if not top:
        return None, None

    # Remove obvious metadata-table lines from candidate blocks
    filtered = []
    for ln in top:
        if _is_metadata_line(ln):
            continue
        filtered.append(ln)

    # Split into blocks by blank lines
    blocks = []
    cur = []
    for ln in filtered:
        if not ln:
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(ln)
    if cur:
        blocks.append(cur)

    # Keep only plausible address/name blocks
    plausible = []
    for block in blocks:
        joined = " ".join(block)
        if re.search(r"\bINVOICE\b", joined, re.I):
            continue
        if len(joined) < 4:
            continue
        plausible.append(block)

    if not plausible:
        return None, None

    issuer = None
    recipient = None

    # First plausible block: issuer
    first = plausible[0]
    if first:
        issuer = _cleanup_party_value(first[0])
        if issuer:
            # If first line is too short or OCR-damaged, try combining first two lines
            if len(issuer.split()) == 1 and len(first) >= 2:
                combo = _cleanup_party_value(f"{first[0]} {first[1]}")
                if combo:
                    issuer = combo

    # Second plausible block: recipient
    if len(plausible) >= 2:
        second = plausible[1]

        # first non-addressish line is usually company/person name
        for ln in second:
            cand = _cleanup_party_value(ln)
            if not cand:
                continue
            if _is_metadata_line(cand):
                continue
            if "@" in cand:
                continue
            if not _is_addressish_line(cand) or re.search(_LEGAL_SUFFIX, cand, re.I):
                recipient = cand
                break

    return issuer, recipient

def _extract_total_from_ocr_total_to_pay(text: str) -> Optional[str]:
    """
    Handle OCR like:
        Total topay€ 42542
        Total to pay € 42542
        Total topay 42542
    """
    m = re.search(
        r"total\s*to\s*pay\s*[€$£=:]?\s*(\d{3,6})\b",
        text,
        re.IGNORECASE,
    )
    if not m:
        m = re.search(
            r"total\s*topay\s*[€$£=:]?\s*(\d{3,6})\b",
            text,
            re.IGNORECASE,
        )

    if m:
        raw = m.group(1)
        if raw.isdigit() and len(raw) >= 3:
            return f"{raw[:-2]}.{raw[-2:]}"
    return None

# ─────────────────────────────────────────────────────────────────────────────
# AMOUNT NORMALISATION
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_amount(raw: str) -> str:
    raw = raw.strip()

    # remove currency symbols/codes
    raw = re.sub(r"[£$€]|USD|EUR|GBP|CHF", "", raw, flags=re.IGNORECASE).strip()

    # remove internal spaces
    raw = re.sub(r"\s+", " ", raw).strip()

    # European thousands + decimal: 1.234,56 or 1 234,56
    if re.fullmatch(r"\d{1,3}(?:[.\s]\d{3})+,\d{2}", raw):
        return raw.replace(".", "").replace(" ", "").replace(",", ".")

    # US/intl thousands + decimal: 1,234.56 or 1 234.56
    if re.fullmatch(r"\d{1,3}(?:[,\s]\d{3})+\.\d{2}", raw):
        return raw.replace(",", "").replace(" ", "")

    # plain comma decimal: 618,01
    if re.fullmatch(r"\d+,\d{2}", raw):
        return raw.replace(",", ".")

    # plain decimal: 618.01
    if re.fullmatch(r"\d+\.\d{2}", raw):
        return raw

    # OCR missing decimal separator: 42542 -> 425.42
    if re.fullmatch(r"\d{4,6}", raw):
        return f"{raw[:-2]}.{raw[-2:]}"

    # fallback cleanup
    raw = raw.replace(" ", "")
    raw = re.sub(r",(?=\d{3}\b)", "", raw)  # thousands commas only
    if re.fullmatch(r"\d+,\d{2}", raw):
        raw = raw.replace(",", ".")
    return raw


def _normalise_ocr_text(text: str) -> str:
    t = text
    t = re.sub(r"\b1nvoice\b", "invoice", t, flags=re.IGNORECASE)
    t = re.sub(r"\b1nv\b", "inv", t, flags=re.IGNORECASE)
    t = re.sub(r"\btota[1l]\b", "total", t, flags=re.IGNORECASE)
    t = re.sub(r"\bam0unt\b", "amount", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdue\s*da[tf]e\b", "due date", t, flags=re.IGNORECASE)
    t = re.sub(r"\bi[1l]\s*va\b", "iva", t, flags=re.IGNORECASE)
    t = re.sub(r"\bcif\b", "CHF", t, flags=re.IGNORECASE)
    t = re.sub(r"[ \t]+", " ", t)
    return t

def _normalise_label_line(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[:\-\s]+", " ", s)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# PATTERNS
# ─────────────────────────────────────────────────────────────────────────────

_INV_NUMBER_PATTERNS = [
    # strict "Invoice number: XXX"
    r"\binvoice\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-/]{2,30})\b",

    # header form: "INVOICE INV/33-57/240"
    r"\bINVOICE\s+([A-Z0-9][A-Z0-9\-/]{2,30})\b",
]

_DATE_VALUE = (
    r"(\d{4}\s*[\/\-\.]\s*\d{1,2}\s*[\/\-\.]\s*\d{1,2}"
    r"|\d{1,2}\s*[\/\-\.]\s*\d{1,2}\s*[\/\-\.]\s*\d{2,4}"
    r"|\d{1,2}\s*[-\s]\s*[A-Za-z]{3,9}\s*[-\s]\s*\d{4}"   # 🔥 ADD THIS
    r"|\d{1,2}\s+[A-Za-z]{3,9}\.?\s+\d{4}"
    r"|[A-Za-z]{3,9}\.?\s+\d{1,2},?\s+\d{4})"
)

_INV_DATE_PATTERNS = [
    rf"invoice\s*date\s*[:\-]?\s*{_DATE_VALUE}",
    rf"date\s*of\s*(?:invoice|issue)\s*[:\-]?\s*{_DATE_VALUE}",
    rf"(?:bill|billing)\s*date\s*[:\-]?\s*{_DATE_VALUE}",
    rf"issue\s*date\s*[:\-]?\s*{_DATE_VALUE}",
    rf"issued\s*(?:on|date)?\s*[:\-]?\s*{_DATE_VALUE}",
    rf"(?<!due )\bdate\s*[:\-]?\s*{_DATE_VALUE}",
]


_DUE_DATE_PATTERNS = [
    rf"due\s*date\s*[:\-]?\s*{_DATE_VALUE}",
    rf"payment\s*due\s*[:\-]?\s*{_DATE_VALUE}",
    rf"date\s*due\s*[:\-]?\s*{_DATE_VALUE}",
    rf"due\s*by\s*[:\-]?\s*{_DATE_VALUE}",
    rf"pay(?:ment)?\s*(?:by|before|on)\s*[:\-]?\s*{_DATE_VALUE}",
]

_AMOUNT_VALUE = (
    r"((?:[£$€]|USD|EUR|GBP|CHF)\s*[\d]{1,3}(?:[,\.\s]\d{3})*(?:[,\.]\d{2})?"
    r"|[\d]{1,3}(?:[,\.]\d{3})*(?:[,\.]\d{2})?\s*(?:[£$€]|USD|EUR|GBP|CHF)?)"
)

_TOTAL_PATTERNS = [
    rf"total\s*to\s*pay\s*[:=\-]?\s*{_AMOUNT_VALUE}",
    rf"total\s*topay\s*[:=\-]?\s*{_AMOUNT_VALUE}",
    rf"amount\s*due\s*[:=\-]?\s*{_AMOUNT_VALUE}",
    rf"grand\s*total\s*[:=\-]?\s*{_AMOUNT_VALUE}",
    rf"total\s*payable\s*[:=\-]?\s*{_AMOUNT_VALUE}",
    rf"(?:balance|outstanding)\s*due\s*[:=\-]?\s*{_AMOUNT_VALUE}",
    rf"total\s+(?:amount\s+)?(?:due|payable|owed)\s*[:=\-]?\s*{_AMOUNT_VALUE}",
    rf"(?:invoice|bill)\s*total\s*[:=\-]?\s*{_AMOUNT_VALUE}",
    rf"total\s*(?:amount)?\s*[:=\-]\s*{_AMOUNT_VALUE}",
    rf"(?m)^[ \t]*total(?!\s+(?:exclu|excl|before|net|tax))[^\n\d]*{_AMOUNT_VALUE}",
]

_AMOUNT_RE = re.compile(
    r"(?:[£$€]|USD|EUR|GBP|CHF)?\s*\d{1,3}(?:[,\.\s]\d{3})*(?:[,\.\s]\d{2})"
    r"|\d+(?:[,\.\s]\d{2})\s*(?:[£$€]|USD|EUR|GBP|CHF)?",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
# PARTY EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

_GENERIC_HEADERS = {
    "invoice", "receipt", "bill", "statement", "quotation", "quote",
    "estimate", "proforma", "purchase", "order",
}

_GENERIC_PARTY_VALUES = {
    "seller", "client", "buyer", "customer", "vendor", "supplier",
    "company", "billto", "shipto", "recipient", "from", "to",
    "clientname", "bill", "billto", "invoice to", "ship to",
}

_PARTY_STOP_LABELS = [
    "seller", "client", "buyer", "customer", "vendor", "supplier",
    "bill to", "billed to", "ship to", "shipped to", "recipient",
    "invoice", "invoice no", "invoice number", "invoice date",
    "date", "due date", "tax id", "vat", "iban", "account",
    "subtotal", "total", "amount due", "gross", "net",
]

_LEGAL_SUFFIX = r"(?:Co|Co\.|Ltd|LLC|Inc|GmbH|SL|SA|Limited|Corp|Corp\.|BV|AG|PLC)\.?"


def _cleanup_party_value(v: Optional[str]) -> Optional[str]:
    if not v:
        return None

    v = v.strip().strip(".,;:")
    
    v = _strip_trailing_field_labels(v)

    # collapse spaces
    v = re.sub(r"\s+", " ", v)
    
    v = re.split(
        r"\b(?:invoice\s*number|invoice\s*date|due\s*date|delivery\s*date|payment\s*terms|"
        r"our\s*reference|your\s*reference|buyer'?s\s*order\s*number|penalty\s*interest|"
        r"customer'?s\s*business\s*id|customer\s*number|amount\s*due|subtotal|total|tax)\b",
        v,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,;:")

    # if OCR captured another label after the value, cut it off
    v = re.split(
        r"\b(?:seller|client|buyer|customer|vendor|supplier|bill(?:ed)?\s*to|ship(?:ped)?\s*to|recipient|from|to)\s*:",
        v,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip(" ,;:")

    # remove obvious garbage-only cases
    low = re.sub(r"[^a-z]", "", v.lower())
    if not low or low in _GENERIC_PARTY_VALUES:
        return None

    # skip address-like lines if possible
    if re.search(r"\b(?:street|st|road|rd|avenue|ave|apt|suite|floor|po box|zip|postcode|tel|phone)\b", v, re.I):
        return None

    # too numeric => probably not a company/person name
    digit_count = sum(ch.isdigit() for ch in v)
    alpha_count = sum(ch.isalpha() for ch in v)
    if digit_count > 0 and alpha_count and digit_count >= alpha_count:
        return None
    
    # reject table headers / metadata junk
    if re.search(r"\b(order|date|number|item|qty|price|total)\b", v, re.I):
        # too many metadata words → not a company
        tokens = re.findall(r"[a-z]+", v.lower())
        if sum(t in {"order", "date", "number", "item", "qty", "price", "total"} for t in tokens) >= 2:
            return None
    
    # reject URLs
    if re.search(r"http[s]?://|www\.", v, re.I):
        return None

    return v


def _looks_generic_party(v: Optional[str]) -> bool:
    if not v:
        return True
    x = re.sub(r"[^a-z]", "", v.lower())
    return x in _GENERIC_PARTY_VALUES


def _first_match(text: str, patterns: list[str]) -> Optional[str]:
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            val = m.group(1).strip().strip(".,;:")
            if val:
                return val
    return None


def _first_match_any(texts: list[str], patterns: list[str]) -> Optional[str]:
    for t in texts:
        val = _first_match(t, patterns)
        if val:
            return val
    return None


def _extract_name_after_label(lines: list[str], label_patterns: list[str]) -> Optional[str]:
    """
    Finds labels like 'Seller:' or 'Bill To:' and extracts a plausible
    name either from the same line or the next few lines.
    """
    compiled = [re.compile(p, re.I) for p in label_patterns]

    for i, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue

        for cp in compiled:
            m = cp.search(line)
            if not m:
                continue

            remainder = line[m.end():].strip(" :-\t")
            cleaned = _cleanup_party_value(remainder)
            if cleaned:
                return cleaned

            # look ahead a few lines for the actual name
            for j in range(i + 1, min(i + 5, len(lines))):
                cand = lines[j].strip()
                if not cand:
                    continue

                cand_low = cand.lower().strip(" :")
                if any(re.fullmatch(rf"{lbl}\s*:?", cand_low) for lbl in _PARTY_STOP_LABELS):
                    continue

                cleaned = _cleanup_party_value(cand)
                if cleaned:
                    return cleaned

            break

    return None


def _extract_party_pair_from_seller_client_layout(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    Handle OCR where headers appear as:
        Seller: Client:
        Andrews, Kirby and Valdez Becker Ltd
    or where seller/client are in adjacent blocks.
    """
    lines = [ln.strip() for ln in text.splitlines()]

    for i, ln in enumerate(lines):
        l = ln.lower()

        if "seller" in l and "client" in l:
            # first meaningful line after merged headers
            nxt = None
            for j in range(i + 1, min(i + 6, len(lines))):
                cand = lines[j].strip()
                if cand:
                    nxt = cand
                    break

            if not nxt:
                continue

            issuer, recipient = _split_merged_party_line(nxt)
            if issuer and recipient:
                return issuer, recipient

            # fallback: split by very large spacing if OCR preserved columns
            parts = re.split(r"\s{3,}", nxt)
            if len(parts) >= 2:
                issuer = _cleanup_party_value(parts[0])
                recipient = _cleanup_party_value(parts[1])
                if issuer and recipient:
                    return issuer, recipient

    return None, None


def _extract_issuer_name(text: str) -> Optional[str]:
    # 🔥 Fallback: first company-like line near top (before Bill To)
    top_lines = [ln.strip() for ln in text.splitlines()[:12] if ln.strip()]

    for ln in top_lines:
        if re.search(r"\b(invoice|bill to|ship to|date|gst|invoice number)\b", ln, re.I):
            continue

        cleaned = _cleanup_party_value(ln)
        if not cleaned:
            continue

        # strong company pattern
        if re.search(_LEGAL_SUFFIX, cleaned, re.I):
            return cleaned

        # fallback: multi-word capitalized line
        if len(cleaned.split()) >= 2 and re.match(r"^[A-Z][A-Za-z&.,'\- ]+$", cleaned):
            return cleaned
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # Strong top-of-document issuer extraction:
    # stop before recipient/order sections
    stop_re = re.compile(
        r"\b(?:ship\s*to|bill\s*to|customer\s*id|order\s*date|order\s*number|due\s*date)\b",
        re.I,
    )

    top_block = []
    for ln in lines[:15]:
        if stop_re.search(ln):
            break
        top_block.append(ln)

    # 1. Prefer the first clean company-like line in the top block
    for ln in top_block:
        # skip obvious non-company lines
        if re.search(r"\b(invoice|date|email|phone|tel)\b", ln, re.I):
            continue

        # block metadata/table headers explicitly
        if re.search(r"\b(order|date|number|item|qty|price|total)\b", ln, re.I):
            continue

        cleaned = _cleanup_party_value(ln)
        if not cleaned:
            continue

        # ConIncorporated / Mycompany Ltd / Turnpike Designs Co.
        if re.search(_LEGAL_SUFFIX, cleaned, re.I):
            return cleaned
        if re.match(r"^[A-Z][A-Za-z&.,'\- ]{3,}$", cleaned):
            return cleaned

    # 2. Label-based block extraction
    issuer = _extract_party_from_block(lines, [
        "seller", "vendor", "supplier", "biller", "from", "company", "issued by"
    ])
    if issuer:
        return issuer

    # 3. Same-line labels
    issuer = _extract_name_after_label(lines, [
        r"\bfrom\s*:",
        r"\bissued\s*by\s*:",
        r"\bseller\s*:",
        r"\bvendor\s*:",
        r"\bsupplier\s*:",
        r"\bbiller\s*:",
        r"\bcompany\s*:",
    ])
    if issuer:
        return issuer

    # 4. OCR-specific fallbacks
    issuer = _extract_issuer_from_top_ocr(text)
    if issuer:
        return issuer

    ocr_issuer, _ = _extract_parties_from_collapsed_top_ocr(text)
    if ocr_issuer:
        return ocr_issuer

    block_issuer, _ = _extract_top_left_blocks(text)
    if block_issuer:
        return block_issuer

    block_issuer, _ = _extract_two_top_blocks(text)
    if block_issuer:
        return block_issuer

    return None
    return None


def _extract_recipient_name(text: str) -> Optional[str]:
    lines = text.splitlines()

    # 1. Labelled blocks
    recipient = _extract_party_from_block(lines, [
        "client",
        "customer",
        "buyer",
        "recipient",
        "bill to",
        "billed to",
        "invoice to",
        "ship to",
        "shipped to",
        "to",
        "attn",
        "attention",
    ])
    if recipient:
        return recipient

    # 2. Same-line labels
    recipient = _extract_name_after_label(lines, [
        r"\bbill(?:ed)?\s*to\s*:",
        r"\binvoice\s*to\s*:",
        r"\bship(?:ped)?\s*to\s*:",
        r"\brecipient\s*:",
        r"\bbuyer\s*:",
        r"\bcustomer\s*:",
        r"\bclient\s*:",
        r"\bto\s*:",
        r"\battn\.?\s*:",
        r"\battention\s*:",
    ])
    if recipient:
        return recipient

    # 3. OCR BILLTO fallback
    recipient = _extract_recipient_from_billto_ocr(text)
    if recipient:
        return recipient

    # 4. Collapsed OCR fallback
    _, ocr_recipient = _extract_parties_from_collapsed_top_ocr(text)
    if ocr_recipient:
        return ocr_recipient

    # 5. Unlabelled top-left block fallback
    _, block_recipient = _extract_top_left_blocks(text)
    if block_recipient:
        return block_recipient

    # 6. Very simple two-block fallback
    _, block_recipient = _extract_two_top_blocks(text)
    if block_recipient:
        return block_recipient

    return None

def _extract_two_top_blocks(text: str) -> tuple[Optional[str], Optional[str]]:
    """
    For layouts like:

    SpiceImporter Ltd., Kaisaniemenkatu 6A, 00100 Helsinki

    Bering Catering
    Marc Miller
    Bulevardi 15
    00180 Helsinki

    INVOICE
    Invoice number ...
    """

    lines = [ln.strip() for ln in text.splitlines()]

    # keep only early/top content before invoice metadata or item table
    top = []
    for ln in lines[:30]:
        low = ln.lower()

        if re.search(r"\b(invoice number|invoice date|due date|delivery date|payment terms|our reference|your reference|buyer's order number|customer number)\b", low):
            break
        if re.search(r"\b(product no|description|unit price|qty|vat %|total to pay)\b", low):
            break

        top.append(ln)

    # split into blocks by blank lines
    blocks = []
    cur = []
    for ln in top:
        if ln:
            cur.append(ln)
        else:
            if cur:
                blocks.append(cur)
                cur = []
    if cur:
        blocks.append(cur)

    # keep plausible blocks only
    plausible = []
    for block in blocks:
        joined = " ".join(block).strip()
        if not joined:
            continue
        if re.search(r"\binvoice\b", joined, re.I):
            continue
        plausible.append(block)

    if not plausible:
        return None, None

    issuer = None
    recipient = None

    # issuer = first line of first block
    if len(plausible) >= 1:
        issuer = plausible[0][0].strip(" ,;:")
        if issuer:
            issuer = _cleanup_party_value(issuer)

    # recipient = first line of second block
    if len(plausible) >= 2:
        recipient = plausible[1][0].strip(" ,;:")
        if recipient:
            recipient = _cleanup_party_value(recipient)

    return issuer, recipient



# ─────────────────────────────────────────────────────────────────────────────
# TOTAL EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

def _extract_total_from_total_line(text: str) -> Optional[str]:
    """
    Robust total extraction:
    - prefer strong total labels
    - skip subtotal/tax/vat/excluding lines
    - choose the best amount on the winning line
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    strong = []
    medium = []

    for line in lines:
        ll = line.lower()

        if "total" not in ll and "amount due" not in ll:
            continue

        # reject non-grand-total lines
        if "subtotal" in ll or "sub_total" in ll or "sub total" in ll:
            continue
        if "vat" in ll and "total" in ll and "grand total" not in ll and "total due" not in ll and "amount due" not in ll and "total to pay" not in ll:
            continue
        if "tax" in ll and "total" in ll and "grand total" not in ll and "total due" not in ll and "amount due" not in ll and "total to pay" not in ll:
            continue
        if "excluding" in ll or "excl" in ll or "before tax" in ll:
            continue

        if (
            "grand total" in ll
            or "total due" in ll
            or "amount due" in ll
            or "total to pay" in ll
            or "total payable" in ll
            or "invoice total" in ll
        ):
            strong.append(line)
        elif re.search(r"\btotal\b", ll):
            medium.append(line)

    for bucket in (strong, medium):
        for line in reversed(bucket):  # prefer later totals on page
            amounts = _AMOUNT_RE.findall(line)
            if not amounts:
                continue
            try:
                parsed = []
                for amt in amounts:
                    norm = _normalise_amount(amt)
                    parsed.append((float(norm), norm))
                # usually the rightmost/largest is the grand total on that line
                parsed.sort(key=lambda x: x[0], reverse=True)
                return parsed[0][1]
            except Exception:
                continue

    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────

def extract_invoice_fields(text: str) -> dict:
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text_norm = _normalise_ocr_text(text)
    candidates = [text, text_norm]

    inv_number = _first_match_any(candidates, _INV_NUMBER_PATTERNS)
    inv_date_raw = _first_match_any(candidates, _INV_DATE_PATTERNS)
    due_date_raw = _first_match_any(candidates, _DUE_DATE_PATTERNS)

    pair_issuer, pair_recipient = _extract_party_pair_from_seller_client_layout(text_norm)

    issuer = _extract_issuer_name(text_norm)
    recipient = _extract_recipient_name(text_norm)

    if pair_issuer and pair_recipient:
        issuer = pair_issuer
        recipient = pair_recipient

    if issuer and recipient and issuer == recipient:
        s1, s2 = _split_merged_party_line(issuer)
        if s1 and s2:
            issuer, recipient = s1, s2

    total_raw = _extract_total_from_total_line(text_norm)
    if not total_raw:
        total_raw = _first_match_any(candidates, _TOTAL_PATTERNS)

    if not total_raw:
        total_raw = _extract_total_from_ocr_total_to_pay(text_norm)

    # safer last-resort fallback: only accept largest amount if it appears near "total"
    if not total_raw:
        all_amounts = re.findall(
            r"(?:[£$€]|USD|EUR|GBP|CHF)?\s*([\d]+(?:[,\.\s]\d{3})*(?:[,\.]\d{2}))",
            text_norm,
            re.IGNORECASE,
        )
        if all_amounts:
            def _to_float(s):
                s = re.sub(r",(?=\d{3})", "", s)
                return float(s.replace(",", "."))

            try:
                max_val = max(all_amounts, key=_to_float)
                if re.search(rf"total[^\n]*{re.escape(max_val)}", text_norm, re.I):
                    total_raw = max_val
            except (ValueError, AttributeError):
                pass

    inv_date = _normalise_date(inv_date_raw) if inv_date_raw else None
    due_date = _normalise_date(due_date_raw) if due_date_raw else None
    total = _normalise_amount(total_raw) if total_raw else None

    if issuer:
        issuer = issuer.splitlines()[0].strip().strip(".,;:")
    if recipient:
        recipient = recipient.splitlines()[0].strip().strip(".,;:")

    return {
        "invoice_number": inv_number,
        "invoice_date": inv_date,
        "due_date": due_date,
        "issuer_name": issuer,
        "recipient_name": recipient,
        "total_amount": total,
    }

def _split_merged_party_line(line: str) -> tuple[Optional[str], Optional[str]]:
    """
    Split OCR-merged line like:
        'Andrews, Kirby and Valdez Becker Ltd'
    into:
        ('Andrews, Kirby and Valdez', 'Becker Ltd')

    Strategy:
    - work from right to left
    - look for the shortest plausible company tail ending in legal suffix
    - reject tails that are too long unless necessary
    """
    if not line:
        return None, None

    line = re.sub(r"\s+", " ", line).strip(" ,;:")
    tokens = line.split()
    if len(tokens) < 2:
        return None, None

    suffix_re = re.compile(rf"^{_LEGAL_SUFFIX}$", re.I)

    candidates = []

    for end_len in range(2, min(5, len(tokens)) + 1):
        right_tokens = tokens[-end_len:]
        left_tokens = tokens[:-end_len]

        if not left_tokens or len(right_tokens) < 2:
            continue

        right = " ".join(right_tokens)
        left = " ".join(left_tokens)

        # recipient must end with legal suffix
        if not re.search(rf"\b{_LEGAL_SUFFIX}$", right, re.I):
            continue

        left_clean = _cleanup_party_value(left)
        right_clean = _cleanup_party_value(right)

        if not left_clean or not right_clean:
            continue

        # Heuristics:
        # - prefer short recipient tails like "Becker Ltd"
        # - avoid swallowing too much of issuer
        # - recipient usually 2-3 words
        right_word_count = len(right_clean.split())
        left_word_count = len(left_clean.split())

        score = 0

        if right_word_count == 2:
            score += 100
        elif right_word_count == 3:
            score += 80
        elif right_word_count == 4:
            score += 50
        else:
            score += 10

        # reward issuer still looking like a real name/company
        if left_word_count >= 3:
            score += 20

        # penalize recipient tails starting with connectors that likely belong to issuer
        if right_tokens[0].lower() in {"and", "&"}:
            score -= 100

        candidates.append((score, left_clean, right_clean))

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        _, left_best, right_best = candidates[0]
        return left_best, right_best

    return None, None

def _extract_party_from_block(lines: list[str], start_labels: list[str]) -> Optional[str]:
    """
    Extract first plausible party name from a labelled block, e.g.

    BILL TO
    Jiro Doi
    1954 Bloor Street West
    Toronto, ON ...

    or

    Client:
    Becker Ltd
    ...

    Returns the first plausible name line after the label.
    """
    wanted = {_normalise_label_line(x) for x in start_labels}

    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line:
            continue

        norm = _normalise_label_line(line)
        if norm not in wanted:
            continue

        for j in range(i + 1, min(i + 7, len(lines))):
            cand = lines[j].strip()
            if not cand:
                continue

            cand_norm = _normalise_label_line(cand)
            if cand_norm in wanted:
                break

            # stop if we hit another obvious field header
            if re.match(
                r"^(invoice number|invoice date|payment due|p\.?o\.?/s\.?o\.? number|"
                r"tax|vat|subtotal|total|amount due)\b",
                cand,
                re.I,
            ):
                break

            cleaned = _cleanup_party_value(cand)
            if not cleaned:
                continue

            # address-ish lines are often not the name; prefer a cleaner name first
            if re.search(r"\b\d{2,}\b", cleaned) and not re.search(_LEGAL_SUFFIX, cleaned, re.I):
                continue

            return cleaned

    return None

def _extract_issuer_from_top_ocr(text: str) -> Optional[str]:
    """
    Extract issuer from the top section only, before recipient/payment fields.

    Example OCR:
        INVOICE
        Turnpike =
        Designs Co. 156 University Ave, Toronto
        ON, Canada, M5H 2H7
        416-555-1212
        BILLTO Invoice Number: 14
        Jiro Doi ...

    -> Turnpike Designs Co.
    """
    raw_lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    if not raw_lines:
        return None

    stop_re = re.compile(
        r"\b(?:bill\s*to|billto|billed\s*to|invoice\s*to|ship\s*to|customer|client|recipient|invoice\s*number|payment\s*due|amount\s*due)\b",
        re.I,
    )

    top_section = []
    for line in raw_lines[:12]:
        if stop_re.search(line):
            break
        top_section.append(line)

    if not top_section:
        top_section = raw_lines[:5]

    cleaned_lines = []
    for line in top_section:
        line = line.replace("=", " ")
        line = re.sub(r"\s+", " ", line).strip()
        if not line:
            continue
        if re.search(r"\bINVOICE\b", line, re.I):
            continue
        cleaned_lines.append(line)

    if not cleaned_lines:
        return None

    joined = " ".join(cleaned_lines)

    # Best case: explicit company suffix
    m = re.search(
        r"([A-Z][A-Za-z&'\-]+(?:\s+[A-Z][A-Za-z&'\-\.]+){0,4}\s+(?:Co\.|Ltd|LLC|Inc|Corporation|Corp\.|Limited|GmbH|PLC|BV|AG))",
        joined,
        re.I,
    )
    if m:
        return m.group(1).strip(" ,;:")

    # Special case: company split across lines, e.g. "Turnpike" + "Designs Co."
    if len(cleaned_lines) >= 2:
        first = re.sub(r"[^A-Za-z&'\- ]", " ", cleaned_lines[0]).strip()
        second = re.sub(r"\b\d{2,}.*$", "", cleaned_lines[1]).strip()
        second = re.sub(r"\s+", " ", second)

        combo = f"{first} {second}".strip()
        m = re.search(
            r"([A-Z][A-Za-z&'\-]+(?:\s+[A-Z][A-Za-z&'\-\.]+){0,4}\s+(?:Co\.|Ltd|LLC|Inc|Corporation|Corp\.|Limited|GmbH|PLC|BV|AG))",
            combo,
            re.I,
        )
        if m:
            return m.group(1).strip(" ,;:")

    # Fallback: longest title-case/company-like phrase before address/phone
    candidates = []
    for line in cleaned_lines[:4]:
        # remove phone/address tails
        line = re.split(r"\b(?:\d{3}[-\s]\d{3}[-\s]\d{4}|\d{2,} [A-Za-z])", line, maxsplit=1)[0].strip(" ,;:")
        if not line:
            continue
        if re.search(r"\b(?:canada|invoice|amount due|payment due)\b", line, re.I):
            continue
        if len(line.split()) >= 2 and re.match(r"^[A-Za-z&'\-\. ]+$", line):
            candidates.append(line)

    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0].strip(" ,;:")

    return None

# ─────────────────────────────────────────────────────────────────────────────
# REALISTIC TESTS BASED ON YOUR EXAMPLES
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = [
        {
            "name": "Example 1 - seller/client two-column invoice",
            "text": """
Invoice no: 51109338
Date of issue: 04/13/2013

Seller:
Andrews, Kirby and Valdez
58861 Gonzalez Prairie
Lake Daniellefturt, IN 57228
Tax Id: 945-82-2137

Client:
Becker Ltd
8012 Stewart Summit Apt. 455
North Douglas, AZ 95355
Tax Id: 942-80-0517

SUMMARY
Total $ 6 204,19
""",
            "expected": {
                "invoice_number": "51109338",
                "invoice_date": "2013-04-13",
                "issuer_name": "Andrews, Kirby and Valdez",
                "recipient_name": "Becker Ltd",
                "total_amount": "6204.19",
            },
        },
        {
            "name": "Example 2 - OCR-merged seller/client headers",
            "text": """
Invoice no: 51109338
Date of issue: 04/13/2013

Seller: Client:
Andrews, Kirby and Valdez Becker Ltd
58861 Gonzalez Prairie 8012 Stewart Summit Apt. 455
Lake Daniellefturt, IN 57228 North Douglas, AZ 95355

Gross worth 6 204,19
""",
            "expected": {
                "invoice_number": "51109338",
                "invoice_date": "2013-04-13",
                "issuer_name": "Andrews, Kirby and Valdez",
                "recipient_name": "Becker Ltd",
                "total_amount": "6204.19",
            },
        },
        {
            "name": "Example 3 - Mycompany invoice",
            "text": """
Mycompany Ltd
57 Goodwood St
Eastwood
Woodshire, WE99 9EE
Tel: 01234 567890

INVOICE
VAT No: 123 4567 89

Client Name
1 High St
Newtown
EU Country

INVOICE No. 1
DATE/TAXPOINT 31/01/2000
REF/ACCOUNT No. CL201

Total £175.00
""",
            "expected": {
                "invoice_number": "1",
                "invoice_date": "2000-01-31",
                "issuer_name": "Mycompany Ltd",
                "recipient_name": None,   # because the example literally says "Client Name"
                "total_amount": "175.00",
            },
        },
    ]

    for sample in samples:
        print("=" * 80)
        print(sample["name"])
        print("=" * 80)
        fields = extract_invoice_fields(sample["text"])
        for k, v in fields.items():
            print(f"{k:16} -> {v}")
        print("EXPECTED:")
        for k, v in sample["expected"].items():
            print(f"{k:16} -> {v}")
        print()
        
