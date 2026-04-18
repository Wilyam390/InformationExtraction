"""
rule_classifier.py
==================
Stage 0: Rule-based document classifier using handcrafted regular-expression
patterns.

Each document is scored against keyword/pattern sets for every class.
The class with the highest number of matching patterns wins.

This is an intentionally simple baseline that:
  - Requires no training data and no learning
  - Encodes explicit domain knowledge about document structure
  - Demonstrates the limits of rule-based NLP vs. statistical ML

Usage
-----
    from src.rule_classifier import rule_based_classify, RULE_PATTERNS

    label = rule_based_classify(text)
    # → 'invoice' | 'email' | 'scientific_report' | 'letter'
"""

import re
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# PATTERN DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

RULE_PATTERNS: dict[str, list[str]] = {
    # ── Invoice ──────────────────────────────────────────────────────────────
    # Structural markers: billing keywords, numeric totals, payment terms
    "invoice": [
        r"\binvoice\b",
        r"\binv[-#]\w+",                          # INV-0042, INV#001
        r"total\s*(?:amount|due|payable)",
        r"billing\s*date",
        r"due\s*date",
        r"\bvat\b",
        r"amount\s*due",
        r"bill\s*to",
        r"net\s*\d+\s*days",
        r"(?:subtotal|grand\s*total)",
        r"(?:USD|EUR|GBP)\s*[\d,]+\.\d{2}",      # currency + amount
        r"tax\s*\(\d+\s*%\)",                     # Tax (20%)
        r"payment\s*terms",
        r"account\s*number",
        r"receipt\s*(?:number|no\.?|#)",
    ],

    # ── Email ─────────────────────────────────────────────────────────────────
    # Structural markers: From/To headers, subject line, email address patterns
    "email": [
        r"from:\s*\S+@\S+",
        r"to:\s*\S+@\S+",
        r"subject:",
        r"\bcc:\b",
        r"\bbcc:\b",
        r"dear\s+\w+,",
        r"(?:best|kind)\s*regards",
        r"sincerely\s*yours",
        r"reply[-\s]to",
        r"unsubscribe",
        r"sent\s*from\s*my",
        r"date:\s+\w+,\s+\d+\s+\w+\s+\d{4}",    # Date: Fri, 12 Apr 2024
    ],

    # ── Scientific Report ─────────────────────────────────────────────────────
    # Section headings, citation patterns, statistical notation
    "scientific_report": [
        r"\babstract\b",
        r"\bintroduction\b",
        r"\bmethodology\b",
        r"\bconclusion\b",
        r"et\s+al\.",
        r"p\s*[<>]\s*0\.0[0-9]",                 # p < 0.05
        r"\breferences\b",
        r"\bfigure\s+\d",
        r"\btable\s+\d",
        r"(?:accuracy|precision|recall|f1)[\s\-]*score",
        r"neural\s*network",
        r"\bdataset\b",
        r"baseline",
        r"(?:train|test)\s+(?:set|split)",
        r"proposed\s+(?:method|approach|framework)",
    ],

    # ── Letter ────────────────────────────────────────────────────────────────
    # Formal salutations, closings, and letter-specific phrasing
    "letter": [
        r"dear\s+(?:sir|madam|hiring)",
        r"yours\s+(?:sincerely|faithfully)",
        r"i\s+am\s+writing\s+to",
        r"hereby\s+certif",
        r"tenancy\s+agreement",
        r"offer\s+(?:of\s+)?employment",
        r"notice\s+period",
        r"formal\s+complaint",
        r"\blandlord\b",
        r"hiring\s+manager",
        r"i\s+look\s+forward",
        r"please\s+(?:find|do\s+not\s+hesitate)",
        r"ref(?:erence)?:\s*[A-Z0-9\-/]+",       # REF: GOV/2024/1234
    ],
}

# Default class when all scores are zero (least distinctive class)
_DEFAULT_CLASS = "letter"


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

def rule_based_classify(text: str) -> str:
    """
    Classify a document using keyword/regex pattern matching.

    Parameters
    ----------
    text : str
        Raw document text.

    Returns
    -------
    str
        Predicted label: 'invoice' | 'email' | 'scientific_report' | 'letter'
    """
    t = text.lower()
    scores: dict[str, int] = {
        label: sum(
            1 for pat in patterns
            if re.search(pat, t)
        )
        for label, patterns in RULE_PATTERNS.items()
    }
    best_label = max(scores, key=scores.get)

    # Tie-break: if all scores are 0, return default
    if scores[best_label] == 0:
        return _DEFAULT_CLASS

    return best_label


def rule_based_classify_with_scores(text: str) -> dict:
    """
    Like rule_based_classify but returns all scores for interpretability.

    Returns
    -------
    dict with:
        label       : predicted class
        scores      : dict {class: match_count}
        matched     : dict {class: list of matched pattern strings}
    """
    t = text.lower()
    scores   : dict[str, int]        = {}
    matched  : dict[str, list[str]]  = {}

    for label, patterns in RULE_PATTERNS.items():
        hits = [pat for pat in patterns if re.search(pat, t)]
        scores[label]  = len(hits)
        matched[label] = hits

    best_label = max(scores, key=scores.get)
    if scores[best_label] == 0:
        best_label = _DEFAULT_CLASS

    return {
        "label":   best_label,
        "scores":  scores,
        "matched": matched,
    }


# ─────────────────────────────────────────────────────────────────────────────
# QUICK SELF-TEST
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    samples = {
        "invoice": "INVOICE\nInvoice Number: INV-4821\nInvoice Date: 2024-03-15\nDue Date: 2024-04-15\nFrom: Acme Corp\nTo: John Smith\nTOTAL USD 4,200.00",
        "email":   "From: morgan@startup.io\nTo: all@startup.io\nSubject: Q3 Update\n\nHi team,\nBest regards,\nMorgan",
        "scientific_report": "Abstract\nThis paper proposes a novel deep learning framework. et al. Table 1 shows accuracy. References",
        "letter":  "Dear Hiring Manager,\nI am writing to express my interest in the Software Engineer position.\nYours sincerely,\nJohn",
    }

    print("Rule-Based Classifier Self-Test\n" + "─"*40)
    for true_label, text in samples.items():
        result = rule_based_classify_with_scores(text)
        pred   = result["label"]
        status = "✓" if pred == true_label else "✗"
        print(f"{status} True: {true_label:<20} Predicted: {pred}")
        print(f"   Scores: {result['scores']}")
        print()
