"""
predict.py
==========
End-to-end inference pipeline:
  1. Accept text input (or a file path: .txt / .pdf)
  2. Classify the document into one of four categories
  3. If classified as 'invoice', extract structured fields

Usage
-----
    # From Python
    from src.predict import predict

    result = predict("path/to/document.pdf")
    result = predict("raw text of a document...")

    # From CLI
    python3 src/predict.py "Invoice Number: INV-001 ..."
    python3 src/predict.py path/to/document.pdf
    python3 src/predict.py path/to/document.txt
"""

import sys
import json
import joblib
import re
from pathlib import Path
from scipy.sparse import hstack

ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"

# ── lazy-load model artefacts ─────────────────────────────────────────────────
_word_vec  = None
_char_vec  = None
_clf       = None
_meta      = None

def _load_models():
    global _word_vec, _char_vec, _clf, _meta
    if _clf is None:
        _word_vec = joblib.load(MODELS_DIR / "word_vectorizer.joblib")
        _char_vec = joblib.load(MODELS_DIR / "char_vectorizer.joblib")
        _clf      = joblib.load(MODELS_DIR / "classifier.joblib")
        with open(MODELS_DIR / "meta.json") as f:
            _meta = json.load(f)


# ── text extraction from files ────────────────────────────────────────────────

def _read_pdf(path: Path) -> str:
    """Extract text from PDF using pdfplumber."""
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        return "\n".join(text_parts)
    except ImportError:
        raise ImportError("pdfplumber is required to read PDF files: pip install pdfplumber")
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF {path}: {e}")


def _read_file(path: Path) -> str:
    """Extract text from .txt or .pdf files."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if path.suffix.lower() == ".pdf":
        return _read_pdf(path)
    else:
        return path.read_text(encoding="utf-8", errors="replace")


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# ── classify ──────────────────────────────────────────────────────────────────

def classify(text: str) -> dict:
    """
    Classify document text.

    Returns
    -------
    dict with:
        label       : predicted class string
        confidence  : dict {label: decision_function_score}
    """
    _load_models()
    clean_text = _clean(text)
    feat = hstack([
        _word_vec.transform([clean_text]),
        _char_vec.transform([clean_text]),
    ])
    label = _clf.predict(feat)[0]

    # Decision function scores (higher = more confident)
    scores = _clf.decision_function(feat)[0]
    labels = _clf.classes_
    confidence = {l: round(float(s), 3) for l, s in zip(labels, scores)}

    return {"label": label, "confidence": confidence}


# ── full pipeline ─────────────────────────────────────────────────────────────

def predict(input_: str) -> dict:
    """
    Full pipeline: read → classify → extract (if invoice).

    Parameters
    ----------
    input_ : str
        Either a file path (.txt, .pdf) or raw document text.

    Returns
    -------
    dict with:
        label       : predicted document type
        confidence  : per-class decision scores
        extraction  : dict of invoice fields (only if label == 'invoice')
        text_preview: first 300 chars of the input text
    """
    # Resolve input: treat as file path only if short enough and exists on disk
    text, source = input_, "<raw_text>"
    if len(input_) < 512:
        p = Path(input_)
        try:
            if p.exists() and p.is_file():
                text  = _read_file(p)
                source = str(p)
        except OSError:
            pass  # too long or invalid path — treat as raw text

    if not text.strip():
        return {"error": "Empty input — no text to classify."}

    # Classify
    cls_result = classify(text)
    label      = cls_result["label"]

    result = {
        "source":       source,
        "label":        label,
        "confidence":   cls_result["confidence"],
        "text_preview": text[:300].replace("\n", " "),
    }

    # Extract fields if invoice
    if label == "invoice":
        import importlib, sys as _sys
        # Support both `python3 src/predict.py` and `import src.predict`
        try:
            from src.extract import extract_invoice_fields
        except ModuleNotFoundError:
            import os as _os
            _sys.path.insert(0, str(ROOT / "src"))
            from extract import extract_invoice_fields  # type: ignore
        result["extraction"] = extract_invoice_fields(text)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _pretty_print(result: dict):
    label = result.get("label", "unknown")
    print(f"\n{'═'*55}")
    print(f"  Classification : {label.upper()}")

    # Confidence bar
    conf = result.get("confidence", {})
    best = result.get("label")
    print(f"  Scores         :")
    for lbl, score in sorted(conf.items(), key=lambda x: -x[1]):
        bar_len = max(0, int((score + 3) * 5))  # scale decision value
        bar     = "█" * min(bar_len, 30)
        marker  = " ◀" if lbl == best else ""
        print(f"    {lbl:<20} {score:+.3f}  {bar}{marker}")

    if "extraction" in result:
        print(f"\n  Extracted Invoice Fields:")
        print(f"  {'─'*40}")
        for k, v in result["extraction"].items():
            status = "✓" if v else "✗"
            print(f"  {status} {k:<20} {v or 'not found'}")

    print(f"\n  Preview: {result.get('text_preview','')[:120]}…")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 src/predict.py <text_or_file_path>")
        sys.exit(1)

    inp = " ".join(sys.argv[1:])
    res = predict(inp)

    if "error" in res:
        print(f"[ERROR] {res['error']}")
        sys.exit(1)

    _pretty_print(res)

    # Also dump JSON for piping
    print(json.dumps(res, indent=2, ensure_ascii=False))
