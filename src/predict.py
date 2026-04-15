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

# Minimum characters returned by pdfplumber before we consider the PDF
# "text-based".  Scanned PDFs often return 0-20 chars of garbage.
_PDF_TEXT_MIN_CHARS = 50

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

def _ocr_image(image) -> str:
    import pytesseract
    lang_arg = "eng"
    print("[OCR] Running Tesseract...", flush=True)
    text = pytesseract.image_to_string(
        image,
        lang=lang_arg,
        config="--oem 1 --psm 4",
        timeout=60,
    )
    print(f"[OCR] Tesseract finished (chars={len(text.strip())}).", flush=True)
    return text


def _print_extracted_text(source: str, text: str, tag: str = "OCR TEXT") -> None:
    """Print extracted text in terminal for debugging/inspection."""
    extracted = (text or "").strip()
    print("\n" + "=" * 80, flush=True)
    print(f"[{tag}] Source: {source}", flush=True)
    print("-" * 80, flush=True)
    print(extracted if extracted else "<empty OCR output>", flush=True)
    print("=" * 80 + "\n", flush=True)


def _ocr_pdf(path: Path) -> str:
    """
    Convert each PDF page to an image and run Tesseract OCR on it.
    Used as a fallback when pdfplumber finds no text layer (scanned PDF).
    Requires: pdf2image + poppler   (brew install poppler)
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is required for scanned PDFs: pip install pdf2image\n"
            "You also need poppler: brew install poppler"
        )

    pages = convert_from_path(str(path), dpi=300)
    text_parts = []
    for i, page_img in enumerate(pages, 1):
        # Optional: pre-process image for better OCR accuracy
        page_img = _preprocess_for_ocr(page_img)
        text = _ocr_image(page_img)
        _print_extracted_text(f"{path.name} | page {i}", text, tag="OCR TEXT")
        if text.strip():
            text_parts.append(text)

    return "\n\n".join(text_parts)


def _preprocess_for_ocr(image):
    from PIL import Image, ImageOps, ImageFilter
    image = image.convert("L")
    if image.width < 1000:
        scale = 1000 / image.width
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.LANCZOS
        )
    image = ImageOps.autocontrast(image, cutoff=2)
    image = image.filter(ImageFilter.SHARPEN)
    return image


def _read_image(path: Path) -> str:
    """Run Tesseract OCR on a standalone image file (PNG, JPG, TIFF, BMP)."""
    try:
        from PIL import Image
    except ImportError:
        raise ImportError("Pillow is required to read image files: pip install Pillow")
    img  = Image.open(path)
    img  = _preprocess_for_ocr(img)
    text = _ocr_image(img)
    _print_extracted_text(path.name, text, tag="OCR TEXT")
    return text


def _read_pdf(path: Path):
    """
    Extract text from a PDF.  Returns (text, ocr_used: bool).

    Strategy
    --------
    1. Try pdfplumber — fast, lossless for digital/typed PDFs.
    2. If fewer than _PDF_TEXT_MIN_CHARS are found (scanned / image-only PDF),
       fall back to pdf2image + Tesseract OCR.
    """
    # Step 1: digital text layer
    try:
        import pdfplumber
        text_parts = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        text = "\n".join(text_parts).strip()
    except ImportError:
        raise ImportError("pdfplumber is required: pip install pdfplumber")
    except Exception as e:
        raise RuntimeError(f"Failed to read PDF {path}: {e}")

    # Step 2: fallback to OCR if text layer is empty / too short
    if len(text) < _PDF_TEXT_MIN_CHARS:
        print(f"[OCR] No text layer in {path.name} — running Tesseract OCR…")
        text = _ocr_pdf(path)
        return text, True

    # Also print extracted text for text-based PDFs (not OCR fallback).
    _print_extracted_text(path.name, text, tag="PDF TEXT")
    return text, False


def _read_file(path: Path):
    """
    Dispatch to the correct reader.  Returns (text: str, ocr_used: bool).
    Supported formats: .txt, .pdf, .png, .jpg, .jpeg, .tiff, .tif, .bmp, .webp
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _read_pdf(path)   # already returns (text, ocr_used)
    elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
        print(f"[OCR] Image file — running Tesseract on {path.name}…")
        return _read_image(path), True
    else:
        return path.read_text(encoding="utf-8", errors="replace"), False


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
    # Resolve input: treat as file path only if short enough and exists on disk.
    # Do not swallow OCR/file-read errors, otherwise we may silently fall back
    # to classifying the filename string as raw text.
    text, source, ocr_used = input_, "<raw_text>", False
    if len(input_) < 512:
        p = Path(input_)
        try:
            is_file = p.exists() and p.is_file()
        except OSError:
            is_file = False  # invalid path syntax on this platform

        if is_file:
            text, ocr_used = _read_file(p)
            source = str(p)

    if not text.strip():
        return {"error": "Empty input — no text to classify."}

    # Classify
    cls_result = classify(text)
    label      = cls_result["label"]

    result = {
        "source":       source,
        "label":        label,
        "confidence":   cls_result["confidence"],
        "ocr_used":     ocr_used,
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
